import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

import speech_recognition as sr
import pyttsx3

listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# For speak
def talk(text):
    engine.say(text)
    engine.runAndWait()


# For speech recognize
def get_command():
    command = ""
    try:
        with sr.Microphone() as source:
            print(f"\nNow Listening...")
            listener.adjust_for_ambient_noise(source)
            voice = listener.listen(source)
            print(f"Listen Over...")
            command = listener.recognize_google(voice)
            command = command.lower()

    except:
        pass

    return command


if __name__ == "__main__":

    # Open json file for get response
    with open('data.json', 'r') as json_data:
        intents = json.load(json_data)

    # Load saved model
    FILE = "data.pth"
    data = torch.load(FILE)

    # Get details from saved model
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    # Initialize model
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    bot_name = "Bot"
    print("Let's chat! (type 'EXIT' to exit)")

    while True:

        sentence = get_command()
        print(f"You:  {sentence}")

        if sentence == "exit":
            break

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents['intents']:
                lower_tag = intent["tag"].lower()

                if tag == lower_tag:
                    response = random.choice(intent['responses'])

                    print(f"{bot_name}: {response}")
                    talk(response)  # Speak response
                    break
        else:
            print(f"{bot_name}: I do not understand...")
