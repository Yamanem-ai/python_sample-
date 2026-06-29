# voice_assistant.py

from sensevoice_api import listen
from gemma_api import ask

def main():

    print("Voice Assistant Ready")
    print("Speak to the microphone")
    print("Ctrl+C to quit")

    while True:
    
        print("")
        print("Listning...")

        user_text = listen()

        print("")
        print("You>")
        print(user_text)

        answer = ask(user_text)

        print("")
        print("Gemma>")
        print(answer)

if __name__ == "__main__":
    main()
