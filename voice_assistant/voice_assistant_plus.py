# voice_assistant_plus.py

import time
from sensevoice_api import initialize as stt_initialize
from sensevoice_api import listen
from gemma_api import initialize as llm_initialize
from gemma_api import ask
from bert_vits2_api import initialize as tts_initialize
from bert_vits2_api import tts

def main():

    print("Initializing...")
    
    stt_initialize()
    llm_initialize()
    tts_initialize()
    
    print("")
    print("Voice Assistant Ready")
    print("Speak to the microphone")
    print("Ctrl+C to quit")

    while True:
    
        print("")
        print("Listning...")

        user_text = listen()
        
        if not user_text:
            continue

        print("")
        print("You>")
        print(user_text)

        answer = ask(user_text)
        
        print("")
        print("Gemma>")
        print(answer)
        
        tts(answer)

if __name__ == "__main__":
    main()
