import tkinter as tk
import threading

from sensevoice_api import initialize as stt_initialize
from sensevoice_api import listen
from gemma_api import initialize as llm_initialize
from gemma_api import ask
from gpt_sovits_ailia_api import initialize as tts_initialize
from gpt_sovits_ailia_api import tts


class VoiceAssistantGUI:

    def __init__(self):

        self.root = tk.Tk()

        self.root.title("Voice Assistant")
        self.root.geometry("400x400")

        self.status = tk.StringVar()
        self.status.set("Ready")

        self.button = tk.Button(
            self.root,
            text="🎤 TALK",
            font=("Arial", 20),
            width=10,
            height=3,
            command=self.start_voice
        )

#        self.button.pack(expand=True)
        self.button.pack(pady=30)
        
        self.exit_button = tk.Button(
            self.root,
            text="EXIT",
            font=("Arial", 12),
            command=self.quit_app
        )
        
        self.exit_button.pack(pady=10)

        self.status_label = tk.Label(
            self.root,
            textvariable=self.status
        )

        self.status_label.pack(pady=10)

    def start_voice(self):

        self.button.config(state="disabled")

        threading.Thread(
            target=self.voice_loop,
            daemon=True
        ).start()

    def voice_loop(self):

        try:

            self.status.set("Listening")

            user_text = listen()
            
            print("")
            print("You>")
            print(user_text)

            self.status.set("Thinking")

            answer = ask(user_text)
            
            print("")
            print("Gemma>")
            print(answer)

            self.status.set("Speaking")

            tts(answer)

            self.status.set("Ready")

        finally:

            self.button.config(state="normal")

    def run(self):

        self.root.mainloop()

    def quit_app(self):
        print("Voice Assistant terminated.")
        
        self.root.destroy()

if __name__ == "__main__":

    print("Initializing...")
    
    stt_initialize()
    llm_initialize()
    tts_initialize()
    
    app = VoiceAssistantGUI()

    app.run()
