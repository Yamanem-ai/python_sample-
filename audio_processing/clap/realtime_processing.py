import tkinter as tk
import threading

from whisper_api import initialize
from whisper_api import listen
from whisper_api import get_subtitle



class RealtimeProcessingGUI:

    def __init__(self):

        self.root = tk.Tk()

        self.root.title("Whisper Realtime")
        self.root.geometry("700x400")

        self.status = tk.StringVar(value="Ready")
        self.subtitle = tk.StringVar(value="")

        self.button = tk.Button(
            self.root,
            text="🎤 START",
            font=("Arial", 20),
            width=10,
            height=3,
            command=self.start_voice
        )
        
        self.button.pack(pady=20)
        
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status,
            font=("Arial", 12)
        )
        
        self.status_label.pack()
        
        tk.Label(
            self.root,
            text="Subtitle",
            font=("Arial", 14, "bold")
        ).pack(pady=(20, 5))
        
        self.subtitle_label = tk.Label(
            self.root,
            textvariable=self.subtitle,
            font=("Arial", 16),
            wraplength=650,
            justify="left",
            anchor="w"
        )
        self.subtitle_label.pack(fill="x", padx=20)
        
        self.exit_button = tk.Button(
            self.root,
            text="EXIT",
            font=("Arial", 12),
            command=self.quit_app
        )
        
        self.exit_button.pack(pady=20)
        
        # 50msごとに字幕更新
        self.update_subtitle()

    def update_subtitle(self):
        
        self.subtitle.set(get_subtitle())
        
        self.root.after(
            50,
            self.update_subtitle
        )

    def start_voice(self):

        self.button.config(state="disabled")

        threading.Thread(
            target=self.voice_loop,
            daemon=True
        ).start()

    def voice_loop(self):

        try:

            self.status.set("Listening")

            result = listen()
            
            print()
            print("Recognized")
            print(result)

            self.status.set("Finished")
            
        finally:
        
            self.button.config(state="normal")
            
            self.status.set("Ready")

    def run(self):

        self.root.mainloop()

    def quit_app(self):
        print("Application terminated.")
        
        self.root.destroy()

if __name__ == "__main__":

    print("Initializing...")
    
    initialize()
    
    app = RealtimeProcessingGUI()

    app.run()
