import tkinter as tk
from clap_ex import (
    initialize_clap,
    authenticate_whistle
)

#========================
#CLAP initialize
#========================
models = initialize_clap()
net_audio = models["net_audio"]

#=========================
#GUI
#=========================
root = tk.Tk()
root.title("CLAP SMART LOCK")

root.geometry("800x600")

status_label = tk.Label(
    root,
    text="WAITING...",
    font=("Arial", 32)
)
status_label.pack(pady=40)

result_label = tk.Label(
    root,
    text="",
    font=("Arial", 64, "bold")
)
result_label.pack(pady=40)

score_label = tk.Label(
    root,
    text="",
    font=("Arial", 24)
)
score_label.pack(pady=20)

#=========================
# Authentication callback
#=========================

def authenticate():

    status_label.config(
        text="Talking NOW..."
    )

    root.update()

    success, score = authenticate_whistle(
        net_audio,
        "open.m4a"
    )

    if success:
        root.configure(bg="green")
        status_label.config(
            text="owner_recognized",
            bg="green"
        )

        result_label.config(
            text="KEY_OPEN",
            bg="green"
        )

    else:
        root.configure(bg="red")
        status_label.config(
            text="unrecognized_person",
            bg="red"
        )

        result_label.config(
            text="FAILURE",
            bg="red"
        )

    score_label.config(
        text=f"similarity = {score:.3f}",
        bg=root["bg"]
    )
#=======================
# BUTTON Area
#=======================
button = tk.Button(
    root,
    text="START AUTH",
    font=("Arial", 28),
    command=authenticate
)

button.pack(pady=40)

exit_button = tk.Button(
    root,
    text="EXIT",
    font=("Arial", 24),
    command=root.destroy
)

exit_button.pack(pady=20)

#=======================
# Main loop
#=======================

root.mainloop()
