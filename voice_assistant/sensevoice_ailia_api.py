import ailia_speech
import sounddevice as sd
import time
# ===============================================
# Global
# ===============================================
_speech = None
_mic = None

# ===============================================
# Initialize
# ===============================================
def initialize():

    global _speech
    global _mic

    if _speech is not None:
        return

    speech = ailia_speech.SenseVoice()

    speech.initialize_model(
        model_path="./models/",
        model_type=(
            ailia_speech.
            AILIA_SPEECH_MODEL_TYPE_SENSEVOICE_SMALL
        ),
        vad_version="6_2"
    )

    _speech = speech

    _mic = sd.InputStream(
        channels=1,
        dtype="float32",
        samplerate=16000
    )

    _mic.start()

    print("ailia Speech SenseVoice Ready")

# ===============================================
# API
# ===============================================
def listen():

    initialize()

    global _speech
    global _mic

    while True:

        chunk, _ = _mic.read(1600)
        result = list(
            _speech.transcribe_step(
                chunk[:, 0],
                16000,
                False
            )
        )

        if len(result) > 0:

            text = result[0]["text"]

            if text.strip():

                #print(
                #    f"SenseVoice : {text}"
                #)

                return text

# ===============================================
# Test
# ===============================================
if __name__ == "__main__":

    while True:

        text = listen()

        print("")
        print("Recognized:")
        print(text)
        print("")
