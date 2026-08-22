import ailia_speech
import sounddevice as sd
import time
import librosa
import numpy as np

# ===============================================
# Global
# ===============================================
_speech = None
_mic = None
_subtitle = ""

# ===============================================
# Initialize
# ===============================================
def initialize():

    global _speech
    global _mic

    if _speech is not None:
        return

    speech = ailia_speech.Whisper(
        task=ailia_speech.AILIA_SPEECH_TASK_TRANSCRIBE,
        flags=ailia_speech.AILIA_SPEECH_FLAG_LIVE,
        callback=intermediate_callback
    )
    
    #speech.set_silent_threshold(
    #    0.8,
    #    8,
    #    1
    #)

    speech.initialize_model(
        model_path="./models/",
        model_type=(
            ailia_speech.
            AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_SMALL
            #AILIA_SPEECH_MODEL_TYPE_SENSEVOICE_SMALL
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

    print("ailia Speech Whisper Ready")

#================
# Intermediate Callback
#================
def intermediate_callback(text):
    
    global _subtitle
    
    _subtitle = text
    
    #print(f"\r{text}", end="", flush=True)
    print("callback:", repr(text))
# ===============================================
# API
# ===============================================
def get_subtitle():

    return _subtitle
    
def listen():

    initialize()

    global _speech
    global _mic
    global _subtitle

    count = 0
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
            
            r = result[0]
            
            #最終結果で字幕を確定
            _subtitle = r["text"]

            return {
                "text": r["text"],
                "confidence": r["confidence"],
            }

# ===============================================
# Test
# ===============================================
if __name__ == "__main__":

    while True:

        result = listen()

        print("")
        print("Recognized:")
        print(f"Confidence: {result['confidence']}")
        print(f"Text    : {result['text']}")
        print("")

#============
# Audio file
#============
#if __name__ == "__main__":

#    initialize()

#    audio, sr = librosa.load("sample.wav", sr=16000, mono=True)
 
#    results = list(_speech.transcribe(audio, sr))

#    print()

#    for r in results:
#        print(f"Speaker : {r['speaker_id']}")
#        print(f"Begin   : {r['time_stamp_begin']:.2f}")
#        print(f"End     : {r['time_stamp_end']:.2f}")
#        print(f"Text    : {r['text']}")
#        print()
