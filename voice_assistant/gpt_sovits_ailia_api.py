import ailia_voice

import librosa
import time
import soundfile
import sounddevice as sd
import numpy as np
import time

import os
import urllib.request

_voice = None

# Load reference audio
ref_text = "水をマレーシアから買わなくてはならない。"
ref_file_path = "reference_audio_captured_by_ax.wav"
if not os.path.exists(ref_file_path):
	urllib.request.urlretrieve(
		"https://github.com/ailia-ai/ailia-models/raw/refs/heads/master/audio_processing/gpt-sovits/reference_audio_captured_by_ax.wav",
		"reference_audio_girl.wav"
	)
audio_waveform, sampling_rate = librosa.load(ref_file_path, mono=True)

def initialize():

    global _voice
    
    if _voice is not None:
        return
        
    audio_waveform, sampling_rate = (
        librosa.load(
            ref_file_path,
            mono=True
        )
    )

    voice = ailia_voice.GPTSoVITS()
    voice.initialize_model(model_path = "./models/")
    voice.set_reference_audio(ref_text, ailia_voice.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA, audio_waveform, sampling_rate)
    
    _voice = voice

    print("ailia Voice GPT-SoVITS Ready")

def tts(text):

    global _voice
    
    initialize()
    
    t0 = time.time()
    buf, sampling_rate = (
        _voice.synthesize_voice(
            text,
            ailia_voice.
            AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA
        )
    )

    print(
        f"GPT-SoVITS inference : "
        f"{time.time()-t0:.2f}s"
    )

    sd.play(
        np.asarray(buf),
        samplerate=sampling_rate
    )

    sd.wait()

    return buf

if __name__ == "__main__":

    initialize()

    text = input("Text > ")

    tts(text)
