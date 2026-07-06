
import sys
import time
from logging import getLogger

from funasr_ailia import SenseVoiceSmall
from funasr_ailia.utils.postprocess_utils import rich_transcription_postprocess
from funasr_ailia import Fsmn_vad_online

import soundfile
import librosa

import numpy as np

# import original modules
sys.path.append("../../util")

from model_utils import check_and_download_models, check_and_download_file  # noqa
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WAV_PATH = "ja.wav"
SAVE_TEXT_PATH = "output.txt"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("SenseVoice", WAV_PATH, SAVE_TEXT_PATH, input_ftype="audio", fp16_support = False)
parser.add_argument(
	"-V",
	action="store_true",
	help="use microphone input",
)
parser.add_argument(
	"--disable_ailia_audio", action="store_true", help="disable ailia_audio. (Require kaldi_native_fbank)"
)
parser.add_argument(
	"--disable_ailia_tokenizer", action="store_true", help="disable ailia_tokenizer. (Require sentencepiece)"
)
parser.add_argument(
	"--fp16", action="store_true", help="use fp16 model (default : fp32 model)."
)
parser.add_argument(
	"--onnx", action="store_true", help="use onnx runtime."
)
args = update_parser(parser)

# ======================
# Models
# ======================

if args.fp16:
	WEIGHT_PATH = "sensevoice_small_fp16.onnx"
	VAD_WEIGHT_PATH = "speech_fsmn_vad_zh-cn-16k-common_fp16.onnx"
else:
	WEIGHT_PATH = "sensevoice_small.onnx"
	VAD_WEIGHT_PATH = "speech_fsmn_vad_zh-cn-16k-common.onnx"

MODEL_PATH = WEIGHT_PATH + ".prototxt"
VAD_MODEL_PATH = VAD_WEIGHT_PATH + ".prototxt"

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/sensevoice/"

def format_timestamp(seconds: float, always_include_hours: bool = False):
	assert seconds >= 0, "non-negative timestamp expected"
	milliseconds = round(seconds * 1000.0)

	hours = milliseconds // 3_600_000
	milliseconds -= hours * 3_600_000

	minutes = milliseconds // 60_000
	milliseconds -= minutes * 60_000

	seconds = milliseconds // 1_000
	milliseconds -= seconds * 1_000

	hours_marker = f"{hours}:" if always_include_hours or hours > 0 else ""

	return f"{hours_marker}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def recognize_from_mic():
	# input audio loop
	logger.info("Start inference...")

	model = SenseVoiceSmall(env_id=args.env_id, onnx=args.onnx, ailia_audio=not args.disable_ailia_audio, ailia_tokenizer=not args.disable_ailia_tokenizer, profile=args.profile, model_file=WEIGHT_PATH)
	vad = Fsmn_vad_online(env_id=args.env_id, onnx=args.onnx, ailia_audio=not args.disable_ailia_audio, profile=args.profile, model_file=VAD_WEIGHT_PATH)

	import sounddevice

	devices = sounddevice.query_devices()
	if len(devices) == 0:
		logger.error("No microphone devices found")
		return

	#logger.info(devices)
	default_input_device_idx = sounddevice.default.device[0]
	logger.info(f'Input device: {devices[default_input_device_idx]["name"]}')

	param_dict = {"in_cache": []}
	param_dict["is_final"] = False

	start = -1
	end = -1

	samples_per_read = int(0.1 * 16000)
	target_sr = 16000
	samples = np.zeros((0))
	
	logger.info("Waiting microphone input...")

	try:
		with sounddevice.InputStream(channels=1, dtype="float32", samplerate=16000) as mic:
			while True:
				speech, _ = mic.read(samples_per_read)
				speech = speech[:, 0]
				samples = np.concatenate([samples, speech])
				segments_result = vad(
					audio_in=speech, param_dict=param_dict
				)
				for segment in segments_result:
					for s in segment:
						if s[0] != -1:
							start = s[0]
						if s[1] != -1:
							end = s[1]
						if start != -1 and end != -1:
							start_int = int(start / 1000 * target_sr)
							end_int = int(end / 1000 * target_sr)
							audio = samples[start_int:end_int]

							res = model(audio, language="auto", use_itn=True)
							
							for i in res:
								print("[" + format_timestamp(start / 1000) + " --> " + format_timestamp(end / 1000) + "] " + rich_transcription_postprocess(i))

							start = -1
							end = -1
	except KeyboardInterrupt:
		pass

	logger.info("Script finished successfully.")
 
#===============================================
# Global
#===============================================
_model = None
_vad = None
_mic = None

def initialize():

    global _model
    global _vad
    global _mic

    if _model is not None:
        return

    check_and_download_models(
        WEIGHT_PATH,
        MODEL_PATH,
        REMOTE_PATH
    )

    check_and_download_models(
        VAD_WEIGHT_PATH,
        VAD_MODEL_PATH,
        REMOTE_PATH
    )

    _model = SenseVoiceSmall(
        env_id=args.env_id,
        onnx=args.onnx,
        ailia_audio=not args.disable_ailia_audio,
        ailia_tokenizer=not args.disable_ailia_tokenizer,
        profile=args.profile,
        model_file=WEIGHT_PATH
    )

    _vad = Fsmn_vad_online(
        env_id=args.env_id,
        onnx=args.onnx,
        ailia_audio=not args.disable_ailia_audio,
        profile=args.profile,
        model_file=VAD_WEIGHT_PATH
    )
    
    if _mic is None:
    
        import sounddevice
    
        _mic = sounddevice.InputStream(
            channels=1,
            dtype="float32",
            samplerate=16000
        )
     
        _mic.start()

#今回の修正
    devices = sounddevice.query_devices()
    
    default_input_device_idx = (
        sounddevice.default.device[0]
    )
    
    logger.info(
        f'Input device: '
        f'{devices[default_input_device_idx]["name"]}'
    )

    print("SenseVoice Ready")
    
#=======================================
# API
#=======================================
def listen():
    
    initialize()
    
    global _model
    global _vad
    global _mic
    
#    logger.info("Start inference...")

#    import sounddevice

#    devices = sounddevice.query_devices()
#    if len(devices) == 0:
#        logger.error("No microphone devices found")
#        return

#    #logger.info(devices)
#    default_input_device_idx = sounddevice.default.device[0]
#    logger.info(f'Input device: {devices[default_input_device_idx]["name"]}')

    param_dict = {"in_cache": []}
    param_dict["is_final"] = False

    start = -1
    end = -1

    samples_per_read = int(0.1 * 16000)
    target_sr = 16000
    samples = np.zeros((0))
    
    logger.info("Waiting microphone input...")
    try:
        while True:
            speech, _ = _mic.read(samples_per_read)
            speech = speech[:, 0]
            samples = np.concatenate([samples, speech])
            segments_result = _vad(
                audio_in=speech, param_dict=param_dict
            )
            for segment in segments_result:
                for s in segment:
                    if s[0] != -1:
                        start = s[0]
                    if s[1] != -1:
                        end = s[1]
                    if start != -1 and end != -1:
                        start_int = int(start / 1000 * target_sr)
                        end_int = int(end / 1000 * target_sr)
                        audio = samples[start_int:end_int]

                        t0 = time.time()
                        res = _model(audio, language="auto", use_itn=True)
                        print(
                            f"SenseVoice inference * "
                            f"{time.time()-t0:.2f}s"
                        )
                            
                        for i in res:
                            text = rich_transcription_postprocess(i)
                                
                            if text.strip():
                                return text

                        start = -1
                        end = -1
    except KeyboardInterrupt:
        pass

    logger.info("Script finished successfully.")
 
def main():
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    check_and_download_models(VAD_WEIGHT_PATH, VAD_MODEL_PATH, REMOTE_PATH)
 
    recognize_from_mic()

if __name__ == "__main__":

    print("Speak now...")
    
    text = listen()
    
    print("")
    print("Recognized:")
    print(text)
