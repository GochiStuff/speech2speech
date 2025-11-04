from TTS.api import TTS

# Load the model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Generate speech
tts.tts_to_file(text="Hello, this is a multilingual text to speech demo.", file_path="output.wav")
