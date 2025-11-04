import numpy as np
import sounddevice as sd
import resampy
import torch
import torch.hub
import time
import threading
import queue
import keyboard
import traceback
from faster_whisper import WhisperModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from TTS.api import TTS

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
SAMPLE_RATE = 16000
TARGET_LANG = "fr"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASR_MODEL_NAME = "small"
TRANSLATOR_MODEL = "facebook/m2m100_418M"
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SPEAKER_WAV_PATH = "./male.wav"  

# VAD
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 700
VAD_MAX_PHRASE_S = 7
VAD_SAMPLES_PER_CHUNK = 512

# ---------------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------------
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log(f"🚀 Using device={DEVICE}")

log("Loading ASR model...")
asr = WhisperModel(ASR_MODEL_NAME, device=DEVICE, compute_type="int8")

log("Loading Translator...")
tokenizer = M2M100Tokenizer.from_pretrained(TRANSLATOR_MODEL)
translator = M2M100ForConditionalGeneration.from_pretrained(TRANSLATOR_MODEL).to(DEVICE)

log("Loading TTS...")
tts = TTS(model_name=TTS_MODEL_NAME, gpu=torch.cuda.is_available())
sr_tts = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", 24000)

log("Loading VAD...")
vad_model, vad_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    onnx=False
)
(get_speech_timestamps, _, _, VADIterator, _) = vad_utils
vad_model = vad_model.to("cpu")

log("✅ All models loaded.")

# ---------------------------------------------------------------
# STATE
# ---------------------------------------------------------------
audio_queue = queue.Queue()
text_queue = queue.Queue()
playback_queue = queue.Queue()
stop_flag = threading.Event()
playback_buffer = np.zeros(0, dtype=np.float32)
stream = None

# ---------------------------------------------------------------
# AUDIO INPUT
# ---------------------------------------------------------------
def record_audio():
    def callback(indata, frames, time_info, status):
        if status:
            log(f"⚠️ Input status: {status}")
        if not stop_flag.is_set():
            audio_queue.put(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        log("🎙️ Listening...")
        stop_flag.wait()
        log("🎙️ Stopped listening.")

# ---------------------------------------------------------------
# ASR + TRANSLATE
# ---------------------------------------------------------------
def asr_translator_thread():
    global vad_model
    
    phrase_buffer = np.zeros(0, dtype=np.float32)
    buffer = np.zeros(0, dtype=np.float32)
    last_text = ""
    last_speech_time = time.time()
    
    log("🎤 Translation thread running...")

    while not stop_flag.is_set():
        try:
            while not audio_queue.empty():
                data = audio_queue.get()
                buffer = np.concatenate((buffer, data.flatten()))

            if len(buffer) < VAD_SAMPLES_PER_CHUNK:
                time.sleep(0.01)
                continue

            vad_chunk = buffer[:VAD_SAMPLES_PER_CHUNK]
            buffer = buffer[VAD_SAMPLES_PER_CHUNK:]
            speech_prob = vad_model(torch.from_numpy(vad_chunk), SAMPLE_RATE).item()
            is_speech = speech_prob > VAD_THRESHOLD

            if is_speech:
                phrase_buffer = np.concatenate((phrase_buffer, vad_chunk))
                last_speech_time = time.time()
            
            trigger_process = False
            if not is_speech and len(phrase_buffer) > 0:
                if time.time() - last_speech_time > (VAD_MIN_SILENCE_MS / 1000.0):
                    trigger_process = True

            if len(phrase_buffer) > VAD_MAX_PHRASE_S * SAMPLE_RATE:
                trigger_process = True
                log("⚠️ Long phrase forced processing.")

            if trigger_process and len(phrase_buffer) > 0:
                audio_to_process = phrase_buffer.copy()
                phrase_buffer = np.zeros(0, dtype=np.float32)
                
                if np.mean(np.abs(audio_to_process)) < 1e-4:
                    continue

                segments, info = asr.transcribe(audio_to_process, beam_size=1)
                text = " ".join([s.text for s in segments]).strip()
                
                if not text or text.lower() == last_text.lower():
                    continue
                last_text = text
                log(f"🗣️ {info.language}: {text}")

                try:
                    tokenizer.src_lang = info.language
                    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
                    bos = tokenizer.get_lang_id(TARGET_LANG)
                    outputs = translator.generate(**inputs, forced_bos_token_id=bos)
                    translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                    
                    log(f"💬 {TARGET_LANG}: {translated}")
                    text_queue.put(translated)

                except Exception as e:
                    log(f"⚠️ Translation failed: {e}")
            
        except Exception as e:
            log(f"💥 ASR/Translate error: {e}")
            traceback.print_exc()
            time.sleep(0.5)
            
    log("🛑 Translation thread stopped.")

# ---------------------------------------------------------------
# TTS GENERATOR
# ---------------------------------------------------------------
def tts_generator_thread():
    while not stop_flag.is_set():
        try:
            text = text_queue.get(timeout=0.1)
            wav = tts.tts(text=text, language=TARGET_LANG, speaker_wav=SPEAKER_WAV_PATH)
            wav = np.array(wav, dtype=np.float32).flatten()
            
            if sr_tts != SAMPLE_RATE:
                wav = resampy.resample(wav, sr_tts, SAMPLE_RATE)
                
            wav_max = np.max(np.abs(wav))
            if wav_max > 0:
                wav /= wav_max
            
            wav_padded = np.concatenate([wav, np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)])
            playback_queue.put(wav_padded)

        except queue.Empty:
            continue
        except Exception as e:
            log(f"❌ TTS failed: {e}")
            traceback.print_exc()
    log("🛑 TTS thread stopped.")

# ---------------------------------------------------------------
# AUDIO OUTPUT
# ---------------------------------------------------------------
def playback_callback(outdata, frames, time_info, status):
    global playback_buffer
    if status:
        log(f"⚠️ Output status: {status}")

    while not playback_queue.empty():
        new_wav = playback_queue.get()
        playback_buffer = np.concatenate((playback_buffer, new_wav))

    chunk_len = min(len(playback_buffer), frames)
    
    if chunk_len > 0:
        outdata[:chunk_len] = playback_buffer[:chunk_len].reshape(-1, 1)
        playback_buffer = playback_buffer[chunk_len:]
    
    if chunk_len < frames:
        outdata[chunk_len:] = 0.0

# ---------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------
def main():
    log("🎧 Realtime Speech ↔ Speech Translator")
    log("Press SPACE to start/stop, ESC to exit")
    
    global stream, playback_buffer
    is_recording = False

    try:
        while True:
            if keyboard.is_pressed("esc"):
                log("🛑 Exiting...")
                stop_flag.set()
                break

            if keyboard.is_pressed("space"):
                if not is_recording:
                    is_recording = True
                    stop_flag.clear()
                    for q in [audio_queue, text_queue, playback_queue]:
                        q.queue.clear()
                    playback_buffer = np.zeros(0, dtype=np.float32)

                    threading.Thread(target=record_audio, daemon=True).start()
                    threading.Thread(target=asr_translator_thread, daemon=True).start()
                    threading.Thread(target=tts_generator_thread, daemon=True).start()
                    
                    stream = sd.OutputStream(
                        samplerate=SAMPLE_RATE, 
                        channels=1, 
                        dtype="float32", 
                        callback=playback_callback
                    )
                    stream.start()
                    log("✅ Session STARTED.")
                    
                else:
                    is_recording = False
                    log("🧹 Stopping session...")
                    stop_flag.set()
                    
                    if stream:
                        stream.stop()
                        stream.close()
                        stream = None
                    
                    sd.stop()
                    log("🛑 Session STOPPED.")
                time.sleep(0.5)
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        log("🛑 Interrupted.")
    finally:
        stop_flag.set()
        if stream:
            stream.stop()
            stream.close()
        sd.stop()
        log("Cleanup complete. Exiting.")

if __name__ == "__main__":
    main()
