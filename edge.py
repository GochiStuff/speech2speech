#!/usr/bin/env python3

"""
Real-time Speech-to-Speech Translation Pipeline.

This script captures audio from the microphone, detects speech, transcribes it,
translates it to a target language, synthesizes the translation back to speech,
and plays it back with minimal latency.

It uses a pre-buffering and robust decoding approach (via ffmpeg) to handle
various TTS output formats and ensure smooth playback.

Requirements: ffmpeg on PATH, and Python packages:
pip install edge-tts faster-whisper transformers sounddevice numpy torch
"""

import numpy as np
import sounddevice as sd
import torch
import torch.hub
import time
import threading
import queue
import keyboard
import traceback
import asyncio
import edge_tts
from faster_whisper import WhisperModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

import shutil
import subprocess

# --------------------------
# --- CONFIGURATION
# --------------------------

SAMPLE_RATE = 16000
TARGET_LANG = "fr"  # Target language code (e.g., "fr", "es", "de")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ASR (Whisper) model
# "base" is faster than "small". Use "tiny" for even more speed, but less accuracy.
ASR_MODEL_NAME = "base"

# Translation model
TRANSLATOR_MODEL = "facebook/m2m100_418M"

# TTS (Edge-TTS) voice
TTS_VOICE = "fr-FR-HenriNeural"  # Example for French
TTS_FORMAT = "raw-16khz-16bit-mono-pcm"

# How many seconds of audio to buffer for playback
RING_SECONDS = 10

# This setting is present in the original code but not actively used.
PREBUFFER_SECS = 0.12

# --- VAD (Voice Activity Detection) ---
VAD_THRESHOLD = 0.5  # Confidence threshold for speech
# LOWERED for faster response: Milliseconds of silence to trigger end-of-phrase
VAD_MIN_SILENCE_MS = 400
VAD_MAX_PHRASE_S = 7  # Maximum length of a phrase in seconds
VAD_SAMPLES_PER_CHUNK = 512  # Audio chunk size for VAD processing

# --- Audio Device ---
SD_BLOCKSIZE = 512  # Frames per audio callback.


# --------------------------
# --- LOGGING
# --------------------------

def log(msg: str):
    """Prints a message with a timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------
# --- AUDIO DECODING
# --------------------------

def ffmpeg_decode_to_pcm_s16le(input_bytes: bytes, rate: int = SAMPLE_RATE) -> bytes:
    """
    Decodes any audio format (from bytes) into raw 16-bit PCM.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise FileNotFoundError("ffmpeg binary not found on PATH. Please install ffmpeg.")

    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",  # Input from stdin
        "-f", "s16le",  # Format: signed 16-bit little-endian
        "-ar", str(rate),  # Sample rate
        "-ac", "1",  # Mono channel
        "pipe:1",  # Output to stdout
    ]

    proc = subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if proc.returncode != 0:
        err_msg = proc.stderr.decode('utf-8', errors='ignore')
        raise RuntimeError(f"ffmpeg decode failed. Error: {err_msg}")

    return proc.stdout


# --------------------------
# --- RING BUFFER
# --------------------------

class RingBuffer:
    """
    A thread-safe, overwriting ring buffer for audio (float32 samples).
    """

    def __init__(self, capacity_samples: int):
        self.capacity = int(capacity_samples)
        self.buf = np.zeros(self.capacity, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.available = 0  # Samples available to read
        self.lock = threading.Lock()

    def write(self, data: np.ndarray):
        """
        Write data to the buffer.
        
        If the buffer is full, it overwrites the *oldest* data.
        """
        with self.lock:
            n = data.shape[0]

            if n >= self.capacity:
                data = data[-self.capacity:]
                n = data.shape[0]

            free_space = self.capacity - self.available

            if n > free_space:
                overflow = n - free_space
                self.read_pos = (self.read_pos + overflow) % self.capacity
                self.available = self.capacity - n

            end = self.write_pos + n
            if end <= self.capacity:
                self.buf[self.write_pos:end] = data
            else:
                first_part_len = self.capacity - self.write_pos
                self.buf[self.write_pos:] = data[:first_part_len]

                second_part_len = end % self.capacity
                self.buf[:second_part_len] = data[first_part_len:]

            self.write_pos = (self.write_pos + n) % self.capacity
            self.available = min(self.capacity, self.available + n)

    def read(self, n: int) -> np.ndarray:
        """
        Read up to 'n' samples from the buffer.
        
        Returns an array of size 'n', zero-padded if silent.
        """
        with self.lock:
            samples_to_read = min(n, self.available)
            out = np.zeros(n, dtype=np.float32)

            if samples_to_read > 0:
                end = self.read_pos + samples_to_read
                if end <= self.capacity:
                    out[:samples_to_read] = self.buf[self.read_pos:end]
                else:
                    first_part_len = self.capacity - self.read_pos
                    out[:first_part_len] = self.buf[self.read_pos:]

                    second_part_len = end % self.capacity
                    out[first_part_len:samples_to_read] = self.buf[:second_part_len]

                self.read_pos = (self.read_pos + samples_to_read) % self.capacity
                self.available -= samples_to_read

            return out

    def get_available(self):
        with self.lock:
            return self.available

    def get_free_space(self):
        with self.lock:
            return self.capacity - self.available

# --------------------------
# --- GLOBAL STATE
# --------------------------

ring = RingBuffer(int(RING_SECONDS * SAMPLE_RATE))
audio_queue = queue.Queue()
text_queue = queue.Queue()
stop_flag = threading.Event()
stream = None

# --------------------------
# --- MODEL LOADING
# --------------------------

log(f"🚀 Using device: {DEVICE}")
log(f"Loading ASR model ({ASR_MODEL_NAME})...")
# You can also experiment with compute_type="float16" if you have a good GPU
asr = WhisperModel(ASR_MODEL_NAME, device=DEVICE, compute_type="int8")

log("Loading translation model (M2M100)...")
tokenizer = M2M100Tokenizer.from_pretrained(TRANSLATOR_MODEL)
translator = M2M100ForConditionalGeneration.from_pretrained(TRANSLATOR_MODEL).to(DEVICE)

log("Loading VAD model (Silero)...")
vad_model, vad_utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False, onnx=False)
(get_speech_timestamps, _, _, VADIterator, _) = vad_utils
vad_model = vad_model.to("cpu")

log("✅ All models loaded and ready.")

# --------------------------
# --- AUDIO INPUT THREAD
# --------------------------

def record_audio():
    """
    Continuously captures audio from the microphone and puts it into
    the 'audio_queue'.
    """
    def callback(indata, frames, time_info, status):
        if status:
            log(f"⚠️ Microphone input status: {status}")
        if not stop_flag.is_set():
            audio_queue.put(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        log("🎙️ Listening... (Press SPACE to stop)")
        stop_flag.wait()

    log("🎙️ Stopped listening.")


# --------------------------
# --- ASR & TRANSLATE THREAD
# --------------------------

def asr_translator_thread():
    """
    Pulls audio from 'audio_queue', runs VAD, ASR, and Translation.
    Puts translated text into 'text_queue'.
    """
    global vad_model

    phrase_buffer = np.zeros(0, dtype=np.float32)
    buffer = np.zeros(0, dtype=np.float32)
    last_text = ""
    last_speech_time = time.time()

    log("🎤 ASR/Translation thread started.")

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
                log("⚠️ Phrase buffer full. Forcing translation.")

            if trigger_process and len(phrase_buffer) > 0:
                audio_to_process = phrase_buffer.copy()
                phrase_buffer = np.zeros(0, dtype=np.float32)

                if np.mean(np.abs(audio_to_process)) < 1e-4:
                    continue

                t_start = time.time()
                segments, info = asr.transcribe(audio_to_process, beam_size=1)
                text = " ".join([s.text for s in segments]).strip()
                t_asr = time.time() - t_start

                if not text or text.lower() == last_text.lower():
                    continue

                last_text = text
                log(f"🗣️  [ASR in {t_asr:.2f}s] ({info.language}): {text}")

                t_start = time.time()
                try:
                    tokenizer.src_lang = info.language
                    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
                    bos = tokenizer.get_lang_id(TARGET_LANG)

                    outputs = translator.generate(**inputs, forced_bos_token_id=bos)
                    translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

                    t_trans = time.time() - t_start
                    log(f"💬  [Translate in {t_trans:.2f}s] ({TARGET_LANG}): {translated}")

                    text_queue.put(translated)

                except Exception as e:
                    log(f"⚠️ Translation step failed: {e}")

        except Exception as e:
            log(f"💥 Error in ASR/Translate loop: {e}")
            traceback.print_exc()
            time.sleep(0.5)

    log("🛑 ASR/Translation thread stopped.")

# --------------------------
# --- TTS STREAMER
# --------------------------

async def safe_edge_tts_collect(text: str) -> bytes:
    """
    Collect the full audio bytes for one utterance from edge-tts.
    
    This simplified version uses the one reliable method that works
    with the current edge-tts API, removing the noisy errors.
    """
    try:
        # This is the "third attempt" from the old code, which works.
        comm = edge_tts.Communicate(text, TTS_VOICE)
        bytes_acc = bytearray()

        async for chunk in comm.stream():
            if isinstance(chunk, dict) and chunk.get("type") == "audio" and chunk.get("data"):
                bytes_acc.extend(chunk["data"])

        if bytes_acc:
            return bytes(bytes_acc)
        else:
            log(f"⚠️ TTS returned no audio bytes for text: {text[:50]}...")
            return b"" # Return empty bytes

    except Exception as e:
        log(f"❌ All edge-tts streaming attempts failed for text: {text[:50]}...")
        log(f"   Error: {e}")
        # Re-raise the exception so the outer loop can handle it
        raise e


# --------------------------
# --- TTS GENERATOR THREAD
# --------------------------

async def amain_tts_generator():
    """
    Pulls translated text from 'text_queue', generates audio,
    decodes it, and writes it to the 'ring' buffer.
    """
    log("▶️ Async TTS Generator started.")

    while not stop_flag.is_set():
        try:
            try:
                text = await asyncio.to_thread(text_queue.get, timeout=0.5)
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

            t_start = time.time()
            log(f"Synthesizing: \"{text[:60]}...\"")

            # 2. Generate audio bytes
            raw_bytes = await safe_edge_tts_collect(text)

            if not raw_bytes:
                log("⚠️ TTS returned no audio bytes.")
                continue

            t_tts = time.time()
            log(f"   ... TTS generation complete ({len(raw_bytes)} bytes) in {t_tts - t_start:.2f}s")

            # 3. Decode bytes using ffmpeg
            try:
                pcm = await asyncio.to_thread(ffmpeg_decode_to_pcm_s16le, raw_bytes, SAMPLE_RATE)
            except Exception as e:
                log(f"❌ ffmpeg decode failed for: \"{text[:50]}...\"")
                log(f"   Error: {e}")
                traceback.print_exc()
                continue

            t_decode = time.time()
            log(f"   ... ffmpeg decode complete in {t_decode - t_tts:.2f}s")

            # 4. Convert to float32 and write to ring buffer
            arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

            fade_len = int(SAMPLE_RATE * 0.01)  # 10ms
            if arr.size > 2 * fade_len:
                arr[:fade_len] *= np.linspace(0.0, 1.0, fade_len)
                arr[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)

            ring.write(arr)
            log(f"🎧 Wrote {arr.shape[0] / SAMPLE_RATE:.2f}s of audio to playback buffer.")

        except Exception as e:
            log(f"❌ Unexpected error in TTS generator loop: {e}")
            traceback.print_exc()
            await asyncio.sleep(0.2)

    log("🛑 Async TTS Generator stopped.")


def tts_generator_thread():
    """Wrapper to run the async TTS generator in a dedicated thread."""
    try:
        asyncio.run(amain_tts_generator())
    except Exception as e:
        log(f"💥 TTS Generator Thread CRASHED: {e}")
        traceback.print_exc()


# --------------------------
# --- PLAYBACK CALLBACK
# --------------------------

def playback_callback(outdata, frames, time_info, status):
    """
    The audio output callback. Reads from the 'ring' buffer.
    """
    global ring
    if status:
        log(f"⚠️ Playback stream status: {status}")

    data = ring.read(frames)
    outdata[:] = data.reshape(-1, 1)


# --------------------------
# --- MAIN CONTROL LOOP
# --------------------------

def main():
    log("🎧 Realtime Speech-to-Speech Translator (Smooth Playback)")
    log("Press SPACE to start/stop the translation session.")
    log("Press ESC to exit the program.")

    global stream
    is_recording = False

    try:
        while True:
            if keyboard.is_pressed("esc"):
                log("🛑 ESC pressed. Shutting down...")
                stop_flag.set()
                break

            if keyboard.is_pressed("space"):
                if not is_recording:
                    is_recording = True
                    stop_flag.clear()

                    log("Resetting audio buffer and queues...")
                    with ring.lock:
                        ring.buf[:] = 0.0
                        ring.read_pos = ring.write_pos = ring.available = 0
                    while not audio_queue.empty(): audio_queue.get()
                    while not text_queue.empty(): text_queue.get()

                    log("Starting processing threads...")
                    threading.Thread(target=record_audio, daemon=True).start()
                    threading.Thread(target=asr_translator_thread, daemon=True).start()
                    threading.Thread(target=tts_generator_thread, daemon=True).start()

                    log("Starting audio playback stream...")
                    stream = sd.OutputStream(
                        samplerate=SAMPLE_RATE,
                        channels=1,
                        dtype="float32",
                        blocksize=SD_BLOCKSIZE,
                        callback=playback_callback
                    )
                    stream.start()
                    log("✅ Session STARTED. Translation is active.")

                else:
                    is_recording = False
                    log("🧹 SPACE pressed. Stopping session...")
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
        log("🛑 Interrupted by user (Ctrl+C).")
    except Exception as e:
        log(f"💥 An unexpected error occurred in the main loop: {e}")
        traceback.print_exc()
    finally:
        log("Cleaning up resources...")
        stop_flag.set()
        if stream:
            stream.stop()
            stream.close()
        sd.stop() 
        log("Cleanup complete. Goodbye!")


if __name__ == "__main__":
    main()