# test.py — FINAL FIXED VERSION
# ✅ Handles multi-speaker TTS (xtts_v2) correctly using speaker_wav fallback
# ✅ Includes robust ASR, translation, and TTS pipeline with clear logging

import asyncio
import numpy as np
import sounddevice as sd
import resampy
import time
import traceback
from faster_whisper import WhisperModel
from TTS.api import TTS
import torch

# Optional translator
HAS_TRANSLATOR = True
try:
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
except Exception as e:
    HAS_TRANSLATOR = False
    TRANS_IMPORT_ERROR = e

# ----------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------
SAMPLE_RATE = 16000
CHUNK_SEC = 5.0
USE_GPU = True
ASR_MODEL_NAME = "medium"
TARGET_LANG = "fr"

TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# ✅ Provide your own WAV or URL — used for xtts_v2 voice cloning fallback
SPEAKER_WAV_PATH = "./male.wav"

SILENCE_THRESHOLD = 1e-4
TRANSLATOR_MODEL = "facebook/m2m100_418M"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def record_audio(duration: float = CHUNK_SEC):
    try:
        log(f"🎙️ Recording {duration:.1f}s...")
        audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        audio = np.squeeze(audio)
        if audio.size == 0 or np.mean(np.abs(audio)) < SILENCE_THRESHOLD:
            log("⚠️ Silence detected.")
            return None
        log("✅ Recording complete.")
        return audio
    except Exception as e:
        log(f"❌ Recording failed: {e}")
        traceback.print_exc()
        return None


def ensure_wav_array(wav):
    if wav is None:
        return None
    if isinstance(wav, list):
        parts = []
        for p in wav:
            try:
                arr = np.asarray(p, dtype=np.float32).flatten()
                parts.append(arr)
            except Exception:
                pass
        return np.concatenate(parts) if parts else None
    return np.asarray(wav, dtype=np.float32).flatten()


def probe_speakers(tts):
    names = []
    try:
        sm = getattr(tts, "synthesizer", None)
        if sm and hasattr(sm, "speaker_manager"):
            spm = sm.speaker_manager
            if hasattr(spm, "speaker_ids"):
                names.extend(spm.speaker_ids)
            if hasattr(spm, "speakers") and isinstance(spm.speakers, dict):
                names.extend(spm.speakers.keys())
    except Exception:
        pass
    return list(set(str(n) for n in names if n))


async def main():
    device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
    log(f"🚀 Using device={device}")

    # ---------- ASR ----------
    log("🧠 Loading ASR model...")
    asr_model = WhisperModel(ASR_MODEL_NAME, device=device)
    log("✅ ASR ready.")

    # ---------- TRANSLATOR ----------
    translator = None
    tokenizer = None
    if HAS_TRANSLATOR:
        try:
            log("🔁 Loading translator...")
            tokenizer = M2M100Tokenizer.from_pretrained(TRANSLATOR_MODEL)
            translator = M2M100ForConditionalGeneration.from_pretrained(TRANSLATOR_MODEL).to(device)
            log("✅ Translator ready.")
        except Exception as e:
            log(f"⚠️ Translator failed: {e}")
            traceback.print_exc()

    # ---------- TTS ----------
    log("🎤 Loading TTS model...")
    tts = TTS(model_name=TTS_MODEL_NAME, progress_bar=False, gpu=USE_GPU)
    try:
        tts.to(device)
    except Exception:
        pass
    sr = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", 22050)
    log(f"✅ TTS ready (sample_rate={sr})")

    speakers = probe_speakers(tts)
    log(f"🗣️ Speakers found: {speakers}" if speakers else "⚠️ No speaker list found.")
    speaker = speakers[0] if speakers else None

    # ---------- MAIN LOOP ----------
    log(f"🚀 Translator active → {TARGET_LANG}")
    while True:
        try:
            audio = await record_audio(CHUNK_SEC)
            if audio is None:
                continue

            segments, info = asr_model.transcribe(audio, beam_size=1)
            text = " ".join([s.text for s in segments]).strip()
            if not text:
                log("⚠️ Empty ASR result.")
                continue

            lang = getattr(info, "language", None)
            log(f"🗣️ {lang}: {text}")

            translated = text
            if translator and tokenizer and lang != TARGET_LANG:
                try:
                    tokenizer.src_lang = lang
                    inputs = tokenizer(text, return_tensors="pt").to(device)
                    bos = tokenizer.get_lang_id(TARGET_LANG)
                    out = translator.generate(**inputs, forced_bos_token_id=bos)
                    translated = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
                    log(f"💬 {TARGET_LANG}: {translated}")
                except Exception as e:
                    log(f"⚠️ Translate failed: {e}")
                    traceback.print_exc()

            # ---------- TTS ----------
            try:
                log("🎧 Synthesizing...")
                try:
                    wav = tts.tts(text=translated, speaker=speaker, language=TARGET_LANG)
                except Exception:
                    log("🎯 Retrying with speaker_wav fallback...")
                    wav = tts.tts(text=translated, speaker_wav=SPEAKER_WAV_PATH, language=TARGET_LANG)

                wav = ensure_wav_array(wav)
                if wav is None:
                    raise RuntimeError("Empty TTS output")

                if sr != SAMPLE_RATE:
                    wav = resampy.resample(wav, sr, SAMPLE_RATE)
                wav = wav / np.max(np.abs(wav))
                sd.play(wav, SAMPLE_RATE)
                sd.wait()
                log("✅ Done.\n")
            except Exception as e:
                log(f"❌ TTS failed: {e}")
                traceback.print_exc()

        except KeyboardInterrupt:
            log("🛑 Exiting...")
            break
        except Exception as e:
            log(f"💥 Loop error: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("🧹 Stopped by user.")
