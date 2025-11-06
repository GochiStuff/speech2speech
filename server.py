# server.py
import asyncio
import time
import tempfile
import os
import traceback
import shutil
from typing import Dict

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from faster_whisper import WhisperModel
from transformers import (
    MarianTokenizer,
    MarianMTModel,
    M2M100Tokenizer,
    M2M100ForConditionalGeneration,
)
import edge_tts

# ---------- CONFIG ----------
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASR_MODEL_NAME = "small"   # change to base/medium if GPU permits

TRANSLATORS = {
    "en→fr": {"type": "marian", "model": "Helsinki-NLP/opus-mt-en-fr", "target": "fr", "tts_voice": "fr-FR-HenriNeural"},
    "en→hi": {"type": "marian", "model": "Helsinki-NLP/opus-mt-en-hi", "target": "hi", "tts_voice": "hi-IN-MadhurNeural"},
    "en→fr-m2m": {"type": "m2m100", "model": "facebook/m2m100_418M", "target": "fr", "tts_voice": "fr-FR-HenriNeural"},
    "en→hi-m2m": {"type": "m2m100", "model": "facebook/m2m100_418M", "target": "hi", "tts_voice": "hi-IN-MadhurNeural"},
}
DEFAULT_PAIR = "en→fr"

FFMPEG = shutil.which("ffmpeg")
if not FFMPEG:
    raise FileNotFoundError("ffmpeg not found on PATH. Install ffmpeg.")

app = FastAPI()

@app.get("/")
async def index():
    try:
        with open("client.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h3>Put client.html next to server.py and open this page</h3>")

def now(): return time.strftime("%H:%M:%S")
def log(s): print(f"[{now()}] {s}", flush=True)

log(f"Starting server — device={DEVICE}")
log(f"Loading Whisper ASR '{ASR_MODEL_NAME}' (compute_type={'float16' if DEVICE=='cuda' else 'int8'})...)")
asr_model = WhisperModel(ASR_MODEL_NAME, device=DEVICE, compute_type=("float16" if DEVICE=="cuda" else "int8"))
log("Whisper loaded.")

translator_cache: Dict[str, Dict] = {}

log("Loading Silero VAD (server-side fallback)...")
vad_model, vad_utils = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False, onnx=False)
(get_speech_timestamps, _, _, VADIterator, _) = vad_utils
vad_model = vad_model.to("cpu")
log("Silero VAD loaded (fallback).")

def _safe_from_pretrained(cls, model_id: str, **kwargs):
    """
    Try to load using safetensors first (use_safetensors=True). If that fails,
    fall back to the normal load. Returns the loaded model object.
    """
    # Try safetensors first (avoids torch.load checks). Not all model hosting provides safetensors.
    try:
        log(f"[model load] trying safetensors for {model_id}")
        return cls.from_pretrained(model_id, use_safetensors=True, **kwargs)
    except Exception as e:
        # If the error mentions safetensors not available or not found, fall back.
        log(f"[model load] safetensors load failed for {model_id}: {e}")
        # Try normal load (this may trigger the torch.load vulnerability block if torch < 2.6 and the model uses .bin)
        try:
            log(f"[model load] falling back to standard weights for {model_id}")
            return cls.from_pretrained(model_id, **kwargs)
        except Exception as e2:
            log(f"[model load] standard load FAILED for {model_id}: {e2}")
            raise

def ensure_translator(pair_key: str):
    """
    Load translator model + tokenizer on demand. Prefers safetensors if available.
    """
    if pair_key in translator_cache:
        return translator_cache[pair_key]

    cfg = TRANSLATORS.get(pair_key)
    if cfg is None:
        raise ValueError("Unknown translator pair: " + str(pair_key))

    try:
        if cfg["type"] == "marian":
            log(f"Loading tokenizer for {pair_key} -> {cfg['model']}")
            tok = MarianTokenizer.from_pretrained(cfg["model"])
            log(f"Loading model for {pair_key} -> {cfg['model']}")
            model = _safe_from_pretrained(MarianMTModel, cfg["model"])
        else:
            log(f"Loading tokenizer for {pair_key} -> {cfg['model']}")
            tok = M2M100Tokenizer.from_pretrained(cfg["model"])
            log(f"Loading model for {pair_key} -> {cfg['model']}")
            model = _safe_from_pretrained(M2M100ForConditionalGeneration, cfg["model"])
    except ValueError as ve:
        # This is often the transformers-enforced message requiring torch >= 2.6.
        log(f"[ensure_translator] ValueError while loading model {cfg['model']}: {ve}")
        log("You can either install a torch >= 2.6 (stable or nightly) or download a safetensors build of the model.")
        raise
    except Exception as e:
        log(f"[ensure_translator] Unexpected error loading {cfg['model']}: {e}")
        raise

    # Move model to device
    try:
        model = model.to(DEVICE)
    except Exception as e:
        log(f"[ensure_translator] Warning: failed to move model to {DEVICE}: {e}")

    translator_cache[pair_key] = {"cfg": cfg, "tokenizer": tok, "model": model}
    log(f"Loaded translator {pair_key}")
    return translator_cache[pair_key]

async def generate_tts_bytes(text: str, voice: str) -> bytes:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            path = tmp.name
        comm = edge_tts.Communicate(text, voice)
        await comm.save(path)
        with open(path, "rb") as f:
            b = f.read()
        os.remove(path)
        return b
    except Exception as e:
        log(f"TTS generation FAILED for text='{text}' voice='{voice}': {e}")
        return b""  # Return empty bytes on failure

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    client = f"{ws.client.host}:{ws.client.port}" if ws.client else "unknown"
    log(f"WS connected: {client}")
    current_pair = DEFAULT_PAIR

    async def handle_phrase_bytes(b: bytes):
        try:
            arr = np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0
            secs = arr.shape[0] / SAMPLE_RATE
            if secs < 0.06:
                log(f"Ignoring tiny phrase ({secs:.3f}s, {arr.shape[0]} samples)")
                return
            log(f"Processing phrase: {arr.shape[0]} samples -> {secs:.3f}s")
            t0 = time.time()
            segments, info = asr_model.transcribe(arr, beam_size=1)
            asr_text = " ".join([s.text for s in segments]).strip()
            t_asr = time.time() - t0
            log(f"ASR ({t_asr:.2f}s): {asr_text}")
            # send ASR result (may be empty)
            try:
                await ws.send_json({"type":"asr","text":asr_text,"lang":getattr(info,"language",None),"asr_time":t_asr})
            except Exception:
                log("[handle_phrase_bytes] couldn't send ASR JSON (client probably disconnected)")

            if not asr_text or len(asr_text.strip()) < 2:
                log("Skipping translation/TTS for empty or short ASR result.")
                return  # short ASR result

            # ensure translator available
            bundle = ensure_translator(current_pair)
            cfg = bundle["cfg"]; tok = bundle["tokenizer"]; model = bundle["model"]

            t0 = time.time()
            if cfg["type"] == "m2m100":
                tok.src_lang = getattr(info,"language","en") or "en"
                inputs = tok(asr_text, return_tensors="pt").to(DEVICE)
                bos = tok.get_lang_id(cfg["target"])
                outputs = model.generate(**inputs, forced_bos_token_id=bos)
                translated = tok.batch_decode(outputs, skip_special_tokens=True)[0]
            else:
                inputs = tok(asr_text, return_tensors="pt").to(DEVICE)
                outputs = model.generate(**inputs)
                translated = tok.batch_decode(outputs, skip_special_tokens=True)[0]
            t_trans = time.time() - t0
            log(f"Translate ({t_trans:.2f}s): {translated}")
            try:
                await ws.send_json({"type":"translate","text":translated,"target":cfg["target"],"translate_time":t_trans})
            except Exception:
                log("[handle_phrase_bytes] couldn't send translate JSON (client probably disconnected)")

            t0 = time.time()
            tts_bytes = await generate_tts_bytes(translated or asr_text, cfg["tts_voice"])
            t_tts = time.time() - t0

            if not tts_bytes:
                log(f"TTS returned empty bytes (t_tts={t_tts:.2f}s). Skipping audio send.")
                return  # Stop processing

            log(f"TTS ({t_tts:.2f}s), {len(tts_bytes)} bytes")
            try:
                await ws.send_json({"type":"audio_start","format":"audio/mpeg","len_bytes":len(tts_bytes)})
                CHUNK = 32 * 1024
                for i in range(0, len(tts_bytes), CHUNK):
                    await ws.send_bytes(tts_bytes[i:i+CHUNK])
                await ws.send_json({"type":"audio_end"})
            except Exception:
                log("[handle_phrase_bytes] couldn't send audio to client (disconnected mid-stream?)")
        except Exception as e:
            log(f"handle_phrase_bytes error: {e}")
            traceback.print_exc()

    def task_done_cb(task: asyncio.Task):
        # log exceptions from background tasks
        try:
            exc = task.exception()
            if exc:
                log(f"[background task] exception: {exc}")
                traceback.print_exception(type(exc), exc, exc.__traceback__)
        except asyncio.CancelledError:
            log("[background task] cancelled")
        except Exception as e:
            log(f"[background task] error retrieving exception: {e}")

    try:
        while True:
            try:
                msg = await ws.receive()
            except RuntimeError as re:
                # This happens when receive is called after a disconnect message was received
                log(f"WS receive runtime error (likely client disconnected): {re}")
                break
            except WebSocketDisconnect:
                log(f"Client disconnected (exception) during receive: {client}")
                break
            except Exception as e:
                log(f"WS receive unexpected error: {e}")
                traceback.print_exc()
                break

            # text messages (controls)
            if isinstance(msg, dict) and "text" in msg and msg["text"] is not None:
                t = msg["text"]
                try:
                    import json
                    parsed = json.loads(t)
                    if parsed.get("type") == "set_lang" and parsed.get("pair"):
                        p = parsed["pair"]
                        if p in TRANSLATORS:
                            current_pair = p
                            # load translator (may raise; bubble up)
                            try:
                                ensure_translator(p)
                                await ws.send_json({"type":"info","msg":f"lang set {p}"})
                                log(f"Client set lang -> {p}")
                            except Exception as e:
                                await ws.send_json({"type":"error","msg":f"failed to set lang: {e}"})
                        else:
                            await ws.send_json({"type":"error","msg":"unknown lang pair"})
                    else:
                        # other JSON control messages (speech_start/end/ping)
                        if parsed.get("type") == "ping":
                            await ws.send_json({"type":"pong"})
                except Exception:
                    # not JSON or parse error -> send as info
                    try:
                        await ws.send_json({"type":"info","msg":t})
                    except Exception:
                        log("[ws loop] couldn't send info (client disconnected)")
                continue

            # bytes (audio phrases)
            if isinstance(msg, dict) and "bytes" in msg and msg["bytes"] is not None:
                b = msg["bytes"]
                # schedule handling in background, but ensure exceptions are logged
                task = asyncio.create_task(handle_phrase_bytes(b))
                task.add_done_callback(task_done_cb)
    except WebSocketDisconnect:
        log(f"Client disconnected (outer): {client}")
    except Exception as e:
        log(f"WS loop error: {e}")
        traceback.print_exc()
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        log(f"WS closed: {client}")
