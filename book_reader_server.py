#!/usr/bin/env python3
"""Book-page → OCR → Kokoro TTS → audio playback (mobile camera helper).
Run with:  python book_reader_server.py  and open http://<ip>:5004 on phone.
"""
from __future__ import annotations

import io
import uuid
import json
import subprocess, shutil
from datetime import datetime
from pathlib import Path

import soundfile as sf
from flask import Flask, jsonify, render_template, request, send_from_directory, Response, stream_with_context
from PIL import Image
import cv2
import numpy as np
import easyocr
import base64, requests, json
import time
import re  # added for sentence splitting
from collections import OrderedDict
from functools import lru_cache

from kokoro_onnx import Kokoro

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
KOKORO_DIR = BASE_DIR / "kokoro_models"
SAVE_DIR = Path.home() / "Downloads" / "ocr_tts"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH = KOKORO_DIR / "kokoro-v1.0.int8.onnx"
VOICE_BANK = KOKORO_DIR / "voices-v1.0.bin"
DEFAULT_VOICE = "af_sky"  # default Kokoro voice present in your model
SYNTH_SPEED = 1.0  # faster generation
PLAYBACK_RATE = 1  # compensate for faster synthesis (1.3×) to restore natural pace

LM_STUDIO_OCR_URL = "http://localhost:1234/v1/chat/completions"

# ---------------------------------------------------------------------------
# HEAVY LOAD – happens once
# ---------------------------------------------------------------------------
print("📖 Loading EasyOCR (CPU)…")
ocr_reader = easyocr.Reader(['en'], gpu=False, recog_network='english_g2')

# ---------------------------------------------------------------------------
# Preprocess helper – deskew, adaptive threshold, upscale
# ---------------------------------------------------------------------------

import math

def preprocess(pil_img):
    import cv2, numpy as _np
    cv = cv2.cvtColor(_np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    # deskew small tilt
    coords = _np.column_stack(_np.where(gray < 250))
    if coords.size:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.3:  # skip negligible
            (h, w) = cv.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            cv = cv2.warpAffine(cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    # adaptive threshold
    thr = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,31,11)
    # upscale 1.3x to help recogniser
    thr_big = cv2.resize(thr, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_LINEAR)
    return thr_big

print("🔊 Loading Kokoro TTS…")
kokoro = Kokoro(str(ONNX_PATH), str(VOICE_BANK))

# ---------------------------------------------------------------------------
# TTS helper – split long text into safe chunks and stitch audio
# ---------------------------------------------------------------------------

# Rough heuristic: limit ~350 words per chunk so that phoneme tokens stay under 510
MAX_WORDS_PER_CHUNK = 120  # tighter safety to avoid phoneme overflow


def _split_text(text: str, max_words: int = MAX_WORDS_PER_CHUNK) -> list[str]:
    """Split *text* into chunks no longer than *max_words* words, preferably on sentence boundaries."""
    # First split by punctuation that marks sentence endings.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        words = sent.strip().split()
        if not words:
            continue
        if current_len + len(words) > max_words:
            if current:
                # Flush existing chunk first
                chunks.append(" ".join(current).strip())
                current = words
                current_len = len(words)
            else:
                # Single sentence exceeds max_words; split inside sentence
                while len(words) > max_words:
                    chunks.append(" ".join(words[:max_words]))
                    words = words[max_words:]
                current = words
                current_len = len(words)
        else:
            current.extend(words)
            current_len += len(words)

    if current:
        chunks.append(" ".join(current).strip())

    return [c for c in chunks if c]


def synthesize_text(text: str, voice: str = DEFAULT_VOICE, speed: float = SYNTH_SPEED):
    """Safely synthesize potentially long *text* by chunking and concatenating audio."""
    chunks = _split_text(text)
    all_audio = []
    sr = None

    # Helper to pick a fallback voice once instead of repeating lookup in each loop
    fallback_voice = next(iter(kokoro.voices.keys())) if voice not in kokoro.voices else voice

    for idx, chunk in enumerate(chunks):
        try:
            audio, sr = kokoro.create(chunk, voice=voice, speed=speed, lang="en-us")
        except Exception as e:
            # Likely length-related (token overflow) or voice not found – attempt fallback, then split recursively
            print(f"Kokoro chunk {idx} failed: {e}.")
            try:
                audio, sr = kokoro.create(chunk, voice=fallback_voice, speed=speed, lang="en-us")
            except Exception as err:
                # Still failing – split the text further until it succeeds
                words = chunk.split()
                if len(words) <= 5:
                    # Too small to split further – re-raise original error
                    raise
                mid = len(words)//2
                left = " ".join(words[:mid])
                right = " ".join(words[mid:])
                sub_a, sr = synthesize_text(left, voice, speed)
                sub_b, _ = synthesize_text(right, voice, speed)
                audio = np.concatenate([sub_a, sub_b])
        # Trim silence is already handled inside kokoro.create
        all_audio.append(audio)

    if not all_audio:
        raise RuntimeError("No audio chunks synthesised")

    return np.concatenate(all_audio), sr

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

# EasyOCR wrapper
def run_ocr(img_arr):
    try:
        return ocr_reader.readtext(img_arr)
    except Exception as e:
        print("EasyOCR error", e)
        return []

# Vision LLM OCR via LM Studio
def llm_ocr(img_bytes: bytes) -> str:
    b64 = base64.b64encode(img_bytes).decode()
    payload = {
        "model": "unsloth/Nanonets-OCR-s-GGUF",
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": "Extract the plain text from the page."}
            ]}
        ],
        "max_tokens": 3500,
        "temperature": 0.0
    }
    try:
        # Increased timeout to 5 minutes to allow large pages to process
        r = requests.post(LM_STUDIO_OCR_URL, json=payload, timeout=300)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("LM OCR error", e)
    return ""

# ---------------------------------------------------------------------------
# Text post-processing helpers
# ---------------------------------------------------------------------------

import re as _re


def deduplicate_text(text: str) -> str:
    """Remove duplicate paragraphs and sentences (case-insensitive) while preserving order."""
    if not text:
        return text

    # First, normalise line breaks to detect paragraph breaks
    pars = _re.split(r"\n\s*\n|\r\n\s*\r\n", text.strip())
    if len(pars) == 1:
        # OCR may return all in one line – fallback to sentence dedup
        pars = [_re.sub(r"\s+", " ", text.strip())]

    seen_par = set()
    cleaned_pars = []
    for par in pars:
        norm = _re.sub(r"\s+", " ", par).strip().lower()
        if not norm:
            continue
        if norm in seen_par:
            continue  # exact duplicate paragraph
        seen_par.add(norm)

        # Sentence-level dedup inside the paragraph
        sentences = _re.split(r"(?<=[.!?])\s+", par)
        seen_sent = set()
        dedup_sent = []
        for sent in sentences:
            s = sent.strip()
            if not s:
                continue
            low = s.lower()
            if low in seen_sent:
                continue
            seen_sent.add(low)
            dedup_sent.append(s)
        cleaned_pars.append(" ".join(dedup_sent))

    return "\n\n".join(cleaned_pars)

# ---------------------------------------------------------------------------
# Chunking helper – ensure <=450 phoneme tokens per chunk
# ---------------------------------------------------------------------------

MAX_WORDS_PER_CHUNK = 100  # simple word limit (~<510 phoneme tokens)


def split_safe_chunks(text: str, max_words: int = MAX_WORDS_PER_CHUNK) -> list[str]:
    """Split text into chunks of <= max_words, respecting sentence boundaries."""
    if not text:
        return []
    sentences = _re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = []
    word_count = 0
    for sent in sentences:
        words = sent.strip().split()
        if not words:
            continue
        if word_count + len(words) > max_words and current:
            # flush current chunk
            chunks.append(" ".join(current).strip())
            current = words
            word_count = len(words)
        else:
            current.extend(words)
            word_count += len(words)
    if current:
        chunks.append(" ".join(current).strip())
    return chunks

# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("book_reader.html")


# Streaming upload


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image supplied"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    img_obj = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_size_kb = len(img_bytes)//1024

    # OCR
    t0 = time.perf_counter()
    text = deduplicate_text(llm_ocr(img_bytes))
    # Fallback to EasyOCR temporarily disabled for debugging – we only rely on Nanonets OCR.
    # if not text:
    #     cv_img = preprocess(img_obj)
    #     result = run_ocr(cv_img)
    #     text = " ".join(t for (_, t, _conf) in result)
    t_ocr = (time.perf_counter()-t0)*1000
    if not text:
        return jsonify({"error": "OCR failed"}), 200

    # TTS – use safe chunked synthesis
    try:
        t1 = time.perf_counter()
        samples, sr = synthesize_text(text, DEFAULT_VOICE, SYNTH_SPEED)
        tts_ms = (time.perf_counter() - t1) * 1000
    except Exception as e:
        print("TTS error:", e)
        return jsonify({"error": "TTS failed", "detail": str(e)}), 200

    wav_path = SAVE_DIR / f"raw_{datetime.now():%Y%m%d_%H%M%S}.wav"
    sf.write(wav_path, samples, sr)

    mp3_name = wav_path.with_suffix('.mp3').name
    mp3_path = SAVE_DIR / mp3_name
    try:
        import subprocess, shutil
        if shutil.which('ffmpeg'):
            subprocess.run(['ffmpeg','-y','-i',wav_path,'-codec:a','libmp3lame','-b:a','96k', mp3_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            wav_path.unlink(missing_ok=True)
        else:
            mp3_path = wav_path
    except Exception as e:
        print("Encoding error:", e)
        return jsonify({"error": "Encoding failed", "detail": str(e)}), 200

    print(f"[Analytics] size={img_size_kb}KB  OCR={t_ocr:.0f}ms  TTS={tts_ms:.0f}ms  chars={len(text)}")

    return jsonify({
        "text": text,
        "audio_url": f"/download/{mp3_path.name}",
        "playback": PLAYBACK_RATE
    })


@app.route("/upload_stream", methods=["POST"])
def upload_stream():
    if "image" not in request.files:
        return jsonify({"error": "No image supplied"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    # OCR and clean
    text = deduplicate_text(llm_ocr(img_bytes))
    if not text:
        return jsonify({"error": "OCR failed"}), 200

    chunks = split_safe_chunks(text)
    session_id = uuid.uuid4().hex

    def generate():
        # meta event
        meta = {"type": "meta", "text": text, "chunks": len(chunks)}
        yield f"data:{json.dumps(meta)}\n\n"

        for idx, chunk in enumerate(chunks):
            try:
                samples, sr = kokoro.create(chunk, voice=DEFAULT_VOICE, speed=SYNTH_SPEED, lang="en-us")
            except Exception as e:
                samples, sr = kokoro.create(chunk, voice=next(iter(kokoro.voices.keys())), lang="en-us")

            wav_path = SAVE_DIR / f"{session_id}_{idx}.wav"
            sf.write(wav_path, samples, sr)

            mp3_path = wav_path.with_suffix('.mp3')
            if shutil.which('ffmpeg'):
                subprocess.run(['ffmpeg','-y','-i',wav_path,'-codec:a','libmp3lame','-b:a','96k', mp3_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                wav_path.unlink(missing_ok=True)
            else:
                mp3_path = wav_path

            evt = {"type": "audio", "url": f"/download/{mp3_path.name}", "idx": idx}
            yield f"data:{json.dumps(evt)}\n\n"

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return Response(stream_with_context(generate()), headers=headers, mimetype="text/event-stream")


@app.route("/download/<path:fname>")
def download(fname: str):
    return send_from_directory(SAVE_DIR, fname, as_attachment=False)

# ---------------------------------------------------------------------------
# Return JSON for any uncaught exception instead of HTML stacktrace
# ---------------------------------------------------------------------------

@app.errorhandler(Exception)
def handle_exc(err):
    import traceback, sys
    tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
    print(tb, file=sys.stderr)
    return jsonify({"error": "Server error", "detail": str(err)}), 500

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Book Reader server at http://0.0.0.0:5004")
    app.run(host="0.0.0.0", port=5004, debug=False) 