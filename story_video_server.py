import os
import io
import json
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont
import soundfile as sf
from flask import Flask, render_template, request, jsonify, send_from_directory

try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend – safe in threads / Flask
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    # MoviePy <=1.0 had submodule editor; 2.x exports classes at top level
    try:
        from moviepy.editor import AudioFileClip, ImageClip, ColorClip, TextClip, CompositeVideoClip
    except ModuleNotFoundError:
        from moviepy import AudioFileClip, ImageClip, ColorClip, TextClip, CompositeVideoClip  # type: ignore
except Exception:
    AudioFileClip = None

from kokoro_onnx import Kokoro

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
BASE_DIR = Path(__file__).parent
KOKORO_MODELS_DIR = BASE_DIR / "kokoro_models"
SAVE_DIR = Path.home() / "Downloads" / "video-pipeline"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = KOKORO_MODELS_DIR / "kokoro-v1.0.int8.onnx"
VOICES_PATH = KOKORO_MODELS_DIR / "voices-v1.0.bin"

# -----------------------------------------------------------------------------
# INITIALISE
# -----------------------------------------------------------------------------
print("🔊 Loading Kokoro ONNX…")
kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))
voice_list = sorted(list(kokoro.voices.keys()))
print(f"✅ Kokoro loaded with {len(voice_list)} voices")

app = Flask(__name__)

# -----------------------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("story_video.html")

@app.route("/voices")
def voices():
    return jsonify(voice_list)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    story_override = data.get("story_text", "").strip()
    voice = data.get("voice", voice_list[0])

    if not prompt and not story_override:
        return jsonify({"error": "Either prompt or story_text is required"}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"story_{timestamp}"

    # ------- 1. generate story text -------
    if story_override:
        story_text = story_override
    else:
        story_text = get_story_from_llm(prompt)
        if not story_text:
            return jsonify({"error": "Failed to get story from LLM"}), 500

    # ------- 2. synthesize narration -------
    audio_bytes = synthesize_tts(story_text, voice)
    audio_path = SAVE_DIR / f"{base_name}.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    # ------- 3. optional waveform img -------
    waveform_img = None
    if plt is not None:
        waveform_img = render_waveform_image(audio_path, base_name)

    # ------- 4. build video (if moviepy available) -------
    video_path = None
    if AudioFileClip is not None:
        video_path = build_video(story_text, audio_path, waveform_img, base_name)

    return jsonify({
        "story": story_text,
        "audio_file": str(audio_path),
        "video_file": str(video_path) if video_path else None
    })

@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory(SAVE_DIR, filename, as_attachment=True)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def get_story_from_llm(prompt: str) -> str:
    payload = {
        "model": "llama-3.2-3b-instruct",
        "messages": [
            {"role": "system", "content": "You are a creative storyteller. Write a vivid, engaging short story based on the user's prompt."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.9,
        "max_tokens": 600
    }
    try:
        resp = requests.post(LM_STUDIO_URL, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("LLM error", e)
    return ""


def synthesize_tts(text: str, voice: str, speed: float = 1.0) -> bytes:
    samples, sr = kokoro.create(text, voice=voice, speed=speed, lang="en-us")
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    return buf.getvalue()


def render_waveform_image(audio_path: Path, base_name: str):
    try:
        import numpy as np
        data, sr = sf.read(audio_path)
        stride = max(1, int(sr / 1000))
        data_ds = data[::stride]
        t = np.linspace(0, len(data_ds) / (sr / stride), len(data_ds))
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, data_ds, color="#00caff")
        ax.set_axis_off()
        img_path = SAVE_DIR / f"{base_name}.png"
        fig.savefig(img_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return img_path
    except Exception as e:
        print("waveform render failed", e)
        return None


def build_video(story_text: str, audio_path: Path, waveform_img: Path | None, base_name: str):
    audio_clip = AudioFileClip(str(audio_path))
    duration = audio_clip.duration
    width, height = 1280, 720

    if waveform_img and waveform_img.exists():
        bg_clip = ImageClip(str(waveform_img))
    else:
        bg_clip = ColorClip(size=(width, height), color=(0, 0, 0))

    # make sure clip has correct duration with compat helper
    if hasattr(bg_clip, "set_duration"):
        bg_clip = bg_clip.set_duration(duration)
    else:
        bg_clip = bg_clip.with_duration(duration)  # MoviePy 2.x

    # Build TextClip with fallback fonts
    def make_text_clip():
        font_args = [None, "DejaVu-Sans", "Helvetica", "Arial"]
        last_err = None
        for f in font_args:
            try:
                kwargs = {"color": "white", "size": (int(width*0.8), None), "method": "caption"}
                if hasattr(TextClip, "font_size"):
                    kwargs["font_size"] = 40
                else:
                    kwargs["fontsize"] = 40
                if f:
                    kwargs["font"] = f
                return TextClip(story_text, **kwargs)
            except Exception as e:
                last_err = e
        raise last_err

    # Split story into sentences for subtitle effect
    import re, textwrap, tempfile
    sentences = re.split(r"(?<=[.!?])\s+", story_text.strip())
    # guard
    if not sentences:
        sentences = [story_text]
    chunk_dur = duration / len(sentences)

    subtitle_clips = []
    font = ImageFont.load_default()
    font_size = 60

    for idx, sent in enumerate(sentences):
        start = idx * chunk_dur
        end = (idx + 1) * chunk_dur
        wrapper = textwrap.TextWrapper(width=40)
        wrapped = "\n".join(wrapper.wrap(sent))
        # Render with Pillow
        img = Image.new("RGBA", (int(width*0.9), 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            font_pillow = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except IOError:
            font_pillow = ImageFont.load_default()
        w_text, h_text = draw.multiline_textbbox((0,0), wrapped, font=font_pillow)[2:4]
        draw.multiline_text(((img.width - w_text)//2, (200 - h_text)//2), wrapped, font=font_pillow, fill="white", align="center")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        clip = ImageClip(tmp.name)
        if hasattr(clip, "set_start"):
            clip = clip.set_start(start)
        else:
            clip = clip.with_start(start)
        clip = clip.set_duration(chunk_dur).set_position(("center", height*0.75))
        subtitle_clips.append(clip)

    video_layers = [bg_clip] + subtitle_clips
    video = CompositeVideoClip(video_layers)
    if hasattr(video, "set_audio"):
        video = video.set_audio(audio_clip)
    else:
        video = video.with_audio(audio_clip)
    video_path = SAVE_DIR / f"{base_name}.mp4"
    video.write_videofile(str(video_path), fps=24, codec="libx264", audio_codec="aac")
    return video_path

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("🌐 Story Video server running at http://localhost:5003")
    app.run(host="0.0.0.0", port=5003, debug=False) 