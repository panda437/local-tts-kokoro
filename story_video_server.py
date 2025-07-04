import os
import io
import json
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
from moviepy.video.VideoClip import VideoClip
import soundfile as sf
from flask import Flask, render_template, request, jsonify, send_from_directory
from moviepy.video.io.VideoFileClip import VideoFileClip
# for creating clip from GIF frames
try:
    from moviepy.editor import ImageSequenceClip
except Exception:
    ImageSequenceClip = None
# MoviePy vfx loop for compatibility
try:
    import moviepy.video.fx.all as vfx
except Exception:
    vfx = None

# Placeholder to satisfy legacy check; waveform rendering no longer needs matplotlib
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

    # ------- 3. waveform image (for fallback) -------
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


def render_waveform_image(audio_path: Path, base_name: str, width: int = 1280, height: int = 360):
    """Render a stylized vertical-bar waveform inspired by the design sample."""
    try:
        import numpy as _np
        # read mono
        data, sr = sf.read(audio_path)
        if data.ndim > 1:
            data = data.mean(axis=1)

        # samples represented by each pixel column
        samples_per_px = max(1, len(data) // width)
        amps = _np.abs(data[:samples_per_px * width].reshape(-1, samples_per_px)).max(axis=1)

        max_amp = amps.max() or 1.0

        img = Image.new("RGB", (width, height), (20, 22, 25))  # near-black bg
        draw = ImageDraw.Draw(img)
        mid = height // 2

        # colour gradient top (cyan) to bottom (indigo)
        top_col = _np.array([0, 240, 255])
        bot_col = _np.array([55, 60, 180])

        for x, a in enumerate(amps):
            norm = a / max_amp
            bar_h = int(norm * (height * 0.9) / 2)
            # simple two-tone: taller bar uses lighter colour
            col = tuple((top_col * norm + bot_col * (1 - norm)).astype(int))
            draw.line((x, mid - bar_h, x, mid + bar_h), fill=col, width=2)

        # dotted baseline
        for x in range(0, width, 4):
            draw.point((x, mid), fill=(120, 120, 120))

        img_path = SAVE_DIR / f"{base_name}.png"
        img.save(img_path)
        return img_path
    except Exception as e:
        print("waveform render failed", e)
        return None


def build_video(story_text: str, audio_path: Path, waveform_img: Path | None, base_name: str):
    audio_clip = AudioFileClip(str(audio_path))
    duration = audio_clip.duration
    width, height = 1280, 720
    wave_h = 360  # height reserved for waveform top half

    # Base black background
    bg_clip = ColorClip(size=(width, height), color=(0, 0, 0))
    if hasattr(bg_clip, "set_duration"):
        bg_clip = bg_clip.set_duration(duration)
    else:
        bg_clip = bg_clip.with_duration(duration)

    # Dynamic bar waveform animation --------------------------------------
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)

    bar_width = 4
    gap = 2
    bars = width // (bar_width + gap)
    window = int(sr * 0.02)  # 20 ms per amplitude sample
    amp_env = np.abs(data).astype(float)
    # compute max amplitude per window
    env = np.maximum.reduceat(amp_env, np.arange(0, len(amp_env), window))

    # pad env to at least bars + frames
    total_frames = int(duration * 30)  # assume ~30 fps for effect scroll
    needed = total_frames + bars + 1
    if len(env) < needed:
        env = np.pad(env, (0, needed - len(env)), constant_values=0)

    scale_h = (wave_h * 0.9) / (env.max() or 1.0)

    def bar_frame(t):
        idx = int(t * 30)  # 30 fps scroll index
        arr = np.zeros((wave_h, width, 3), dtype=np.uint8)
        mid = wave_h // 2
        for j in range(bars):
            a = env[idx + j] * scale_h
            # random mod for some dynamic flicker
            a *= 0.85 + 0.3 * random.random()
            h = int(a)
            x0 = j * (bar_width + gap)
            col_top = np.array([0, 240, 255], dtype=np.uint8)
            col_bot = np.array([55, 60, 180], dtype=np.uint8)
            # color depending on height
            norm = h / (wave_h/2) if wave_h else 0
            if norm > 1:
                norm = 1.0
            col = (col_top * norm + col_bot * (1 - norm)).clip(0, 255).astype(np.uint8)
            # draw vertical bar
            arr[mid - h:mid + h, x0:x0 + bar_width] = col
        return arr

    wave_clip = VideoClip(bar_frame, duration=duration)
    if hasattr(wave_clip, "set_position"):
        wave_clip = wave_clip.set_position((0, 0))
    else:
        wave_clip = wave_clip.with_position((0, 0))

    # No progress bar when using animated waveform

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
    font_size = 80  # reduced by ~20%

    for idx, sent in enumerate(sentences):
        start = idx * chunk_dur
        end = (idx + 1) * chunk_dur
        wrapper = textwrap.TextWrapper(width=20)
        wrapped = "\n".join(wrapper.wrap(sent))
        # Render with Pillow
        img = Image.new("RGBA", (int(width*0.9), 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # robust font loading
        try:
            from pathlib import Path as _Path
            dejavu_path = (_Path(ImageFont.__file__).parent / "DejaVuSans.ttf")
            font_pillow = ImageFont.truetype(str(dejavu_path), font_size)
        except Exception:
            try:
                font_pillow = ImageFont.truetype("Arial.ttf", font_size)
            except Exception:
                font_pillow = ImageFont.load_default()
                # upscale default bitmap font 8x via resize later
        w_text, h_text = draw.multiline_textbbox((0,0), wrapped, font=font_pillow)[2:4]
        draw.multiline_text(((img.width - w_text)//2, (200 - h_text)//2), wrapped, font=font_pillow, fill="white", align="center")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        # If default font, scale image 3x to compensate small size
        if font_pillow == ImageFont.load_default():
            img = img.resize((img.width*3, img.height*3), resample=Image.NEAREST)
            img.save(tmp.name)
        else:
            img.save(tmp.name)
        clip = ImageClip(tmp.name)
        if hasattr(clip, "set_start"):
            clip = clip.set_start(start)
        else:
            clip = clip.with_start(start)
        if hasattr(clip, "set_duration"):
            clip = clip.set_duration(chunk_dur)
        else:
            clip = clip.with_duration(chunk_dur)
        # Place subtitles below waveform area, centered horizontally
        subtitle_y = wave_h + 40  # 40px margin
        if hasattr(clip, "set_position"):
            clip = clip.set_position(("center", subtitle_y))
        else:
            clip = clip.with_position(("center", subtitle_y))
        subtitle_clips.append(clip)

    video_layers = [bg_clip]
    if wave_clip is not None:
        video_layers.append(wave_clip)
    video_layers.extend(subtitle_clips)
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