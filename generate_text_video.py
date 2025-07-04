# NOTE: Updated for MoviePy 2.x – uses TextClip.from_text
import os, sys, traceback, textwrap, tempfile
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
try:
    from moviepy.editor import AudioFileClip, CompositeVideoClip, ColorClip, ImageClip
except ModuleNotFoundError:
    from moviepy import AudioFileClip, CompositeVideoClip, ColorClip, ImageClip

TEXT = (
    "Under a huge mango tree lived Minoo, a tiny ant who dreamed of flying. "
    "Each morning she watched butterflies flutter across the garden, their wings painting patterns in the sun. "
    "One day a wise old beetle advised, “Find friends, and impossible becomes possible.” "
    "Minoo gathered silky spider threads, dewdrops, and petals; the ladybugs stitched, bees glued, and fireflies tested balance. "
    "At dawn they unveiled a shimmering petal-glider. Minoo climbed aboard; gentle wind lifted her above surprised grasshoppers. "
    "She circled the tall mango leaves, greeting sparrows, then glided down to cheers from every garden friend. "
    "She landed softly, heart soaring skyward."
)

# find an audio file in current folder (wav or mp3)
audio_file = None
for ext in (".wav", ".mp3", ".aac"):
    found = list(Path('.').glob(f"*{ext}"))
    if found:
        audio_file = str(found[0])
        break

if not audio_file:
    print("No audio file (.wav/.mp3/.aac) found in current directory. Place one and retry.")
    sys.exit(1)

print("Using audio:", audio_file)

try:
    pass # No specific imports needed for this block
except ModuleNotFoundError:
    print("MoviePy not installed -> pip install moviepy imageio-ffmpeg pillow")
    sys.exit(1)

# Load audio clip
clip_audio = AudioFileClip(audio_file)
dur = clip_audio.duration
w, h = 1280, 720

bg = ColorClip(size=(w, h), color=(0, 0, 0))
if hasattr(bg, "set_duration"):
    bg = bg.set_duration(dur)
else:
    bg = bg.with_duration(dur)

def make_pillow_text_clip(text:str, width:int, height:int):
    # Simple word wrapping
    wrapper = textwrap.TextWrapper(width=60)
    lines = wrapper.wrap(text)
    font = ImageFont.load_default()
    line_height = font.getbbox("Ay")[3] + 8
    img_h = line_height * len(lines) + 20
    img = Image.new("RGBA", (width, img_h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    y=10
    for line in lines:
        w_line = draw.textlength(line, font=font)
        draw.text(((width-w_line)//2, y), line, font=font, fill="white")
        y += line_height
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    img.save(tmp.name)
    return ImageClip(tmp.name)

try:
    txt = make_pillow_text_clip(TEXT, int(w*0.8), h)
except Exception as e:
    # Fall back to old API loops if using MoviePy 1.x
    font_args = [None, "DejaVu-Sans", "Helvetica", "Arial", "Liberation-Sans", "Verdana"]
    txt = None
    for f in font_args:
        try:
            kwargs = {"color": "white", "size": (int(w*0.8), None), "method": "caption"}
            if hasattr(TextClip, "font_size"):
                kwargs["font_size"] = 40
            else:
                kwargs["fontsize"] = 40
            if f:
                kwargs["font"] = f
            txt = TextClip(TEXT, **kwargs)
            break
        except Exception:
            continue
    if txt is None:
        txt = make_pillow_text_clip(TEXT, int(w*0.8), h)

txt_clip = txt

if hasattr(txt_clip, "set_duration"):
    txt_clip = txt_clip.set_duration(dur).set_position("center")
else:
    txt_clip = txt_clip.with_duration(dur).with_position("center")
video = CompositeVideoClip([bg, txt_clip])
if hasattr(video, "set_audio"):
    video = video.set_audio(clip_audio)
else:
    video = video.with_audio(clip_audio)
out = f"text_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
video.write_videofile(out, fps=24, codec="libx264", audio_codec="aac", logger=None)
print("✅ Video created with font", "->", out) 