import os
import io
import json
import threading
import tempfile
from datetime import datetime

import requests
import numpy as np
import soundfile as sf

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # waveform rendering will be skipped if matplotlib missing

try:
    from moviepy.editor import AudioFileClip, ImageClip, TextClip, CompositeVideoClip, ColorClip
except ImportError:
    AudioFileClip = None  # will disable video if moviepy not installed

try:
    from kokoro_onnx import Kokoro
except ImportError:
    raise RuntimeError("kokoro_onnx is required for this app. Install with: pip install kokoro-onnx")

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
KOKORO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "kokoro_models", "kokoro-v1.0.int8.onnx")
KOKORO_VOICES_PATH = os.path.join(os.path.dirname(__file__), "kokoro_models", "voices-v1.0.bin")
SAVE_DIR = os.path.expanduser("~/Downloads/video-pipeline")

os.makedirs(SAVE_DIR, exist_ok=True)

class StoryVideoApp:
    def __init__(self):
        # Load Kokoro
        self.kokoro = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
        self.voices = sorted(list(self.kokoro.voices.keys()))

        # Tkinter UI
        self.root = tk.Tk()
        self.root.title("Story Video Generator")
        self.root.geometry("800x600")

        prompt_label = ttk.Label(self.root, text="Story prompt:")
        prompt_label.pack(anchor="w", padx=10, pady=(10, 0))

        self.prompt_entry = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=6)
        self.prompt_entry.pack(fill="x", padx=10)

        # Voice selector
        voice_frame = ttk.Frame(self.root)
        voice_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(voice_frame, text="Narrator voice:").pack(side="left")
        self.voice_var = tk.StringVar(value=self.voices[0])
        self.voice_selector = ttk.Combobox(voice_frame, values=self.voices, textvariable=self.voice_var, state="readonly")
        self.voice_selector.pack(side="left", padx=5)

        # Generate button
        self.generate_btn = ttk.Button(self.root, text="Generate Story Video", command=self.start_generation)
        self.generate_btn.pack(pady=10)

        # Log window
        self.log_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=20, state="disabled")
        self.log_area.pack(fill="both", expand=True, padx=10, pady=5)

    # Utility to log messages to UI
    def log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_area.configure(state="normal")
        self.log_area.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.log_area.see(tk.END)
        self.log_area.configure(state="disabled")
        self.root.update()

    def start_generation(self):
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Input missing", "Please enter a prompt for the story.")
            return
        voice = self.voice_var.get()
        self.generate_btn.config(state="disabled")
        threading.Thread(target=self.run_pipeline, args=(prompt, voice), daemon=True).start()

    def run_pipeline(self, prompt: str, voice: str):
        try:
            # 1. Get story from LLM
            self.log("Contacting LLM to write the story…")
            story_text = self.generate_story_with_llm(prompt)
            if not story_text:
                raise RuntimeError("LLM returned no story text")
            self.log("Story received ({} characters)".format(len(story_text)))

            # 2. TTS with Kokoro
            self.log(f"Synthesizing narration with voice '{voice}'…")
            audio_bytes = self.kokoro_to_wav(story_text, voice)
            if audio_bytes is None:
                raise RuntimeError("TTS synthesis failed")

            # Save audio tmp file
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_path = os.path.join(SAVE_DIR, f"story_{ts}.wav")
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            self.log(f"Audio saved to {audio_path}")

            # 3. Create waveform background (optional)
            waveform_img = None
            if plt is not None:
                self.log("Rendering waveform image…")
                waveform_img = self.render_waveform_image(audio_path, ts)
                if waveform_img:
                    self.log("Waveform image created")
            else:
                self.log("matplotlib not available – skipping waveform rendering")

            # 4. Build video
            if AudioFileClip is None:
                self.log("moviepy not installed – skipping video generation")
                video_path = None
            else:
                self.log("Compositing video…")
                video_path = self.build_video(story_text, audio_path, waveform_img, ts)
                self.log(f"Video saved to {video_path}")

            messagebox.showinfo("Done", "Story generation complete!\n\nAudio: {}\nVideo: {}".format(audio_path, video_path or "(not created)"))
        except Exception as e:
            self.log(f"Error: {e}")
            messagebox.showerror("Error", str(e))
        finally:
            self.generate_btn.config(state="normal")

    # ------- Helper methods -------

    def generate_story_with_llm(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "llama-3.2-3b-instruct",
            "messages": [
                {"role": "system", "content": "You are a creative storyteller. Write a vivid, engaging short story based on the user's prompt."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.9,
            "max_tokens": 600
        }
        try:
            resp = requests.post(LM_STUDIO_URL, json=data, headers=headers, timeout=20)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            else:
                self.log(f"LLM responded with status {resp.status_code}")
                return ""
        except Exception as e:
            self.log(f"LLM request failed: {e}")
            return ""

    def kokoro_to_wav(self, text: str, voice: str, speed: float = 1.0) -> bytes:
        samples, sr = self.kokoro.create(text, voice=voice, speed=speed, lang="en-us")
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="WAV")
        return buf.getvalue()

    def render_waveform_image(self, audio_path: str, ts: str):
        try:
            data, sr = sf.read(audio_path)
            duration = len(data) / sr
            # Downsample for large files
            stride = max(1, int(sr / 1000))
            data_ds = data[::stride]
            t = np.linspace(0, duration, len(data_ds))
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(t, data_ds, color="#33c9ff")
            ax.set_axis_off()
            img_path = os.path.join(SAVE_DIR, f"waveform_{ts}.png")
            fig.savefig(img_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            return img_path
        except Exception as e:
            self.log(f"Waveform render failed: {e}")
            return None

    def build_video(self, story_text: str, audio_path: str, waveform_img: str, ts: str):
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        width, height = 1280, 720

        # Background
        if waveform_img and os.path.exists(waveform_img):
            bg_clip = ImageClip(waveform_img).set_duration(duration).resize(width=width)
        else:
            bg_clip = ColorClip(size=(width, height), color=(0, 0, 0)).set_duration(duration)

        # Text overlay (centered)
        txt_clip = TextClip(story_text, fontsize=40, color="white", size=(int(width*0.8), None), method="caption")
        txt_clip = txt_clip.set_position("center").set_duration(duration)

        video = CompositeVideoClip([bg_clip, txt_clip]).set_audio(audio_clip)

        video_path = os.path.join(SAVE_DIR, f"story_{ts}.mp4")
        video.write_videofile(video_path, fps=24, codec="libx264", audio_codec="aac")
        return video_path

    def run(self):
        self.log("Application started. Enter a prompt and click 'Generate'.")
        self.root.mainloop()

if __name__ == "__main__":
    StoryVideoApp().run() 