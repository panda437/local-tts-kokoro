#!/usr/bin/env python3
"""
🎤 Voice Assistant App
Talk to AI and hear responses!

Uses:
- Whisper (speech-to-text)
- LM Studio LLM (chat)
- Kokoro FastAPI (text-to-speech)
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import requests
import json
import whisper
import pyaudio
import wave
import pygame
import io
import tempfile
from datetime import datetime

class VoiceAssistant:
    def __init__(self):
        self.setup_ui()
        self.setup_audio()
        self.setup_models()
        
        # Server endpoints
        self.llm_url = "http://localhost:1234/v1/chat/completions"
        self.tts_url = "http://localhost:8000/tts"  # Kokoro FastAPI default
        
        # Audio settings
        self.is_recording = False
        self.is_speaking = False
        
    def setup_ui(self):
        """Create the user interface"""
        self.root = tk.Tk()
        self.root.title("🎤 Voice Assistant")
        self.root.geometry("600x500")
        self.root.configure(bg="#2c3e50")
        
        # Title
        title = tk.Label(
            self.root, 
            text="🎤 Voice Assistant", 
            font=("Arial", 24, "bold"),
            bg="#2c3e50", 
            fg="#ecf0f1"
        )
        title.pack(pady=20)
        
        # Status
        self.status_label = tk.Label(
            self.root, 
            text="Ready to chat!", 
            font=("Arial", 12),
            bg="#2c3e50", 
            fg="#95a5a6"
        )
        self.status_label.pack(pady=5)
        
        # Chat area
        chat_frame = tk.Frame(self.root, bg="#2c3e50")
        chat_frame.pack(expand=True, fill="both", padx=20, pady=10)
        
        self.chat_area = scrolledtext.ScrolledText(
            chat_frame,
            height=15,
            font=("Arial", 11),
            bg="#34495e",
            fg="#ecf0f1",
            insertbackground="#ecf0f1",
            wrap=tk.WORD
        )
        self.chat_area.pack(expand=True, fill="both")
        
        # Control buttons
        button_frame = tk.Frame(self.root, bg="#2c3e50")
        button_frame.pack(pady=20)
        
        self.record_button = tk.Button(
            button_frame,
            text="🎤 Hold to Talk",
            font=("Arial", 14, "bold"),
            bg="#e74c3c",
            fg="white",
            activebackground="#c0392b",
            width=15,
            height=2
        )
        self.record_button.pack(side=tk.LEFT, padx=10)
        self.record_button.bind("<Button-1>", self.start_recording)
        self.record_button.bind("<ButtonRelease-1>", self.stop_recording)
        
        self.stop_button = tk.Button(
            button_frame,
            text="🛑 Stop AI",
            font=("Arial", 14, "bold"),
            bg="#f39c12",
            fg="white",
            activebackground="#e67e22",
            width=15,
            height=2,
            command=self.stop_speaking
        )
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
        # Connection status
        self.connection_frame = tk.Frame(self.root, bg="#2c3e50")
        self.connection_frame.pack(pady=5)
        
        self.llm_status = tk.Label(
            self.connection_frame, 
            text="LLM: Checking...", 
            font=("Arial", 10),
            bg="#2c3e50", 
            fg="#95a5a6"
        )
        self.llm_status.pack(side=tk.LEFT, padx=10)
        
        self.tts_status = tk.Label(
            self.connection_frame, 
            text="TTS: Checking...", 
            font=("Arial", 10),
            bg="#2c3e50", 
            fg="#95a5a6"
        )
        self.tts_status.pack(side=tk.LEFT, padx=10)
        
    def setup_audio(self):
        """Initialize audio components"""
        pygame.mixer.init()
        self.audio = pyaudio.PyAudio()
        
    def setup_models(self):
        """Load Whisper model"""
        self.update_status("Loading Whisper model...")
        try:
            self.whisper_model = whisper.load_model("base")
            self.log_message("System", "Whisper model loaded successfully!")
        except Exception as e:
            self.log_message("Error", f"Failed to load Whisper: {e}")
            
    def check_connections(self):
        """Check if services are running"""
        # Check LLM Studio
        try:
            response = requests.get("http://localhost:1234/v1/models", timeout=2)
            if response.status_code == 200:
                self.llm_status.config(text="LLM: ✅ Connected", fg="#27ae60")
            else:
                self.llm_status.config(text="LLM: ❌ Error", fg="#e74c3c")
        except:
            self.llm_status.config(text="LLM: ❌ Offline", fg="#e74c3c")
            
        # Check Kokoro TTS
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            self.tts_status.config(text="TTS: ✅ Connected", fg="#27ae60")
        except:
            self.tts_status.config(text="TTS: ❌ Offline", fg="#e74c3c")
            
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update()
        
    def log_message(self, sender, message):
        """Add message to chat area"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.chat_area.insert(tk.END, f"[{timestamp}] {sender}: {message}\\n\\n")
        self.chat_area.see(tk.END)
        self.root.update()
        
    def start_recording(self, event):
        """Start recording audio"""
        if self.is_speaking:
            return
            
        self.is_recording = True
        self.record_button.config(text="🔴 Recording...", bg="#27ae60")
        self.update_status("Recording... Release button when done")
        
        # Start recording in thread
        threading.Thread(target=self.record_audio, daemon=True).start()
        
    def stop_recording(self, event):
        """Stop recording and process"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        self.record_button.config(text="🎤 Hold to Talk", bg="#e74c3c")
        self.update_status("Processing speech...")
        
    def record_audio(self):
        """Record audio from microphone"""
        chunk = 1024
        format = pyaudio.paInt16
        channels = 1
        rate = 16000
        
        stream = self.audio.open(
            format=format,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk
        )
        
        frames = []
        
        while self.is_recording:
            data = stream.read(chunk)
            frames.append(data)
            
        stream.stop_stream()
        stream.close()
        
        if frames:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                wf = wave.open(tmp_file.name, 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(self.audio.get_sample_size(format))
                wf.setframerate(rate)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                # Process the audio
                self.process_speech(tmp_file.name)
                
    def process_speech(self, audio_file):
        """Convert speech to text and get AI response"""
        try:
            # Speech to text with Whisper
            self.update_status("Converting speech to text...")
            result = self.whisper_model.transcribe(audio_file)
            user_text = result["text"].strip()
            
            if not user_text:
                self.update_status("No speech detected. Try again.")
                return
                
            self.log_message("You", user_text)
            
            # Get AI response
            self.update_status("Getting AI response...")
            ai_response = self.get_ai_response(user_text)
            
            if ai_response:
                self.log_message("AI", ai_response)
                
                # Convert to speech
                self.update_status("Converting to speech...")
                self.text_to_speech(ai_response)
            else:
                self.update_status("Failed to get AI response")
                
        except Exception as e:
            self.log_message("Error", f"Speech processing failed: {e}")
            self.update_status("Ready to chat!")
            
    def get_ai_response(self, text):
        """Get response from LM Studio"""
        try:
            payload = {
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a helpful voice assistant. Keep responses concise and conversational."
                    },
                    {
                        "role": "user", 
                        "content": text
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 150
            }
            
            response = requests.post(
                self.llm_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                return f"LLM Error: {response.status_code}"
                
        except Exception as e:
            return f"Failed to connect to LLM: {e}"
            
    def text_to_speech(self, text):
        """Convert text to speech using Kokoro"""
        try:
            self.is_speaking = True
            
            payload = {
                "text": text,
                "voice": "af_sarah",  # Default voice
                "speed": 1.0
            }
            
            response = requests.post(
                self.tts_url,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                # Play audio
                audio_data = io.BytesIO(response.content)
                pygame.mixer.music.load(audio_data)
                pygame.mixer.music.play()
                
                # Wait for audio to finish
                while pygame.mixer.music.get_busy() and self.is_speaking:
                    time.sleep(0.1)
                    
            else:
                self.log_message("Error", f"TTS failed: {response.status_code}")
                
        except Exception as e:
            self.log_message("Error", f"TTS error: {e}")
        finally:
            self.is_speaking = False
            self.update_status("Ready to chat!")
            
    def stop_speaking(self):
        """Stop AI speech"""
        self.is_speaking = False
        pygame.mixer.music.stop()
        self.update_status("Speech stopped")
        
    def run(self):
        """Start the application"""
        # Check connections on startup
        self.root.after(1000, self.check_connections)
        
        # Check connections periodically
        def periodic_check():
            self.check_connections()
            self.root.after(10000, periodic_check)  # Every 10 seconds
            
        self.root.after(2000, periodic_check)
        
        self.log_message("System", "Voice Assistant started! Hold the microphone button to talk.")
        self.root.mainloop()

if __name__ == "__main__":
    app = VoiceAssistant()
    app.run()
