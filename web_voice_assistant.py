#!/usr/bin/env python3
"""
🎤 Web Voice Assistant with Integrated Spark TTS
Talk to AI in your browser!

Uses:
- Whisper (speech-to-text)
- LM Studio LLM (chat)
- Spark TTS (text-to-speech) - Latest cutting-edge TTS integrated directly!
"""

import os
import sys
import torch
import whisper
import requests
import tempfile
import wave
import io
import base64
import soundfile as sf
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import threading
from kokoro_onnx import Kokoro

app = Flask(__name__)

# Global progress tracker (for demo: only one user at a time)
progress = {
    'step': 'idle',
    'timers': {'whisper': 0, 'llm': 0, 'tts': 0, 'total': 0},
    'logs': [],
    'done': False
}
progress_lock = threading.Lock()

class VoiceAssistant:
    def __init__(self):
        self.llm_url = "http://localhost:1234/v1/chat/completions"
        self.whisper_model = None
        self.tts_pipe = None
        self.load_models()
        
    def load_models(self):
        """Load Whisper model and Spark TTS"""
        print("🎨 Loading Whisper model...")
        try:
            self.whisper_model = whisper.load_model("base")
            print("✅ Whisper model loaded!")
        except Exception as e:
            print(f"❌ Whisper loading failed: {e}")
            
        print("🔊 Loading Kokoro ONNX TTS…")
        # Look for model files in ./kokoro_models (downloaded separately)
        base_dir = os.path.join(os.path.dirname(__file__), "kokoro_models")
        model_path = os.path.join(base_dir, "kokoro-v1.0.int8.onnx")
        voices_path = os.path.join(base_dir, "voices-v1.0.bin")

        if not (os.path.exists(model_path) and os.path.exists(voices_path)):
            print("⚠️ Kokoro model files not found – TTS disabled. Make sure to download them into ./kokoro_models/")
            self.tts_pipe = None
        else:
            try:
                self.tts_pipe = Kokoro(model_path, voices_path)
                print("✅ Kokoro TTS loaded with", len(self.tts_pipe.voices), "voices!")
            except Exception as e:
                print(f"❌ Kokoro TTS loading failed: {e}")
                self.tts_pipe = None
            
    def transcribe_audio(self, audio_data):
        """Convert audio to text using Whisper"""
        if not self.whisper_model:
            return None
            
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(temp_path)
            text = result["text"].strip()
            
            # Clean up temp file
            os.unlink(temp_path)
            
            return text if text else None
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None

    def get_llm_response(self, user_input):
        """Get response from LM Studio"""
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model": "llama-3.2-3b-instruct",
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant. Keep responses concise and friendly."},
                    {"role": "user", "content": user_input}
                ],
                "temperature": 0.7,
                "max_tokens": 150
            }
            
            response = requests.post(self.llm_url, json=data, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return "Sorry, I'm having trouble connecting to the AI service."
                
        except Exception as e:
            print(f"❌ LLM error: {e}")
            return "Sorry, I encountered an error processing your request."

    def text_to_speech_tts(self, text, voice="af_heart", speed=1.0):
        """Convert text to speech using Kokoro ONNX"""
        if not self.tts_pipe:
            print("❌ Kokoro TTS not available")
            return None

        try:
            print(f"🎵 Kokoro generating speech… voice={voice}, text[:40]={text[:40]}…")
            samples, sample_rate = self.tts_pipe.create(text, voice=voice, speed=speed, lang="en-us")
            buffer = io.BytesIO()
            sf.write(buffer, samples, samplerate=sample_rate, format='WAV')
            print("✅ Speech generated successfully!")
            return buffer.getvalue()
        except Exception as e:
            print(f"❌ Kokoro TTS error: {e}")
            return None

    def check_services(self):
        """Check if services are available"""
        status = {"llm": False, "tts": False}
        
        # Check LLM
        try:
            response = requests.get("http://localhost:1234/v1/models", timeout=2)
            status["llm"] = response.status_code == 200
        except:
            pass
            
        # Check Kokoro TTS
        status["tts"] = self.tts_pipe is not None
            
        return status

# Initialize voice assistant
assistant = VoiceAssistant()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/status')
def status():
    """Service status endpoint"""
    return jsonify(assistant.check_services())

@app.route('/progress')
def get_progress():
    with progress_lock:
        return jsonify(progress)

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Process uploaded audio file"""
    try:
        import time
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        timings = {}
        start_total = time.time()
        with progress_lock:
            progress['step'] = 'whisper'
            progress['timers'] = {'whisper': 0, 'llm': 0, 'tts': 0, 'total': 0}
            progress['logs'] = ['Starting Whisper (STT)...']
            progress['done'] = False
        # Whisper timing
        start_whisper = time.time()
        user_text = assistant.transcribe_audio(audio_data)
        end_whisper = time.time()
        timings['whisper'] = round(end_whisper - start_whisper, 3)
        with progress_lock:
            progress['timers']['whisper'] = timings['whisper']
            progress['logs'].append(f'Whisper done: {timings["whisper"]}s')
            progress['step'] = 'llm'
            progress['logs'].append('Starting LM Studio (LLM)...')
        if not user_text:
            with progress_lock:
                progress['logs'].append('No speech detected.')
                progress['done'] = True
            return jsonify({"error": "No speech detected"})
        # LLM timing
        start_llm = time.time()
        ai_response = assistant.get_llm_response(user_text)
        end_llm = time.time()
        timings['llm'] = round(end_llm - start_llm, 3)
        with progress_lock:
            progress['timers']['llm'] = timings['llm']
            progress['logs'].append(f'LLM done: {timings["llm"]}s')
            progress['step'] = 'tts'
            progress['logs'].append('Starting Kokoro TTS...')
        # SparkTTS timing
        start_tts = time.time()
        # Convert to speech using Kyutai TTS
        audio_response = assistant.text_to_speech_tts(ai_response)
        end_tts = time.time()
        timings['tts'] = round(end_tts - start_tts, 3)
        timings['total'] = round(time.time() - start_total, 3)
        with progress_lock:
            progress['timers']['tts'] = timings['tts']
            progress['timers']['total'] = timings['total']
            progress['logs'].append(f'Kokoro TTS done: {timings["tts"]}s')
            progress['logs'].append(f'Total: {timings["total"]}s')
            progress['step'] = 'done'
            progress['done'] = True
        audio_base64 = None
        if audio_response:
            audio_base64 = base64.b64encode(audio_response).decode('utf-8')
        return jsonify({
            "user_text": user_text,
            "ai_response": ai_response,
            "audio": audio_base64,
            "timings": timings
        })
    except Exception as e:
        print(f"❌ Processing error: {e}")
        with progress_lock:
            progress['logs'].append(f'Error: {e}')
            progress['step'] = 'error'
            progress['done'] = True
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    print("🎤 Starting Web Voice Assistant with Kokoro TTS…")
    print("🌐 Open your browser to: http://localhost:5002")
    app.run(debug=True, host='0.0.0.0', port=5002)
