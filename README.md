# 🎤 Voice Assistant

**Talk to AI and hear responses!**

Complete voice assistant using:
- **Whisper** (speech-to-text)
- **Your LM Studio LLM** (chat responses) 
- **Kokoro FastAPI** (text-to-speech)

## 🚀 Quick Start

### 1. Setup Dependencies
```bash
cd /Users/asifkabeer/Documents/voice-assistant
python3 setup.py
```

### 2. Start the Services

**Terminal 1 - Start Kokoro TTS:**
```bash
./start_kokoro.sh
```
Wait for: `Application startup complete`

**Terminal 2 - Your LM Studio is already running on port 1234 ✅**

### 3. Start Voice Assistant
```bash
python3 voice_assistant.py
```

## 🎯 How to Use

1. **Hold** the 🎤 button to record your voice
2. **Release** when done speaking  
3. AI will respond with voice!

## 📱 UI Features

- **🎤 Hold to Talk** - Press and hold to record
- **🛑 Stop AI** - Stop AI speech
- **Connection Status** - Shows LLM/TTS status
- **Chat History** - See all conversations
- **Real-time Status** - Shows what's happening

## 🔧 Settings

The app connects to:
- **LM Studio**: `http://localhost:1234` (your current setup)
- **Kokoro TTS**: `http://localhost:8000` (default)

## 🎭 Available Voices

You can change the voice in the code:
- `af_sarah` (default)
- `af_bella` 
- `af_nicole`
- `af_sky`
- And more!

## 🔊 Example Conversation

```
You: "What's the weather like today?"
AI: "I'm sorry, I don't have access to real-time weather data..."
```

## 🛠️ Troubleshooting

**"LLM: ❌ Offline"**
- Make sure LM Studio is running on port 1234
- Check that a model is loaded

**"TTS: ❌ Offline"** 
- Start Kokoro with `./start_kokoro.sh`
- Wait for startup to complete

**PyAudio errors on macOS:**
```bash
brew install portaudio
pip3 install pyaudio
```

**No microphone access:**
- Grant microphone permission in System Preferences

## ✨ Features

- ✅ Real-time voice recognition
- ✅ Natural AI conversations  
- ✅ High-quality voice synthesis
- ✅ Visual chat history
- ✅ Connection monitoring
- ✅ Easy push-to-talk interface

**Just talk and listen! 🎉**

## 🚀 Setup

1. Clone repository
2. Create Python env & install requirements (see `setup.py` or `Spark-TTS/requirements.txt`).
3. Download Kokoro models (~400 MB) which are **git-ignored**:

```bash
mkdir -p kokoro_models
curl -L -o kokoro_models/kokoro-v1.0.int8.onnx "https://huggingface.co/k2-fsa/kokoro-onnx/resolve/main/kokoro-v1.0.int8.onnx"
curl -L -o kokoro_models/voices-v1.0.bin "https://huggingface.co/k2-fsa/kokoro-onnx/resolve/main/voices-v1.0.bin"
```

Alternatively, run the helper script:

```bash
python scripts/download_kokoro_models.py
```
