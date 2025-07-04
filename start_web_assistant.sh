#!/bin/bash
echo "🎤 Starting Web Voice Assistant..."
echo "🌐 Will open in your browser at: http://localhost:5002"

cd /Users/asifkabeer/Documents/voice-assistant
source voice-env/bin/activate

# Start the web server
python3 web_voice_assistant.py
