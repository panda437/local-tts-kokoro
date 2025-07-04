#!/bin/bash
echo "🗣️ Starting Kokoro FastAPI TTS Server..."

cd /Users/asifkabeer/Documents/Kokoro-FastAPI

# Check if we're on Mac with Apple Silicon
if [[ $(uname -m) == "arm64" ]]; then
    echo "Detected Apple Silicon Mac - using GPU script"
    ./start-gpu_mac.sh
else
    echo "Using CPU script"
    ./start-cpu.sh
fi
