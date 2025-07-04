#!/usr/bin/env python3
"""
Voice Assistant Setup
Installs all dependencies
"""

import subprocess
import sys
import platform

def run_command(cmd):
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def install_system_dependencies():
    """Install system dependencies if needed"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        print("🍎 Detected macOS")
        print("Note: Make sure you have PortAudio installed")
        print("If you get errors, run: brew install portaudio")
    
def main():
    print("🎤 Setting up Voice Assistant...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Need Python 3.8 or higher")
        return
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install system dependencies
    install_system_dependencies()
    
    # Python packages
    packages = [
        "openai-whisper",
        "pyaudio", 
        "pygame",
        "requests"
    ]
    
    print("\n📦 Installing Python packages...")
    for package in packages:
        print(f"Installing {package}...")
        if not run_command(f"pip3 install {package}"):
            print(f"❌ Failed to install {package}")
            if package == "pyaudio":
                print("💡 If PyAudio fails on macOS, try:")
                print("   brew install portaudio")
                print("   pip3 install pyaudio")
            return
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Start Kokoro FastAPI server (see Kokoro-FastAPI folder)")
    print("2. Start LM Studio with your model on port 1234")
    print("3. Run: python3 voice_assistant.py")

if __name__ == "__main__":
    main()
