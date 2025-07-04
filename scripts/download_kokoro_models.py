import os
from pathlib import Path
import urllib.request

URLS = {
    "kokoro-v1.0.int8.onnx": "https://huggingface.co/k2-fsa/kokoro-onnx/resolve/main/kokoro-v1.0.int8.onnx",
    "voices-v1.0.bin": "https://huggingface.co/k2-fsa/kokoro-onnx/resolve/main/voices-v1.0.bin",
}


def download(url: str, dest: Path):
    print(f"📥 Downloading {dest.name}…")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print("✅ Done")


def main():
    base_dir = Path(__file__).resolve().parent.parent / "kokoro_models"
    for filename, url in URLS.items():
        dest = base_dir / filename
        if dest.exists():
            print(f"✔️ {filename} already exists, skipping")
            continue
        download(url, dest)


if __name__ == "__main__":
    main() 