#!/usr/bin/env python3
"""Quick CLI to test PaddleOCR on a single image.

Usage:  python scripts/test_ocr.py [image_path]
Outputs recognised text to stdout and saves to <image>.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image
import cv2
import numpy as np
import easyocr

# ---------------------------------------------------------------------------
# model init once
# ---------------------------------------------------------------------------
print("📖 Loading EasyOCR …")
reader = easyocr.Reader(['en'], gpu=False, recog_network='english_g2')

# ---------------------------------------------------------------------------
# util
# ---------------------------------------------------------------------------

def preprocess(pil_img):
    import cv2, numpy as _np
    cv = cv2.cvtColor(_np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,31,11)
    thr_big = cv2.resize(thr, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_LINEAR)
    return thr_big

def ocr_image(img_arr):
    result = reader.readtext(img_arr)
    lines: list[str] = []
    for (_, text, _conf) in result:
        lines.append(text)
    return "\n".join(lines).strip()

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("book sample 2.jpg")
    if not img_path.exists():
        sys.exit(f"❌ File {img_path} not found")

    print(f"🔍 OCR on {img_path} …")
    img_pil = Image.open(img_path).convert("RGB")
    img_proc = preprocess(img_pil)

    # 1) raw
    text = ocr_image(img_proc)
    if text:
        print("\n==== Raw OCR ====\n" + text)
    else:
        # 2) adaptive threshold fallback
        # cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        # gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                cv2.THRESH_BINARY,35,11)
        # text = ocr_image(thresh)
        print("⚠️  No text recognised.")
        sys.exit(1)

    # save txt
    out_path = img_path.with_suffix(".txt")
    out_path.write_text(text, encoding="utf-8")
    print(f"\nSaved to {out_path}") 