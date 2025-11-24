from PIL import Image, ImageDraw, ImageFont
import string
import json
import numpy as np
from datetime import datetime

ASCII_SPACE = " "
ASCII_LETTERS = string.ascii_letters
ASCII_DIGITS = string.digits
ASCII_PUNCTUATION = string.punctuation
ASCII_ALL = ASCII_LETTERS + ASCII_DIGITS + ASCII_PUNCTUATION + ASCII_SPACE

ASCII_LETTERS_ONLY = ASCII_LETTERS + ASCII_SPACE
ASCII_DIGITS_ONLY = ASCII_DIGITS + ASCII_SPACE
ASCII_PUNCTUATION_ONLY = ASCII_PUNCTUATION + ASCII_SPACE


def build_glyph_metrics(chars, font_path, font_size, canvas_size=32, thr=200):
    font = ImageFont.truetype(str(font_path), font_size)
    metrics = {}

    for ch in chars:
        img = Image.new("L", (canvas_size, canvas_size), 255)
        d = ImageDraw.Draw(img)

        try:
            bbox = d.textbbox((0, 0), ch, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            w, h = d.textsize(ch, font=font)

        x = (canvas_size - w) // 2
        y = (canvas_size - h) // 2
        d.text((x, y), ch, font=font, fill=0)

        arr = np.array(img, dtype=np.uint8)
        darkness = float((arr < thr).mean())
        metrics[ch] = darkness

    return metrics


def make_ramp(metrics, invert=False):
    items = sorted(metrics.items(), key=lambda kv: kv[1], reverse=invert)
    return [ch for ch, _ in items]


def make_lut(ramp, levels=256):
    n = len(ramp)
    return [ramp[int(v * (n - 1) / (levels - 1))] for v in range(levels)]


def save_metrics_json(path, font_meta, metrics, ramp, lut):
    data = {
        "meta": {
            **font_meta,
            "chars_count": len(metrics),
            "created_at": datetime.now().isoformat(),
        },
        "metrics": metrics,
        "ramp": ramp,
        "lut": lut,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved: {path}")
