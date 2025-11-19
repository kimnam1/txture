from pathlib import Path
from txture.charsets import (
    ascii_all,
    ascii_letters_only,
    ascii_punctuation_only,
    ascii_digits_only,
    ascii_letters_digits_punct,
)
from txture.ascii_maps import (
    build_glyph_metrics,
    make_ramp,
    make_lut,
    save_metrics_json,
)

BASE = Path(__file__).resolve().parents[1]
FONT = BASE / "data" / "fonts" / "DejaVuSansMono-Bold.ttf"
OUT_DIR = BASE / "data" / "metrics"


def main():
    # label, chars = ascii_all()
    # label, chars = ascii_letters_digits_punct()
    # label, chars = ascii_letters_only()
    # label, chars = ascii_punctuation_only()
    label, chars = ascii_digits_only()

    font_size = 16
    canvas_size = 32
    thr = 200

    metrics = build_glyph_metrics(
        chars, FONT, font_size=font_size, canvas_size=canvas_size, thr=thr
    )

    ramp = make_ramp(metrics, invert=False)

    lut = make_lut(ramp, levels=256)

    meta = {
        "charset_label": label,
        "font_name": FONT.name,
        "font_path": str(FONT),
        "font_size": font_size,
        "canvas_size": canvas_size,
        "threshold": thr,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{label}__{FONT.stem}_{font_size}.json"

    save_metrics_json(out_path, meta, metrics, ramp, lut)

    print(f"JSON created: {out_path}")


if __name__ == "__main__":
    main()
