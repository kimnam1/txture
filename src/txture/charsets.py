from pathlib import Path
import unicodedata as ud
from wcwidth import wcwidth

from txture.ascii_maps import (
    build_glyph_metrics,
    make_ramp,
    make_lut,
    save_metrics_json,
)

BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "data" / "metrics"

HANZI_PATH = BASE / "data" / "charsets" / "hanzi_10k.txt"

FONT = BASE / "data" / "fonts" / "DejaVuSansMono-Bold.ttf"


def _sanitize(chars: str, target_width: int = 2) -> str:
    safe: list[str] = []
    for ch in chars:
        if ud.category(ch)[0] == "C":
            continue
        if ud.combining(ch):
            continue
        if wcwidth(ch) != target_width:
            continue
        safe.append(ch)
    # 유니코드 code point 기준 정렬 + 중복 제거
    return "".join(sorted(set(safe), key=lambda c: ord(c)))


def _load_hanzi_charset(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    # 파일이 한 줄에 1글자든, 줄바꿈이 있든 대응
    raw = (
        raw.replace("\n", "")
        .replace("\r", "")
        .replace("\t", "")
        .replace(" ", "")
    )

    # 공백은 별도로 넣는다
    chars = _sanitize(raw, target_width=2)
    if " " not in chars:
        chars = chars + " "
    return chars


def main() -> None:
    label = "hanzi_10k"

    if not HANZI_PATH.exists():
        raise FileNotFoundError(f"Missing charset file: {HANZI_PATH}")

    chars = _load_hanzi_charset(HANZI_PATH)

    # 한자는 글자 폭이 2인 경우가 많다. 폰트/캔버스 크기를 크게 잡는 편이 안정적이다.
    font_size = 20
    canvas_size = 48
    thr = 200

    metrics = build_glyph_metrics(
        chars,
        FONT,
        font_size=font_size,
        canvas_size=canvas_size,
        thr=thr,
    )

    ramp = make_ramp(metrics, invert=False)
    lut = make_lut(ramp, levels=256)

    meta = {
        "charset_label": label,
        "charset_path": str(HANZI_PATH),
        "charset_count": len(chars),
        "font_name": FONT.name,
        "font_path": str(FONT),
        "font_size": font_size,
        "canvas_size": canvas_size,
        "threshold": thr,
        "target_width": 2,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{label}__{FONT.stem}_{font_size}.json"

    save_metrics_json(out_path, meta, metrics, ramp, lut)

    print(f"JSON created: {out_path}")


if __name__ == "__main__":
    main()
