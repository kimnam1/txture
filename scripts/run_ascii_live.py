import sys
import time
import shutil
import argparse
import glob
from pathlib import Path
import cv2
from txture.loaders import load_lut
from txture.ascii_render import frame_to_ascii

BASE = Path(__file__).resolve().parents[1]
METRIC_DIR = BASE / "data" / "metrics"


def discover_metric_files():
    paths = glob.glob(str(METRIC_DIR / "*.json"))
    mapping = {}
    for p in paths:
        name = Path(p).name
        label = name.split("__")[0]
        mapping.setdefault(label, Path(p))
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--set",
        default="ascii_all",
        help="ascii_all | ascii_punctuation_only | ascii_letters_only | ascii_digits_only | ascii_letters_digits_punct",
    )
    ap.add_argument("--lut", default=None, help="Path to custom LUT JSON file")
    ap.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Target FPS for ASCII rendering",
    )
    ap.add_argument(
        "--color", action="store_true", help="Colorize ASCII output"
    )
    ap.add_argument(
        "--cols", type=int, default=0, help="columns count. 0=auto"
    )
    ap.add_argument("--aspect", type=float, default=2.0)
    ap.add_argument(
        "--device", type=int, default=0, help="Camera device index"
    )
    ap.add_argument(
        "--equalize",
        action="store_true",
        help="Apply histogram equalization to the frame before processing",
    )

    args = ap.parse_args()

    if args.lut:
        lut_path = Path(args.lut)
        lut = load_lut(lut_path)
        label = lut_path.stem

    else:
        files = discover_metric_files()
        if args.set not in files:
            print(f'[red]Error:[/red] Metric set "{args.set}" not found.')
            return
        lut_path = files[args.set]
        lut = load_lut(lut_path)
        label = args.set

    print(f"[info] set = {label} color = {args.color} fps = {args.fps} ")

    cap = cv2.VideoCapture(args.device, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print("[red]Error:[/red] Cannot open camera")
        return

    boot_ok = False
    for _ in range(30):
        ok, frame = cap.read()
        if ok:
            boot_ok = True
            break
        time.sleep(0.1)
    if not boot_ok:
        print("[red]Error:[/red] Cannot read from camera")
        return

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    sys.stdout.write("\x1b[2J")
    sys.stdout.write("\x1b[?25l")
    sys.stdout.flush()

    print(
        f"set: {label}, color: {args.color}, fps: {args.fps}, cols: {args.cols}, aspect: {args.aspect}",
        file=sys.stderr,
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            if args.cols > 0:
                cols = args.cols
            else:
                cols = max(20, shutil.get_terminal_size((80, 24)).columns - 2)

            if args.equalize:
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                g = cv2.equalizeHist(g)
                frame = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

            lines, colors = frame_to_ascii(
                frame,
                lut,
                cols=cols,
                char_aspect=args.aspect,
                colorize=args.color,
            )

            sys.stdout.write("\x1b[H")

            if args.color and colors is not None:
                out_lines = []
                for y, line in enumerate(lines):
                    segs = []
                    for x, ch in enumerate(line):
                        r, g, b = colors[y][x]
                        segs.append(f"\x1b[38;2;{r};{g};{b}m{ch}")
                    out_lines.append("".join(segs) + "\x1b[0m")
                sys.stdout.write("\n".join(out_lines) + "\n")
            else:
                for line in lines:
                    sys.stdout.write(line + "\n")
            sys.stdout.flush()

            k = cv2.waitKey(1) & 0xFF

            if k == 27:  # esc
                break
            elif k in (ord("p"), ord("l"), ord("d"), ord("a")):
                label_map = {
                    ord("p"): "ascii_punctuation_only",
                    ord("l"): "ascii_letters_only",
                    ord("d"): "ascii_digits_only",
                    ord("a"): "ascii_all",
                }
                new_label = label_map[k]
                files = discover_metric_files()
                if new_label in files:
                    lut = load_lut(files[new_label])
                    label = new_label
            elif k == ord("c"):
                args.color = not args.color

            time.sleep(max(0.0, 1.0 / args.fps))

    finally:
        cap.release()
        sys.stdout.write("\x1b[?25h")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
