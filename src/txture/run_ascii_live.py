import sys
from pathlib import Path

import time
import shutil
import argparse
import glob
from txture.loaders import load_lut
from txture.ascii_render import frame_to_ascii
from txture.devices import open_auto_camera
from txture.pipeline import process_frame
import cv2
import numpy as np


BASE = Path(__file__).resolve().parents[2]
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
        default="ascii_punctuation_only",
        help="ascii_all | ascii_punctuation_only | ascii_letters_only | ascii_digits_only | ascii_letters_digits_punct",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Target FPS for ASCII rendering",
    )
    ap.add_argument(
        "--color", action="store_true", help="Colorize ASCII output"
    )
    ap.add_argument(
        "--cols", type=int, default=0, help="columns count. 0=auto"
    )
    ap.add_argument(
        "--preview",
        action="store_true",
        help="Show processed frame in OpenCV window",
    )
    ap.add_argument(
        "--outline", action="store_true", help="Use outline mode (edge based)"
    )
    ap.add_argument("--aspect", type=float, default=2.0)

    args = ap.parse_args()

    files = discover_metric_files()
    if args.set not in files:
        print(f'[red]Error:[/red] Metric set "{args.set}" not found.')
        return
    lut = load_lut(files[args.set])
    label = args.set

    print(f"[info] set = {label} color = {args.color} fps = {args.fps} ")

    cap, cam_info = open_auto_camera(max_devices=5)
    print(
        f"using camera index={cam_info.index} backend={cam_info.backend} "
        f"score = {cam_info.score:.2f}"
    )

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

    sys.stdout.write("\x1b[?1049h\x1b[H\x1b[2J\x1b[?25l")
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

            terminal_size = shutil.get_terminal_size((80, 24))
            term_cols = terminal_size.columns
            term_rows = terminal_size.lines

            if args.cols > 0:
                cols = args.cols
            else:
                cols = max(20, term_cols - 2)

            features = process_frame(frame, outline_mode=args.outline)

            if args.preview:
                vis_orig = features.orig
                vis_det = features.det_vis
                vis_proc = features.processed

                target_height = 240
                h, w = vis_orig.shape[:2]
                new_w = int(w * target_height / h)

                vis_orig = cv2.resize(vis_orig, (new_w, target_height))
                vis_det = cv2.resize(vis_det, (new_w, target_height))
                vis_proc = cv2.resize(vis_proc, (new_w, target_height))

                tiles = [vis_orig, vis_det, vis_proc]
                preview = np.hstack(tiles)

                cv2.imshow("preview", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                # continue

            lines, colors = frame_to_ascii(
                features.processed,
                lut,
                cols=cols,
                char_aspect=args.aspect,
                colorize=args.color,
            )

            max_ascii_rows = max(1, term_rows - 2)
            if len(lines) > max_ascii_rows:
                lines = lines[:max_ascii_rows]
                if colors is not None:
                    colors = colors[:max_ascii_rows]

            sys.stdout.write("\x1b[H\x1b[2J")

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

            mode = "outline" if args.outline else "normal"
            status = f"[mode={mode}] [fps={args.fps:.1f}] [cols={cols}] [charset={label}]"
            padded = status.center(term_cols)

            current_line = len(lines) + 1
            while current_line < term_rows - 1:
                sys.stdout.write("\n")
                current_line += 1
            sys.stdout.write(padded + "\n")

            sys.stdout.flush()

            time.sleep(max(0.0, 1.0 / args.fps))

    finally:
        cap.release()
        cv2.destroyAllWindows()
        sys.stdout.write("\x1b[?1049l\x1b[?25h")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
