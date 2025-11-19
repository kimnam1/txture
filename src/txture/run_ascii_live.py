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

from txture.control import (
    state as ctrl_state,
    handle_key,
    format_mode_line,
    format_info_line,
    format_help_line,
)


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
    ap.add_argument(
        "--control",
        action="store_true",
        help="Enable controller window",
    )
    ap.add_argument("--aspect", type=float, default=2.0)

    args = ap.parse_args()

    current_set = args.set
    use_color = args.color
    use_outline = args.outline

    files = discover_metric_files()
    if current_set not in files:
        print(f'[red]Error:[/red] Metric set "{current_set}" not found.')
        return
    lut = load_lut(files[current_set])

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

    if args.control:
        ctrl_state.charset = current_set
        ctrl_state.outline = use_outline
        ctrl_state.color = use_color

    sys.stdout.write("\x1b[?1049h\x1b[H\x1b[2J\x1b[?25l")
    sys.stdout.flush()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            terminal_size = shutil.get_terminal_size((80, 24))
            term_cols = terminal_size.columns
            term_rows = terminal_size.lines

            if args.control:
                desired_set = ctrl_state.charset
                if desired_set != current_set and desired_set in files:
                    lut = load_lut(files[desired_set])
                    current_set = desired_set

            if args.cols > 0:
                cols = args.cols
            else:
                cols = max(20, term_cols - 2)

            effective_outline = (
                ctrl_state.outline if args.control else use_outline
            )
            effective_color = ctrl_state.color if args.control else use_color
            sat_gain = ctrl_state.saturation_gain if args.control else 1

            features = process_frame(frame, outline_mode=effective_outline)

            key = -1

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

            if args.control:
                cv2.imshow("controller", np.zeros((1, 1, 3), dtype=np.uint8))

            if args.preview or args.control:
                key = cv2.waitKey(1) & 0xFF
            else:
                key = -1

            if args.control and key != 255 and key != -1:
                handle_key(ctrl_state, key)
                if not ctrl_state.running:
                    break

            if not args.control and key == 27:
                break

            lines, colors = frame_to_ascii(
                features.processed,
                lut,
                cols=cols,
                char_aspect=args.aspect,
                colorize=effective_color,
                saturation_gain=sat_gain,
            )

            max_ascii_rows = max(1, term_rows - 3)
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

            mode_line = format_mode_line(ctrl_state) + format_help_line(
                ctrl_state
            )
            info_line = format_info_line(ctrl_state)

            term_cols = terminal_size.columns

            current_line = len(lines) + 1
            while current_line < term_rows - 2:
                sys.stdout.write("\n")
                current_line += 1

            sys.stdout.write(mode_line.center(term_cols) + "\n")
            sys.stdout.write(info_line.center(term_cols) + "\n")

            sys.stdout.flush()

            time.sleep(max(0.0, 1.0 / args.fps))

    finally:
        cap.release()
        cv2.destroyAllWindows()
        sys.stdout.write("\x1b[?1049l\x1b[?25h")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
