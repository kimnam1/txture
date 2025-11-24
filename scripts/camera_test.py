import sys
import argparse

from txture.devices import auto_scan_devices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-devices",
        type=int,
        default=5,
    )
    args = ap.parse_args()
    os_name, backends, devices = auto_scan_devices(args.max_devices)

    sys.stderr.write(f'os = "{os_name}"\n')
    sys.stderr.write(f"backends = {backends}\n")

    if not devices:
        sys.stderr.write("No camera devices found\n")
        return

    sys.stderr.write("Detected camera devices:\n")
    for info in devices:
        sys.stderr.write(
            f"  index = {info.index} backend = {info.backend} score = {info.score:.2f}\n"
        )


if __name__ == "__main__":
    main()
