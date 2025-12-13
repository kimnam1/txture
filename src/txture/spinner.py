import itertools
import sys
import time
import threading
from typing import Callable, Optional


class Spinner(object):
    spinner_cycle = itertools.cycle(
        ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    )

    def __init__(
        self,
        message: str = "Loading",
        *,
        callback: Optional[Callable[[str], None]] = None,
        interval: float = 0.2,
    ):
        self.message = message
        self.callback = callback
        self.interval = interval
        self.stop_running = threading.Event()
        self.spin_thread = threading.Thread(target=self.init_spin, daemon=True)

    def start(self):
        self.spin_thread.start()

    def stop(self):
        self.stop_running.set()
        self.spin_thread.join()
        if self.callback is None:
            sys.stdout.write("\r\x1b[2K")
            sys.stdout.flush()

    def init_spin(self):
        while not self.stop_running.is_set():
            frame = next(self.spinner_cycle)
            if self.callback is not None:
                self.callback(f"{frame} {self.message}")
                time.sleep(self.interval)
                continue
            bold_frame = f"\x1b[1m{frame}\x1b[0m"
            sys.stdout.write(f"\r{bold_frame} {self.message}")
            sys.stdout.flush()
            time.sleep(self.interval)
            sys.stdout.write("\b")
