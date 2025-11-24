import itertools
import sys
import time
import threading


class Spinner(object):
    spinner_cycle = itertools.cycle(
        ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    )

    def __init__(self, message="Loading"):
        self.message = message
        self.stop_running = threading.Event()
        self.spin_thread = threading.Thread(target=self.init_spin)

    def start(self):
        self.spin_thread.start()

    def stop(self):
        self.stop_running.set()
        self.spin_thread.join()
        sys.stdout.write("\r\x1b[2K")
        sys.stdout.flush()

    def init_spin(self):
        while not self.stop_running.is_set():
            frame = next(self.spinner_cycle)
            bold_frame = f"\x1b[1m{frame}\x1b[0m"
            sys.stdout.write(f"\r{bold_frame} {self.message}")
            sys.stdout.flush()
            time.sleep(0.25)
            sys.stdout.write("\b")
