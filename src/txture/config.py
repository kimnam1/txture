# Edge detection and processing parameters
CANNY_LOW = 30
CANNY_HIGH = 100
BLUR_KERNEL_SIZE = (5, 5)
SOBEL_KERNEL_SIZE = 5
MORPH_KERNEL_SIZE = (3, 3)


# Controller
CONTROLLER_WINDOW_SIZE = (1, 1)
CONTROLLER_WINDOW_POS = (-100, -100)
KEY_HELP_DICT = {
    "NORMAL": "(o) outline | (v) visual | (t) tone | (c) color ON/OFF | (h) help | (esc) quit | (y option) copy",
    "HELP": "(backspace/q) -> close help",
    "OUTLINE": "(o) outline OFF | (backspace/q) -> NORMAL",
    "VISUAL": "(l) letters | (p) punctuation | (d) digits | (.) dots | (backspace/q) -> NORMAL",
    "TONE": "(s) saturation | (g) gamma | (b) brightness | (backspace/q) -> NORMAL",
    "SATURATION": "(-/right) UP | (+/left) DOWN | (backspace/q) -> TONE",
    "GAMMA": "(-/right) UP | (+/left) DOWN | (backspace/q) -> TONE",
    "BRIGHTNESS": "(-/right) UP | (+/left) DOWN | (backspace/q) -> TONE",
}
KEY_COOLDOWN_S = 0.1
Y_SEQUENCE_WINDOW_S = 0.5


# Default settings
DEFAULT_FPS = 6  # Textual rendering so slow..
DEFAULT_OUTLINE = False
DEFAULT_COLOR = True
DEFAULT_BRIGHTNESS_THRESHOLD = 127
