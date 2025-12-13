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
    "LIVE": "(v) -> visual | (t) -> tone",
    "VISUAL": "(o) -> outline ON/OFF | (c) -> color ON/OFF |  (.) -> dots only | (p) -> punctuation | (l) -> letters | (d) -> digits | (a) -> all | (backspace) -> previous",
    "TONE": "(s) -> saturation | (g) -> gamma | (b) -> brightness pass | (backspace) -> previous",
    "SATURATION": "(left/right) -> adjust saturation | (backspace) -> previous",
    "GAMMA": "(left/right) -> adjust gamma | (backspace) -> previous",
    "BRIGHTNESS": "(left/right) -> adjust brightness | (backspace) -> previous",
}


# Default settings
DEFAULT_FPS = 6  # Textual rendering so slow..
DEFAULT_OUTLINE = False
DEFAULT_COLOR = False
DEFAULT_OUTLINE = False
DEFAULT_BRIGHTNESS_THRESHOLD = 100
