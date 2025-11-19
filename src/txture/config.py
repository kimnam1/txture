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
    "LIVE": "v : visual | tone : tone | esc : exit",
    "VISUAL": "o: outline ON/OFF | p : punctuation | l: letters | d: digits | a: all | backspace : previous | esc : exit",
    "TONE": "s: saturation | g: gamma | b: brightness | backspace : previous | esc : exit",
    "SATURATION": "left/right: adjust saturation | backspace : previous | esc : exit",
    "GAMMA": "left/right: adjust gamma | backspace : previous | esc : exit",
    "BRIGHTNESS": "left/right: adjust brightness | backspace : previous | esc : exit",
}


# Default settings
DEFAULT_FPS = 30
DEFAULT_OUTLINE = False
DEFAULT_COLOR = False
DEFAULT_OUTLINE = False
