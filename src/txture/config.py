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
    "NORMAL": "[o] outline ON | [c] color ON/OFF | [v] visual",
    "HELP": "[backspace/q] close help",
    "OUTLINE": "[o] outline OFF | [backspace/q] Back to NORMAL",
    "VISUAL": "[l] letters | [p] punctuation | [d] digits | [./o] dots | [backspace/q] Back to NORMAL",
}
KEY_COOLDOWN_S = 0.1
Y_SEQUENCE_WINDOW_S = 0.5


# Default settings
DEFAULT_FPS = 20  # App FPS
DEFAULT_RENDER_FPS = 3  # Rendering so slow... Seperate FPS for text rendering
DEFAULT_OUTLINE = False
DEFAULT_COLOR = True
DEFAULT_GESTURE_CONFIDENCE = 0.5
DEFAULT_FACE_CONFIDENCE = 0.5
DEFAULT_STABLE_FRAMES = 3
DEFAULT_BRIGHTNESS_THRESHOLD = 127

# FACE MAP
FACE_MAP = {
    "neutral": "😐",
    "happy": "😄",
    "sad": "😢",
    "surprise": "😲",
}

# Effect settings
RAINBOW_RIPPLE_SECONDS = 1
