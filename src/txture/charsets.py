import string
import unicodedata as ud
from typing import Tuple
from wcwidth import wcwidth


def _sanitize(chars: str, target_width: int = 1) -> str:
    safe = []
    for ch in chars:
        if ud.category(ch)[0] == "C":
            continue
        if ud.combining(ch):
            continue
        if wcwidth(ch) != target_width:
            continue
        safe.append(ch)
    return "".join(sorted(set(safe), key=lambda c: ord(c)))


def ascii_all() -> Tuple[str, str]:
    raw = "".join(ch for ch in string.printable if ch not in "\t\n\r\x0b\x0c")
    return "ascii_all", _sanitize(raw, target_width=1)


def ascii_letters_only() -> Tuple[str, str]:
    raw = string.ascii_letters + " "
    return "ascii_letters_only", _sanitize(raw, target_width=1)


def ascii_punctuation_only() -> Tuple[str, str]:
    raw = string.punctuation + " "
    return "ascii_punctuation_only", _sanitize(raw, target_width=1)


UNICODE_PUNCT = (
    string.punctuation + "¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿×÷“”‘’„‟‹›–—…•"
)


def unicode_punctuation_only() -> tuple[str, str]:
    return "unicode_punctuation_only", _sanitize(UNICODE_PUNCT, target_width=1)


def ascii_dots_only() -> Tuple[str, str]:
    raw = "•" + " "
    return "ascii_dots_only", _sanitize(raw, target_width=1)


def ascii_digits_only() -> Tuple[str, str]:
    raw = string.digits + " "
    return "ascii_digits_only", _sanitize(raw, target_width=1)


def ascii_letters_digits_punct() -> Tuple[str, str]:
    raw = string.ascii_letters + string.digits + string.punctuation
    return "ascii_letters_digits_punct", _sanitize(raw, target_width=1)


# _CJK_SEED_HANGUL = " ·.:;가각간갈감갑값강개객거걱건검겁것게겨결경고과곽관광교구국군권귀규극근금기"
# _CJK_SEED_HANZI = " 一二三上下大小中日月田目口山川林森雨風海国图爱梦"


# def hangul_subset() -> Tuple[str, str]:
#     return "hangul_subset", _sanitize(_CJK_SEED_HANGUL, target_width=2)


# def hanzi_subset() -> Tuple[str, str]:
#     return "hanzi_subset", _sanitize(_CJK_SEED_HANZI, target_width=2)


__all__ = [
    "ascii_all",
    "ascii_letters_digits_punct",
    "ascii_punctuation_only",
    "ascii_digits_only",
    "ascii_letters_only",
    #     "hangul_subset",
    #     "hanzi_subset",
]
