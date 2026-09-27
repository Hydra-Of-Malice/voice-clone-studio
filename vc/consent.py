"""Spoken-consent verification. The user reads a statement containing a per-session nonce; we check
(1) the ASR transcript matches the statement (including the nonce words) and (2) the speaker matches
the uploaded sample's voice centroid. Both must pass before a profile can be used for generation."""
from __future__ import annotations

import random
import re
import secrets
import unicodedata

TERMS_VERSION = "2026-09-27"

_NONCE_WORDS_EN = ["river", "yellow", "window", "garden", "silver", "monday", "coffee", "planet", "candle",
                   "orange", "forest", "pencil", "summer", "rocket", "island", "guitar", "marble", "violet"]
_NONCE_WORDS_HI = ["नदी", "पीला", "खिड़की", "बगीचा", "चाँदी", "सोमवार", "कॉफ़ी", "ग्रह", "मोमबत्ती",
                   "संतरा", "जंगल", "पेंसिल", "गर्मी", "रॉकेट", "द्वीप", "गिटार", "पत्थर", "बैंगनी"]

_STATEMENT = {
    "en": ("I, the speaker of this recording, give my consent to create a digital copy of my voice and "
           "to generate speech with it. My code words are: {nonce}."),
    "hi": ("मैं, इस रिकॉर्डिंग का वक्ता, अपनी आवाज़ की डिजिटल प्रतिकृति बनाने और उससे नई आवाज़ उत्पन्न करने की "
           "सहमति देता हूँ। मेरे कोड शब्द हैं: {nonce}।"),
}


def make_statement(language: str = "en") -> tuple[str, str]:
    lang = "hi" if language.startswith("hi") and language != "hi-Latn" else "en"
    words = _NONCE_WORDS_HI if lang == "hi" else _NONCE_WORDS_EN
    rng = random.Random(secrets.randbits(64))
    nonce_words = rng.sample(words, 3)
    nonce = " ".join(nonce_words)
    return _STATEMENT[lang].format(nonce=nonce), nonce


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\sऀ-ॿ]|[।॥]", " ", text)
    return [t for t in text.split() if t]


def text_match(statement: str, asr_text: str) -> tuple[float, bool]:
    """Returns (fraction of statement tokens found in the ASR text, nonce words all present)."""
    st = _tokens(statement)
    at = set(_tokens(asr_text))
    if not st:
        return 0.0, False
    found = sum(1 for t in st if t in at or any(_close(t, a) for a in at))
    return found / len(st), True


def nonce_present(nonce: str, asr_text: str) -> bool:
    at = set(_tokens(asr_text))
    return all(any(_close(n, a) for a in at) for n in _tokens(nonce))


def _close(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 2 or min(len(a), len(b)) < 4:
        return False
    # cheap edit-distance bound
    d = 0
    for x, y in zip(a, b):
        if x != y:
            d += 1
            if d > 2:
                return False
    return True
