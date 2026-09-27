"""Text front end for new scripts: language detection (script-based, Hinglish-aware), normalisation
of numbers / currency / abbreviations, and sentence-preserving chunking.

Nothing here translates. Hindi stays Hindi, English stays English, mixed stays mixed."""
from __future__ import annotations

import re
from dataclasses import dataclass

from vc.asr.base import script_language

_ABBREV_EN = {
    "Dr.": "Doctor", "Mr.": "Mister", "Mrs.": "Missus", "Ms.": "Miss", "Prof.": "Professor",
    "St.": "Saint", "vs.": "versus", "etc.": "et cetera", "e.g.": "for example", "i.e.": "that is",
    "AI": "A I", "ML": "M L", "API": "A P I", "UI": "U I", "URL": "U R L", "GPU": "G P U", "CPU": "C P U",
    "OTP": "O T P", "EMI": "E M I", "UPI": "U P I", "SMS": "S M S", "ATM": "A T M", "PDF": "P D F",
}
_ABBREV_HI = {
    "डॉ.": "डॉक्टर", "श्री.": "श्री",
}
_SENT_END = re.compile(r"(?<=[.!?।॥])\s+|\n+")
_SCALE = r"करोड़|लाख|हज़ार|हजार|अरब|crores?|lakhs?|lacs?|thousand|million|billion|trillion"
_NUM = re.compile(r"(?<![\w.])(₹|\$|Rs\.?\s?)?(\d+(?:,\d{2,3})*)(\.\d+)?(\s?%)?(\s(?:" + _SCALE + r")(?![\wऀ-ॿ]))?",
                  re.IGNORECASE)
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Small vocabulary to spot romanised Hindi so the right language id is passed to the engine.
_ROMAN_HI = {"hai", "hain", "nahi", "nahin", "kya", "aur", "hum", "aap", "mein", "main", "ke", "ki", "ka", "ko",
             "se", "par", "yeh", "woh", "kar", "karenge", "karte", "raha", "rahe", "tha", "thi", "the", "hoga",
             "kaise", "kyun", "kyon", "bahut", "accha", "acha", "theek", "chalo", "matlab", "lekin", "abhi",
             "toh", "bhi", "sab", "kuch", "apna", "apne", "humare", "tumhare", "unka", "iska", "baare", "liye"}


@dataclass
class Chunk:
    text: str
    language: str
    pause_after_s: float


def detect_language(text: str, hint: str | None = None) -> str:
    if hint and hint != "auto":
        return hint
    lang = script_language(text, fallback="en")
    if lang == "en":
        from vc.text.hinglish import hindi_ratio
        words = re.findall(r"[a-zA-Z]+", text.lower())
        if words:
            hits = sum(1 for w in words if w in _ROMAN_HI)
            if hits / len(words) >= 0.2 or hindi_ratio(text) >= 0.2:
                return "hi-Latn"
    return lang


def prepare_script(script: str, language_hint: str | None, pause_stats: dict | None = None) -> dict:
    """Everything the TTS stage needs, also returned by /api/text/preview so the user can see and
    correct exactly what will be spoken."""
    hint = language_hint if language_hint not in (None, "", "auto") else None
    language = detect_language(script, hint)
    note = None
    text = script
    if language == "hi-Latn":
        from vc.text.hinglish import romanized_to_mixed
        text, n_conv, n_kept = romanized_to_mixed(script)
        note = f"{n_conv} romanised Hindi words converted to Devanagari, {n_kept} words kept in Latin"
    engine_lang = "hi" if language.startswith("hi") else language
    text = normalize(text, "hi" if engine_lang == "hi" else "en")
    chunks = chunk(text, language, pause_stats=pause_stats)
    return {"language": language, "engine_language": engine_lang, "text": text, "chunks": chunks,
            "transliteration": note}


def _num_to_words(num: str, frac: str | None, lang: str) -> str:
    from num2words import num2words

    if lang == "hi":
        from vc.text.hindi_numbers import number_to_hindi
        try:
            return number_to_hindi(num, frac)
        except ValueError:
            return num
    n = num.replace(",", "")
    try:
        value = float(n + (frac or ""))
    except ValueError:
        return num
    l = "hi" if lang.startswith("hi") and lang != "hi-Latn" else "en_IN" if lang == "hi-Latn" else "en"
    try:
        if frac:
            return num2words(value, lang=l)
        return num2words(int(value), lang=l)
    except Exception:
        try:
            return num2words(int(value), lang="en")
        except Exception:
            return num


def normalize(text: str, language: str) -> str:
    text = text.translate(_DEV_DIGITS)
    text = text.replace("​", "").replace("﻿", "")
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"\.{3,}|…", ", ", text)
    text = re.sub(r"[–—]", ", ", text)

    abbrev = _ABBREV_HI if language == "hi" else _ABBREV_EN
    for k, v in abbrev.items():
        text = re.sub(rf"(?<!\w){re.escape(k)}(?!\w)", v, text)
    if language != "hi":
        for k, v in _ABBREV_EN.items():
            if k.isupper():
                text = re.sub(rf"(?<!\w){re.escape(k)}(?!\w)", v, text)

    def repl(m: re.Match) -> str:
        cur, num, frac, pct, scale = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        words = _num_to_words(num, frac, language) + (scale or "")      # "25,000 crore" before the unit
        if cur:
            unit = "रुपये" if language == "hi" else "rupees" if "₹" in cur or "Rs" in cur else "dollars"
            words = f"{words} {unit}"
        if pct:
            words = f"{words} {'प्रतिशत' if language == 'hi' else 'percent'}"
        return words

    text = _NUM.sub(repl, text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def chunk(text: str, language: str, max_chars: int = 220, pause_stats: dict | None = None) -> list[Chunk]:
    """Split into sentence-aligned chunks. Pause after each chunk follows the speaker's own pause
    statistics when available (longer at paragraph breaks)."""
    base_pause = float((pause_stats or {}).get("pause_mean_s") or 0.45)
    base_pause = min(max(base_pause, 0.25), 0.9)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    for pi, para in enumerate(paragraphs):
        sentences = [s.strip() for s in _SENT_END.split(para) if s and s.strip()]
        buf = ""
        for s in sentences:
            if len(s) > max_chars:                       # very long sentence: split on commas
                parts = re.split(r"(?<=[,;:])\s+", s)
                sentences_extra = []
                cur = ""
                for p in parts:
                    if len(cur) + len(p) + 1 > max_chars and cur:
                        sentences_extra.append(cur)
                        cur = p
                    else:
                        cur = f"{cur} {p}".strip()
                if cur:
                    sentences_extra.append(cur)
                # still too long (no commas): fall back to word boundaries
                split_more = []
                for p in sentences_extra:
                    while len(p) > max_chars:
                        cut = p.rfind(" ", 0, max_chars)
                        cut = cut if cut > 0 else max_chars
                        split_more.append(p[:cut].strip())
                        p = p[cut:].strip()
                    if p:
                        split_more.append(p)
                sentences_extra = split_more
            else:
                sentences_extra = [s]
            for s2 in sentences_extra:
                if buf and len(buf) + len(s2) + 1 > max_chars:
                    chunks.append(Chunk(buf, detect_language(buf, language if language != "auto" else None), base_pause))
                    buf = s2
                else:
                    buf = f"{buf} {s2}".strip()
        if buf:
            chunks.append(Chunk(buf, detect_language(buf, language if language != "auto" else None), base_pause))
        if chunks and pi < len(paragraphs) - 1:
            chunks[-1].pause_after_s = base_pause * 1.8
    if chunks:
        chunks[-1].pause_after_s = 0.0
    return chunks
