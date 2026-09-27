"""Open-vocabulary romanised Hindi -> Devanagari (rule-based, no model download).

Casual romanisation is ambiguous (t/d dental vs retroflex, a vs aa), so this aims at a spelling the
TTS reads with the right sounds rather than at dictionary-perfect orthography. The lexicon in
`vc.text.hinglish` takes precedence for common words; this handles everything else that is judged
to be Hindi rather than English."""
from __future__ import annotations

_HALANT = "्"
_ANUSVARA = "ं"

# longest match first
_CONS = [
    ("ksh", "क्ष"), ("chh", "छ"), ("shr", "श्र"),
    ("kh", "ख"), ("gh", "घ"), ("ch", "च"), ("jh", "झ"), ("th", "थ"), ("dh", "ध"), ("ph", "फ"), ("bh", "भ"),
    ("sh", "श"), ("gy", "ज्ञ"),
    ("k", "क"), ("g", "ग"), ("c", "क"), ("j", "ज"), ("t", "त"), ("d", "द"), ("n", "न"), ("p", "प"), ("b", "ब"),
    ("m", "म"), ("y", "य"), ("r", "र"), ("l", "ल"), ("v", "व"), ("w", "व"), ("s", "स"), ("h", "ह"),
    ("f", "फ़"), ("z", "ज़"), ("q", "क़"), ("x", "क्स"),
]
# (roman, independent form, matra)
_VOWELS = [
    ("aa", "आ", "ा"), ("ai", "ऐ", "ै"), ("au", "औ", "ौ"), ("ou", "औ", "ौ"), ("ee", "ई", "ी"), ("ii", "ई", "ी"),
    ("oo", "ऊ", "ू"), ("uu", "ऊ", "ू"), ("ei", "ए", "े"),
    ("a", "अ", ""), ("i", "इ", "ि"), ("u", "उ", "ु"), ("e", "ए", "े"), ("o", "ओ", "ो"),
]
_FINAL_LONG = {"a": ("आ", "ा"), "i": ("ई", "ी"), "u": ("ऊ", "ू")}
_NASAL_BEFORE = {"n": {"k", "kh", "g", "gh", "ch", "chh", "j", "jh", "t", "th", "d", "dh", "s", "sh", "z"},
                 "m": {"p", "ph", "b", "bh"}}
_CONJUNCT_SECOND = {"r", "y"}
_S_CLUSTER = {"t", "th", "k", "p"}


def _units(word: str) -> list[tuple[str, str]]:
    """Split into ('C', roman) / ('V', roman) units, greedy longest match."""
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(word):
        for rom, *_ in _VOWELS:
            if word.startswith(rom, i):
                out.append(("V", rom))
                i += len(rom)
                break
        else:
            for rom, _ in _CONS:
                if word.startswith(rom, i):
                    out.append(("C", rom))
                    i += len(rom)
                    break
            else:
                i += 1          # unknown character: skip
    return out


# ---------------------------------------------------------------- dictionary lookup ----------
# Real Devanagari spellings from the Hindi frequency list, indexed by a fuzzy phonetic key that
# ignores what casual romanisation cannot express (vowel length, dental/retroflex, nukta).
_DEV_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "f", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l", "क़": "k", "ख़": "kh", "ग़": "g", "ज़": "j", "ड़": "d", "ढ़": "dh", "फ़": "f",
}
_DEV_VOW = {"अ": "", "आ": "", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ए": "e", "ऐ": "e", "ओ": "o", "औ": "o",
            "ऑ": "o", "ऋ": "ri", "ा": "", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "े": "e", "ै": "e", "ो": "o",
            "ौ": "o", "ॉ": "o", "ॅ": "e", "ृ": "ri"}
_ROM_CONS_KEY = {"ksh": ["k", "sh"], "chh": ["chh"], "shr": ["sh", "r"], "ph": ["f"], "gy": ["j", "n"], "c": ["k"],
                 "w": ["v"], "z": ["j"], "q": ["k"], "x": ["k", "s"]}
_ROM_VOW_KEY = {"aa": "", "a": "", "ai": "e", "ei": "e", "e": "e", "au": "o", "ou": "o", "o": "o",
                "ee": "i", "ii": "i", "i": "i", "oo": "u", "uu": "u", "u": "u"}
_LABIALS = {"p", "b", "f", "bh"}
_index: dict[str, list[str]] | None = None


def _finish_key(units: list[str]) -> str:
    out: list[str] = []
    for i, u in enumerate(units):
        if not u:
            continue
        if u == "m" and i + 1 < len(units) and units[i + 1] in _LABIALS:
            u = "n"
        if out and u == "y" and out[-1] == "i":
            continue
        if out and out[-1] == u:
            continue
        if out and out[-1] == "ch" and u == "chh":
            out[-1] = "chh"
            continue
        out.append(u)
    return "-".join(out)


def dev_key(word: str) -> str:
    units: list[str] = []
    chars = list(word)
    for i, ch in enumerate(chars):
        if ch in _DEV_CONS:
            units.append(_DEV_CONS[ch])
        elif ch in _DEV_VOW:
            units.append(_DEV_VOW[ch])
        elif ch in "ंँ":
            if i < len(chars) - 1:
                units.append("n")
    return _finish_key(units)


def roman_keys(word: str) -> list[str]:
    units: list[str] = []
    for kind, rom in _units(word.lower()):
        if kind == "V":
            units.append(_ROM_VOW_KEY[rom])
        else:
            units.extend(_ROM_CONS_KEY.get(rom, [rom]))
    keys = [_finish_key(units)]
    w = word.lower()
    if w.endswith("ai") and len(w) > 3:                               # padhai = padh-aa-ii, not padh-ai
        keys.insert(0, _finish_key(units[:-1] + ["i"]))
    if len(w) > 3 and w.endswith("n") and w[-2] in "aeiou":          # nahin, kahan, main: nasalised ending
        keys.append(_finish_key(units[:-1]))
    return [k for k in dict.fromkeys(keys) if k]


def _load_index() -> dict[str, list[str]]:
    global _index
    if _index is None:
        _index = {}
        try:
            import re
            import wordfreq
            dev = re.compile(r"^[ऀ-ॿ]+$")
            for w in wordfreq.iter_wordlist("hi"):
                if dev.match(w) and len(w) > 1:
                    _index.setdefault(dev_key(w), []).append(w)       # already frequency-ordered
        except Exception:  # noqa: BLE001  (wordfreq missing: rules only)
            _index = {}
    return _index


def dictionary_lookup(word: str) -> str | None:
    """Most plausible real Hindi word for a romanised token, or None."""
    from wordfreq import zipf_frequency

    index = _load_index()
    w = word.lower()
    explicit_long = "aa" in w
    starts_vowel = w[0] in "aeiou"
    best, best_score = None, -1.0
    for key in roman_keys(w):
        for cand in index.get(key, [])[:12]:
            score = zipf_frequency(cand, "hi")
            if explicit_long == (("ा" in cand) or ("आ" in cand)):
                score += 0.4
            if starts_vowel != (cand[0] in "अआइईउऊएऐओऔऑ"):
                score -= 3.0
            if ("chh" in w) == ("छ" in cand):
                score += 0.2
            if abs(len(cand) - len(w)) > 4:
                score -= 1.0
            if score > best_score:
                best, best_score = cand, score
    return best if best_score >= 1.5 else None


def transliterate_word(word: str) -> str:
    w = word.lower()
    units = _units(w)
    if not units:
        return word
    cons = dict(_CONS)
    vow = {r: (ind, mat) for r, ind, mat in _VOWELS}
    out: list[str] = []
    seen_vowel = False
    n = len(units)
    for i, (kind, rom) in enumerate(units):
        nxt = units[i + 1] if i + 1 < n else None
        prev = units[i - 1] if i > 0 else None
        last = i == n - 1
        if kind == "V":
            ind, mat = vow[rom]
            if last and rom in _FINAL_LONG and n > 1:
                ind, mat = _FINAL_LONG[rom]
            out.append(mat if prev and prev[0] == "C" else ind)
            seen_vowel = True
            continue

        # consonant
        if nxt is None or nxt[0] == "V":
            out.append(cons[rom])
            continue
        nrom = nxt[1]
        rest_is_suffix = (i + 2 == n - 1 and units[i + 2][0] == "V")      # C + C + final vowel
        # nasal before a homorganic consonant -> anusvara (but keep verb forms like "sunta", "banta")
        if seen_vowel and rom in _NASAL_BEFORE and nrom in _NASAL_BEFORE[rom] \
                and not (rom == "n" and nrom in ("t", "th") and rest_is_suffix):
            out.append(_ANUSVARA)
            continue
        tail_cluster = all(k == "C" for k, _ in units[i + 1:])             # cluster at the end of the word
        geminate = nrom == rom or (rom == "c" and nrom in ("ch", "chh")) or (rom == "t" and nrom == "th") \
            or (rom == "d" and nrom == "dh") or (rom == "k" and nrom == "kh")
        if rom == "c" and nrom in ("ch", "chh"):
            out.append("च" + _HALANT)
            continue
        if (not seen_vowel) or tail_cluster or geminate or nrom in _CONJUNCT_SECOND \
                or (rom in ("s", "sh") and nrom in _S_CLUSTER):
            out.append(cons[rom] + _HALANT)
        else:
            out.append(cons[rom])
    return "".join(out)
