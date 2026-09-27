"""Hindi number spelling with Indian grouping (हज़ार, लाख, करोड़, अरब). num2words has no Hindi."""
from __future__ import annotations

_ONES = ("शून्य एक दो तीन चार पाँच छह सात आठ नौ दस "
         "ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस "
         "इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस "
         "इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस चालीस "
         "इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास "
         "इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ "
         "इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर सत्तर "
         "इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी अस्सी "
         "इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे "
         "इक्यानवे बानवे तिरानवे चौरानवे पंचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे").split()
assert len(_ONES) == 100

_GROUPS = [(10 ** 9, "अरब"), (10 ** 7, "करोड़"), (10 ** 5, "लाख"), (10 ** 3, "हज़ार"), (100, "सौ")]


def int_to_hindi(n: int) -> str:
    if n < 0:
        return "ऋण " + int_to_hindi(-n)
    if n < 100:
        return _ONES[n]
    parts = []
    for value, name in _GROUPS:
        if n >= value:
            q, n = divmod(n, value)
            parts.append(f"{int_to_hindi(q)} {name}")
    if n:
        parts.append(_ONES[n])
    return " ".join(parts)


def number_to_hindi(num: str, frac: str | None = None) -> str:
    """`num` is the integer part (commas allowed), `frac` the optional '.digits' part."""
    words = int_to_hindi(int(num.replace(",", "")))
    if frac:
        words += " दशमलव " + " ".join(_ONES[int(d)] for d in frac.lstrip("."))
    return words
