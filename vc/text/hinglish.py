"""Romanised Hinglish -> mixed script. Common Hindi words typed in Latin letters are converted to
Devanagari; everything else (English words, names) stays in Latin, which the Hindi TTS path reads
well. This is a lexicon, not a full transliterator: unknown romanised Hindi words are left as typed,
so Devanagari input remains the most reliable way to write Hindi.

Swap in AI4Bharat IndicXlit behind `romanized_to_mixed` for open-vocabulary coverage."""
from __future__ import annotations

import re

_LEX_RAW = """
aaj आज|kal कल|abhi अभी|ab अब|tab तब|jab जब|kab कब|phir फिर|fir फिर|kabhi कभी|hamesha हमेशा|pehle पहले|pahle पहले|baad बाद|
hum हम|main मैं|mai मैं|mein में|me में|tum तुम|aap आप|ap आप|tu तू|yeh यह|ye ये|yah यह|woh वह|wo वो|vo वो|voh वह|
hai है|hain हैं|ho हो|hoon हूँ|hun हूँ|hu हूँ|tha था|thi थी|the थे|hoga होगा|hogi होगी|honge होंगे|hota होता|hoti होती|hote होते|hua हुआ|hui हुई|hue हुए|
ka का|ki की|ke के|ko को|se से|par पर|pe पे|tak तक|ne ने|liye लिए|liya लिया|saath साथ|sath साथ|bina बिना|
aur और|ya या|lekin लेकिन|magar मगर|par पर|kyunki क्योंकि|kyonki क्योंकि|isliye इसलिए|toh तो|to तो|bhi भी|hi ही|na ना|nahi नहीं|nahin नहीं|nhi नहीं|mat मत|
kya क्या|kyun क्यों|kyon क्यों|kyu क्यों|kaise कैसे|kaisa कैसा|kaisi कैसी|kahan कहाँ|kaha कहाँ|kaun कौन|kitna कितना|kitne कितने|kitni कितनी|kaunsa कौनसा|
mera मेरा|meri मेरी|mere मेरे|tera तेरा|teri तेरी|tere तेरे|hamara हमारा|humara हमारा|hamari हमारी|humari हमारी|hamare हमारे|humare हमारे|
tumhara तुम्हारा|tumhari तुम्हारी|tumhare तुम्हारे|aapka आपका|aapki आपकी|aapke आपके|apna अपना|apni अपनी|apne अपने|
uska उसका|uski उसकी|uske उसके|unka उनका|unki उनकी|unke उनके|iska इसका|iski इसकी|iske इसके|inka इनका|inki इनकी|inke इनके|
mujhe मुझे|mujhko मुझको|tumhe तुम्हें|tumhein तुम्हें|aapko आपको|hame हमें|hamein हमें|humein हमें|use उसे|usko उसको|unhe उन्हें|unko उनको|isko इसको|ise इसे|
yahan यहाँ|yaha यहाँ|wahan वहाँ|waha वहाँ|idhar इधर|udhar उधर|andar अंदर|bahar बाहर|upar ऊपर|neeche नीचे|niche नीचे|paas पास|pas पास|door दूर|
baare बारे|bare बारे|baat बात|baatein बातें|cheez चीज़|chiz चीज़|kaam काम|log लोग|logon लोगों|din दिन|raat रात|samay समय|waqt वक़्त|saal साल|mahina महीना|
ghar घर|desh देश|duniya दुनिया|zindagi ज़िंदगी|jindagi ज़िंदगी|paisa पैसा|paise पैसे|naam नाम|dost दोस्त|bhai भाई|behen बहन|
accha अच्छा|acha अच्छा|achha अच्छा|acchi अच्छी|achi अच्छी|acche अच्छे|bura बुरा|bada बड़ा|badi बड़ी|bade बड़े|chhota छोटा|chota छोटा|choti छोटी|
bahut बहुत|bohot बहुत|bohat बहुत|thoda थोड़ा|thodi थोड़ी|zyada ज़्यादा|jyada ज़्यादा|kam कम|sab सब|sabhi सभी|kuch कुछ|kuchh कुछ|koi कोई|har हर|
naya नया|nayi नई|naye नए|purana पुराना|sahi सही|galat ग़लत|theek ठीक|thik ठीक|zaroor ज़रूर|jarur ज़रूर|zaroori ज़रूरी|jaruri ज़रूरी|shayad शायद|bilkul बिल्कुल|
matlab मतलब|yaani यानी|jaise जैसे|waise वैसे|aise ऐसे|aisa ऐसा|aisi ऐसी|kaafi काफ़ी|kafi काफ़ी|sirf सिर्फ़|bas बस|ek एक|do दो|teen तीन|
kar कर|karo करो|karna करना|karne करने|karta करता|karti करती|karte करते|karenge करेंगे|karega करेगा|karegi करेगी|karunga करूँगा|karungi करूँगी|kiya किया|kiye किए|ki की|
raha रहा|rahi रही|rahe रहे|rehna रहना|rahega रहेगा|rahenge रहेंगे|
ja जा|jao जाओ|jana जाना|jaata जाता|jata जाता|jati जाती|jate जाते|jayenge जाएँगे|jayega जाएगा|gaya गया|gayi गई|gaye गए|
aa आ|aao आओ|aana आना|aata आता|aati आती|aate आते|aayega आएगा|aayenge आएँगे|aaya आया|aayi आई|aaye आए|
de दे|do दो|dena देना|deta देता|deti देती|dete देते|denge देंगे|dega देगा|diya दिया|diye दिए|di दी|
le ले|lo लो|lena लेना|leta लेता|leti लेती|lete लेते|lenge लेंगे|lega लेगा|
dekh देख|dekho देखो|dekhna देखना|dekha देखा|dekhte देखते|dekhenge देखेंगे|sun सुन|suno सुनो|sunna सुनना|suna सुना|
bol बोल|bolo बोलो|bolna बोलना|bola बोला|bolte बोलते|keh कह|kehna कहना|kaha कहा|kehte कहते|bata बता|batao बताओ|batana बताना|bataya बताया|
samajh समझ|samjho समझो|samajhna समझना|samjha समझा|samjhe समझे|seekh सीख|seekhna सीखना|seekhenge सीखेंगे|sikhna सीखना|padh पढ़|padhna पढ़ना|likh लिख|likhna लिखना|
chal चल|chalo चलो|chalna चलना|chalta चलता|chalte चलते|chalega चलेगा|chalenge चलेंगे|ruk रुक|ruko रुको|
mil मिल|milna मिलना|milta मिलता|milte मिलते|milega मिलेगा|mila मिला|soch सोच|socho सोचो|sochna सोचना|socha सोचा|
chahiye चाहिए|chahte चाहते|chahta चाहता|chahti चाहती|sakta सकता|sakti सकती|sakte सकते|pata पता|lagta लगता|lagti लगती|laga लगा|
banana बनाना|banata बनाता|banate बनाते|banaya बनाया|banayenge बनाएँगे|shuru शुरू|khatam ख़त्म|khatm ख़त्म|istemal इस्तेमाल|madad मदद|
namaste नमस्ते|namaskar नमस्कार|dhanyavaad धन्यवाद|dhanyavad धन्यवाद|shukriya शुक्रिया|swagat स्वागत|haan हाँ|han हाँ|ji जी|
is इस|us उस|in इन|un उन|isse इससे|usse उससे|ki कि|
dosto दोस्तों|doston दोस्तों|sawal सवाल|jawab जवाब|tarika तरीक़ा|tareeka तरीक़ा|wajah वजह|zariye ज़रिए|dwara द्वारा|
"""

# Words that are also common English words: only convert when the sentence is clearly Hindi.
_AMBIGUOUS = {"is", "us", "in", "un", "to", "do", "the", "me", "use", "hi", "par", "ho", "log", "kam", "ab", "bas", "sun", "bold", "mat",
              "ki", "de", "le", "ja", "aa", "na", "pe", "ne", "se", "main", "hue", "din", "pas", "sab", "chal", "bol",
              "mil", "kar", "bare", "pata", "bade", "banana", "laga", "ruk", "han", "tu", "ye", "wo", "vo", "ya", "ka",
              "ke", "ko", "ek", "tab", "jab", "har", "the", "di", "lo", "ap", "hu", "hun", "fir"}

LEXICON: dict[str, str] = {}
for _entry in _LEX_RAW.replace("\n", "").split("|"):
    _entry = _entry.strip()
    if _entry:
        _k, _v = _entry.split(" ", 1)
        LEXICON.setdefault(_k, _v.strip())

_WORD = re.compile(r"[A-Za-z]+")


def hindi_ratio(text: str) -> float:
    words = [w.lower() for w in _WORD.findall(text)]
    if not words:
        return 0.0
    return sum(1 for w in words if w in LEXICON and w not in _AMBIGUOUS) / len(words)


def _english_zipf(word: str) -> float:
    try:
        from wordfreq import zipf_frequency
        return zipf_frequency(word, "en")
    except ImportError:
        return 9.0


def is_english_word(word: str) -> bool:
    """English vs romanised Hindi by corpus frequency (wordfreq). Unknown words count as Hindi only
    when they are clearly absent from English."""
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return True            # without the vocabulary we never guess: keep the word as typed
    w = word.lower()
    en, hi = zipf_frequency(w, "en"), zipf_frequency(w, "hi")
    if en >= 3.0 and en >= hi:
        return True
    if hi > en:
        return False
    return en >= 2.0 and len(w) <= 3       # short rare tokens (abbreviations) stay Latin


def romanized_to_mixed(text: str, open_vocabulary: bool = True) -> tuple[str, int, int]:
    """Returns (converted text, words converted, Latin words left)."""
    sentence_is_hindi = hindi_ratio(text) >= 0.15
    converted = kept = 0

    def repl(m: re.Match) -> str:
        nonlocal converted, kept
        w = m.group(0)
        lw = w.lower()
        acronym = w.isupper() and len(w) > 1
        if lw in LEXICON and (lw not in _AMBIGUOUS or sentence_is_hindi) and not acronym:
            converted += 1
            return LEXICON[lw]
        if open_vocabulary and sentence_is_hindi and not acronym and len(lw) > 2 and lw not in _AMBIGUOUS \
                and not is_english_word(lw):
            from vc.text.translit import dictionary_lookup, transliterate_word
            found = dictionary_lookup(lw)
            if found:
                converted += 1
                return found
            if _english_zipf(lw) < 2.0:          # unknown to both vocabularies: spell it by rule
                converted += 1
                return transliterate_word(lw)
        kept += 1
        return w

    out = _WORD.sub(repl, text)
    # Latin full stop after Devanagari reads better as a danda
    out = re.sub(r"(?<=[ऀ-ॿ])\.(?=\s|$)", "।", out)
    return out, converted, kept
