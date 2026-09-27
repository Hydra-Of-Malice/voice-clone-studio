import numpy as np

from vc import consent
from vc.audio.preprocess import slice_clips
from vc.audio.quality import assess, effective_bandwidth_hz
from vc.audio.vad import SpeechSpan
from vc.postprocess import join_chunks, normalize_loudness
from vc.text.hindi_numbers import int_to_hindi, number_to_hindi
from vc.text.normalize import chunk, detect_language, normalize


def test_hindi_numbers():
    assert int_to_hindi(0) == "शून्य"
    assert int_to_hindi(25) == "पच्चीस"
    assert int_to_hindi(2030) == "दो हज़ार तीस"
    assert int_to_hindi(150000) == "एक लाख पचास हज़ार"
    assert int_to_hindi(10000000) == "एक करोड़"
    assert number_to_hindi("12", ".5") == "बारह दशमलव पाँच"


def test_detect_language():
    assert detect_language("Today we learn about AI.") == "en"
    assert detect_language("आज हम artificial intelligence के बारे में discuss करेंगे।") == "hi"
    assert detect_language("Aaj hum artificial intelligence ke baare mein discuss karenge.") == "hi-Latn"
    assert detect_language("anything", "en") == "en"


def test_normalize_keeps_punctuation_and_language():
    out = normalize("By 2030, experts expect 5 billion users.", "en")
    assert out == "By two thousand and thirty, experts expect five billion users."
    hi = normalize("यह 2030 तक 5 अरब लोगों तक पहुँचेगा।", "hi")
    assert "दो हज़ार तीस" in hi and "पाँच अरब" in hi and hi.endswith("।")
    assert "रुपये" in normalize("कीमत ₹500 है", "hi")
    assert normalize("revenue ₹25,000 करोड़ रहा", "hi") == "revenue पच्चीस हज़ार करोड़ रुपये रहा"
    assert normalize("It cost $3 million.", "en") == "It cost three million dollars."
    assert "percent" in normalize("Growth was 12.5% last year", "en")
    assert "A I" in normalize("AI is here", "en")


def test_hinglish_to_mixed_script():
    from vc.text.hinglish import romanized_to_mixed
    out, conv, kept = romanized_to_mixed("Aaj hum artificial intelligence ke baare mein discuss karenge. Yeh bahut interesting topic hai.")
    assert out == "आज हम artificial intelligence के बारे में discuss करेंगे। यह बहुत interesting topic है।"
    assert conv == 9 and kept == 5
    english, conv_en, _ = romanized_to_mixed("We need to do the main task and use the log.")
    assert english == "We need to do the main task and use the log." and conv_en == 0


def test_open_vocabulary_transliteration():
    from vc.text.hinglish import romanized_to_mixed
    from vc.text.translit import dev_key, roman_keys, transliterate_word
    assert dev_key("राहुल") in roman_keys("rahul")
    assert dev_key("नहीं") in roman_keys("nahin")
    assert dev_key("ज़िंदगी") in roman_keys("zindagi")
    out, _, _ = romanized_to_mixed("Rahul ne bola ki barish mein zindagi khubsurat hai.")
    assert "राहुल" in out and "बारिश" in out and "खूबसूरत" in out
    assert transliterate_word("dost") == "दोस्त" and transliterate_word("andar") == "अंदर"
    kept, conv, _ = romanized_to_mixed("Please send me the report by Monday, we need to finalise the budget.")
    assert conv == 0 and kept.startswith("Please send")


def test_merge_spans_covers_all_speech():
    from vc.audio.preprocess import merge_spans
    spans = [SpeechSpan(0, 4), SpeechSpan(4.3, 9), SpeechSpan(9.4, 14), SpeechSpan(20, 50), SpeechSpan(51, 53)]
    chunks = merge_spans(spans, max_s=12.0)
    assert all(c.duration <= 12.01 for c in chunks)
    assert abs(sum(c.duration for c in chunks) - (14 - 0 + 30 + 2)) < 1.0
    assert chunks[0].start == 0 and chunks[-1].end == 53


def test_adapter_gate_and_cer():
    from vc.finetune import char_error_rate, gate
    assert char_error_rate("आज मौसम अच्छा है।", "आज मौसम अच्छा है") == 0.0
    assert 0 < char_error_rate("hello world", "hello word") < 0.2
    ok, _ = gate({"similarity": 0.70, "cer": 0.02}, {"similarity": 0.75, "cer": 0.03}, 0.01, 0.03)
    assert ok
    assert not gate({"similarity": 0.70, "cer": 0.02}, {"similarity": 0.80, "cer": 0.12}, 0.01, 0.03)[0]
    assert not gate({"similarity": 0.70, "cer": 0.02}, {"similarity": 0.70, "cer": 0.02}, 0.01, 0.03)[0]


def test_audioseal_payload_is_16_bit():
    from vc.provenance import payload_for
    p = payload_for("au_abc")
    assert 0 <= p < 65536 and p == payload_for("au_abc") and p != payload_for("au_abd")


def test_chunk_preserves_sentences():
    text = "पहला वाक्य। दूसरा वाक्य है! Third sentence here? " + "Long sentence " * 30 + "."
    chunks = chunk(text, "auto", max_chars=120)
    assert len(chunks) >= 3
    assert all(len(c.text) <= 130 for c in chunks)
    assert chunks[0].text.startswith("पहला वाक्य।")
    assert chunks[-1].pause_after_s == 0.0


def test_consent_matching():
    statement, nonce = consent.make_statement("en")
    assert nonce in statement
    ratio, _ = consent.text_match(statement, statement.replace(",", "").lower())
    assert ratio > 0.95 and consent.nonce_present(nonce, statement)
    ratio2, _ = consent.text_match(statement, "mister quilter is the apostle of the middle classes")
    assert ratio2 < 0.4 and not consent.nonce_present(nonce, "hello there")
    hi_statement, hi_nonce = consent.make_statement("hi")
    assert consent.nonce_present(hi_nonce, hi_statement)


def _tone_speech(seconds: float, sr: int = 16000, noise: float = 0.001) -> np.ndarray:
    """Broadband 'speech' for 9 s out of every 10 s, background noise elsewhere."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(0)
    active = (t % 10.0) < 9.0
    sig = 0.1 * rng.standard_normal(n) * active
    return (sig + noise * rng.standard_normal(n)).astype(np.float32)


def test_quality_scores_and_rejects():
    sr = 16000
    clean = _tone_speech(400, sr)
    spans = [SpeechSpan(i * 10.0, i * 10.0 + 9.0) for i in range(40)]
    rep = assess(clean, sr, clean, spans)
    assert rep.duration_s == 400 and rep.speech_s == 360
    assert rep.snr_db > 25 and rep.noise_level == "Low" and rep.accepted

    noisy = _tone_speech(400, sr, noise=0.15)
    rep_noisy = assess(noisy, sr, noisy, spans)
    assert rep_noisy.score < rep.score and rep_noisy.noise_level != "Low"

    short = assess(clean[: sr * 30], sr, clean[: sr * 30], spans[:3])
    assert not short.accepted and any("short" in i.lower() for i in short.issues)

    two = assess(clean, sr, clean, spans, speakers_detected=2)
    assert not two.accepted


def test_bandwidth_detects_band_limited_audio():
    sr = 48000
    t = np.arange(sr * 2) / sr
    narrow = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    assert effective_bandwidth_hz(narrow, sr) < 3000


def test_slice_clips_lengths():
    sr = 16000
    audio = np.zeros(sr * 60, dtype=np.float32)
    spans = [SpeechSpan(i * 3.0, i * 3.0 + 2.6) for i in range(20)]
    clips = slice_clips(audio, sr, spans, 6.0, 12.0)
    assert clips and all(5.9 <= c.duration <= 13.0 for c in clips)


def test_loudness_and_join():
    sr = 24000
    a = _tone_speech(3, sr, 0.0005)
    out = join_chunks([a, a], [0.4, 0.0], sr)
    assert len(out) > len(a) * 1.9
    norm = normalize_loudness(out, sr, -16.0, -1.0)
    assert np.max(np.abs(norm)) <= 10 ** (-1.0 / 20) + 1e-3
