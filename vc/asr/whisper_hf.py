"""Hindi / Hinglish ASR: Trelis whisper-hinglish-preview (Apache-2.0), a Whisper-large-v3 fine-tune
that writes code-switched speech in mixed script (Devanagari Hindi + Latin English).

The model was trained without timestamp tokens, so long audio is transcribed chunk by chunk along
VAD pause boundaries; segment times are therefore exact, word times inside a segment are
interpolated. Never translates: task is always `transcribe`."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from vc.asr.base import Segment, Transcript, Word, script_language
from vc.audio.vad import SpeechSpan
from vc.registry import Capabilities, registry

log = logging.getLogger("vc.asr.hf")
DEFAULT_REPO = "Trelis/whisper-hinglish-preview"


@registry.register(Capabilities(
    name="whisper-hinglish", kind="asr", license="Apache-2.0", commercial_ok=True, languages=["hi", "en"],
    approx_vram_gb=3.5, notes="mixed-script Hinglish; segment-level timestamps from VAD chunks"))
class WhisperHinglishASR:
    name = "whisper-hinglish"

    def __init__(self, model: str | None = None, device: str = "cuda", download_root: str | None = None, **_):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        src = model if model and model != "large-v3-turbo" else DEFAULT_REPO
        if download_root and not Path(src).exists():
            local = Path(download_root) / src.split("/")[-1]
            if not (local / "config.json").exists():
                from huggingface_hub import snapshot_download
                snapshot_download(src, local_dir=str(local))
            src = str(local)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.proc = WhisperProcessor.from_pretrained(src)
        self.model = WhisperForConditionalGeneration.from_pretrained(src, torch_dtype=self.dtype)
        self.model = self.model.to(self.device).eval()
        self.model_name = Path(src).name
        tok = self.proc.tokenizer
        self._ids = tok.convert_tokens_to_ids
        self._mixed = tok("<|mixedcode|>", add_special_tokens=False).input_ids

    def _prompt(self, language: str, mixed: bool) -> list[int]:
        p = [self._ids("<|startoftranscript|>"), self._ids(f"<|{language}|>")]
        if mixed:
            p += self._mixed
        return p + [self._ids("<|transcribe|>"), self._ids("<|notimestamps|>")]

    def transcribe(self, audio16k: np.ndarray, language: str | None = None,
                   chunks: list[SpeechSpan] | None = None, batch_size: int = 6) -> Transcript:
        import torch

        language = (language or "hi").split("-")[0]
        if language not in ("hi", "en"):
            language = "hi"
        if chunks is None:
            from vc.audio.preprocess import merge_spans
            from vc.audio.vad import speech_spans
            chunks = merge_spans(speech_spans(audio16k), max_s=12.0)
        # Whisper's window is 30 s
        windows: list[SpeechSpan] = []
        for c in chunks:
            if c.duration <= 29.0:
                windows.append(c)
            else:
                n = int(np.ceil(c.duration / 25.0))
                step = c.duration / n
                windows.extend(SpeechSpan(c.start + i * step, c.start + (i + 1) * step) for i in range(n))

        prompt = self._prompt(language, mixed=(language == "hi"))
        segs: list[Segment] = []
        for b in range(0, len(windows), batch_size):
            batch = windows[b: b + batch_size]
            clips = [audio16k[max(0, int((w.start - 0.1) * 16000)): int((w.end + 0.2) * 16000)] for w in batch]
            feats = self.proc.feature_extractor(clips, sampling_rate=16000, return_tensors="pt").input_features
            feats = feats.to(self.device, self.dtype)
            dec = torch.tensor([prompt] * len(batch), device=self.device)
            with torch.inference_mode():
                out = self.model.generate(input_features=feats, decoder_input_ids=dec, max_new_tokens=440,
                                          num_beams=1, do_sample=False, output_scores=True,
                                          return_dict_in_generate=True)
            seqs = out.sequences[:, len(prompt):]
            # token probabilities -> per-segment confidence
            probs = torch.stack([s.float().softmax(-1) for s in out.scores], dim=1)      # B x T x V
            eos = self.model.generation_config.eos_token_id
            for i, w in enumerate(batch):
                ids = seqs[i]
                n_tok = int((ids != eos).sum().item()) if eos is not None else len(ids)
                n_tok = max(1, min(n_tok, probs.shape[1]))
                p = probs[i, :n_tok].gather(1, ids[:n_tok].unsqueeze(1)).squeeze(1)
                conf = float(p.mean().item())
                text = self.proc.tokenizer.decode(ids, skip_special_tokens=True).strip()
                if not text:
                    continue
                segs.append(Segment(round(w.start, 3), round(w.end, 3), text, round(min(conf, 1.0), 3),
                                    script_language(text, language), _interpolate_words(text, w, conf)))
        transcript = " ".join(s.text for s in segs).strip()
        if segs:
            weights = np.array([max(s.end - s.start, 0.1) for s in segs])
            overall = float(np.average([s.confidence for s in segs], weights=weights))
        else:
            overall = 0.0
        return Transcript(transcript=transcript, language=script_language(transcript, language),
                          language_confidence=1.0, confidence=overall, segments=segs,
                          engine=f"whisper-hinglish:{self.model_name}")

    def unload(self) -> None:
        self.model = None
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def _interpolate_words(text: str, span: SpeechSpan, conf: float) -> list[Word]:
    words = text.split()
    total = sum(len(w) for w in words) or 1
    out, t = [], span.start
    for w in words:
        d = span.duration * len(w) / total
        out.append(Word(round(t, 3), round(t + d, 3), w, round(conf, 3)))
        t += d
    return out
