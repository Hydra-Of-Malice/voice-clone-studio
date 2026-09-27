"""Resemble Chatterbox Multilingual (MIT). 0.5B, ~10 s reference clip, no reference transcript needed,
Perth watermark embedded in every output. Hindi ('hi') and English ('en') are both supported.

Per-speaker LoRA adapters (see `vc.finetune`) are merged into the T3 transformer on demand and
removed again, so one loaded model serves every profile."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from vc.registry import Capabilities, registry
from vc.tts.base import ReferencePrompt, StylePreset

log = logging.getLogger("vc.tts.chatterbox")
LANGS = ["ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no",
         "pl", "pt", "ru", "sv", "sw", "tr", "zh"]


@registry.register(Capabilities(
    name="chatterbox-multilingual", kind="tts", license="MIT", commercial_ok=True, languages=LANGS,
    needs_ref_transcript=False, native_sr=24000, approx_vram_gb=5.0, supports_finetune=True,
    notes="Perth watermark built in; one language_id per call; LoRA adapters on T3"))
class ChatterboxMultilingual:
    name = "chatterbox-multilingual"
    sample_rate = 24000

    def __init__(self, device: str = "cuda"):
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        self.sample_rate = int(self.model.sr)
        self._adapter: str | None = None
        self._deltas: dict[str, "torch.Tensor"] = {}

    def supports_language(self, language: str) -> bool:
        return language.split("-")[0] in LANGS

    # ------------------------------------------------------------------ adapters -------------
    def set_adapter(self, adapter_dir: str | None) -> None:
        """Merge the LoRA deltas of `adapter_dir` into T3 (None restores the base model)."""
        import torch

        if adapter_dir == self._adapter:
            return
        params = dict(self.model.t3.tfmr.named_parameters())
        with torch.no_grad():
            for name, delta in self._deltas.items():
                params[name].sub_(delta.to(params[name].device, params[name].dtype))
            self._deltas = {}
            self._adapter = None
            if adapter_dir:
                for name, delta in load_lora_deltas(Path(adapter_dir)).items():
                    if name not in params:
                        raise KeyError(f"Adapter weight {name} does not match the model")
                    params[name].add_(delta.to(params[name].device, params[name].dtype))
                    self._deltas[name] = delta
                self._adapter = adapter_dir
                log.info("Applied adapter %s (%d matrices)", adapter_dir, len(self._deltas))

    def synthesize(self, text: str, language: str, ref: ReferencePrompt, style: StylePreset,
                   seed: int | None = None) -> np.ndarray:
        import torch

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        lang = language.split("-")[0]
        with torch.inference_mode():
            wav = self.model.generate(text, language_id=lang, audio_prompt_path=ref.audio_path,
                                      exaggeration=style.exaggeration, cfg_weight=style.cfg_weight,
                                      temperature=style.temperature)
        return wav.squeeze().float().cpu().numpy().astype(np.float32)

    def unload(self) -> None:
        self.model = None
        self._deltas = {}
        self._adapter = None
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def load_lora_deltas(adapter_dir: Path) -> dict:
    """Read a PEFT LoRA adapter and return {parameter name: scale * B @ A} on CPU."""
    from safetensors.torch import load_file

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    state = load_file(str(adapter_dir / "adapter_model.safetensors"))
    deltas = {}
    for key, a in state.items():
        if ".lora_A" not in key:
            continue
        b = state[key.replace(".lora_A", ".lora_B")]
        target = key.split(".lora_A")[0].removeprefix("base_model.model.") + ".weight"
        deltas[target] = (b.float() @ a.float()) * scale
    return deltas
