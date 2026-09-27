"""Module registry: every replaceable stage (ASR, speaker encoder, TTS, enhancer) registers a
factory plus a capability manifest. The pipeline only talks to the abstract interfaces in
`vc.asr.base`, `vc.speaker.base`, `vc.tts.base`, so engines can be swapped through settings."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Capabilities:
    name: str
    kind: str                       # asr | speaker | tts | enhancer
    license: str
    commercial_ok: bool
    languages: list[str] = field(default_factory=list)   # ISO codes, "*" for any
    needs_ref_transcript: bool = False
    native_sr: int = 0
    approx_vram_gb: float = 0.0
    supports_finetune: bool = False
    notes: str = ""

    def supports_language(self, lang: str) -> bool:
        return "*" in self.languages or lang in self.languages


class Registry:
    def __init__(self) -> None:
        self._factories: dict[tuple[str, str], Callable[..., Any]] = {}
        self._caps: dict[tuple[str, str], Capabilities] = {}

    def register(self, caps: Capabilities) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def deco(factory: Callable[..., Any]) -> Callable[..., Any]:
            key = (caps.kind, caps.name)
            self._factories[key] = factory
            self._caps[key] = caps
            return factory
        return deco

    def create(self, kind: str, name: str, **kwargs: Any) -> Any:
        try:
            factory = self._factories[(kind, name)]
        except KeyError as e:
            available = [n for k, n in self._factories if k == kind]
            raise KeyError(f"No {kind} engine named {name!r}; available: {available}") from e
        return factory(**kwargs)

    def capabilities(self, kind: str | None = None) -> list[Capabilities]:
        return [c for (k, _), c in self._caps.items() if kind is None or k == kind]


registry = Registry()


def load_builtin_engines() -> None:
    """Import engine modules so their @registry.register decorators run."""
    import vc.asr.router  # noqa: F401
    import vc.asr.whisper_fw  # noqa: F401
    import vc.asr.whisper_hf  # noqa: F401
    import vc.audio.enhance  # noqa: F401
    import vc.speaker.ecapa  # noqa: F401
    import vc.tts.chatterbox_engine  # noqa: F401
