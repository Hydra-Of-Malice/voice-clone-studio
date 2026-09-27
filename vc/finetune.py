"""Few-shot speaker adaptation: a LoRA on Chatterbox's T3 (text -> speech-token transformer),
trained on the speaker's own 5-10 minutes.

Evidence says adaptation buys a few hundredths of similarity and can cost intelligibility, so an
adapter is only *accepted* when it beats zero-shot on speaker similarity without raising the
character error rate (see `gate`). Only the attention projections are adapted; the acoustic decoder
and vocoder stay frozen."""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger("vc.finetune")
Progress = Callable[[float, str], None]


@dataclass
class TrainItem:
    text: str
    language: str
    audio16k: np.ndarray


@dataclass
class TrainResult:
    adapter_dir: str
    steps: int
    train_loss: float
    val_loss_start: float
    val_loss_best: float
    clips: int
    minutes: float
    seconds: float


def train_lora(tts_model, items: list[TrainItem], out_dir: Path, rank: int = 16, alpha: int = 32,
               lr: float = 1e-4, epochs: int = 12, max_steps: int = 600, seed: int = 0,
               progress: Progress | None = None) -> TrainResult:
    """`tts_model` is a loaded ChatterboxMultilingualTTS. Its T3 weights are unchanged on return."""
    import torch
    import torch.nn.functional as F
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.mtl_tts import punc_norm
    from peft import LoraConfig, get_peft_model

    t0 = time.time()
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = tts_model.device
    t3 = tts_model.t3
    hp = t3.hp
    progress = progress or (lambda f, m: None)

    # ---- 1. features (frozen tokenizer / voice encoder) -------------------------------------
    progress(0.02, "Preparing training clips")
    feats = []
    s3_tok = tts_model.s3gen.tokenizer
    enc_len = tts_model.ENC_COND_LEN
    with torch.no_grad():
        for it in items:
            speech, _ = s3_tok.forward([it.audio16k])
            speech = torch.atleast_2d(speech)[0].long().cpu()
            prompt, _ = s3_tok.forward([it.audio16k[:enc_len]], max_len=hp.speech_cond_prompt_len)
            prompt = torch.atleast_2d(prompt)[0].long().cpu()
            ve = torch.from_numpy(tts_model.ve.embeds_from_wavs([it.audio16k], sample_rate=16000)).mean(0).float()
            text = tts_model.tokenizer.text_to_tokens(punc_norm(it.text), language_id=it.language)[0].long().cpu()
            text = F.pad(F.pad(text, (1, 0), value=hp.start_text_token), (0, 1), value=hp.stop_text_token)
            speech = F.pad(F.pad(speech, (1, 0), value=hp.start_speech_token), (0, 1), value=hp.stop_speech_token)
            if len(text) > hp.max_text_tokens or len(speech) > hp.max_speech_tokens:
                continue
            feats.append({"text": text, "speech": speech, "prompt": prompt, "ve": ve})
    # tensors produced inside library inference_mode blocks cannot take part in autograd
    feats = [{k: v.clone() for k, v in f.items()} for f in feats]
    if len(feats) < 8:
        raise ValueError(f"Only {len(feats)} usable clips; at least 8 are needed for fine-tuning")
    torch.cuda.empty_cache()

    rng.shuffle(feats)
    n_val = max(2, len(feats) // 10)
    val, train = feats[:n_val], feats[n_val:]

    # ---- 2. LoRA on the attention projections ------------------------------------------------
    for p in t3.parameters():
        p.requires_grad_(False)
    base_tfmr = t3.tfmr
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    peft_model = get_peft_model(base_tfmr, cfg)
    t3.tfmr = peft_model
    params = [p for p in peft_model.parameters() if p.requires_grad]
    log.info("LoRA parameters: %.2f M", sum(p.numel() for p in params) / 1e6)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    total = min(max_steps, epochs * len(train))
    warm = max(5, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    def batch_loss(target: dict, ref: dict) -> "torch.Tensor":
        cond = T3Cond(speaker_emb=ref["ve"].unsqueeze(0), cond_prompt_speech_tokens=ref["prompt"].unsqueeze(0),
                      emotion_adv=0.5 * torch.ones(1, 1, 1)).to(device=device)
        text = target["text"].unsqueeze(0).to(device)
        speech = target["speech"].unsqueeze(0).to(device)
        # Teacher forcing: the state at position i predicts token i + 1. (The library's own
        # `T3.loss` neither shifts the targets nor orders the logits for cross_entropy.)
        speech_in, speech_tgt = speech[:, :-1], speech[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(str(device) != "cpu")):
            out = t3.forward(t3_cond=cond, text_tokens=text,
                             text_token_lens=torch.tensor([text.size(1)], device=device),
                             speech_tokens=speech_in,
                             speech_token_lens=torch.tensor([speech_in.size(1)], device=device), training=True)
        ls = F.cross_entropy(out.speech_logits.float().transpose(1, 2), speech_tgt)
        lt = F.cross_entropy(out.text_logits[:, :-1].float().transpose(1, 2), text[:, 1:])
        return ls + 0.1 * lt

    def validate() -> float:
        peft_model.eval()
        with torch.no_grad():
            losses = [float(batch_loss(v, train[i % len(train)])) for i, v in enumerate(val)]
        peft_model.train()
        return float(np.mean(losses))

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        val_start = validate()
        best_val, best_step, running = val_start, 0, []
        peft_model.train()
        accum = 2
        step = 0
        order: list[int] = []
        while step < total:
            if not order:
                order = list(range(len(train)))
                rng.shuffle(order)
            target = train[order.pop()]
            # condition on a different clip of the same speaker, as at inference time
            ref = train[rng.randrange(len(train))]
            loss = batch_loss(target, ref) / accum
            loss.backward()
            running.append(float(loss) * accum)
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 50 == 0 or step == total:
                v = validate()
                log.info("step %d/%d train %.3f val %.3f", step, total, np.mean(running[-50:]), v)
                if v < best_val:
                    best_val, best_step = v, step
                    peft_model.save_pretrained(str(out_dir))
                elif step - best_step >= 150:
                    log.info("early stop at step %d (best %d)", step, best_step)
                    break
                progress(0.05 + 0.6 * step / total, f"Training {step}/{total} (val loss {v:.3f})")
        if best_step == 0:
            raise ValueError("Fine-tuning did not improve the validation loss; keeping the zero-shot voice")
    finally:
        # restore the untouched base transformer (LoRA layers wrap, they do not modify, the weights)
        t3.tfmr = peft_model.unload() if hasattr(peft_model, "unload") else base_tfmr
        del opt
        torch.cuda.empty_cache()
    minutes = sum(len(i.audio16k) for i in items) / 16000 / 60
    return TrainResult(str(out_dir), best_step, float(np.mean(running[-50:])), val_start, best_val,
                       len(feats), round(minutes, 2), round(time.time() - t0, 1))


# ------------------------------------------------------------------ acceptance gate ---------
EVAL_SENTENCES = {
    "hi": ["आज का मौसम बहुत अच्छा है और हम बाहर घूमने जा रहे हैं।",
           "कृपया अपना नाम और पता साफ़ आवाज़ में बताइए।",
           "इस project की deadline अगले हफ़्ते है, इसलिए हमें जल्दी काम करना होगा।",
           "विज्ञान और तकनीक ने हमारी ज़िंदगी को पूरी तरह बदल दिया है।"],
    "en": ["The weather is lovely today, so we are going out for a walk.",
           "Please say your name and address in a clear voice.",
           "The deadline for this project is next week, so we need to move quickly.",
           "Science and technology have completely changed the way we live."],
}


def char_error_rate(reference: str, hypothesis: str) -> float:
    import re
    import unicodedata

    def norm(s: str) -> str:
        s = unicodedata.normalize("NFC", s.lower())
        s = re.sub(r"[^\w\sऀ-ॿ]|[।॥_]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    r, h = norm(reference), norm(hypothesis)
    if not r:
        return 0.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return min(1.0, prev[-1] / len(r))


def gate(base: dict, adapted: dict, min_gain: float, max_cer_regression: float) -> tuple[bool, str]:
    gain = adapted["similarity"] - base["similarity"]
    cer_delta = adapted["cer"] - base["cer"]
    if cer_delta > max_cer_regression:
        return False, f"rejected: intelligibility dropped (CER {base['cer']:.3f} -> {adapted['cer']:.3f})"
    if gain < min_gain:
        return False, f"rejected: no similarity gain over zero-shot ({base['similarity']:.3f} -> {adapted['similarity']:.3f})"
    return True, f"accepted: similarity {base['similarity']:.3f} -> {adapted['similarity']:.3f}, CER {base['cer']:.3f} -> {adapted['cer']:.3f}"
