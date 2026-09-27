"""Pipeline steps executed by the GPU worker (`vc.worker`).

Models are loaded lazily and evicted to fit an 8 GB card: ASR and TTS never co-reside."""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from vc import consent as consent_mod
from vc import finetune, postprocess, provenance
from vc.asr.base import Segment, Transcript, Word
from vc.audio import quality as quality_mod
from vc.audio.io import load_audio, resample, save_wav
from vc.audio.preprocess import maybe_enhance, merge_spans, normalize_level, slice_clips
from vc.audio.prosody import ProsodyProfile, analyze as analyze_prosody
from vc.audio.vad import SpeechSpan, speech_spans
from vc.config import Settings
from vc.db import Database, new_id
from vc.profile import ReferenceOpener, SpeakerProfile, build_profile, load_profile, similarity_to_profile
from vc.registry import registry
from vc.security import KeyStore
from vc.speaker.base import cosine, count_speakers
from vc.text.normalize import normalize as normalize_text, prepare_script
from vc.tts.base import STYLE_PRESETS, ReferencePrompt

log = logging.getLogger("vc.jobs")
Progress = Callable[[float, str], None]


class ModelManager:
    """Lazy loading with a simple VRAM policy: TTS and ASR never co-reside."""

    def __init__(self, settings: Settings):
        self.s = settings
        self._asr = None
        self._spk = None
        self._tts = None
        self.lock = threading.RLock()

    def asr(self):
        with self.lock:
            if self._asr is None:
                self.release_tts()
                log.info("Loading ASR %s (%s)", self.s.asr_engine, self.s.asr_model)
                self._asr = registry.create("asr", self.s.asr_engine, model=self.s.asr_model, device=self.s.device,
                                            download_root=str(self.s.models_dir / "asr"),
                                            hindi_model=self.s.asr_hindi_model)
            return self._asr

    def speaker(self):
        with self.lock:
            if self._spk is None:
                log.info("Loading speaker encoder %s", self.s.speaker_engine)
                self._spk = registry.create("speaker", self.s.speaker_engine, device=self.s.device,
                                            savedir=str(self.s.models_dir / "speaker"))
            return self._spk

    def tts(self):
        with self.lock:
            if self._tts is None:
                self.release_asr()
                log.info("Loading TTS %s", self.s.tts_engine)
                self._tts = registry.create("tts", self.s.tts_engine, device=self.s.device)
            return self._tts

    def release_asr(self):
        if self._asr is not None:
            self._asr.unload()
            self._asr = None

    def release_tts(self):
        if self._tts is not None:
            self._tts.unload()
            self._tts = None

    def status(self) -> dict:
        return {"asr_loaded": self._asr is not None, "speaker_loaded": self._spk is not None,
                "tts_loaded": self._tts is not None}


class Pipeline:
    def __init__(self, settings: Settings, db: Database, models: ModelManager):
        self.s = settings
        self.db = db
        self.models = models
        self.keystore = KeyStore(settings.data_dir)
        self._signer: provenance.C2PASigner | None = None

    def signer(self) -> provenance.C2PASigner:
        if self._signer is None:
            self._signer = provenance.C2PASigner(self.s.data_dir, self.s.c2pa_cert, self.s.c2pa_key,
                                                 self.s.c2pa_timestamp_url)
        return self._signer

    # ------------------------------------------------------------------ analyze --------------
    @staticmethod
    def _synthetic_marks(audio: np.ndarray, sr: int, seconds: int = 45) -> list[str]:
        """AI-speech watermarks found in a recording (used to refuse cloned input and cloned consent)."""
        found = []
        mid = max(0, len(audio) // 2 - seconds * sr // 2)
        seg = audio[mid: mid + seconds * sr]
        try:
            if provenance.detect_audioseal(seg, sr)["detected"]:
                found.append("AudioSeal")
            if provenance.detect_perth(seg, sr).get("detected"):
                found.append("Perth")
        except Exception:  # noqa: BLE001
            log.exception("watermark check failed")
        return found

    def _speaker_check(self, audio16: np.ndarray, spans: list[SpeechSpan]) -> tuple[int, float]:
        clips = slice_clips(audio16, 16000, spans, self.s.ref_clip_min_s, self.s.ref_clip_max_s)
        if len(clips) < 4:
            return 1, 1.0
        enc = self.models.speaker()
        return count_speakers(np.stack([enc.embed(c.audio) for c in clips[:60]]))

    def _assess(self, audio, sr, audio16, spans, n_spk, consistency) -> quality_mod.QualityReport:
        return quality_mod.assess(audio, sr, audio16, spans, n_spk, consistency, self.s.min_sample_seconds,
                                  self.s.recommended_min_seconds, self.s.recommended_max_seconds,
                                  self.s.min_quality_score)

    def analyze_sample(self, sample_id: str, progress: Progress) -> dict[str, Any]:
        sample = self.db.get("voice_samples", sample_id)
        if not sample:
            raise ValueError("Unknown sample")
        progress(0.05, "Decoding audio")
        audio, sr = load_audio(sample["path"])
        if len(audio) / sr > self.s.max_upload_seconds:
            audio = audio[: int(self.s.max_upload_seconds * sr)]
        audio = normalize_level(audio)
        audio16 = resample(audio, sr, 16000)

        progress(0.12, "Detecting speech")
        spans = speech_spans(audio16)
        if not spans:
            raise ValueError("No speech detected in the recording")

        progress(0.2, "Checking speaker consistency")
        n_spk, consistency = self._speaker_check(audio16, spans)

        progress(0.3, "Scoring quality")
        report = self._assess(audio, sr, audio16, spans, n_spk, consistency)
        synthetic = self._synthetic_marks(audio, sr)

        # Conditional enhancement: only for noisy uploads, kept only if it helps and keeps the voice
        enc = self.models.speaker()

        def snr_of(x: np.ndarray) -> float:
            x16 = resample(x, sr, 16000)
            return quality_mod.estimate_snr_db(x16, spans)

        def identity(raw: np.ndarray, enh: np.ndarray) -> float:
            sims = []
            for c in slice_clips(raw, sr, spans, self.s.ref_clip_min_s, self.s.ref_clip_max_s)[:6]:
                a, b = int(c.start * sr), int(c.end * sr)
                sims.append(cosine(enc.embed(resample(raw[a:b], sr, 16000)), enc.embed(resample(enh[a:b], sr, 16000))))
            return float(np.mean(sims)) if sims else 1.0

        if report.snr_db < self.s.enhance_below_snr_db:
            progress(0.38, "Reducing background noise")
        enhanced, enh = maybe_enhance(audio, sr, report.snr_db, self.s.enhance_below_snr_db, snr_of, identity)
        raw_quality = None
        if enh["applied"]:
            raw_quality = report.to_dict()
            audio = normalize_level(enhanced)
            audio16 = resample(audio, sr, 16000)
            spans = speech_spans(audio16)
            n_spk, consistency = self._speaker_check(audio16, spans)
            report = self._assess(audio, sr, audio16, spans, n_spk, consistency)
            report.issues.insert(0, f"Background noise was reduced automatically "
                                    f"({enh['snr_before_db']} dB to {enh.get('snr_after_db', '?')} dB SNR)")
        if synthetic:
            report.accepted = False
            report.issues.insert(0, "This recording carries an AI-speech watermark (" + ", ".join(synthetic) + ")")
            report.suggestions.insert(0, "Voice profiles can only be created from a real recording of your own voice.")
        clean_path = Path(sample["path"]).with_suffix(".clean.wav")
        save_wav(clean_path, audio, sr)

        progress(0.5, "Transcribing")
        chunks = merge_spans(spans, self.s.ref_clip_max_s)
        transcript = self.models.asr().transcribe(audio16, chunks=chunks)

        progress(0.88, "Analysing speaking style")
        prosody = analyze_prosody(audio16, spans, transcript.word_count)

        analysis = {"quality": report.to_dict(), "quality_before_enhancement": raw_quality,
                    "transcript": transcript.to_dict(), "prosody": prosody.to_dict(), "enhancement": enh,
                    "synthetic_marks": synthetic,
                    "clean_path": str(clean_path),
                    "n_clips": len(slice_clips(audio16, 16000, spans, self.s.ref_clip_min_s, self.s.ref_clip_max_s)),
                    "spans": [[round(s.start, 3), round(s.end, 3)] for s in spans]}
        self.db.update("voice_samples", sample_id, analysis_json=json.dumps(analysis, ensure_ascii=False),
                       duration_s=report.duration_s, sample_rate=sr)
        self.db.insert("transcriptions", dict(id=new_id("tr_"), sample_id=sample_id, engine=transcript.engine,
                                              language=transcript.language, confidence=transcript.confidence,
                                              transcript=transcript.transcript,
                                              segments_json=json.dumps(analysis["transcript"]["segments"], ensure_ascii=False),
                                              created_at=time.time()))
        progress(1.0, "Done")
        return {"sample_id": sample_id, "quality": analysis["quality"], "prosody": analysis["prosody"],
                "quality_before_enhancement": raw_quality, "enhancement": enh, "synthetic_marks": synthetic,
                "transcript": {k: v for k, v in analysis["transcript"].items() if k != "segments"},
                "segments": [{k: v for k, v in s.items() if k != "words"} for s in analysis["transcript"]["segments"][:400]]}

    # ------------------------------------------------------------------ profile --------------
    def _sample_material(self, sample_id: str):
        sample = self.db.get("voice_samples", sample_id)
        if not sample or not sample.get("analysis_json"):
            raise ValueError("Sample has not been analysed yet")
        analysis = json.loads(sample["analysis_json"])
        audio, sr = load_audio(analysis["clean_path"])
        spans = [SpeechSpan(a, b) for a, b in analysis["spans"]]
        tr = analysis["transcript"]
        segments = [Segment(s["start"], s["end"], s["text"], s["confidence"], s["language"],
                            [Word(**w) for w in s.get("words", [])]) for s in tr["segments"]]
        transcript = Transcript(tr["transcript"], tr["language"], tr["language_confidence"], tr["confidence"],
                                segments, tr["engine"])
        return sample, analysis, audio, sr, spans, transcript

    def create_profile(self, sample_id: str, name: str, progress: Progress) -> dict[str, Any]:
        progress(0.1, "Loading cleaned audio")
        sample, analysis, audio, sr, spans, transcript = self._sample_material(sample_id)
        if not analysis["quality"]["accepted"]:
            raise ValueError("Sample quality too low: " + "; ".join(analysis["quality"]["issues"]))
        clips = slice_clips(audio, sr, spans, self.s.ref_clip_min_s, self.s.ref_clip_max_s)
        if len(clips) < 3:
            raise ValueError("Could not cut enough clean reference clips from this sample")

        progress(0.3, "Embedding reference clips")
        prosody = ProsodyProfile(**analysis["prosody"])
        profile_id = new_id("vp_")
        prof, wrapped = build_profile(profile_id, name, clips, sr, transcript, prosody, self.models.speaker(),
                                      analysis["quality"]["score"], self.s.profiles_dir, self.keystore,
                                      self.s.ref_bank_size)
        progress(0.9, "Saving profile")
        self.db.insert("voice_profiles", dict(
            id=profile_id, user_id=sample["user_id"], sample_id=sample_id, name=name,
            embedding_reference=wrapped.decode(), language=prof.language, quality_score=prof.quality_score,
            consent_status="pending", profile_json=json.dumps(prof.to_dict(), ensure_ascii=False),
            created_at=time.time()))
        progress(1.0, "Done")
        return {"profile_id": profile_id, "name": name, "language": prof.language,
                "clips": len(prof.clips), "quality_score": prof.quality_score,
                "consent_status": "pending", "style": prof.style_features, "pitch": prof.pitch_statistics,
                "speaking_rate": prof.speaking_rate}

    # ------------------------------------------------------------------ consent --------------
    def verify_consent(self, profile_id: str, consent_id: str, audio_path: str, client_info: str,
                       progress: Progress) -> dict[str, Any]:
        row = self.db.get("voice_profiles", profile_id)
        rec = self.db.get("consent_records", consent_id)
        if not row or not rec or rec["profile_id"] != profile_id:
            raise ValueError("Unknown profile or consent record")
        prof = load_profile(self.s.profiles_dir, profile_id)
        progress(0.2, "Transcribing consent statement")
        audio16, _ = load_audio(audio_path, target_sr=16000)
        if len(audio16) < 16000 * 2:
            raise ValueError("Consent recording is too short")
        lang = "hi" if consent_mod._tokens(rec["statement_text"]) and any(
            "ऀ" <= ch <= "ॿ" for ch in rec["statement_text"]) else "en"
        tr = self.models.asr().transcribe(audio16, language=lang)
        match, _ = consent_mod.text_match(rec["statement_text"], tr.transcript)
        nonce_ok = consent_mod.nonce_present(rec["nonce"], tr.transcript)
        progress(0.7, "Verifying speaker")
        sim = similarity_to_profile(prof, self.models.speaker(), audio16)
        raw, raw_sr = load_audio(audio_path)
        synthetic = self._synthetic_marks(raw, raw_sr, seconds=30)
        verified = (match >= self.s.consent_min_text_match and nonce_ok and sim >= self.s.same_speaker_cosine
                    and not synthetic)
        self.db.update("consent_records", consent_id, audio_path=audio_path, asr_text=tr.transcript,
                       text_match=round(match, 3), speaker_cosine=round(sim, 3), verified=int(verified),
                       client_info=client_info)
        if verified:
            self.db.update("voice_profiles", profile_id, consent_status="verified")
        progress(1.0, "Done")
        reasons = []
        if match < self.s.consent_min_text_match:
            reasons.append("The statement was not read closely enough; please read it exactly as shown.")
        if not nonce_ok:
            reasons.append("The code words were not recognised.")
        if sim < self.s.same_speaker_cosine:
            reasons.append("The voice does not match the uploaded sample.")
        if synthetic:
            reasons.append("The recording carries an AI-speech watermark; consent must be spoken live by the speaker.")
        return {"verified": verified, "synthetic_marks": synthetic, "text_match": round(match, 3), "nonce_ok": nonce_ok,
                "speaker_match": sim >= self.s.same_speaker_cosine, "heard": tr.transcript, "reasons": reasons}

    # ------------------------------------------------------------------ generate -------------
    def _authorised_profile(self, profile_id: str) -> tuple[dict, SpeakerProfile, bytes]:
        row = self.db.get("voice_profiles", profile_id)
        if not row:
            raise ValueError("Unknown voice profile")
        if row["consent_status"] != "verified":
            raise PermissionError("This voice profile has no verified consent. Record the consent statement first.")
        return row, load_profile(self.s.profiles_dir, row["id"]), self.keystore.unwrap(row["embedding_reference"].encode())

    def _latest_consent(self, profile_id: str) -> str | None:
        rows = self.db.all("consent_records", "profile_id=? AND verified=1", (profile_id,))
        return rows[0]["id"] if rows else None

    def _active_adapter(self, profile_id: str, tts_name: str) -> dict | None:
        rows = self.db.all("voice_adapters", "profile_id=? AND engine=? AND status='accepted' AND active=1",
                           (profile_id, tts_name))
        return rows[0] if rows else None

    def _synthesize(self, tts, prof: SpeakerProfile, key: bytes, chunks, engine_lang: str, language: str,
                    style_name: str, seed: int, progress: Progress | None = None, lo: float = 0.1,
                    hi: float = 0.8) -> tuple[list[np.ndarray], list[float], Any]:
        style = STYLE_PRESETS.get(style_name or "natural", STYLE_PRESETS["natural"])
        clip = prof.best_clip(style.name, language=engine_lang)
        wavs, pauses = [], []
        with ReferenceOpener(self.s.profiles_dir, prof, clip, key) as ref_path:
            ref = ReferencePrompt(str(ref_path), clip.transcript, clip.language)
            for i, ch in enumerate(chunks):
                if progress:
                    progress(lo + (hi - lo) * i / len(chunks), f"Generating {i + 1}/{len(chunks)}")
                # A fully English sentence inside a Hindi script is read with English phonology
                ch_lang = "en" if (ch.language == "en" and language != "hi-Latn") else engine_lang
                wavs.append(tts.synthesize(ch.text, ch_lang, ref, style, seed=seed))
                pauses.append(ch.pause_after_s)
        return wavs, pauses, clip

    def generate(self, gen_job_id: str, progress: Progress) -> dict[str, Any]:
        gj = self.db.get("generation_jobs", gen_job_id)
        if not gj:
            raise ValueError("Unknown generation job")
        row, prof, key = self._authorised_profile(gj["voice_profile_id"])
        self.db.update("generation_jobs", gen_job_id, status="processing", updated_at=time.time())

        progress(0.05, "Preparing text")
        prep = prepare_script(gj["script"], gj["language"], prof.style_features)
        chunks, language, engine_lang = prep["chunks"], prep["language"], prep["engine_language"]
        if not chunks:
            raise ValueError("Empty script")

        tts = self.models.tts()
        if not tts.supports_language(engine_lang):
            raise ValueError(f"TTS engine {tts.name} does not support language {engine_lang}")
        adapter = self._active_adapter(prof.speaker_id, tts.name) if gj.get("adapter_id") != "none" else None
        if hasattr(tts, "set_adapter"):
            tts.set_adapter(adapter["path"] if adapter else None)
        seed = int(time.time()) % 100000
        wavs, pauses, clip = self._synthesize(tts, prof, key, chunks, engine_lang, language, gj["style"], seed, progress)

        progress(0.85, "Post-processing")
        out_id = new_id("au_")
        payload = provenance.payload_for(out_id) if self.s.audioseal else None
        details = {"audio_id": out_id, "profile_id": prof.speaker_id, "consent_id": self._latest_consent(prof.speaker_id),
                   "engine": tts.name, "adapter_id": adapter["id"] if adapter else None,
                   "language": language, "audioseal_payload": payload,
                   "watermarks": [w for w, on in (("perth", "chatterbox" in tts.name), ("audioseal", payload is not None)) if on]}
        wav_path, mp3_path, metrics, prov = postprocess.finalize(
            wavs, pauses, tts.sample_rate, self.s.outputs_dir, out_id, speed=float(gj["speed"] or 1.0),
            out_sr=self.s.output_sr, target_lufs=self.s.target_lufs, true_peak_dbtp=self.s.true_peak_dbtp,
            mp3_bitrate=self.s.mp3_bitrate, audioseal_payload=payload,
            signer=self.signer() if self.s.c2pa else None, manifest_details=details)

        progress(0.95, "Checking similarity")
        gen16, _ = load_audio(wav_path, target_sr=16000)
        sim = cosine(prof.voice_embedding, self.models.speaker().embed(gen16[: 16000 * 30]))
        band_low = prof.intra_speaker_cosine_mean - 2 * max(prof.intra_speaker_cosine_std, 0.03)
        m = metrics.to_dict()
        m.update({"speaker_similarity": round(sim, 3), "similarity_band_low": round(band_low, 3),
                  "similarity_ok": sim >= band_low, "language": language, "engine": tts.name,
                  "reference_clip": clip.file, "style": gj["style"] or "natural",
                  "fine_tuned": bool(adapter), "transliteration": prep["transliteration"], "provenance": prov})
        self.db.insert("generated_audio", dict(id=out_id, job_id=gen_job_id, profile_id=prof.speaker_id,
                                               wav_path=str(wav_path), mp3_path=str(mp3_path),
                                               duration_s=metrics.duration_s, metrics_json=json.dumps(m),
                                               watermark=",".join(details["watermarks"]),
                                               audioseal_payload=payload, provenance_json=json.dumps(prov),
                                               created_at=time.time()))
        self.db.update("generation_jobs", gen_job_id, status="completed", output_id=out_id, updated_at=time.time())
        progress(1.0, "Done")
        return {"audio_id": out_id, "wav_url": f"/api/audio/{out_id}?fmt=wav", "mp3_url": f"/api/audio/{out_id}?fmt=mp3",
                "metrics": m, "spoken_text": prep["text"], "chunks": [c.text for c in chunks]}

    # ------------------------------------------------------------------ verify audio ---------
    def verify_audio(self, path: str, progress: Progress) -> dict[str, Any]:
        """Was this file made here? Checks the C2PA manifest and both watermarks."""
        progress(0.2, "Reading Content Credentials")
        c2 = provenance.read_c2pa(Path(path))
        progress(0.5, "Looking for watermarks")
        audio, sr = load_audio(path)
        seal = provenance.detect_audioseal(audio[: sr * 120], sr)
        perth_res = provenance.detect_perth(audio[: sr * 60], sr)
        match = None
        # 1) the signed manifest names the output directly; 2) otherwise the watermark payload,
        #    tolerating up to two flipped bits from lossy re-encoding
        claimed = (c2.get("details") or {}).get("audio_id")
        row = self.db.get("generated_audio", claimed) if claimed else None
        how = "content credentials"
        if row is None and seal["detected"]:
            how = "watermark"
            best = 3
            for r in self.db.all("generated_audio", "audioseal_payload IS NOT NULL"):
                d = bin(int(r["audioseal_payload"]) ^ seal["payload"]).count("1")
                if d < best:
                    best, row = d, r
        if row is not None:
            match = {"audio_id": row["id"], "profile_id": row["profile_id"], "created_at": row["created_at"],
                     "matched_by": how}
        ai = bool(seal["detected"] or perth_res.get("detected") or c2.get("ai_generated"))
        verdict = ("Generated by this system" if match else
                   "AI-generated speech (watermark or Content Credentials found)" if ai else
                   "No watermark or Content Credentials found")
        progress(1.0, "Done")
        return {"verdict": verdict, "ai_generated": ai, "matched_output": match, "audioseal": seal,
                "perth": perth_res, "c2pa": c2,
                "note": "Absence of a mark does not prove a recording is genuine: marks can be removed by "
                        "re-recording or neural codecs."}

    # ------------------------------------------------------------------ fine-tune ------------
    def fine_tune(self, profile_id: str, adapter_id: str, progress: Progress) -> dict[str, Any]:
        row, prof, key = self._authorised_profile(profile_id)
        _, analysis, audio, sr, spans, transcript = self._sample_material(row["sample_id"])
        adapter_dir = self.s.profiles_dir / profile_id / "adapters" / adapter_id

        by_start = {round(s.start, 2): s for s in transcript.segments}
        items, held_out_text = [], []
        for c in slice_clips(audio, sr, spans, 3.0, self.s.ref_clip_max_s, tail_silence_s=0.2):
            seg = by_start.get(round(c.start, 2))
            text = (seg.text if seg else transcript.text_between(c.start, c.end)).strip()
            conf = seg.confidence if seg else transcript.confidence
            if len(text.split()) < 3 or conf < self.s.ft_min_clip_confidence:
                continue
            lang = "hi" if any("ऀ" <= ch <= "ॿ" for ch in text) else \
                ("hi" if prof.language.startswith("hi") else "en")
            items.append(finetune.TrainItem(normalize_text(text, lang), lang, resample(c.audio, sr, 16000)))
        if len(items) < 8:
            raise ValueError(f"Only {len(items)} clips have a confident transcript; fine-tuning needs at least 8")
        log.info("Fine-tuning on %d clips", len(items))

        tts = self.models.tts()
        if not hasattr(tts, "set_adapter"):
            raise ValueError(f"TTS engine {tts.name} does not support fine-tuning")
        tts.set_adapter(None)
        res = finetune.train_lora(tts.model, items, adapter_dir, self.s.ft_rank, self.s.ft_alpha, self.s.ft_lr,
                                  self.s.ft_epochs, self.s.ft_max_steps, progress=progress)

        # ---- acceptance gate: zero-shot vs adapter on the same sentences, seeds and reference ----
        lang = "hi" if prof.language.startswith("hi") else "en"
        sentences = finetune.EVAL_SENTENCES[lang]
        audios: dict[str, list[np.ndarray]] = {"base": [], "adapter": []}
        for name, path in (("base", None), ("adapter", str(adapter_dir))):
            tts.set_adapter(path)
            for i, sent in enumerate(sentences):
                progress(0.66 + 0.2 * (len(audios["base"]) + len(audios["adapter"])) / (2 * len(sentences)),
                         f"Comparing with zero-shot ({name} {i + 1}/{len(sentences)})")
                prep = prepare_script(sent, lang, prof.style_features)
                wavs, _, _ = self._synthesize(tts, prof, key, prep["chunks"], lang, lang, "natural", seed=1234 + i)
                audios[name].append(resample(np.concatenate(wavs), tts.sample_rate, 16000))
        tts.set_adapter(None)

        progress(0.88, "Scoring intelligibility")
        enc = self.models.speaker()
        sims = {k: [cosine(prof.voice_embedding, enc.embed(a)) for a in v] for k, v in audios.items()}
        asr = self.models.asr()                      # releases the TTS model
        scores = {}
        for name in ("base", "adapter"):
            cers = [finetune.char_error_rate(normalize_text(s, lang), normalize_text(asr.transcribe(a, language=lang).transcript, lang))
                    for s, a in zip(sentences, audios[name])]
            scores[name] = {"similarity": round(float(np.mean(sims[name])), 4), "cer": round(float(np.mean(cers)), 4)}
        accepted, verdict = finetune.gate(scores["base"], scores["adapter"], self.s.ft_min_similarity_gain,
                                          self.s.ft_max_cer_regression)
        metrics = {"zero_shot": scores["base"], "fine_tuned": scores["adapter"], "verdict": verdict,
                   "clips": res.clips, "minutes": res.minutes, "steps": res.steps,
                   "val_loss_start": round(res.val_loss_start, 4), "val_loss_best": round(res.val_loss_best, 4),
                   "train_seconds": res.seconds, "rank": self.s.ft_rank}
        if accepted:
            with self.db.connect() as con:
                con.execute("UPDATE voice_adapters SET active=0 WHERE profile_id=?", (profile_id,))
        self.db.update("voice_adapters", adapter_id, status="accepted" if accepted else "rejected",
                       active=int(accepted), metrics_json=json.dumps(metrics))
        progress(1.0, "Done")
        return {"adapter_id": adapter_id, "accepted": accepted, **metrics}


HANDLERS: dict[str, Callable[[Pipeline, dict, Progress], dict]] = {
    "analyze": lambda p, pl, pr: p.analyze_sample(pl["sample_id"], pr),
    "create_profile": lambda p, pl, pr: p.create_profile(pl["sample_id"], pl["name"], pr),
    "verify_consent": lambda p, pl, pr: p.verify_consent(pl["profile_id"], pl["consent_id"], pl["audio_path"],
                                                        pl.get("client_info", ""), pr),
    "generate": lambda p, pl, pr: p.generate(pl["generation_job_id"], pr),
    "verify_audio": lambda p, pl, pr: p.verify_audio(pl["path"], pr),
    "fine_tune": lambda p, pl, pr: p.fine_tune(pl["profile_id"], pl["adapter_id"], pr),
}
