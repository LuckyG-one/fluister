"""Speech gate: decide whether a recording holds real speech before
anything gets pasted.

Uses the Silero VAD model that ships with faster-whisper (the same model
Handy uses), run through onnxruntime in ~10 ms per dictation. Two checks:

1. Before ASR: too little detected speech (accidental tap, silence) means
   the dictation is dropped and nothing is pasted.
2. After ASR: classic silence hallucinations ("You", "Thank you.",
   "Mm-hmm") are dropped when the amount of speech was short as well.

Both checks fail open: if the VAD model cannot load, dictation works as
before.
"""
import re

import numpy as np

from .audio import SAMPLE_RATE

_model_failed = False


def speech_seconds(audio: np.ndarray) -> float | None:
    """Seconds of detected speech, or None when the VAD is unavailable."""
    global _model_failed
    if _model_failed:
        return None
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        segments = get_speech_timestamps(
            audio,
            VadOptions(min_silence_duration_ms=300, speech_pad_ms=100),
            sampling_rate=SAMPLE_RATE,
        )
    except Exception as exc:
        _model_failed = True
        print(f"fluister: VAD unavailable, speech gate disabled ({exc})")
        return None
    return sum(seg["end"] - seg["start"] for seg in segments) / SAMPLE_RATE


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s'-]", "", text).strip().lower()


def has_too_little_speech(speech_s: float | None, vad_cfg: dict) -> bool:
    if speech_s is None:
        return False
    return speech_s < vad_cfg["min_speech_seconds"]


def is_hallucination(raw: str, speech_s: float | None, vad_cfg: dict) -> bool:
    """True for an empty transcript, or a known silence phrase spoken in
    under `hallucination_max_speech_s` of detected speech."""
    normalized = _normalize(raw or "")
    if not normalized:
        return True
    if speech_s is not None and speech_s >= vad_cfg["hallucination_max_speech_s"]:
        return False
    phrases = {_normalize(p) for p in vad_cfg["hallucinations"]}
    return normalized in phrases
