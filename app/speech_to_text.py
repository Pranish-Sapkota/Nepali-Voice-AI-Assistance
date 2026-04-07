"""
speech_to_text.py
-----------------
Transcribes audio using faster-whisper.
Supports automatic language detection (Nepali / English / multilingual).
GPU (CUDA) acceleration is used when available.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Model configuration ────────────────────────────────────────────────────────
SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
DEFAULT_MODEL    = "small"   # Good balance of speed and Nepali accuracy

# Nepali ISO code understood by Whisper
NEPALI_LANG = "ne"


# ── Device detection ───────────────────────────────────────────────────────────
def _detect_device() -> tuple[str, str]:
    """
    Returns (device, compute_type) tuned for the hardware.
    RTX 3050 → cuda / int8   (fits in 6 GB VRAM with small model)
    CPU fallback             → cpu / int8
    """
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA detected — using GPU acceleration.")
            return "cuda", "int8"
    except ImportError:
        pass
    logger.info("No CUDA — using CPU.")
    return "cpu", "int8"


# ── Model loader (singleton via lru_cache) ─────────────────────────────────────
@lru_cache(maxsize=1)
def _load_model(model_size: str, device: str, compute_type: str):
    """Load and cache the Whisper model (loaded once per process)."""
    from faster_whisper import WhisperModel

    # Local model path (downloaded on first run into models/)
    model_dir = Path(__file__).parent.parent / "models" / f"whisper-{model_size}"

    if model_dir.exists():
        source = str(model_dir)
        logger.info("Loading Whisper from local path: %s", source)
    else:
        source = model_size
        logger.info(
            "Downloading Whisper '%s' model (first run) …", model_size
        )

    model = WhisperModel(
        source,
        device=device,
        compute_type=compute_type,
        download_root=str(Path(__file__).parent.parent / "models"),
    )
    logger.info("Whisper model loaded ✓")
    return model


# ── Core transcription ─────────────────────────────────────────────────────────
def transcribe(
    audio_path: str,
    model_size: str = DEFAULT_MODEL,
    language: Optional[str] = None,
    beam_size: int = 5,
    vad_filter: bool = True,
) -> dict:
    """
    Transcribe an audio file.

    Parameters
    ----------
    audio_path  : Path to the WAV/MP3 file.
    model_size  : Whisper model variant (tiny/base/small/medium/large-v2).
    language    : ISO language code ('ne' for Nepali, 'en' for English).
                  None → auto-detect.
    beam_size   : Beam search width (higher = more accurate, slower).
    vad_filter  : Use Silero VAD to skip silent sections.

    Returns
    -------
    dict with keys:
        text      : Full transcription string
        language  : Detected / provided language code
        segments  : List of segment dicts (start, end, text)
        confidence: Average log-probability as a rough confidence score
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    device, compute_type = _detect_device()
    model = _load_model(model_size, device, compute_type)

    logger.info("Transcribing: %s (lang=%s)", audio_path, language or "auto")

    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=False,
    )

    segments = []
    full_text_parts = []
    total_log_prob = 0.0

    for seg in segments_iter:
        segments.append(
            {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "avg_logprob": round(seg.avg_logprob, 4),
            }
        )
        full_text_parts.append(seg.text.strip())
        total_log_prob += seg.avg_logprob

    full_text = " ".join(full_text_parts).strip()
    avg_conf  = round(total_log_prob / max(len(segments), 1), 4)

    detected_lang = info.language if language is None else language
    logger.info(
        "Transcription complete | lang=%s | conf=%.3f | text='%s'",
        detected_lang, avg_conf, full_text[:80],
    )

    return {
        "text": full_text,
        "language": detected_lang,
        "language_probability": round(info.language_probability, 3),
        "segments": segments,
        "confidence": avg_conf,
    }


# ── Language detection only ────────────────────────────────────────────────────
def detect_language(audio_path: str, model_size: str = "base") -> dict:
    """
    Quick language detection without full transcription.
    Returns dict: {language, probability}
    """
    from faster_whisper import WhisperModel
    import numpy as np

    device, compute_type = _detect_device()
    model = _load_model(model_size, device, compute_type)

    # Use only first 30 s for detection
    _, info = model.transcribe(audio_path, language=None, beam_size=1)
    return {
        "language": info.language,
        "probability": round(info.language_probability, 3),
    }


# ── Utility ────────────────────────────────────────────────────────────────────
def is_nepali(result: dict, threshold: float = 0.4) -> bool:
    """Heuristic: return True if the transcription is likely Nepali."""
    return (
        result.get("language") == NEPALI_LANG
        and result.get("language_probability", 0) >= threshold
    )


def format_transcript(result: dict) -> str:
    """Return a human-readable summary of the transcription result."""
    lang_label = "Nepali" if result["language"] == NEPALI_LANG else result["language"].upper()
    return (
        f"[{lang_label} | conf {result['confidence']:.2f}]\n"
        f"{result['text']}"
    )
