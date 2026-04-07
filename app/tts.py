"""
tts.py
------
Text-to-Speech synthesis.
Supports:
  • gTTS     — simple, requires internet for synthesis
  • Coqui TTS — fully offline, multilingual (Nepali support via custom model)
  • pyttsx3   — offline system TTS, English only, ultimate fallback
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

TTSBackend = Literal["gtts", "coqui", "pyttsx3", "auto"]

NEPALI_LANG = "ne"
ENGLISH_LANG = "en"

# Coqui model best suited for multi-language
COQUI_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


# ── Backend availability ───────────────────────────────────────────────────────
def _has_gtts() -> bool:
    try:
        import gtts  # noqa
        return True
    except ImportError:
        return False


def _has_coqui() -> bool:
    try:
        from TTS.api import TTS  # noqa
        return True
    except ImportError:
        return False


def _has_pyttsx3() -> bool:
    try:
        import pyttsx3  # noqa
        return True
    except ImportError:
        return False


def available_backends() -> list[str]:
    backends = []
    if _has_coqui():
        backends.append("coqui")
    if _has_gtts():
        backends.append("gtts")
    if _has_pyttsx3():
        backends.append("pyttsx3")
    return backends


# ── gTTS backend ───────────────────────────────────────────────────────────────
def _synthesize_gtts(text: str, language: str, output_path: str) -> str:
    """
    Synthesize using Google TTS (requires internet).
    Falls back to English if Nepali synthesis fails.
    """
    from gtts import gTTS, gTTSError

    lang_code = NEPALI_LANG if language == NEPALI_LANG else ENGLISH_LANG
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save(output_path)
        logger.info("gTTS synthesis complete (%s) → %s", lang_code, output_path)
    except gTTSError as exc:
        logger.warning("gTTS failed for lang=%s: %s. Trying English.", lang_code, exc)
        tts = gTTS(text=text, lang=ENGLISH_LANG, slow=False)
        tts.save(output_path)
    return output_path


# ── Coqui TTS backend ──────────────────────────────────────────────────────────
_coqui_model_cache = {}


def _load_coqui_model(model_name: str):
    if model_name not in _coqui_model_cache:
        from TTS.api import TTS
        logger.info("Loading Coqui model: %s (first run — may download)", model_name)
        _coqui_model_cache[model_name] = TTS(model_name)
    return _coqui_model_cache[model_name]


def _synthesize_coqui(
    text: str,
    language: str,
    output_path: str,
    model: str = COQUI_MODEL,
    speaker_wav: Optional[str] = None,
) -> str:
    """
    Fully offline TTS using Coqui XTTS v2.
    Nepali support requires the multilingual model.
    """
    tts_engine = _load_coqui_model(model)

    # XTTS v2 uses language codes
    lang_code = "ne" if language == NEPALI_LANG else "en"

    try:
        if speaker_wav:
            # Voice cloning mode
            tts_engine.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=speaker_wav,
                language=lang_code,
            )
        else:
            tts_engine.tts_to_file(
                text=text,
                file_path=output_path,
                language=lang_code,
            )
        logger.info("Coqui synthesis complete → %s", output_path)
    except Exception as exc:
        logger.error("Coqui synthesis failed: %s", exc)
        raise

    return output_path


# ── pyttsx3 fallback ───────────────────────────────────────────────────────────
def _synthesize_pyttsx3(text: str, output_path: str) -> str:
    """Offline system TTS (English only, low quality)."""
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    logger.info("pyttsx3 synthesis → %s", output_path)
    return output_path


# ── Public API ─────────────────────────────────────────────────────────────────
def synthesize(
    text: str,
    language: str = ENGLISH_LANG,
    backend: TTSBackend = "auto",
    output_path: Optional[str] = None,
    coqui_model: str = COQUI_MODEL,
    speaker_wav: Optional[str] = None,
) -> Optional[str]:
    """
    Convert text to speech and save to a file.

    Parameters
    ----------
    text         : Text to synthesize.
    language     : ISO language code ('ne' for Nepali, 'en' for English).
    backend      : 'gtts' | 'coqui' | 'pyttsx3' | 'auto'
                   'auto' picks the best available backend.
    output_path  : Destination file path (.wav or .mp3). Temp file if None.
    coqui_model  : Override the default Coqui model.
    speaker_wav  : (Coqui only) Reference audio for voice cloning.

    Returns
    -------
    Path to the synthesized audio file, or None on failure.
    """
    if not text or not text.strip():
        logger.warning("TTS called with empty text — skipping.")
        return None

    # Determine output path
    if output_path is None:
        suffix = ".wav" if backend in ("coqui", "pyttsx3") else ".mp3"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        output_path = tmp.name

    # Resolve "auto" backend
    if backend == "auto":
        if _has_coqui():
            backend = "coqui"
        elif _has_gtts():
            backend = "gtts"
        elif _has_pyttsx3():
            backend = "pyttsx3"
        else:
            logger.error("No TTS backend available. Install gtts, TTS, or pyttsx3.")
            return None

    logger.info("TTS | backend=%s | lang=%s | chars=%d", backend, language, len(text))

    try:
        if backend == "gtts":
            return _synthesize_gtts(text, language, output_path)
        elif backend == "coqui":
            return _synthesize_coqui(text, language, output_path,
                                      model=coqui_model, speaker_wav=speaker_wav)
        elif backend == "pyttsx3":
            return _synthesize_pyttsx3(text, output_path)
        else:
            raise ValueError(f"Unknown TTS backend: {backend}")

    except Exception as exc:
        logger.error("TTS synthesis failed: %s", exc)
        # Attempt chain-fallback
        return _fallback_synthesis(text, output_path)


def _fallback_synthesis(text: str, output_path: str) -> Optional[str]:
    """Try each available backend in order of quality."""
    for backend in ["gtts", "pyttsx3"]:
        try:
            if backend == "gtts" and _has_gtts():
                return _synthesize_gtts(text, ENGLISH_LANG, output_path)
            if backend == "pyttsx3" and _has_pyttsx3():
                return _synthesize_pyttsx3(text, output_path)
        except Exception:
            continue
    logger.error("All TTS backends failed.")
    return None


def chunk_text(text: str, max_chars: int = 500) -> list[str]:
    """
    Split long text into chunks for TTS processing.
    Splits at sentence boundaries to avoid mid-sentence cuts.
    """
    import re
    sentences = re.split(r"(?<=[।!?.]) +", text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) <= max_chars:
            current += (" " if current else "") + s
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks
