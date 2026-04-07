"""
utils/helpers.py
----------------
Shared utility functions: language detection, text cleaning,
formatting, logging setup, and system diagnostics.
"""

import logging
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional


# ── Logging setup ──────────────────────────────────────────────────────────────
def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure application-wide logging."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=handlers,
        force=True,
    )


# ── Language detection ─────────────────────────────────────────────────────────
_DEVANAGARI_RANGE = range(0x0900, 0x097F + 1)


def contains_devanagari(text: str) -> bool:
    """Return True if the text contains any Devanagari codepoints (Nepali/Hindi)."""
    return any(ord(c) in _DEVANAGARI_RANGE for c in text)


def detect_language_heuristic(text: str) -> str:
    """
    Fast heuristic language detection (no model needed).
    Returns 'ne' (Nepali), 'en' (English), or 'mixed'.
    """
    if not text:
        return "en"

    devanagari_chars = sum(1 for c in text if ord(c) in _DEVANAGARI_RANGE)
    latin_chars      = sum(1 for c in text if c.isascii() and c.isalpha())
    total            = devanagari_chars + latin_chars

    if total == 0:
        return "en"

    deva_ratio = devanagari_chars / total

    if deva_ratio > 0.6:
        return "ne"
    elif deva_ratio > 0.2:
        return "mixed"
    return "en"


def get_tts_language(text: str, whisper_lang: Optional[str] = None) -> str:
    """
    Determine the TTS language code.
    Combines Whisper's detection with heuristic analysis.
    """
    heuristic = detect_language_heuristic(text)
    if heuristic == "ne":
        return "ne"
    if whisper_lang == "ne":
        return "ne"
    return "en"


# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_transcript(text: str) -> str:
    """Remove Whisper artifacts and normalize whitespace."""
    if not text:
        return ""
    # Remove common Whisper hallucinations
    noise_patterns = [
        r"\[.*?\]",          # [Music], [Applause], etc.
        r"\(.*?\)",          # (inaudible)
        r"♪.*?♪",            # music notes
        r"<\|.*?\|>",        # Whisper timestamp tokens
    ]
    for pat in noise_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return " ".join(text.split()).strip()


def truncate_text(text: str, max_chars: int = 2000, ellipsis: str = "…") -> str:
    """Truncate text to max_chars, breaking at word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + ellipsis


def format_duration(seconds: float) -> str:
    """Human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


# ── System diagnostics ─────────────────────────────────────────────────────────
def get_system_info() -> dict:
    """Collect system info for the diagnostics panel."""
    info: dict = {"python": sys.version.split()[0]}

    # CUDA
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except ImportError:
        info["cuda_available"] = False

    # RAM
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1e9, 1)
        info["ram_used_gb"]  = round(vm.used  / 1e9, 1)
    except ImportError:
        pass

    # Ollama
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        info["ollama_running"] = r.status_code == 200
        if info["ollama_running"]:
            models = r.json().get("models", [])
            info["ollama_models"] = [m["name"] for m in models]
    except Exception:
        info["ollama_running"] = False

    return info


def check_dependencies() -> dict[str, bool]:
    """Check which optional dependencies are installed."""
    deps = {
        "faster_whisper" : "faster_whisper",
        "sounddevice"    : "sounddevice",
        "pyaudio"        : "pyaudio",
        "gtts"           : "gtts",
        "coqui_tts"      : "TTS",
        "pyttsx3"        : "pyttsx3",
        "faiss"          : "faiss",
        "sentence_transformers": "sentence_transformers",
        "pdfplumber"     : "pdfplumber",
        "torch"          : "torch",
        "streamlit"      : "streamlit",
        "requests"       : "requests",
        "numpy"          : "numpy",
    }
    result = {}
    for friendly, module in deps.items():
        try:
            __import__(module)
            result[friendly] = True
        except ImportError:
            result[friendly] = False
    return result


# ── File utilities ─────────────────────────────────────────────────────────────
def ensure_dir(path: str) -> Path:
    """Create directory (and parents) if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_project_root() -> Path:
    """Return the nepali_voice_ai project root directory."""
    return Path(__file__).parent.parent


def temp_audio_path(suffix: str = ".wav") -> str:
    """Return a path in the data/ directory for temp audio files."""
    data_dir = get_project_root() / "data"
    data_dir.mkdir(exist_ok=True)
    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        suffix=suffix, dir=str(data_dir), delete=False
    )
    tmp.close()
    return tmp.name
