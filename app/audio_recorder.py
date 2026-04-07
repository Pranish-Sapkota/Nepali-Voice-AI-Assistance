"""
audio_recorder.py
-----------------
Handles microphone input and WAV file saving.
Supports both sounddevice and pyaudio as fallbacks.
"""

import os
import logging
import tempfile
import numpy as np
import wave

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000          # Whisper expects 16 kHz
CHANNELS    = 1              # Mono
DTYPE       = "int16"        # 16-bit PCM
DEFAULT_DURATION = 7         # seconds (silence-aware recording goes longer)


# ── Backend detection ──────────────────────────────────────────────────────────
def _get_backend() -> str:
    """Return the available audio backend: 'sounddevice' or 'pyaudio'."""
    try:
        import sounddevice  # noqa: F401
        return "sounddevice"
    except ImportError:
        pass
    try:
        import pyaudio  # noqa: F401
        return "pyaudio"
    except ImportError:
        pass
    raise RuntimeError(
        "No audio backend found. Install sounddevice or pyaudio:\n"
        "  pip install sounddevice\n  pip install pyaudio"
    )


# ── sounddevice implementation ─────────────────────────────────────────────────
def _record_sounddevice(duration: int, output_path: str) -> str:
    import sounddevice as sd

    logger.info("Recording %ds via sounddevice …", duration)
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
    )
    sd.wait()
    _save_wav(audio, output_path)
    return output_path


def _record_sounddevice_vad(max_duration: int, silence_threshold: float,
                             silence_duration: float, output_path: str) -> str:
    """
    Record until the user stops speaking (VAD-lite via energy threshold).
    Falls back to fixed duration after max_duration seconds.
    """
    import sounddevice as sd

    CHUNK = int(SAMPLE_RATE * 0.1)   # 100 ms chunks
    silence_chunks = int(silence_duration / 0.1)
    max_chunks = int(max_duration / 0.1)

    frames: list[np.ndarray] = []
    silent_count = 0
    speaking = False

    logger.info("VAD recording started (max %ds) …", max_duration)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype=DTYPE, blocksize=CHUNK) as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(CHUNK)
            energy = np.abs(chunk).mean()
            frames.append(chunk.copy())

            if energy > silence_threshold:
                speaking = True
                silent_count = 0
            elif speaking:
                silent_count += 1
                if silent_count >= silence_chunks:
                    logger.info("Silence detected — stopping recording.")
                    break

    audio = np.concatenate(frames, axis=0)
    _save_wav(audio, output_path)
    return output_path


# ── pyaudio implementation ─────────────────────────────────────────────────────
def _record_pyaudio(duration: int, output_path: str) -> str:
    import pyaudio

    pa = pyaudio.PyAudio()
    CHUNK = 1024
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    logger.info("Recording %ds via pyaudio …", duration)
    frames = []
    for _ in range(0, int(SAMPLE_RATE / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    pa.terminate()

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    return output_path


# ── Shared helpers ─────────────────────────────────────────────────────────────
def _save_wav(audio: np.ndarray, path: str) -> None:
    """Save a numpy int16 array to a WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)          # 16-bit = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    logger.info("Audio saved → %s", path)


def _make_output_path(output_path: str | None) -> str:
    if output_path:
        return output_path
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    return tmp.name


# ── Public API ─────────────────────────────────────────────────────────────────
def record_audio(
    duration: int = DEFAULT_DURATION,
    output_path: str | None = None,
    use_vad: bool = True,
    silence_threshold: float = 300.0,
    silence_duration: float = 1.5,
    max_duration: int = 30,
) -> str:
    """
    Record audio from the default microphone.

    Parameters
    ----------
    duration          : Fixed recording length (used when use_vad=False or backend=pyaudio).
    output_path       : Where to save the WAV file. Temp file if None.
    use_vad           : Stop early when silence is detected (sounddevice only).
    silence_threshold : RMS energy below which audio is considered silent.
    silence_duration  : Seconds of silence before stopping.
    max_duration      : Hard cap when VAD is enabled.

    Returns
    -------
    Path to the saved WAV file.
    """
    path = _make_output_path(output_path)
    backend = _get_backend()

    try:
        if backend == "sounddevice":
            if use_vad:
                return _record_sounddevice_vad(
                    max_duration, silence_threshold, silence_duration, path
                )
            return _record_sounddevice(duration, path)
        else:
            return _record_pyaudio(duration, path)

    except Exception as exc:
        logger.error("Recording failed: %s", exc)
        raise RuntimeError(f"Audio recording error: {exc}") from exc


def list_microphones() -> list[dict]:
    """Return available input devices (for debugging / UI device picker)."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]
    except Exception as exc:
        logger.warning("Could not list microphones: %s", exc)
        return []
