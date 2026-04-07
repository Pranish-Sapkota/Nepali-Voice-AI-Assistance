"""
tests/test_suite.py
-------------------
Sample test cases for the Nepali Voice AI system.
Run: pytest tests/test_suite.py -v
"""

import os
import sys
import tempfile
import wave
import struct
import math
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════════════════
def _make_wav(path: str, duration_s: float = 1.0, freq: float = 440.0,
              sample_rate: int = 16000) -> str:
    """Create a synthetic sine-wave WAV file for testing."""
    n_samples = int(sample_rate * duration_s)
    amplitude = 16000
    samples = [
        int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
        for i in range(n_samples)
    ]
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))
    return path


@pytest.fixture
def sample_wav(tmp_path):
    path = str(tmp_path / "test.wav")
    return _make_wav(path)


@pytest.fixture
def sample_txt(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text(
        "Nepal is a landlocked country in South Asia. "
        "Kathmandu is the capital city of Nepal. "
        "The Himalayas are located in Nepal, including Mount Everest.\n\n"
        "नेपाल दक्षिण एसियामा अवस्थित एक भूपरिवेष्टित देश हो। "
        "काठमाडौं नेपालको राजधानी शहर हो।",
        encoding="utf-8",
    )
    return str(path)


# ════════════════════════════════════════════════════════════════════════════
# 1. UTILS / LANGUAGE DETECTION
# ════════════════════════════════════════════════════════════════════════════
class TestLanguageDetection:
    def test_detect_english(self):
        from utils.helpers import detect_language_heuristic
        assert detect_language_heuristic("Hello, how are you?") == "en"

    def test_detect_nepali(self):
        from utils.helpers import detect_language_heuristic
        assert detect_language_heuristic("नमस्ते, तपाईंलाई कस्तो छ?") == "ne"

    def test_detect_mixed(self):
        from utils.helpers import detect_language_heuristic
        result = detect_language_heuristic("Hello नमस्ते how are you कस्तो छ")
        assert result in ("mixed", "en", "ne")

    def test_detect_empty(self):
        from utils.helpers import detect_language_heuristic
        assert detect_language_heuristic("") == "en"

    def test_devanagari_check(self):
        from utils.helpers import contains_devanagari
        assert contains_devanagari("नेपाल") is True
        assert contains_devanagari("Nepal") is False
        assert contains_devanagari("Hello नमस्ते") is True

    def test_get_tts_language(self):
        from utils.helpers import get_tts_language
        assert get_tts_language("नमस्ते") == "ne"
        assert get_tts_language("Hello there") == "en"
        assert get_tts_language("Hello", whisper_lang="ne") == "en"  # heuristic wins


# ════════════════════════════════════════════════════════════════════════════
# 2. TEXT CLEANING
# ════════════════════════════════════════════════════════════════════════════
class TestTextCleaning:
    def test_clean_whisper_artifacts(self):
        from utils.helpers import clean_transcript
        dirty = "[Music] Hello there [Applause]"
        clean = clean_transcript(dirty)
        assert "[Music]" not in clean
        assert "[Applause]" not in clean
        assert "Hello there" in clean

    def test_clean_empty(self):
        from utils.helpers import clean_transcript
        assert clean_transcript("") == ""
        assert clean_transcript(None) == ""

    def test_truncate(self):
        from utils.helpers import truncate_text
        long_text = "word " * 500
        result = truncate_text(long_text, max_chars=50)
        assert len(result) <= 55   # allow for ellipsis
        assert result.endswith("…")


# ════════════════════════════════════════════════════════════════════════════
# 3. AUDIO RECORDER (mocked — no real mic needed)
# ════════════════════════════════════════════════════════════════════════════
class TestAudioRecorder:
    def test_save_wav(self, tmp_path):
        from app.audio_recorder import _save_wav
        path = str(tmp_path / "out.wav")
        audio = np.zeros(16000, dtype="int16")
        _save_wav(audio, path)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 44  # at least WAV header

    def test_wav_is_valid(self, tmp_path):
        from app.audio_recorder import _save_wav
        path = str(tmp_path / "out.wav")
        audio = (np.sin(np.linspace(0, 2 * np.pi * 440, 16000)) * 10000).astype("int16")
        _save_wav(audio, path)

        with wave.open(path, "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2

    @patch("sounddevice.rec")
    @patch("sounddevice.wait")
    def test_record_sounddevice(self, mock_wait, mock_rec, tmp_path):
        from app.audio_recorder import _record_sounddevice
        mock_rec.return_value = np.zeros((16000, 1), dtype="int16")
        path = str(tmp_path / "rec.wav")
        result = _record_sounddevice(1, path)
        assert result == path
        mock_rec.assert_called_once()


# ════════════════════════════════════════════════════════════════════════════
# 4. SPEECH TO TEXT (mocked model)
# ════════════════════════════════════════════════════════════════════════════
class TestSpeechToText:
    def test_file_not_found(self):
        from app.speech_to_text import transcribe
        with pytest.raises(FileNotFoundError):
            transcribe("/nonexistent/path/audio.wav")

    @patch("app.speech_to_text._load_model")
    def test_transcribe_returns_dict(self, mock_load, sample_wav):
        """Verify return schema without loading a real model."""
        mock_seg = MagicMock()
        mock_seg.text = " Hello world "
        mock_seg.start = 0.0
        mock_seg.end = 1.5
        mock_seg.avg_logprob = -0.3

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_seg]), mock_info)
        mock_load.return_value = mock_model

        from app.speech_to_text import transcribe
        result = transcribe(sample_wav)

        assert "text" in result
        assert "language" in result
        assert "segments" in result
        assert result["text"] == "Hello world"
        assert result["language"] == "en"

    def test_is_nepali_true(self):
        from app.speech_to_text import is_nepali
        result = {"language": "ne", "language_probability": 0.9}
        assert is_nepali(result) is True

    def test_is_nepali_false(self):
        from app.speech_to_text import is_nepali
        result = {"language": "en", "language_probability": 0.95}
        assert is_nepali(result) is False

    def test_is_nepali_low_confidence(self):
        from app.speech_to_text import is_nepali
        result = {"language": "ne", "language_probability": 0.2}
        assert is_nepali(result) is False


# ════════════════════════════════════════════════════════════════════════════
# 5. LLM ENGINE (mocked Ollama)
# ════════════════════════════════════════════════════════════════════════════
class TestLLMEngine:
    def test_ollama_not_running(self):
        from app.llm_engine import is_ollama_running
        with patch("requests.get", side_effect=ConnectionError):
            assert is_ollama_running() is False

    def test_build_history_entry(self):
        from app.llm_engine import build_history_entry
        entry = build_history_entry("user", "Hello")
        assert entry == {"role": "user", "content": "Hello"}

    def test_build_history_entry_invalid_role(self):
        from app.llm_engine import build_history_entry
        with pytest.raises(AssertionError):
            build_history_entry("system", "test")

    def test_summarize_history(self):
        from app.llm_engine import summarize_history
        history = [{"role": "user", "content": f"msg {i}"} for i in range(30)]
        trimmed = summarize_history(history, max_turns=5)
        assert len(trimmed) <= 10  # 5 turns × 2 messages

    @patch("requests.get")
    def test_list_models(self, mock_get):
        from app.llm_engine import list_available_models
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [{"name": "mistral:7b-instruct"}, {"name": "llama3"}]
        }
        models = list_available_models()
        assert "mistral:7b-instruct" in models

    @patch("requests.post")
    @patch("app.llm_engine.is_ollama_running", return_value=True)
    @patch("app.llm_engine.get_best_model", return_value="mistral:7b-instruct")
    def test_query_llm(self, mock_model, mock_running, mock_post):
        from app.llm_engine import query_llm
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "message": {"content": "Kathmandu is the capital of Nepal."}
        }
        result = query_llm("What is the capital of Nepal?")
        assert "Kathmandu" in result

    def test_query_llm_no_ollama(self):
        from app.llm_engine import query_llm
        with patch("app.llm_engine.is_ollama_running", return_value=False):
            with pytest.raises(ConnectionError):
                query_llm("test prompt")


# ════════════════════════════════════════════════════════════════════════════
# 6. RAG ENGINE
# ════════════════════════════════════════════════════════════════════════════
class TestRAGEngine:
    def test_extract_text_from_txt(self, sample_txt):
        from app.rag_engine import extract_text_from_file
        text = extract_text_from_file(sample_txt)
        assert "Nepal" in text
        assert len(text) > 50

    def test_chunk_text_sizes(self):
        from app.rag_engine import chunk_text
        text = "This is a sentence. " * 100
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        assert all(len(c) <= 250 for c in chunks)  # chunks ≤ size + overlap tolerance
        assert len(chunks) > 1

    def test_chunk_nepali_text(self):
        from app.rag_engine import chunk_text
        nepali = "नेपाल एक सुन्दर देश हो। काठमाडौं यसको राजधानी हो। " * 20
        chunks = chunk_text(nepali, chunk_size=100)
        assert len(chunks) > 0
        assert all(len(c) > 10 for c in chunks)

    def test_chunk_text_min_length_filter(self):
        from app.rag_engine import chunk_text
        text = "Hi. " * 200
        chunks = chunk_text(text, chunk_size=500)
        # trivial chunks (<30 chars) should be filtered
        assert all(len(c) >= 30 for c in chunks)

    def test_vector_store_empty_search(self):
        from app.rag_engine import VectorStore
        vs = VectorStore()
        results = vs.search("What is Nepal?")
        assert results == []

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("sentence_transformers"),
        reason="sentence-transformers not installed",
    )
    def test_vector_store_add_and_search(self, sample_txt):
        from app.rag_engine import VectorStore
        vs = VectorStore()
        count = vs.add_document(sample_txt, "test_doc")
        assert count > 0
        results = vs.search("What is the capital of Nepal?", top_k=2)
        assert len(results) > 0
        assert "source" in results[0]
        assert "text" in results[0]
        assert "score" in results[0]

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("sentence_transformers"),
        reason="sentence-transformers not installed",
    )
    def test_build_context(self, sample_txt):
        from app.rag_engine import VectorStore
        vs = VectorStore()
        vs.add_document(sample_txt, "test")
        ctx = vs.build_context("capital of Nepal")
        assert isinstance(ctx, str)

    def test_unsupported_file_type(self, tmp_path):
        from app.rag_engine import extract_text_from_file
        docx_path = tmp_path / "test.docx"
        docx_path.write_text("test")
        with pytest.raises(ValueError, match="Unsupported"):
            extract_text_from_file(str(docx_path))


# ════════════════════════════════════════════════════════════════════════════
# 7. TTS
# ════════════════════════════════════════════════════════════════════════════
class TestTTS:
    def test_empty_text_returns_none(self):
        from app.tts import synthesize
        result = synthesize("", backend="gtts")
        assert result is None

    def test_chunk_text(self):
        from app.tts import chunk_text
        long_text = "This is a sentence. " * 50
        chunks = chunk_text(long_text, max_chars=100)
        assert len(chunks) > 1
        assert all(len(c) <= 110 for c in chunks)

    @patch("gtts.gTTS.save")
    def test_gtts_synthesis(self, mock_save, tmp_path):
        import gtts
        gtts_path = str(tmp_path / "out.mp3")
        from app.tts import _synthesize_gtts
        with patch("gtts.gTTS") as mock_gtts:
            mock_instance = MagicMock()
            mock_gtts.return_value = mock_instance
            # Create the file so path check passes
            Path(gtts_path).touch()
            _synthesize_gtts("Hello world", "en", gtts_path)
            mock_instance.save.assert_called_once()


# ════════════════════════════════════════════════════════════════════════════
# 8. INTEGRATION: full pipeline (all mocked)
# ════════════════════════════════════════════════════════════════════════════
class TestPipelineIntegration:
    @patch("app.speech_to_text._load_model")
    @patch("app.llm_engine.is_ollama_running", return_value=True)
    @patch("app.llm_engine.get_best_model", return_value="mistral:7b-instruct")
    @patch("requests.post")
    def test_transcribe_and_query(self, mock_post, mock_model,
                                   mock_running, mock_load, sample_wav):
        """End-to-end: WAV → transcription → LLM response."""
        # Mock Whisper
        mock_seg = MagicMock()
        mock_seg.text = " What is the capital of Nepal? "
        mock_seg.start = 0.0; mock_seg.end = 2.0; mock_seg.avg_logprob = -0.2
        mock_info = MagicMock()
        mock_info.language = "en"; mock_info.language_probability = 0.98
        mock_model_inst = MagicMock()
        mock_model_inst.transcribe.return_value = (iter([mock_seg]), mock_info)
        mock_load.return_value = mock_model_inst

        # Mock Ollama
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "message": {"content": "Kathmandu is the capital of Nepal."}
        }

        from app.speech_to_text import transcribe
        from app.llm_engine import query_llm
        from utils.helpers import clean_transcript

        # Step 1: Transcribe
        result = transcribe(sample_wav)
        text = clean_transcript(result["text"])
        assert "capital" in text.lower()

        # Step 2: LLM query
        response = query_llm(text)
        assert "Kathmandu" in response


# ════════════════════════════════════════════════════════════════════════════
# SAMPLE NEPALI QUERIES (documentation-style, not automated)
# ════════════════════════════════════════════════════════════════════════════
SAMPLE_TEST_CASES = [
    {
        "id": "NE_01",
        "input_nepali": "नेपालको राजधानी कहाँ हो?",
        "expected_topic": "Kathmandu / काठमाडौं",
        "language": "ne",
    },
    {
        "id": "NE_02",
        "input_nepali": "माउन्ट एभरेस्टको उचाइ कति हो?",
        "expected_topic": "8848.86 meters",
        "language": "ne",
    },
    {
        "id": "EN_01",
        "input_english": "What is the currency of Nepal?",
        "expected_topic": "Nepali Rupee",
        "language": "en",
    },
    {
        "id": "EN_02",
        "input_english": "Who is the prime minister of Nepal?",
        "expected_topic": "current leader",
        "language": "en",
    },
    {
        "id": "RAG_01",
        "input_english": "Summarize the uploaded document",
        "expected_topic": "document summary",
        "requires_rag": True,
        "language": "en",
    },
    {
        "id": "MIXED_01",
        "input": "Nepal को population कति छ?",
        "expected_topic": "~30 million",
        "language": "mixed",
    },
]


def test_sample_cases_documented():
    """Verify all sample test cases are well-formed."""
    for case in SAMPLE_TEST_CASES:
        assert "id" in case
        assert "expected_topic" in case
        assert "language" in case


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
