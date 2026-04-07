# 🇳🇵 Nepali Voice AI Assistant

> **100% Offline · Nepali + English · Voice-to-Voice AI · RAG-enabled**

A production-grade, fully offline Voice AI system that understands Nepali (नेपाली) and English speech, generates intelligent responses via a local LLM, and speaks back — without a single paid API call.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://python.org)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-orange?style=flat-square)](https://ollama.ai)
[![Whisper](https://img.shields.io/badge/STT-faster--whisper-purple?style=flat-square)](https://github.com/guillaumekln/faster-whisper)
[![FAISS](https://img.shields.io/badge/RAG-FAISS-red?style=flat-square)](https://github.com/facebookresearch/faiss)

---

## ✨ Features

| Feature | Technology | Status |
|---|---|---|
| 🎙️ Voice Recording | sounddevice / pyaudio | ✅ |
| 🧠 Speech-to-Text | faster-whisper (small) | ✅ Nepali + English |
| 🤖 Local LLM | Ollama + Mistral 7B | ✅ Offline |
| 📚 RAG Pipeline | FAISS + sentence-transformers | ✅ PDF + TXT |
| 🔊 Text-to-Speech | gTTS / Coqui TTS | ✅ |
| 🖥️ Web UI | Streamlit | ✅ |
| 🚀 GPU Acceleration | CUDA (CTranslate2) | ✅ RTX 3050 |
| 🌐 Internet required? | — | ❌ **No** |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Nepali Voice AI Pipeline                        │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────┐   WAV    ┌──────────────────┐  text   ┌────────────────┐
  │   Mic    │ ──────▶  │  faster-whisper   │ ──────▶ │  RAG Engine    │
  │(sounddev)│  16kHz   │  (small model)   │         │ FAISS + MiniLM │
  └──────────┘          │  Nepali / EN     │         └───────┬────────┘
                        └──────────────────┘                 │ context
                                                             ▼
  ┌──────────┐  audio   ┌──────────────────┐  prompt ┌────────────────┐
  │  Speaker │ ◀──────  │   TTS Engine     │ ◀────── │ Ollama LLM     │
  │ (gTTS /  │          │ (gTTS/Coqui/py3) │         │ mistral:7b     │
  │  Coqui)  │          └──────────────────┘         │ (quantized)    │
  └──────────┘                                       └────────────────┘

  ──────────────────────────────────────────────────────────────────────
                      Streamlit Web Interface
  ──────────────────────────────────────────────────────────────────────
```

---

## 📁 Project Structure

```
Nepali-Voice-AI-Assistance/
│
├── app/
│   ├── app.py              # Streamlit UI (main entry point)
│   ├── audio_recorder.py   # Microphone recording (VAD-aware)
│   ├── speech_to_text.py   # faster-whisper transcription
│   ├── llm_engine.py       # Ollama LLM client + streaming
│   ├── tts.py              # Text-to-Speech (gTTS / Coqui)
│   └── rag_engine.py       # FAISS RAG pipeline
│
├── utils/
│   ├── __init__.py
│   └── helpers.py          # Language detection, logging, diagnostics
│
├── tests/
│   └── test_suite.py       # Full test suite (pytest)
│
├── models/                 # Whisper model cache (auto-downloaded)
├── data/                   # Audio temp files + FAISS index
│
├── requirements.txt
├── setup.sh                # One-command setup script
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | [Download](https://python.org) |
| Ollama | Latest | [Download](https://ollama.ai) |
| CUDA (optional) | 11.8+ | RTX 3050 supported |
| RAM | 8 GB+ | 16 GB recommended |
| Disk | 10 GB+ | For models |

### 1. Clone the repository

```bash
git clone https://github.com/Pranish-Sapkota/Nepali-Voice-AI-Assistance.git
cd Nepali-Voice-AI-Assistance
```

### 2. Run the setup script (Linux / macOS)

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a virtual environment
- Detect and configure CUDA
- Install all Python dependencies
- Pull the Mistral 7B model via Ollama

### 3. Manual Setup (Windows / advanced)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama (https://ollama.ai/download/windows)
# Then pull the model:
ollama pull mistral:7b-instruct
```

### 4. Launch the app

```bash
# Terminal 1: Start Ollama server
ollama serve

# Terminal 2: Launch Streamlit
source venv/bin/activate
streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🎮 Usage Guide

### Voice Input
1. Click **"🎙️ Start Recording"**
2. Speak in Nepali or English
3. The app automatically stops when you pause (VAD enabled)
4. View transcription → AI response → hear the reply

### Text Input
- Type in the text box and click **"Send ➤"**
- Supports both Nepali (Devanagari) and English

### RAG (Document Q&A)
1. Upload a `.pdf` or `.txt` file in the sidebar
2. Enable **"Use uploaded docs"** toggle
3. Ask questions about the document in voice or text
4. The AI retrieves relevant sections and answers accurately

### Settings Sidebar
| Setting | Description |
|---|---|
| **LLM Model** | Choose from installed Ollama models |
| **Whisper Model** | `tiny` (fast) → `small` (Nepali best) → `medium` (accurate) |
| **Language Override** | Force Nepali or English detection |
| **VAD** | Auto-stop recording on silence |
| **TTS Backend** | `gTTS` (needs internet) / `Coqui` (offline) / `pyttsx3` |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/test_suite.py -v

# Run specific category
pytest tests/test_suite.py::TestLanguageDetection -v
pytest tests/test_suite.py::TestRAGEngine -v

# With coverage
pip install pytest-cov
pytest tests/test_suite.py --cov=app --cov=utils -v
```

### Sample Test Cases

| ID | Input | Expected |
|---|---|---|
| NE_01 | "नेपालको राजधानी कहाँ हो?" | Kathmandu |
| NE_02 | "माउन्ट एभरेस्टको उचाइ कति हो?" | 8848.86 m |
| EN_01 | "What is the currency of Nepal?" | Nepali Rupee |
| EN_02 | "Who is the prime minister of Nepal?" | Current leader |
| RAG_01 | "Summarize the uploaded document" | Document content |
| MIXED_01 | "Nepal को population कति छ?" | ~30 million |

---

## ⚡ Performance

Tested on: **RTX 3050 (6 GB VRAM) + 16 GB RAM**

| Component | Model | Speed | VRAM |
|---|---|---|---|
| Whisper STT | small (int8) | ~2–3s per 7s clip | 1.2 GB |
| Embedding | MiniLM L12 | ~50ms per query | 0.3 GB |
| LLM | Mistral 7B (Q4_K_M) | ~15–25 tok/s | 4.2 GB |
| TTS (gTTS) | — | ~1s | 0 GB |
| **Total** | | | **~5.7 GB** ✅ |

> **Tip for 6 GB VRAM**: Use `tiny` Whisper model + `mistral:7b-instruct` Q4. The system is tuned for this exact configuration.

### Optimization flags

```bash
# Force CPU-only mode (if VRAM is tight)
export CUDA_VISIBLE_DEVICES=-1
streamlit run app/app.py

# Use smallest models
# Set Whisper = "tiny", and choose a 3B model:
ollama pull phi3:mini
```

---

## 🔧 Model Management

```bash
# List installed models
ollama list

# Pull alternative models
ollama pull llama3:8b-instruct
ollama pull phi3:mini           # Lightweight (3.8B)
ollama pull gemma2:9b

# Remove a model
ollama rm mistral:7b-instruct

# Check Ollama status
curl http://localhost:11434/api/tags
```

---

## 🌐 Nepali Language Support

| Component | Nepali Support | Notes |
|---|---|---|
| faster-whisper | ✅ Good | Use `small` or `medium` model |
| MiniLM embeddings | ✅ Multilingual | Paraphrase-multilingual-MiniLM-L12 |
| Mistral 7B | ⚠️ Moderate | Works for common queries; fine-tune for better results |
| gTTS | ✅ Good | Requires internet |
| Coqui XTTS v2 | ✅ Good | Fully offline |

**Tips for better Nepali accuracy:**
- Use `small` or `medium` Whisper model (not `tiny`)
- Speak clearly at normal pace
- Set language to `ne` if auto-detection fails
- For TTS: gTTS produces more natural Nepali than pyttsx3

---

## 🛠️ Troubleshooting

<details>
<summary><strong>🔴 Ollama not running</strong></summary>

```bash
# Check if Ollama is installed
which ollama

# Start the server
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```
</details>

<details>
<summary><strong>🔴 Microphone not detected</strong></summary>

```bash
# List available microphones (Python)
python -c "from app.audio_recorder import list_microphones; print(list_microphones())"

# Linux: Install ALSA/PulseAudio dev headers
sudo apt install portaudio19-dev python3-pyaudio

# macOS: Install portaudio
brew install portaudio
```
</details>

<details>
<summary><strong>🔴 CUDA out of memory</strong></summary>

- Switch to `tiny` Whisper model in sidebar
- Use a smaller LLM: `ollama pull phi3:mini`
- Set `CUDA_VISIBLE_DEVICES=-1` to force CPU
</details>

<details>
<summary><strong>🔴 FAISS install fails</strong></summary>

```bash
# CPU-only FAISS
pip install faiss-cpu

# GPU FAISS (requires CUDA toolkit)
pip install faiss-gpu
```
</details>

<details>
<summary><strong>🔴 gTTS ConnectionError</strong></summary>

gTTS requires internet access. Use Coqui TTS for offline:
```bash
pip install TTS
# Set TTS backend to "coqui" in sidebar
```
</details>

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/add-hindi-support`
3. Commit changes: `git commit -m "Add Hindi language support"`
4. Push: `git push origin feature/add-hindi-support`
5. Open a Pull Request

---

## 🗺️ Roadmap

- [ ] Real-time streaming transcription
- [ ] Fine-tuned Nepali LLM via LoRA
- [ ] Wake word detection ("Hey Sahayak")
- [ ] Mobile app (React Native)
- [ ] Docker containerization
- [ ] Hindi language support
- [ ] Voice cloning (speaker reference)

---

## 🙏 Acknowledgments

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) — CTranslate2-based Whisper
- [Ollama](https://ollama.ai) — Local LLM serving
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [sentence-transformers](https://www.sbert.net/) — Multilingual embeddings
- [Streamlit](https://streamlit.io) — Web UI framework
- [Coqui TTS](https://github.com/coqui-ai/TTS) — Offline text-to-speech

---

If you found this project useful, please give it a ⭐ on GitHub!

---

<div align="center">
  Made with ❤️ by Pranish Pr Sapkota for  🇳🇵 · <strong>नेपाल जिन्दाबाद</strong>
</div>
