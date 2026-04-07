#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# setup.sh — One-command setup for Nepali Voice AI
# Usage:  chmod +x setup.sh && ./setup.sh
# ══════════════════════════════════════════════════════════════════
set -e

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

echo -e "${BOLD}🇳🇵  Nepali Voice AI — Setup Script${RESET}\n"

# ── Python version check ───────────────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python)
PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python:${RESET} $PY_VERSION"

if $PYTHON -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"; then
  echo -e "${GREEN}✓ Python 3.10+ detected${RESET}"
else
  echo -e "${RED}✗ Python 3.10 or higher required${RESET}"
  exit 1
fi

# ── Virtual environment ────────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
  echo -e "\n${YELLOW}→ Creating virtual environment…${RESET}"
  $PYTHON -m venv venv
fi

source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${RESET}"

# ── Upgrade pip ────────────────────────────────────────────────────────────────
pip install --upgrade pip setuptools wheel -q

# ── Detect CUDA ────────────────────────────────────────────────────────────────
CUDA_VERSION=""
if command -v nvcc &>/dev/null; then
  CUDA_VERSION=$(nvcc --version | grep -oP "release \K[\d.]+")
  echo -e "${GREEN}✓ CUDA detected: ${CUDA_VERSION}${RESET}"
else
  echo -e "${YELLOW}⚠  No CUDA detected — using CPU mode${RESET}"
fi

# ── Install PyTorch ────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}→ Installing PyTorch…${RESET}"
if [ -n "$CUDA_VERSION" ]; then
  CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')
  pip install torch torchvision torchaudio --index-url \
    "https://download.pytorch.org/whl/cu${CUDA_MAJOR}" -q
else
  pip install torch torchvision torchaudio --index-url \
    https://download.pytorch.org/whl/cpu -q
fi
echo -e "${GREEN}✓ PyTorch installed${RESET}"

# ── Install requirements ───────────────────────────────────────────────────────
echo -e "\n${YELLOW}→ Installing Python dependencies…${RESET}"
pip install -r requirements.txt -q
echo -e "${GREEN}✓ Dependencies installed${RESET}"

# ── FAISS GPU (optional) ───────────────────────────────────────────────────────
if [ -n "$CUDA_VERSION" ]; then
  echo -e "\n${YELLOW}→ Installing FAISS GPU…${RESET}"
  pip install faiss-gpu -q 2>/dev/null || pip install faiss-cpu -q
else
  pip install faiss-cpu -q
fi
echo -e "${GREEN}✓ FAISS installed${RESET}"

# ── Create directories ─────────────────────────────────────────────────────────
echo -e "\n${YELLOW}→ Creating directories…${RESET}"
mkdir -p models data data/faiss_index
echo -e "${GREEN}✓ Directories ready${RESET}"

# ── Ollama check ───────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}→ Checking Ollama…${RESET}"
if command -v ollama &>/dev/null; then
  echo -e "${GREEN}✓ Ollama is installed${RESET}"
  echo -e "${YELLOW}→ Pulling mistral:7b-instruct (this may take a while)…${RESET}"
  ollama pull mistral:7b-instruct || echo -e "${YELLOW}⚠  Could not pull model — start ollama serve first${RESET}"
else
  echo -e "${RED}✗ Ollama not found${RESET}"
  echo ""
  echo "Install Ollama from: https://ollama.ai"
  echo "  Linux:   curl -fsSL https://ollama.ai/install.sh | sh"
  echo "  macOS:   brew install ollama"
  echo "  Windows: https://ollama.ai/download/windows"
  echo ""
  echo "Then run:"
  echo "  ollama serve"
  echo "  ollama pull mistral:7b-instruct"
fi

# ── sounddevice / portaudio check ─────────────────────────────────────────────
echo -e "\n${YELLOW}→ Checking audio backend…${RESET}"
if $PYTHON -c "import sounddevice" 2>/dev/null; then
  echo -e "${GREEN}✓ sounddevice available${RESET}"
else
  echo -e "${YELLOW}⚠  sounddevice not working. Trying pyaudio…${RESET}"
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get install -y portaudio19-dev 2>/dev/null || true
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install portaudio 2>/dev/null || true
  fi
  pip install pyaudio -q || echo -e "${RED}✗ Neither sounddevice nor pyaudio installed${RESET}"
fi

echo -e "\n${BOLD}${GREEN}════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN}✅  Setup complete!${RESET}"
echo -e "${GREEN}════════════════════════════════════${RESET}\n"
echo -e "Start the app:"
echo -e "  ${BOLD}source venv/bin/activate${RESET}"
echo -e "  ${BOLD}ollama serve &${RESET}"
echo -e "  ${BOLD}streamlit run app/app.py${RESET}\n"
