"""
llm_engine.py
-------------
Handles all interactions with the local Ollama LLM.
Supports streaming, chat history, and RAG context injection.
Completely offline — no paid APIs.
"""

import json
import logging
import time
from typing import Generator, Optional

import requests

logger = logging.getLogger(__name__)

# ── Ollama configuration ───────────────────────────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"
DEFAULT_MODEL    = "mistral:7b-instruct"
FALLBACK_MODELS  = ["mistral", "llama3", "llama2", "phi3", "gemma2"]

# System prompt tuned for Nepali/English bilingual assistant
SYSTEM_PROMPT = """You are a helpful, bilingual AI assistant that speaks both Nepali (नेपाली) and English fluently.

Key behaviors:
1. If the user writes in Nepali (Devanagari script), respond in Nepali.
2. If the user writes in English, respond in English.
3. If the query is mixed, match the dominant language.
4. Be concise, accurate, and culturally aware of Nepali context.
5. When RAG context is provided, prioritize it for answering questions.
6. For general knowledge questions, use your training data.
7. Always be respectful and helpful.

You are running locally on the user's machine — no internet access is needed."""


# ── Health check ───────────────────────────────────────────────────────────────
def is_ollama_running() -> bool:
    """Return True if Ollama server is reachable."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def list_available_models() -> list[str]:
    """Return names of locally installed Ollama models."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception as exc:
        logger.warning("Could not list models: %s", exc)
        return []


def get_best_model() -> str:
    """
    Return the best available model from the preference list.
    Falls back through FALLBACK_MODELS if DEFAULT_MODEL is not installed.
    """
    available = list_available_models()
    if not available:
        return DEFAULT_MODEL   # Optimistic — might get pulled

    # Exact match first
    if DEFAULT_MODEL in available:
        return DEFAULT_MODEL

    # Prefix match (e.g. "mistral" matches "mistral:latest")
    for preferred in [DEFAULT_MODEL] + FALLBACK_MODELS:
        base = preferred.split(":")[0]
        for installed in available:
            if installed.startswith(base):
                logger.info("Using model: %s (preferred: %s)", installed, preferred)
                return installed

    logger.warning("None of the preferred models found. Using: %s", available[0])
    return available[0]


# ── Core generation ────────────────────────────────────────────────────────────
def query_llm(
    prompt: str,
    model: Optional[str] = None,
    system_prompt: str = SYSTEM_PROMPT,
    context: Optional[str] = None,
    history: Optional[list[dict]] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    stream: bool = False,
) -> str:
    """
    Send a prompt to the local Ollama model and return the response.

    Parameters
    ----------
    prompt       : The user's message (text).
    model        : Ollama model name. Auto-selected if None.
    system_prompt: Override the default system prompt.
    context      : RAG context to prepend (retrieved documents).
    history      : List of {role, content} dicts for multi-turn chat.
    temperature  : Sampling temperature (0 = deterministic).
    max_tokens   : Maximum tokens to generate.
    stream       : If True, returns a generator (use query_llm_stream instead).

    Returns
    -------
    The model's response as a plain string.
    """
    if not is_ollama_running():
        raise ConnectionError(
            "Ollama is not running. Start it with:\n  ollama serve\n"
            "Then pull a model:\n  ollama pull mistral:7b-instruct"
        )

    selected_model = model or get_best_model()

    # Build message list ────────────────────────────────────────────────────────
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # Inject RAG context as a system-level knowledge block
    if context and context.strip():
        rag_block = (
            "---\nRELEVANT CONTEXT (from uploaded documents):\n"
            f"{context.strip()}\n---\n"
            "Use the above context to answer the user's question when relevant."
        )
        messages.append({"role": "system", "content": rag_block})

    # Previous conversation turns
    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": selected_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    }

    logger.info("Querying model=%s | tokens≤%d | rag=%s",
                selected_model, max_tokens, bool(context))

    start = time.time()
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
    except requests.Timeout:
        raise TimeoutError("LLM request timed out (120 s). Is the model loaded?")
    except requests.HTTPError as exc:
        raise RuntimeError(f"Ollama API error: {exc}") from exc

    elapsed = time.time() - start
    result = response.json()
    answer = result["message"]["content"].strip()

    logger.info("LLM responded in %.1fs (%d chars)", elapsed, len(answer))
    return answer


def query_llm_stream(
    prompt: str,
    model: Optional[str] = None,
    system_prompt: str = SYSTEM_PROMPT,
    context: Optional[str] = None,
    history: Optional[list[dict]] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> Generator[str, None, None]:
    """
    Stream the LLM response token-by-token.
    Yields string chunks as they arrive from Ollama.
    """
    if not is_ollama_running():
        raise ConnectionError("Ollama is not running.")

    selected_model = model or get_best_model()
    messages = [{"role": "system", "content": system_prompt}]

    if context and context.strip():
        rag_block = (
            "---\nRELEVANT CONTEXT:\n"
            f"{context.strip()}\n---\n"
        )
        messages.append({"role": "system", "content": rag_block})

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": selected_model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    with requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break


# ── Prompt helpers ─────────────────────────────────────────────────────────────
def build_history_entry(role: str, content: str) -> dict:
    """Create a single history entry dict."""
    assert role in ("user", "assistant"), "role must be 'user' or 'assistant'"
    return {"role": role, "content": content}


def summarize_history(history: list[dict], max_turns: int = 10) -> list[dict]:
    """Trim history to last N turns to avoid context overflow."""
    return history[-(max_turns * 2):]


# ── Pull model helper ──────────────────────────────────────────────────────────
def pull_model(model_name: str) -> bool:
    """
    Pull a model from Ollama registry.
    Returns True on success, False on failure.
    Should be run ONCE during setup.
    """
    try:
        logger.info("Pulling model: %s …", model_name)
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": model_name},
            stream=True,
            timeout=600,  # 10 min for large model
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                status = json.loads(line).get("status", "")
                if status:
                    logger.info("[pull] %s", status)
        logger.info("Model '%s' pulled successfully.", model_name)
        return True
    except Exception as exc:
        logger.error("Failed to pull model '%s': %s", model_name, exc)
        return False
