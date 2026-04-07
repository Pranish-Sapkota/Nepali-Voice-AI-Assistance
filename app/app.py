"""
app.py
------
Nepali Voice AI Assistant — Streamlit frontend.

Features:
  • Voice recording (VAD-aware)
  • Nepali + English transcription (faster-whisper)
  • Local LLM responses (Ollama)
  • RAG over uploaded documents (FAISS + sentence-transformers)
  • Text-to-Speech playback (gTTS / Coqui / pyttsx3)
  • Chat history with export
  • System diagnostics sidebar
"""

import sys
import os
import logging
import tempfile
import time
import base64
from pathlib import Path
from typing import Optional

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ── Local modules ──────────────────────────────────────────────────────────────
from app.audio_recorder  import record_audio, list_microphones
from app.speech_to_text  import transcribe, DEFAULT_MODEL as DEFAULT_WHISPER_MODEL
from app.llm_engine      import (
    query_llm, query_llm_stream, is_ollama_running,
    list_available_models, get_best_model, build_history_entry,
    summarize_history,
)
from app.tts             import synthesize, available_backends
from app.rag_engine      import get_vector_store
from utils               import (
    setup_logging, get_system_info, check_dependencies,
    detect_language_heuristic, get_tts_language,
    clean_transcript, truncate_text, temp_audio_path,
)

# ── Logging ────────────────────────────────────────────────────────────────────
setup_logging("INFO", log_file=str(PROJECT_ROOT / "data" / "app.log"))
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Nepali Voice AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Noto+Serif+Devanagari:wght@400;600&display=swap');

  :root {
    --red:    #e63946;
    --blue:   #457b9d;
    --dark:   #1d3557;
    --cream:  #f1faee;
    --light:  #a8dadc;
  }

  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

  /* Header banner */
  .hero-banner {
    background: linear-gradient(135deg, var(--dark) 0%, #2d4a6e 60%, #1a3a5c 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    border-left: 6px solid var(--red);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
  }
  .hero-banner h1 {
    color: #fff; font-size: 2.1rem; font-weight: 700; margin: 0 0 6px;
  }
  .hero-banner p { color: var(--light); margin: 0; font-size: 1rem; }
  .hero-banner .flag { font-size: 1.8rem; }

  /* Chat bubble styles */
  .chat-user {
    background: linear-gradient(135deg, #1d3557, #2d4a6e);
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    word-wrap: break-word;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
  }
  .chat-assistant {
    background: linear-gradient(135deg, #f1faee, #e8f4f5);
    color: #1d3557;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 18px;
    margin: 8px 0;
    max-width: 85%;
    word-wrap: break-word;
    border-left: 4px solid var(--blue);
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }
  .chat-meta {
    font-size: 0.72rem;
    opacity: 0.6;
    margin-top: 4px;
    font-style: italic;
  }

  /* Status badges */
  .badge-online  { background:#16a34a; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.78rem; }
  .badge-offline { background:#dc2626; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.78rem; }
  .badge-warn    { background:#d97706; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.78rem; }

  /* Recording button */
  div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, var(--red), #c1121f) !important;
    border: none !important;
    border-radius: 50px !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    padding: 14px 32px !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(230,57,70,0.5) !important;
    transition: all 0.2s ease !important;
  }

  /* Transcription box */
  .transcript-box {
    background: #0f2035;
    border: 1px solid #2d4a6e;
    border-radius: 12px;
    padding: 16px 20px;
    font-family: 'Noto Serif Devanagari', 'Space Grotesk', serif;
    font-size: 1.05rem;
    color: #a8dadc;
    min-height: 60px;
    margin: 12px 0;
    line-height: 1.7;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.3);
  }

  /* RAG doc list */
  .doc-chip {
    display: inline-block;
    background: #1d3557;
    color: var(--light);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    margin: 3px;
    border: 1px solid #2d4a6e;
  }

  /* Divider */
  hr { border-color: #2d4a6e !important; margin: 18px 0 !important; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2035 0%, #162a42 100%);
  }
  section[data-testid="stSidebar"] * { color: #a8dadc !important; }
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color: #fff !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════
def init_session():
    defaults = {
        "chat_history"    : [],     # [{role, content, lang, timestamp}]
        "llm_history"     : [],     # [{role, content}] — for LLM context
        "last_transcript" : "",
        "last_response"   : "",
        "last_audio_path" : None,
        "tts_path"        : None,
        "is_recording"    : False,
        "recording_status": "",
        "rag_docs"        : [],     # list of uploaded doc names
        "use_rag"         : False,
        "whisper_model"   : DEFAULT_WHISPER_MODEL,
        "llm_model"       : None,
        "tts_backend"     : "auto",
        "use_tts"         : True,
        "stream_response" : False,
        "language_lock"   : None,   # None = auto, 'ne', 'en'
        "recording_duration": 7,
        "use_vad"         : True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def add_chat_message(role: str, content: str, lang: str = "en",
                     meta: str = "") -> None:
    """Append a message to the visual chat history."""
    st.session_state.chat_history.append({
        "role"      : role,
        "content"   : content,
        "lang"      : lang,
        "meta"      : meta,
        "timestamp" : time.strftime("%H:%M:%S"),
    })
    # Also update the LLM context window
    st.session_state.llm_history.append(
        build_history_entry(role, content)
    )
    st.session_state.llm_history = summarize_history(
        st.session_state.llm_history, max_turns=12
    )


def render_chat_bubble(msg: dict) -> None:
    """Render a single chat message as an HTML bubble."""
    role    = msg["role"]
    content = msg["content"].replace("\n", "<br>")
    ts      = msg.get("timestamp", "")
    meta    = msg.get("meta", "")
    lang    = msg.get("lang", "en")

    if role == "user":
        label = "🧑 You"
        css   = "chat-user"
    else:
        label = "🤖 AI"
        css   = "chat-assistant"

    lang_badge = "🇳🇵" if lang == "ne" else "🇬🇧"
    meta_str   = f" · {meta}" if meta else ""

    st.markdown(
        f'<div class="{css}">'
        f'<strong>{label}</strong> {lang_badge}'
        f'<div style="margin-top:6px">{content}</div>'
        f'<div class="chat-meta">{ts}{meta_str}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def play_audio_file(path: str, label: str = "▶ Play response") -> None:
    """Render an HTML5 audio player for a file."""
    if not path or not Path(path).exists():
        return
    ext  = Path(path).suffix.lstrip(".")
    mime = "audio/mpeg" if ext == "mp3" else "audio/wav"
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    st.markdown(
        f'<audio controls style="width:100%;margin-top:8px">'
        f'<source src="data:{mime};base64,{b64}" type="{mime}"></audio>',
        unsafe_allow_html=True,
    )


def export_chat_history() -> str:
    """Serialize chat history to a readable text format."""
    lines = ["Nepali Voice AI — Chat Export", "=" * 40, ""]
    for msg in st.session_state.chat_history:
        role = "You" if msg["role"] == "user" else "AI"
        lines.append(f"[{msg['timestamp']}] {role}: {msg['content']}")
        lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def run_voice_pipeline(duration: int, use_vad: bool) -> None:
    """
    Full pipeline: record → transcribe → LLM → TTS.
    Updates session state throughout.
    """
    # Step 1 — Record ──────────────────────────────────────────────────────────
    status_ph = st.empty()
    status_ph.info("🎙️ Recording… speak now!")
    st.session_state.is_recording = True

    try:
        audio_path = temp_audio_path(".wav")
        record_audio(
            duration=duration,
            output_path=audio_path,
            use_vad=use_vad,
            max_duration=30,
        )
        st.session_state.last_audio_path = audio_path
        st.session_state.is_recording    = False
    except Exception as exc:
        st.session_state.is_recording = False
        st.error(f"❌ Recording error: {exc}")
        logger.error("Recording failed: %s", exc)
        return

    # Step 2 — Transcribe ──────────────────────────────────────────────────────
    status_ph.info("🧠 Transcribing audio…")
    try:
        result = transcribe(
            audio_path,
            model_size=st.session_state.whisper_model,
            language=st.session_state.language_lock,
        )
        transcript = clean_transcript(result["text"])
        lang       = result["language"]
    except Exception as exc:
        st.error(f"❌ Transcription error: {exc}")
        logger.error("Transcription failed: %s", exc)
        status_ph.empty()
        return

    if not transcript:
        st.warning("⚠️ No speech detected. Please try again.")
        status_ph.empty()
        return

    st.session_state.last_transcript = transcript
    add_chat_message("user", transcript, lang=lang, meta=f"whisper/{lang}")

    # Step 3 — RAG context ─────────────────────────────────────────────────────
    rag_context = ""
    if st.session_state.use_rag and st.session_state.rag_docs:
        status_ph.info("📚 Searching documents…")
        try:
            vs = get_vector_store()
            rag_context = vs.build_context(transcript)
            if rag_context:
                logger.info("RAG context injected (%d chars)", len(rag_context))
        except Exception as exc:
            logger.warning("RAG search failed: %s", exc)

    # Step 4 — LLM ─────────────────────────────────────────────────────────────
    status_ph.info("⚙️ Thinking…")
    model = st.session_state.llm_model or get_best_model()

    # Remove last pair from history to avoid duplication (user msg already added)
    history_for_llm = st.session_state.llm_history[:-1]  # exclude just-added user msg

    try:
        response = query_llm(
            prompt    = transcript,
            model     = model,
            context   = rag_context or None,
            history   = history_for_llm,
            max_tokens= 1024,
        )
    except ConnectionError as exc:
        st.error(str(exc))
        status_ph.empty()
        return
    except Exception as exc:
        st.error(f"❌ LLM error: {exc}")
        logger.error("LLM query failed: %s", exc)
        status_ph.empty()
        return

    st.session_state.last_response = response
    resp_lang = detect_language_heuristic(response)
    add_chat_message(
        "assistant", response, lang=resp_lang,
        meta=f"{model}" + (" [RAG]" if rag_context else ""),
    )

    # Step 5 — TTS ─────────────────────────────────────────────────────────────
    if st.session_state.use_tts:
        status_ph.info("🔊 Generating speech…")
        try:
            tts_lang = get_tts_language(response, whisper_lang=lang)
            tts_path = synthesize(
                response,
                language=tts_lang,
                backend=st.session_state.tts_backend,
            )
            st.session_state.tts_path = tts_path
        except Exception as exc:
            logger.warning("TTS failed: %s", exc)
            st.session_state.tts_path = None

    status_ph.empty()


def run_text_pipeline(user_text: str) -> None:
    """Pipeline for typed text input (bypasses STT)."""
    if not user_text.strip():
        return

    lang = detect_language_heuristic(user_text)
    add_chat_message("user", user_text, lang=lang, meta="typed")

    rag_context = ""
    if st.session_state.use_rag and st.session_state.rag_docs:
        try:
            vs = get_vector_store()
            rag_context = vs.build_context(user_text)
        except Exception:
            pass

    model    = st.session_state.llm_model or get_best_model()
    history  = st.session_state.llm_history[:-1]

    with st.spinner("⚙️ Thinking…"):
        try:
            response = query_llm(
                user_text, model=model,
                context=rag_context or None,
                history=history,
            )
        except Exception as exc:
            st.error(str(exc))
            return

    st.session_state.last_response = response
    resp_lang = detect_language_heuristic(response)
    add_chat_message(
        "assistant", response, lang=resp_lang,
        meta=model + (" [RAG]" if rag_context else ""),
    )

    if st.session_state.use_tts:
        try:
            tts_path = synthesize(
                response, language=get_tts_language(response, lang),
                backend=st.session_state.tts_backend,
            )
            st.session_state.tts_path = tts_path
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    # ── Model settings ─────────────────────────────────────────────────────────
    st.markdown("### 🧠 LLM")
    ollama_ok = is_ollama_running()
    if ollama_ok:
        st.markdown('<span class="badge-online">● Ollama Online</span>',
                    unsafe_allow_html=True)
        models = list_available_models()
        if models:
            st.session_state.llm_model = st.selectbox(
                "Model", models,
                index=models.index(get_best_model()) if get_best_model() in models else 0,
            )
        else:
            st.warning("No models found. Run:\n`ollama pull mistral:7b-instruct`")
    else:
        st.markdown('<span class="badge-offline">● Ollama Offline</span>',
                    unsafe_allow_html=True)
        st.info("Start Ollama:\n```\nollama serve\n```")

    st.divider()

    # ── Whisper settings ───────────────────────────────────────────────────────
    st.markdown("### 🎙️ Speech Recognition")
    st.session_state.whisper_model = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small", "medium"],
        index=["tiny", "base", "small", "medium"].index(
            st.session_state.whisper_model
        ),
        help="small = best Nepali accuracy | tiny = fastest",
    )
    st.session_state.language_lock = st.selectbox(
        "Language override",
        [None, "ne", "en"],
        format_func=lambda x: "Auto-detect" if x is None
                    else ("Nepali 🇳🇵" if x == "ne" else "English 🇬🇧"),
    )

    st.divider()

    # ── Recording settings ─────────────────────────────────────────────────────
    st.markdown("### 🔴 Recording")
    st.session_state.use_vad = st.toggle(
        "Smart stop (VAD)", value=st.session_state.use_vad,
        help="Auto-stop when silence detected",
    )
    if not st.session_state.use_vad:
        st.session_state.recording_duration = st.slider(
            "Duration (s)", 3, 30, st.session_state.recording_duration
        )

    st.divider()

    # ── TTS settings ───────────────────────────────────────────────────────────
    st.markdown("### 🔊 Text-to-Speech")
    st.session_state.use_tts = st.toggle(
        "Enable TTS", value=st.session_state.use_tts
    )
    if st.session_state.use_tts:
        backends = available_backends() or ["gtts"]
        st.session_state.tts_backend = st.selectbox(
            "TTS backend",
            ["auto"] + backends,
        )

    st.divider()

    # ── RAG settings ───────────────────────────────────────────────────────────
    st.markdown("### 📚 RAG Documents")
    st.session_state.use_rag = st.toggle(
        "Use uploaded docs", value=st.session_state.use_rag
    )

    vs = get_vector_store()
    if vs.chunk_count > 0:
        st.markdown(
            f"**{vs.document_count}** doc(s) · **{vs.chunk_count}** chunks indexed"
        )
        if st.button("🗑️ Clear index"):
            vs.clear()
            st.session_state.rag_docs = []
            st.success("Index cleared.")

    st.divider()

    # ── Diagnostics ────────────────────────────────────────────────────────────
    with st.expander("🔍 System Info"):
        info = get_system_info()
        st.json(info)

    # ── Export ─────────────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        export_text = export_chat_history()
        st.download_button(
            "📥 Export chat",
            data=export_text,
            file_name="chat_export.txt",
            mime="text/plain",
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
# Header
st.markdown("""
<div class="hero-banner">
  <span class="flag">🇳🇵</span>
  <h1>Nepali Voice AI Assistant</h1>
  <p>नेपाली र अंग्रेजी भाषामा कुराकानी गर्नुहोस् · Speak in Nepali or English · 100% Offline</p>
</div>
""", unsafe_allow_html=True)

# Warn if Ollama is offline
if not is_ollama_running():
    st.error(
        "⚠️ **Ollama is not running.** The AI cannot respond without it.\n\n"
        "```bash\nollama serve\nollama pull mistral:7b-instruct\n```"
    )

# ── Three-column main area ─────────────────────────────────────────────────────
col_left, col_mid, col_right = st.columns([2.5, 4, 2.5])

# ── LEFT: Input controls ───────────────────────────────────────────────────────
with col_left:
    st.markdown("### 🎤 Voice Input")

    # Record button
    rec_label = (
        "⏹ Recording…" if st.session_state.is_recording else "🎙️ Start Recording"
    )
    if st.button(rec_label, type="primary", use_container_width=True,
                 disabled=st.session_state.is_recording):
        run_voice_pipeline(
            st.session_state.recording_duration,
            st.session_state.use_vad,
        )
        st.rerun()

    st.divider()
    st.markdown("### ⌨️ Text Input")
    typed_input = st.text_area(
        "Type your message (Nepali or English):",
        height=100,
        placeholder="आफ्नो प्रश्न यहाँ लेख्नुहोस् …\nOr type in English …",
        label_visibility="collapsed",
    )
    if st.button("Send ➤", use_container_width=True):
        run_text_pipeline(typed_input)
        st.rerun()

    st.divider()
    st.markdown("### 📂 Upload Document")
    uploaded = st.file_uploader(
        "Upload .txt or .pdf for RAG",
        type=["txt", "pdf"],
        label_visibility="collapsed",
    )
    if uploaded and uploaded.name not in st.session_state.rag_docs:
        with st.spinner(f"Indexing {uploaded.name}…"):
            # Save to temp file and index
            suffix = Path(uploaded.name).suffix
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(uploaded.read())
            tmp.close()
            try:
                vs  = get_vector_store()
                cnt = vs.add_document(tmp.name, source_label=uploaded.name)
                st.session_state.rag_docs.append(uploaded.name)
                st.session_state.use_rag = True
                st.success(f"✅ Indexed **{cnt}** chunks from *{uploaded.name}*")
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    # Show indexed docs
    if st.session_state.rag_docs:
        st.markdown("**Indexed documents:**")
        for doc in st.session_state.rag_docs:
            st.markdown(f'<span class="doc-chip">📄 {doc}</span>',
                        unsafe_allow_html=True)

# ── MIDDLE: Chat ───────────────────────────────────────────────────────────────
with col_mid:
    st.markdown("### 💬 Conversation")

    # Transcription display
    if st.session_state.last_transcript:
        st.markdown("**Last transcription:**")
        st.markdown(
            f'<div class="transcript-box">{st.session_state.last_transcript}</div>',
            unsafe_allow_html=True,
        )

    # Chat history
    chat_container = st.container(height=480)
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(
                '<div style="text-align:center;opacity:0.4;padding:60px 0">'
                '<div style="font-size:3rem">🎙️</div>'
                '<p>Press "Start Recording" to begin …</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.chat_history:
                render_chat_bubble(msg)

    # TTS playback
    if st.session_state.tts_path and st.session_state.use_tts:
        st.markdown("**🔊 AI Voice Response:**")
        play_audio_file(st.session_state.tts_path)

    # Clear history button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.chat_history  = []
            st.session_state.llm_history   = []
            st.session_state.last_transcript = ""
            st.session_state.last_response   = ""
            st.session_state.tts_path        = None
            st.rerun()

# ── RIGHT: Response & info ─────────────────────────────────────────────────────
with col_right:
    st.markdown("### 📊 Live Stats")

    if st.session_state.last_response:
        words     = len(st.session_state.last_response.split())
        chars     = len(st.session_state.last_response)
        turns     = len(st.session_state.chat_history) // 2

        st.metric("Turns", turns)
        st.metric("Response words", words)
        st.metric("Response chars", chars)

        resp_lang = detect_language_heuristic(st.session_state.last_response)
        lang_icon = "🇳🇵 Nepali" if resp_lang == "ne" else "🇬🇧 English"
        st.metric("Language", lang_icon)
    else:
        st.info("Stats appear after first response.")

    st.divider()
    st.markdown("### 🚦 System Status")

    # Ollama status
    if is_ollama_running():
        st.markdown('<span class="badge-online">● Ollama</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-offline">● Ollama</span>', unsafe_allow_html=True)

    # TTS status
    tts_backends = available_backends()
    if tts_backends:
        st.markdown(f'<span class="badge-online">● TTS ({tts_backends[0]})</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warn">● TTS (none)</span>', unsafe_allow_html=True)

    # RAG status
    vs = get_vector_store()
    if vs.chunk_count > 0 and st.session_state.use_rag:
        st.markdown(
            f'<span class="badge-online">● RAG ({vs.chunk_count} chunks)</span>',
            unsafe_allow_html=True,
        )
    elif vs.chunk_count > 0:
        st.markdown(
            f'<span class="badge-warn">● RAG (disabled)</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="badge-offline">● RAG (no docs)</span>',
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("### 💡 Quick Help")
    st.markdown("""
**Voice Tips:**
- Speak clearly, ~20 cm from mic
- Pause 1–2s before stopping
- Nepali works best with `small` model

**Keyboard shortcuts:**
- `Enter` in text box → Send

**RAG Tips:**
- Upload `.pdf` or `.txt` first
- Enable "Use uploaded docs" toggle
- Then ask questions about the doc
""")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="text-align:center;opacity:0.5;font-size:0.82rem;padding:8px 0">
    Nepali Voice AI · Built with faster-whisper + Ollama + FAISS · 100% Offline<br>
    Made with ❤️ · Logic by Pranish Pr Sapkota · Code by Claude
</div>
""", unsafe_allow_html=True)
