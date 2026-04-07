"""
rag_engine.py
-------------
Retrieval-Augmented Generation (RAG) pipeline.

Flow:
  1. User uploads .txt or .pdf document.
  2. Text is extracted and split into chunks.
  3. Chunks are embedded using sentence-transformers (offline).
  4. Embeddings are stored in an in-memory FAISS index.
  5. At query time, top-K similar chunks are retrieved.
  6. Retrieved context is injected into the LLM prompt.

Runs completely offline (no OpenAI Embeddings API).
"""

import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
EMBED_MODEL     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# ↑ Multilingual model that handles Nepali + English
CHUNK_SIZE      = 500        # characters per chunk
CHUNK_OVERLAP   = 100        # characters of overlap between chunks
TOP_K           = 4          # number of chunks to retrieve
INDEX_SAVE_DIR  = Path(__file__).parent.parent / "data" / "faiss_index"


# ── Embedding model (singleton) ────────────────────────────────────────────────
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", EMBED_MODEL)
        _embedder = SentenceTransformer(EMBED_MODEL)
        logger.info("Embedding model loaded ✓")
    return _embedder


# ── Text extraction ────────────────────────────────────────────────────────────
def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF using pdfplumber (best for tables/columns)."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except ImportError:
        pass

    # Fallback: pypdf2
    try:
        import pypdf
        text_parts = []
        reader = pypdf.PdfReader(file_path)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        return "\n\n".join(text_parts)
    except ImportError:
        raise ImportError(
            "PDF extraction requires pdfplumber or pypdf.\n"
            "  pip install pdfplumber"
        )


def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from .txt, .pdf, or .md files.
    Returns the full document text as a string.
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext in (".txt", ".md", ".rst"):
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .txt or .pdf")

    logger.info("Extracted %d chars from %s", len(text), file_path)
    return text


# ── Chunking ───────────────────────────────────────────────────────────────────
def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping fixed-size chunks.
    Tries to respect sentence/paragraph boundaries.
    """
    import re

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Split into sentences (handles both English period and Nepali daṇḍa ।)
    sentences = re.split(r"(?<=[।!?.\n]) +", text)

    chunks = []
    current = ""
    buffer = ""   # overlap buffer from previous chunk

    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current += (" " if current else "") + sent
        else:
            if current:
                chunks.append(current.strip())
                # Overlap: keep last `overlap` chars as start of next chunk
                buffer = current[-overlap:] if len(current) > overlap else current
            current = buffer + " " + sent if buffer else sent
            buffer = ""

    if current.strip():
        chunks.append(current.strip())

    logger.info("Chunked document: %d chunks (size≤%d, overlap=%d)",
                len(chunks), chunk_size, overlap)
    return [c for c in chunks if len(c) > 30]   # filter trivial chunks


# ── FAISS index ────────────────────────────────────────────────────────────────
class VectorStore:
    """
    In-memory FAISS index backed by sentence-transformer embeddings.
    Persists to disk for session reuse.
    """

    def __init__(self):
        self.index  = None
        self.chunks : list[str] = []
        self.meta   : list[dict] = []    # {source, chunk_id}
        self._dim   : int = 0

    # ── Building ────────────────────────────────────────────────────────────────
    def add_document(self, file_path: str, source_label: Optional[str] = None) -> int:
        """
        Extract, chunk, embed, and index a document.
        Returns the number of chunks added.
        """
        import faiss

        text   = extract_text_from_file(file_path)
        chunks = chunk_text(text)
        label  = source_label or Path(file_path).name

        embedder   = _get_embedder()
        embeddings = embedder.encode(chunks, show_progress_bar=False,
                                      convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(embeddings)   # cosine similarity via L2 on normalized vecs

        if self.index is None:
            self._dim  = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self._dim)   # Inner Product = cosine

        self.index.add(embeddings)
        start_id = len(self.chunks)
        self.chunks.extend(chunks)
        self.meta.extend(
            [{"source": label, "chunk_id": start_id + i} for i in range(len(chunks))]
        )

        logger.info("Indexed %d chunks from '%s' (total: %d)",
                    len(chunks), label, len(self.chunks))
        return len(chunks)

    # ── Retrieval ───────────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Retrieve the top-K most relevant chunks for a query.

        Returns list of dicts: {text, source, chunk_id, score}
        """
        if self.index is None or len(self.chunks) == 0:
            return []

        import faiss

        embedder = _get_embedder()
        q_vec    = embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text"     : self.chunks[idx],
                "source"   : self.meta[idx]["source"],
                "chunk_id" : int(idx),
                "score"    : round(float(score), 4),
            })

        logger.info("RAG search: '%s' → %d results", query[:60], len(results))
        return results

    def build_context(self, query: str, top_k: int = TOP_K,
                      min_score: float = 0.2) -> str:
        """
        Build a context string from retrieved chunks, ready for LLM injection.
        Returns empty string if no relevant documents found.
        """
        results = self.search(query, top_k=top_k)
        filtered = [r for r in results if r["score"] >= min_score]

        if not filtered:
            return ""

        parts = []
        for i, r in enumerate(filtered, 1):
            parts.append(f"[Source: {r['source']}]\n{r['text']}")

        return "\n\n".join(parts)

    # ── Persistence ─────────────────────────────────────────────────────────────
    def save(self, path: Optional[str] = None) -> str:
        import faiss, pickle

        save_dir = Path(path or INDEX_SAVE_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(save_dir / "index.faiss"))
        with open(save_dir / "metadata.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "meta": self.meta, "dim": self._dim}, f)

        logger.info("FAISS index saved → %s", save_dir)
        return str(save_dir)

    def load(self, path: Optional[str] = None) -> bool:
        import faiss, pickle

        save_dir = Path(path or INDEX_SAVE_DIR)
        idx_file = save_dir / "index.faiss"
        meta_file = save_dir / "metadata.pkl"

        if not idx_file.exists() or not meta_file.exists():
            logger.info("No saved FAISS index found at %s", save_dir)
            return False

        self.index = faiss.read_index(str(idx_file))
        with open(meta_file, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.meta   = data["meta"]
        self._dim   = data["dim"]

        logger.info("FAISS index loaded: %d chunks", len(self.chunks))
        return True

    def clear(self):
        self.index  = None
        self.chunks = []
        self.meta   = []
        self._dim   = 0
        logger.info("Vector store cleared.")

    @property
    def document_count(self) -> int:
        """Number of unique source documents indexed."""
        return len(set(m["source"] for m in self.meta))

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


# ── Module-level singleton ────────────────────────────────────────────────────
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return the global VectorStore singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
        _vector_store.load()   # Try to restore previous session
    return _vector_store
