"""
Embedder: wraps Ollama to generate embeddings locally.
Handles batching, retries, and text truncation.
"""
import ollama
import numpy as np
import time
from typing import List
from src.config import OLLAMA_EMBED_MODEL, OLLAMA_HOST, MAX_TEXT_CHARS

# Reuse a single client — avoids Metal GPU re-init crash on every call
_client: ollama.Client = None

def _get_client() -> ollama.Client:
    global _client
    if _client is None:
        _client = ollama.Client(host=OLLAMA_HOST)
    return _client


def _truncate(text: str) -> str:
    """Keep text within token budget (rough char limit)."""
    return text[:MAX_TEXT_CHARS] if len(text) > MAX_TEXT_CHARS else text


def embed_text(text: str, retries: int = 4) -> List[float]:
    """
    Embed a single string using Ollama.
    Retries with exponential backoff to handle Metal GPU init flakiness.
    """
    text = _truncate(str(text).strip() or "empty")
    client = _get_client()
    for attempt in range(retries):
        try:
            try:
                response = client.embed(model=OLLAMA_EMBED_MODEL, input=text)
                emb = response.get("embeddings", [response.get("embedding", [])])
                return emb[0] if emb else []
            except AttributeError:
                response = client.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
                return response["embedding"]
        except Exception as e:
            wait = 2 ** attempt   # 1s, 2s, 4s, 8s
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                print(f"[embedder] Failed after {retries} attempts: {e}")
    return []


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts. Reuses the shared client."""
    return [embed_text(t) for t in texts]


def check_ollama_ready() -> tuple[bool, str]:
    """Check if Ollama is running and the model is available."""
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.list()
        # Handle both dict-style and object-style SDK responses
        raw_models = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
        model_names = []
        for m in raw_models:
            name = m["name"] if isinstance(m, dict) else getattr(m, "model", getattr(m, "name", ""))
            model_names.append(name.split(":")[0])
        target = OLLAMA_EMBED_MODEL.split(":")[0]
        if target not in model_names:
            return False, (
                f"Model '{OLLAMA_EMBED_MODEL}' not found. "
                f"Run: ollama pull {OLLAMA_EMBED_MODEL}"
            )
        return True, f"Ollama ready with {OLLAMA_EMBED_MODEL}"
    except Exception as e:
        return False, f"Ollama not running: {e}\nStart it with: ollama serve"
