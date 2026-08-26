"""
ChromaDB vector store wrapper.
3 collections: community_reviews | movie_dna | my_lens
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from src.config import (
    EMBEDDINGS_DIR,
    CHROMA_COMMUNITY_COLLECTION,
    CHROMA_MOVIE_DNA_COLLECTION,
    CHROMA_MY_LENS_COLLECTION,
)


def _get_client() -> chromadb.PersistentClient:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(EMBEDDINGS_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(name: str):
    client = _get_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_embedding(collection_name: str, doc_id: str, embedding: List[float],
                     metadata: Dict, text: str):
    """Add or update a single document in a collection."""
    if not embedding:
        return
    col = get_collection(collection_name)
    col.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[metadata],
        documents=[text[:2000]],   # ChromaDB document preview
    )


def doc_exists(collection_name: str, doc_id: str) -> bool:
    """Check if a document already has an embedding stored."""
    try:
        col = get_collection(collection_name)
        result = col.get(ids=[doc_id], include=[])
        return len(result["ids"]) > 0
    except Exception:
        return False


def query_similar(collection_name: str, query_embedding: List[float],
                  n_results: int = 20, where: Optional[Dict] = None) -> Dict:
    """
    Find the n most similar documents.
    Returns a dict with keys: ids, distances, metadatas, documents
    """
    col = get_collection(collection_name)
    count = col.count()
    if count == 0:
        return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

    n_results = min(n_results, count)
    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["distances", "metadatas", "documents"],
    )
    if where:
        kwargs["where"] = where
    return col.query(**kwargs)


def get_all_embeddings(collection_name: str) -> Dict:
    """Return all stored embeddings + metadata (for clustering)."""
    col = get_collection(collection_name)
    if col.count() == 0:
        return {"ids": [], "embeddings": [], "metadatas": [], "documents": []}
    return col.get(include=["embeddings", "metadatas", "documents"])


def collection_count(collection_name: str) -> int:
    try:
        return get_collection(collection_name).count()
    except Exception:
        return 0
