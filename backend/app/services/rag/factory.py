"""
Singleton RAG backend factory.
Backend is chosen by RAG_BACKEND env var: json (default) | faiss | chroma
"""
from __future__ import annotations

from functools import lru_cache

from app.services.rag.base import RAGBackend
from app.settings import settings


@lru_cache(maxsize=1)
def get_backend() -> RAGBackend:
    backend_type = settings.rag_backend.lower()
    data_dir     = settings.rag_data_dir

    if backend_type == "faiss":
        from app.services.rag.faiss_backend import FaissRAGBackend
        return FaissRAGBackend(data_dir / "faiss", embedding_model=settings.rag_embedding_model)

    if backend_type == "chroma":
        from app.services.rag.chroma_backend import ChromaRAGBackend
        host = settings.chroma_host if settings.chroma_host else None
        return ChromaRAGBackend(
            data_dir / "chroma",
            embedding_model=settings.rag_embedding_model,
            host=host,
            port=settings.chroma_port,
        )

    # Default: JSON keyword search
    from app.services.rag.json_backend import JsonRAGBackend
    return JsonRAGBackend(data_dir / "json")
