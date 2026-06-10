"""Abstract interface for pluggable RAG backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RAGBackend(ABC):
    """
    Two collections used by this system:
      "feedback"   — pathologist correction history
      "literature" — scientific paper chunks
    """

    @abstractmethod
    def add(
        self,
        collection: str,
        doc_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        """Upsert a document."""

    @abstractmethod
    def search(
        self,
        collection: str,
        query: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return top-k documents sorted by relevance. Each dict has keys:
        doc_id, text, metadata, score."""

    @abstractmethod
    def delete(self, collection: str, doc_id: str) -> None:
        """Remove a document by id."""

    @abstractmethod
    def list_docs(
        self,
        collection: str,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return all document metadata (no text body)."""

    @abstractmethod
    def count(self, collection: str) -> int:
        """Number of documents in collection."""
