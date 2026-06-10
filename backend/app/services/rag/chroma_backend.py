"""
ChromaDB backend — persistent vector search with built-in embedding functions.
Requirements: chromadb
pip install chromadb
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.rag.base import RAGBackend


class ChromaRAGBackend(RAGBackend):
    def __init__(
        self,
        data_dir: Path,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        host: str | None = None,
        port: int = 8001,
    ) -> None:
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError as e:
            raise ImportError("ChromaDB backend requires: pip install chromadb") from e

        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        if host:
            self._client = chromadb.HttpClient(host=host, port=port)
        else:
            data_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(data_dir))

        self._collections: dict[str, Any] = {}

    def _col(self, collection: str) -> Any:
        if collection not in self._collections:
            self._collections[collection] = self._client.get_or_create_collection(
                name=collection, embedding_function=self._ef
            )
        return self._collections[collection]

    def add(self, collection: str, doc_id: str, text: str, metadata: dict[str, Any]) -> None:
        col = self._col(collection)
        # Chroma upserts by ID
        col.upsert(ids=[doc_id], documents=[text], metadatas=[metadata])

    def search(
        self,
        collection: str,
        query: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        col = self._col(collection)
        n = col.count()
        if n == 0:
            return []
        kwargs: dict[str, Any] = {"query_texts": [query], "n_results": min(k, n)}
        if filter_metadata:
            # Chroma where clause: {"key": {"$eq": value}}
            where = {k: {"$eq": v} for k, v in filter_metadata.items()}
            if len(where) == 1:
                kwargs["where"] = where
            else:
                kwargs["where"] = {"$and": [{k: v} for k, v in where.items()]}
        result = col.query(**kwargs)

        output = []
        for doc_id, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            output.append({
                "doc_id":   doc_id,
                "text":     text,
                "metadata": meta,
                "score":    1.0 - distance,  # convert distance → similarity
            })
        return output

    def delete(self, collection: str, doc_id: str) -> None:
        self._col(collection).delete(ids=[doc_id])

    def list_docs(
        self,
        collection: str,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        col = self._col(collection)
        n = col.count()
        if n == 0:
            return []
        kwargs: dict[str, Any] = {"include": ["metadatas"]}
        if filter_metadata:
            kwargs["where"] = {k: {"$eq": v} for k, v in filter_metadata.items()}
        result = col.get(**kwargs)
        return [
            {"doc_id": did, "metadata": meta}
            for did, meta in zip(result["ids"], result["metadatas"])
        ]

    def count(self, collection: str) -> int:
        return self._col(collection).count()
