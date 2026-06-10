"""
FAISS backend — dense vector search.
Requirements: faiss-cpu (or faiss-gpu), sentence-transformers
pip install faiss-cpu sentence-transformers
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from app.services.rag.base import RAGBackend


class FaissRAGBackend(RAGBackend):
    def __init__(self, data_dir: Path, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            import faiss  # noqa: F401
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "FAISS backend requires: pip install faiss-cpu sentence-transformers"
            ) from e

        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._model_name = embedding_model
        self._encoder = SentenceTransformer(embedding_model)
        self._dim = self._encoder.get_sentence_embedding_dimension()
        self._indexes: dict[str, Any] = {}         # collection → faiss.IndexFlatIP
        self._meta: dict[str, dict[str, Any]] = {} # collection → {doc_id → record}
        self._id_map: dict[str, dict[str, int]] = {} # collection → {doc_id → faiss_index}
        self._rev_map: dict[str, dict[int, str]] = {} # collection → {faiss_index → doc_id}

    def _index_path(self, collection: str) -> Path:
        return self._data_dir / f"{collection}.faiss"

    def _meta_path(self, collection: str) -> Path:
        return self._data_dir / f"{collection}.meta.json"

    def _load(self, collection: str) -> None:
        import faiss
        if collection in self._indexes:
            return
        index_path = self._index_path(collection)
        meta_path  = self._meta_path(collection)
        if index_path.exists() and meta_path.exists():
            self._indexes[collection] = faiss.read_index(str(index_path))
            saved = json.loads(meta_path.read_text(encoding="utf-8"))
            self._meta[collection]    = saved.get("meta", {})
            self._id_map[collection]  = {k: int(v) for k, v in saved.get("id_map", {}).items()}
            self._rev_map[collection] = {int(k): v for k, v in saved.get("rev_map", {}).items()}
        else:
            self._indexes[collection] = faiss.IndexIDMap(faiss.IndexFlatIP(self._dim))
            self._meta[collection]    = {}
            self._id_map[collection]  = {}
            self._rev_map[collection] = {}

    def _save(self, collection: str) -> None:
        import faiss
        faiss.write_index(self._indexes[collection], str(self._index_path(collection)))
        payload = {
            "meta":    self._meta[collection],
            "id_map":  self._id_map[collection],
            "rev_map": {str(k): v for k, v in self._rev_map[collection].items()},
        }
        self._meta_path(collection).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _embed(self, text: str) -> np.ndarray:
        vec = self._encoder.encode([text], normalize_embeddings=True).astype("float32")
        return vec  # shape (1, dim)

    def add(self, collection: str, doc_id: str, text: str, metadata: dict[str, Any]) -> None:
        self._load(collection)
        # Delete old if exists
        if doc_id in self._id_map[collection]:
            self.delete(collection, doc_id)
        vec = self._embed(text)
        # Assign a unique int64 ID
        int_id = hash(doc_id) & 0x7FFFFFFFFFFFFFFF
        self._indexes[collection].add_with_ids(vec, np.array([int_id], dtype="int64"))
        self._id_map[collection][doc_id]   = int_id
        self._rev_map[collection][int_id]  = doc_id
        self._meta[collection][doc_id]     = {"doc_id": doc_id, "text": text, "metadata": metadata}
        self._save(collection)

    def search(
        self,
        collection: str,
        query: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self._load(collection)
        if self._indexes[collection].ntotal == 0:
            return []
        vec = self._embed(query)
        fetch_k = min(k * 4, self._indexes[collection].ntotal)
        scores, int_ids = self._indexes[collection].search(vec, fetch_k)

        results = []
        for score, int_id in zip(scores[0], int_ids[0]):
            if int_id < 0:
                continue
            doc_id = self._rev_map[collection].get(int(int_id))
            if doc_id is None:
                continue
            rec = self._meta[collection].get(doc_id, {})
            if filter_metadata:
                meta = rec.get("metadata", {})
                if not all(meta.get(fk) == fv for fk, fv in filter_metadata.items()):
                    continue
            results.append({
                "doc_id":   doc_id,
                "text":     rec.get("text", ""),
                "metadata": rec.get("metadata", {}),
                "score":    float(score),
            })
            if len(results) >= k:
                break
        return results

    def delete(self, collection: str, doc_id: str) -> None:
        self._load(collection)
        int_id = self._id_map[collection].pop(doc_id, None)
        if int_id is None:
            return
        ids_to_remove = np.array([int_id], dtype="int64")
        self._indexes[collection].remove_ids(ids_to_remove)
        self._rev_map[collection].pop(int_id, None)
        self._meta[collection].pop(doc_id, None)
        self._save(collection)

    def list_docs(
        self,
        collection: str,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self._load(collection)
        results = []
        for doc_id, rec in self._meta[collection].items():
            if filter_metadata:
                meta = rec.get("metadata", {})
                if not all(meta.get(k) == v for k, v in filter_metadata.items()):
                    continue
            results.append({"doc_id": doc_id, "metadata": rec.get("metadata", {})})
        return results

    def count(self, collection: str) -> int:
        self._load(collection)
        return self._indexes[collection].ntotal
