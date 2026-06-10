"""
JSON-file backend — zero extra dependencies.
Simple BM25-style TF-IDF keyword search.
Good for <5 000 documents.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.services.rag.base import RAGBackend


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class JsonRAGBackend(RAGBackend):
    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._docs: dict[str, dict[str, dict]] = {}   # collection → {doc_id → record}
        self._index: dict[str, dict[str, dict[str, int]]] = {}  # collection → {term → {doc_id → freq}}

    def _col_path(self, collection: str) -> Path:
        return self._dir / f"{collection}.jsonl"

    def _load(self, collection: str) -> None:
        if collection in self._docs:
            return
        self._docs[collection] = {}
        self._index[collection] = defaultdict(dict)
        path = self._col_path(collection)
        if not path.exists():
            return
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                did = rec["doc_id"]
                self._docs[collection][did] = rec
                for term, freq in Counter(_tokenize(rec["text"])).items():
                    self._index[collection][term][did] = freq

    def _save(self, collection: str) -> None:
        with self._col_path(collection).open("w", encoding="utf-8") as f:
            for rec in self._docs[collection].values():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def add(self, collection: str, doc_id: str, text: str, metadata: dict[str, Any]) -> None:
        self._load(collection)
        if doc_id in self._docs[collection]:
            # Remove old index entries
            old = self._docs[collection][doc_id]
            for term in _tokenize(old["text"]):
                self._index[collection][term].pop(doc_id, None)
        record = {"doc_id": doc_id, "text": text, "metadata": metadata}
        self._docs[collection][doc_id] = record
        for term, freq in Counter(_tokenize(text)).items():
            self._index[collection].setdefault(term, {})[doc_id] = freq
        self._save(collection)

    def search(
        self,
        collection: str,
        query: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self._load(collection)
        docs = self._docs[collection]
        if not docs:
            return []

        query_terms = set(_tokenize(query))
        scores: dict[str, float] = defaultdict(float)
        N = len(docs)

        for term in query_terms:
            postings = self._index[collection].get(term, {})
            if not postings:
                continue
            idf = math.log((N + 1) / (len(postings) + 0.5))
            for did, freq in postings.items():
                doc_len = sum(self._index[collection].get(t, {}).get(did, 0)
                              for t in _tokenize(docs[did]["text"]))
                tf = freq / max(1, doc_len) * 2.0
                scores[did] += idf * tf

        candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for did, score in candidates:
            rec = docs[did]
            if filter_metadata:
                meta = rec.get("metadata", {})
                if not all(meta.get(k) == v for k, v in filter_metadata.items()):
                    continue
            results.append({
                "doc_id": did,
                "text": rec["text"],
                "metadata": rec.get("metadata", {}),
                "score": round(score, 4),
            })
            if len(results) >= k:
                break
        return results

    def delete(self, collection: str, doc_id: str) -> None:
        self._load(collection)
        if doc_id not in self._docs[collection]:
            return
        old = self._docs[collection].pop(doc_id)
        for term in _tokenize(old["text"]):
            self._index[collection].get(term, {}).pop(doc_id, None)
        self._save(collection)

    def list_docs(
        self,
        collection: str,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self._load(collection)
        results = []
        for rec in self._docs[collection].values():
            if filter_metadata:
                meta = rec.get("metadata", {})
                if not all(meta.get(k) == v for k, v in filter_metadata.items()):
                    continue
            results.append({"doc_id": rec["doc_id"], "metadata": rec.get("metadata", {})})
        return results

    def count(self, collection: str) -> int:
        self._load(collection)
        return len(self._docs.get(collection, {}))
