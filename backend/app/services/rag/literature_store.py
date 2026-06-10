"""
Scientific literature storage and retrieval.
Supports: PDF upload, plain text, DOI/title metadata.
Chunks text for fine-grained retrieval.
"""
from __future__ import annotations

import re
import uuid
from typing import Any

from app.services.rag.factory import get_backend

COLLECTION = "literature"
CHUNK_SIZE   = 600    # characters per chunk
CHUNK_OVERLAP = 80


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def _parse_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF. Tries PyMuPDF, falls back to pdfminer."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "\n\n".join(page.get_text() for page in doc)
    except ImportError:
        pass
    try:
        import io
        from pdfminer.high_level import extract_text
        return extract_text(io.BytesIO(pdf_bytes))
    except ImportError:
        pass
    return "[PDF parsing failed — install PyMuPDF: pip install pymupdf]"


class LiteratureStore:
    """Add / search / delete scientific papers in the RAG backend."""

    def _backend(self):
        return get_backend()

    def add_text(
        self,
        text: str,
        title: str,
        *,
        authors: str = "",
        year: str = "",
        doi: str = "",
        paper_id: str | None = None,
    ) -> dict[str, Any]:
        """Add a paper from raw text. Returns paper summary."""
        pid = paper_id or str(uuid.uuid4())
        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            meta: dict[str, Any] = {
                "type":       "literature",
                "paper_id":   pid,
                "title":      title,
                "authors":    authors,
                "year":       str(year),
                "doi":        doi,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            self._backend().add(
                COLLECTION,
                doc_id=f"{pid}_chunk{i:04d}",
                text=chunk,
                metadata=meta,
            )
        return {
            "paper_id":   pid,
            "title":      title,
            "authors":    authors,
            "year":       year,
            "doi":        doi,
            "num_chunks": len(chunks),
        }

    def add_pdf(
        self,
        pdf_bytes: bytes,
        filename: str,
        *,
        title: str = "",
        authors: str = "",
        year: str = "",
        doi: str = "",
        paper_id: str | None = None,
    ) -> dict[str, Any]:
        """Parse PDF and add to store."""
        text = _parse_pdf(pdf_bytes)
        return self.add_text(
            text,
            title=title or filename,
            authors=authors,
            year=year,
            doi=doi,
            paper_id=paper_id,
        )

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Retrieve top-k relevant chunks."""
        return self._backend().search(COLLECTION, query, k=k)

    def search_by_paper_ids(self, paper_ids: list[str], max_chunks: int = 10) -> list[dict[str, Any]]:
        """Retrieve chunks for specific paper IDs (for explicit citation)."""
        results = []
        for pid in paper_ids:
            chunks = self._backend().list_docs(
                COLLECTION, filter_metadata={"paper_id": pid}
            )
            results.extend(chunks[:max_chunks])
        return results

    def get_full_paper(self, paper_id: str) -> str:
        """Reassemble all chunks of a paper (ordered by chunk_index)."""
        chunks = self._backend().list_docs(
            COLLECTION, filter_metadata={"paper_id": paper_id}
        )
        # Need text too — search with a broad query for this paper
        all_chunks = self._backend().search(
            COLLECTION, paper_id, k=500, filter_metadata={"paper_id": paper_id}
        )
        sorted_chunks = sorted(all_chunks, key=lambda x: x.get("metadata", {}).get("chunk_index", 0))
        return "\n\n".join(c["text"] for c in sorted_chunks)

    def list_papers(self) -> list[dict[str, Any]]:
        """List unique papers (deduplicated by paper_id)."""
        all_docs = self._backend().list_docs(COLLECTION)
        seen: dict[str, dict[str, Any]] = {}
        for doc in all_docs:
            meta = doc.get("metadata", {})
            pid = meta.get("paper_id", doc["doc_id"])
            if pid not in seen:
                seen[pid] = {
                    "paper_id": pid,
                    "title":    meta.get("title", ""),
                    "authors":  meta.get("authors", ""),
                    "year":     meta.get("year", ""),
                    "doi":      meta.get("doi", ""),
                    "num_chunks": int(meta.get("total_chunks", 1)),
                }
        return sorted(seen.values(), key=lambda x: x.get("title", ""))

    def delete_paper(self, paper_id: str) -> int:
        """Delete all chunks of a paper. Returns number of chunks deleted."""
        chunks = self._backend().list_docs(
            COLLECTION, filter_metadata={"paper_id": paper_id}
        )
        for chunk in chunks:
            self._backend().delete(COLLECTION, chunk["doc_id"])
        return len(chunks)

    def build_rag_context(self, query: str, k: int = 5) -> str:
        """Format retrieved literature chunks as context for the agent."""
        results = self.search(query, k=k)
        if not results:
            return ""
        lines = ["[관련 문헌 참조]"]
        for r in results:
            meta = r.get("metadata", {})
            title   = meta.get("title", "Unknown")
            authors = meta.get("authors", "")
            year    = meta.get("year", "")
            doi     = meta.get("doi", "")
            text    = r.get("text", "")
            ref_line = f"• {title}"
            if authors:
                ref_line += f" ({authors}"
                if year:
                    ref_line += f", {year}"
                ref_line += ")"
            if doi:
                ref_line += f" DOI: {doi}"
            lines.append(ref_line)
            lines.append(f"  ...{text[:400]}...")
        return "\n".join(lines)

    def build_explicit_context(self, paper_ids: list[str]) -> str:
        """
        Build full context for explicitly cited papers.
        Injects up to 3 chunks per paper to keep context size manageable.
        """
        if not paper_ids:
            return ""
        lines = ["[지정 참고 문헌 — 아래 문헌을 반드시 인용하여 답변하시오]"]
        store = self._backend()
        for pid in paper_ids:
            chunks = store.search(COLLECTION, pid, k=3, filter_metadata={"paper_id": pid})
            if not chunks:
                continue
            meta = chunks[0].get("metadata", {})
            title   = meta.get("title", "Unknown")
            authors = meta.get("authors", "")
            year    = meta.get("year", "")
            doi     = meta.get("doi", "")
            lines.append(f"\n=== {title} ({authors}, {year}) ===")
            if doi:
                lines.append(f"DOI: {doi}")
            for chunk in chunks:
                lines.append(chunk.get("text", ""))
        return "\n".join(lines)
