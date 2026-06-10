"""
Pathologist feedback storage and retrieval.
Each feedback entry stores: slide_id, feedback_text, prior_report, revised_report, metadata.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from app.services.rag.factory import get_backend

COLLECTION = "feedback"


class FeedbackStore:
    """CRUD + semantic search over pathologist feedback entries."""

    def _backend(self):
        return get_backend()

    def save(
        self,
        slide_id: str,
        feedback_text: str,
        prior_report: str,
        revised_report: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist one feedback entry. Returns feedback_id."""
        feedback_id = str(uuid.uuid4())
        # Searchable text = feedback + prior report summary
        searchable = f"{feedback_text}\n\n[이전 보고서 요약]\n{prior_report[:2000]}"
        meta: dict[str, Any] = {
            "type":           "feedback",
            "feedback_id":    feedback_id,
            "slide_id":       slide_id,
            "timestamp":      time.time(),
            "prior_report_snippet": prior_report[:500],
            "revised_report_snippet": revised_report[:500],
        }
        if metadata:
            meta.update({k: str(v) for k, v in metadata.items()})
        self._backend().add(COLLECTION, feedback_id, searchable, meta)
        return feedback_id

    def get_for_slide(self, slide_id: str) -> list[dict[str, Any]]:
        """All feedback entries for a given slide_id."""
        return self._backend().list_docs(
            COLLECTION, filter_metadata={"slide_id": slide_id}
        )

    def search_similar(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Semantic search across all feedback."""
        return self._backend().search(COLLECTION, query, k=k)

    def search_for_slide(self, slide_id: str, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Semantic search filtered to a specific slide."""
        return self._backend().search(
            COLLECTION, query, k=k, filter_metadata={"slide_id": slide_id}
        )

    def delete(self, feedback_id: str) -> None:
        self._backend().delete(COLLECTION, feedback_id)

    def count(self) -> int:
        return self._backend().count(COLLECTION)

    def build_rag_context(self, query: str, k: int = 3) -> str:
        """
        Retrieve top-k relevant past feedbacks and format as context
        to inject into the agent system prompt.
        """
        results = self.search_similar(query, k=k)
        if not results:
            return ""
        lines = ["[과거 병리학자 피드백 — 유사 케이스]"]
        for r in results:
            meta = r.get("metadata", {})
            slide = meta.get("slide_id", "?")
            snippet = meta.get("prior_report_snippet", "")
            text = r.get("text", "").split("[이전 보고서 요약]")[0].strip()
            lines.append(f"• Slide {slide}: {text}")
            if snippet:
                lines.append(f"  (이전 보고서: {snippet[:200]}…)")
        return "\n".join(lines)
