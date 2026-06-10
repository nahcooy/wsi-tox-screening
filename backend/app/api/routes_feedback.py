"""
Pathologist feedback API.
  POST /api/feedback/{slide_id}         — submit feedback, get revised report
  GET  /api/feedback/{slide_id}         — list past feedback for a slide
  GET  /api/feedback/search?q=...&k=5  — cross-slide semantic search
  DELETE /api/feedback/{feedback_id}    — remove an entry
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.agent_loop import run_feedback_agent
from app.services.rag.feedback_store import FeedbackStore
from app.settings import settings

router = APIRouter()
store  = FeedbackStore()


class FeedbackRequest(BaseModel):
    feedback_text: str
    prior_report: str = ""
    metadata: dict = {}   # e.g. {"compound": "APAP", "dose": "2000 mg/kg", "pathology": "necrosis"}


class FeedbackResponse(BaseModel):
    feedback_id: str
    revised_report: str
    agent: dict


def _load_prior_report(slide_id: str) -> str:
    """Load previously generated report from disk if not provided."""
    report_path = settings.output_dir / "runs" / slide_id / "report" / "diagnostic_report.json"
    if not report_path.exists():
        return ""
    try:
        with report_path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data.get("report_text", "")
    except Exception:
        return ""


@router.post("/feedback/{slide_id}", response_model=FeedbackResponse)
def submit_feedback(slide_id: str, req: FeedbackRequest) -> FeedbackResponse:
    """
    Submit pathologist feedback → agent re-reasons → returns revised report.
    Saves both feedback and revised report to RAG store.
    """
    prior_report = req.prior_report or _load_prior_report(slide_id)
    if not prior_report:
        raise HTTPException(
            status_code=400,
            detail="No prior report found. Generate a report first or provide prior_report in request body.",
        )

    revised_report, agent_meta = run_feedback_agent(
        slide_id=slide_id,
        feedback_text=req.feedback_text,
        prior_report=prior_report,
    )

    # Save to RAG store (Level 2)
    feedback_id = store.save(
        slide_id=slide_id,
        feedback_text=req.feedback_text,
        prior_report=prior_report,
        revised_report=revised_report,
        metadata=req.metadata,
    )

    # Persist revised report to disk
    report_dir = settings.output_dir / "runs" / slide_id / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    feedback_dir = report_dir / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    revised_path = feedback_dir / f"{feedback_id}.json"
    with revised_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "feedback_id":     feedback_id,
                "slide_id":        slide_id,
                "feedback_text":   req.feedback_text,
                "prior_report":    prior_report,
                "revised_report":  revised_report,
                "agent":           agent_meta,
                "metadata":        req.metadata,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return FeedbackResponse(
        feedback_id=feedback_id,
        revised_report=revised_report,
        agent=agent_meta,
    )


@router.get("/feedback/{slide_id}")
def list_feedback(slide_id: str) -> list[dict]:
    """List all past feedback entries for a slide."""
    return store.get_for_slide(slide_id)


@router.get("/feedback")
def search_feedback(q: str, k: int = 5) -> list[dict]:
    """Cross-slide semantic search over all feedback entries."""
    return store.search_similar(q, k=k)


@router.delete("/feedback/entry/{feedback_id}")
def delete_feedback(feedback_id: str) -> dict:
    store.delete(feedback_id)
    return {"deleted": True, "feedback_id": feedback_id}
