"""
Literature management API.
  POST   /api/literature/upload          — upload PDF
  POST   /api/literature/add-text        — add plain text / abstract
  GET    /api/literature                 — list all papers
  GET    /api/literature/search?q=...    — semantic search
  DELETE /api/literature/{paper_id}      — remove paper
"""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.rag.literature_store import LiteratureStore

router = APIRouter()
store  = LiteratureStore()


class AddTextRequest(BaseModel):
    title:   str
    text:    str
    authors: str = ""
    year:    str = ""
    doi:     str = ""


@router.post("/literature/upload")
async def upload_pdf(
    file:    UploadFile = File(...),
    title:   str        = Form(default=""),
    authors: str        = Form(default=""),
    year:    str        = Form(default=""),
    doi:     str        = Form(default=""),
) -> dict:
    """Upload a PDF. Parses and indexes it into the RAG store."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    pdf_bytes = await file.read()
    result = store.add_pdf(
        pdf_bytes,
        filename=file.filename,
        title=title or file.filename,
        authors=authors,
        year=year,
        doi=doi,
    )
    return result


@router.post("/literature/add-text")
def add_text(req: AddTextRequest) -> dict:
    """Add a paper from plain text (abstract, full text, etc.)."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty.")
    return store.add_text(
        req.text,
        title=req.title,
        authors=req.authors,
        year=req.year,
        doi=req.doi,
    )


@router.get("/literature")
def list_papers() -> list[dict]:
    """List all indexed papers."""
    return store.list_papers()


@router.get("/literature/search")
def search_literature(q: str, k: int = 5) -> list[dict]:
    """Semantic search over paper chunks."""
    return store.search(q, k=k)


@router.delete("/literature/{paper_id}")
def delete_paper(paper_id: str) -> dict:
    n = store.delete_paper(paper_id)
    if n == 0:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found.")
    return {"deleted": True, "paper_id": paper_id, "chunks_removed": n}
