from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes_feedback import router as feedback_router
from app.api.routes_literature import router as literature_router
from app.api.routes_mil import router as mil_router
from app.api.routes_mock import router as mock_router
from app.api.routes_preprocess import router as preprocess_router
from app.api.routes_workflow import router as workflow_router
from app.settings import settings

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title=settings.app_name, version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mock_router, prefix="/api", tags=["mock"])
app.include_router(preprocess_router, prefix="/api", tags=["preprocess"])
app.include_router(mil_router, prefix="/api", tags=["mil"])
app.include_router(workflow_router, prefix="/api", tags=["workflow"])
app.include_router(feedback_router, prefix="/api", tags=["feedback"])
app.include_router(literature_router, prefix="/api", tags=["literature"])
settings.output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=settings.output_dir), name="outputs")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
