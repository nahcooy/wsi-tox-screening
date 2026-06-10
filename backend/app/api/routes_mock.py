from pathlib import Path

from fastapi import APIRouter

from app.schemas.slide import SlideMetadata, SlideRegisterRequest
from app.settings import settings

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "mock": settings.enable_mock_mode,
    }


@router.get("/mock/config")
def mock_config() -> dict:
    return {
        "mock": True,
        "workflow": ["slide", "mil", "attention", "nulite", "statistics", "report"],
        "models": {
            "encoders": ["UNI", "Rat-Liver-DINOv2", "custom"],
            "mil": ["abmil", "clam", "dsmil", "rrt", "transmil", "wikg"],
            "nuclei": ["NuLite-H"],
        },
        "message": "Task 1 scaffold is running. Real inference is intentionally not implemented yet.",
    }


@router.post("/slides/register", response_model=SlideMetadata)
def register_slide(payload: SlideRegisterRequest) -> SlideMetadata:
    slide_name = Path(payload.slide_path).stem or "mock_slide"
    slide_id = payload.slide_id or slide_name
    return SlideMetadata(
        slide_id=slide_id,
        slide_path=payload.slide_path,
        width=100_000,
        height=72_000,
        level_count=4,
        level_dimensions=[
            (100_000, 72_000),
            (50_000, 36_000),
            (25_000, 18_000),
            (12_500, 9_000),
        ],
        mpp_x=0.5,
        mpp_y=0.5,
        objective_power=20.0,
        stain=payload.stain,
        species=payload.species,
        organ=payload.organ,
        mock=True,
    )

