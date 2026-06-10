from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from app.pipelines.trident_preprocess import build_trident_command, default_job_dir
from app.schemas.preprocess import (
    JobStatus,
    TridentPreprocessJobResponse,
    TridentPreprocessPlanResponse,
    TridentPreprocessRequest,
)
from app.settings import settings
from app.workers.job_manager import job_manager

router = APIRouter()


@router.get("/preprocess/trident/defaults")
def trident_defaults() -> dict:
    return {
        "dataset_csv": str(settings.default_dataset_csv),
        "wsi_dir": str(settings.default_wsi_dir),
        "job_dir": str(default_job_dir()),
        "trident_root": str(settings.trident_root),
        "trident_python": str(settings.trident_python),
        "pipeline": [
            "grandqc tissue segmentation",
            "grandqc artifact segmentation",
            "20x 256x256 no-overlap coordinate extraction",
            "UNI v1 patch feature extraction",
        ],
    }


@router.post("/preprocess/trident/run", response_model=TridentPreprocessJobResponse)
def run_trident_preprocess(request: TridentPreprocessRequest) -> TridentPreprocessJobResponse:
    try:
        plan = build_trident_command(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = job_manager.start(
        command=plan.command,
        cwd=plan.cwd,
        log_path=plan.log_path,
        env=plan.env,
        metadata={
            "kind": "trident_preprocess",
            "job_dir": str(plan.job_dir),
            "manifest": plan.manifest.model_dump(),
        },
    )

    return TridentPreprocessJobResponse(
        job_id=job.job_id,
        status=job.status,
        command=plan.command,
        job_dir=str(plan.job_dir),
        manifest=plan.manifest,
        log_path=str(plan.log_path),
    )


@router.post("/preprocess/trident/plan", response_model=TridentPreprocessPlanResponse)
def plan_trident_preprocess(request: TridentPreprocessRequest) -> TridentPreprocessPlanResponse:
    try:
        plan = build_trident_command(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return TridentPreprocessPlanResponse(
        command=plan.command,
        job_dir=str(plan.job_dir),
        manifest=plan.manifest,
        log_path=str(plan.log_path),
    )


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@router.post("/jobs/{job_id}/cancel", response_model=JobStatus)
def cancel_job(job_id: str) -> JobStatus:
    job = job_manager.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@router.get("/jobs/{job_id}/logs", response_class=PlainTextResponse)
def get_job_logs(job_id: str) -> str:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    try:
        with open(job.log_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        return ""
