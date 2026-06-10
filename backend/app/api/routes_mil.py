import json

from fastapi import APIRouter, HTTPException

from app.pipelines.abmil_inference import build_abmil_command
from app.schemas.mil import (
    ABMILInferenceJobResponse,
    ABMILInferencePlanResponse,
    ABMILInferenceRequest,
)
from app.workers.job_manager import job_manager
from app.settings import settings

router = APIRouter()


@router.post("/inference/abmil/plan", response_model=ABMILInferencePlanResponse)
def plan_abmil_inference(request: ABMILInferenceRequest) -> ABMILInferencePlanResponse:
    try:
        plan = build_abmil_command(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ABMILInferencePlanResponse(
        command=plan.command,
        output_dir=str(plan.output_dir),
        feature_h5=str(plan.feature_h5),
        slide_path=str(plan.slide_path) if plan.slide_path is not None else None,
        checkpoint_path=str(plan.checkpoint_path),
        expected_outputs=plan.expected_outputs,
    )


@router.post("/inference/abmil/run", response_model=ABMILInferenceJobResponse)
def run_abmil_inference(request: ABMILInferenceRequest) -> ABMILInferenceJobResponse:
    try:
        plan = build_abmil_command(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = job_manager.start(
        command=plan.command,
        cwd=plan.cwd,
        log_path=plan.log_path,
        env=plan.env,
        metadata={
            "kind": "abmil_inference",
            "feature_h5": str(plan.feature_h5),
            "output_dir": str(plan.output_dir),
            "expected_outputs": plan.expected_outputs,
        },
    )

    return ABMILInferenceJobResponse(
        job_id=job.job_id,
        status=job.status,
        command=plan.command,
        output_dir=str(plan.output_dir),
        feature_h5=str(plan.feature_h5),
        slide_path=str(plan.slide_path) if plan.slide_path is not None else None,
        checkpoint_path=str(plan.checkpoint_path),
        expected_outputs=plan.expected_outputs,
        log_path=str(plan.log_path),
    )


@router.get("/inference/abmil/result/{slide_id}")
def get_abmil_result(slide_id: str) -> dict:
    result_path = settings.output_dir / "mil_inference" / slide_id / "mil_result.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"ABMIL result not found: {result_path}")
    with result_path.open("r", encoding="utf-8") as f:
        return json.load(f)
