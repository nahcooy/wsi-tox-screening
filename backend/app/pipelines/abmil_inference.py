from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from app.pipelines.trident_preprocess import default_job_dir
from app.schemas.mil import ABMILInferenceRequest
from app.settings import settings


@dataclass(frozen=True)
class ABMILCommandPlan:
    command: list[str]
    output_dir: Path
    feature_h5: Path
    slide_path: Path | None
    checkpoint_path: Path
    log_path: Path
    cwd: Path
    env: dict[str, str]
    expected_outputs: dict[str, str]


def _slide_id_from_request(request: ABMILInferenceRequest) -> str:
    if request.slide_id:
        return request.slide_id
    if request.slide_path:
        return Path(request.slide_path).stem
    if request.feature_h5:
        return Path(request.feature_h5).stem
    raise ValueError("slide_id, slide_path, or feature_h5 is required.")


def _resolve_feature_path(request: ABMILInferenceRequest, slide_id: str) -> Path:
    if request.feature_h5:
        return Path(request.feature_h5).resolve()
    trident_dir = Path(request.trident_job_dir).resolve() if request.trident_job_dir else default_job_dir()
    return trident_dir / "20x_256px_0px_overlap" / "features_uni_v1" / f"{slide_id}.h5"


def _resolve_coords_path(request: ABMILInferenceRequest, slide_id: str) -> Path | None:
    if request.coords_h5:
        return Path(request.coords_h5).resolve()
    trident_dir = Path(request.trident_job_dir).resolve() if request.trident_job_dir else default_job_dir()
    candidate = trident_dir / "20x_256px_0px_overlap" / "patches" / f"{slide_id}_patches.h5"
    return candidate if candidate.exists() else None


def _resolve_slide_path(request: ABMILInferenceRequest, slide_id: str) -> Path | None:
    if request.slide_path:
        return Path(request.slide_path).resolve()
    candidate = settings.default_wsi_dir / f"{slide_id}.svs"
    return candidate if candidate.exists() else None


def build_abmil_command(request: ABMILInferenceRequest) -> ABMILCommandPlan:
    slide_id = _slide_id_from_request(request)
    feature_h5 = _resolve_feature_path(request, slide_id)
    if not feature_h5.exists():
        raise FileNotFoundError(f"Feature h5 not found: {feature_h5}")

    checkpoint_path = Path(request.checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"ABMIL checkpoint not found: {checkpoint_path}")

    script_path = settings.project_root / "backend" / "scripts" / "run_abmil_inference.py"
    if not script_path.exists():
        raise FileNotFoundError(f"ABMIL inference script not found: {script_path}")

    mil_python = Path(request.mil_python).resolve()
    if not mil_python.exists():
        raise FileNotFoundError(f"MIL python executable not found: {mil_python}")

    output_dir = (
        Path(request.output_dir).resolve()
        if request.output_dir
        else settings.output_dir / "mil_inference" / slide_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    slide_path = _resolve_slide_path(request, slide_id)
    coords_h5 = _resolve_coords_path(request, slide_id)

    command = [
        str(mil_python),
        str(script_path),
        "--feature_h5",
        str(feature_h5),
        "--checkpoint_path",
        str(checkpoint_path),
        "--output_dir",
        str(output_dir),
        "--mil_lab_root",
        request.mil_lab_root,
        "--slide_id",
        slide_id,
        "--top_k",
        str(request.top_k),
        "--thumbnail_max_size",
        str(request.thumbnail_max_size),
        "--device",
        request.device,
    ]
    if slide_path is not None:
        command.extend(["--slide_path", str(slide_path)])
    if coords_h5 is not None:
        command.extend(["--coords_h5", str(coords_h5)])

    expected_outputs = {
        "mil_result": str(output_dir / "mil_result.json"),
        "attention_csv": str(output_dir / "attention_scores.csv"),
        "qupath_geojson": str(output_dir / "attention_heatmap_qupath.geojson"),
        "thumbnail_heatmap": str(output_dir / "attention_heatmap_thumbnail.png"),
        "topk_manifest": str(output_dir / "topk" / "top25_patches.json"),
    }

    return ABMILCommandPlan(
        command=command,
        output_dir=output_dir,
        feature_h5=feature_h5,
        slide_path=slide_path,
        checkpoint_path=checkpoint_path,
        log_path=output_dir / "abmil_inference.log",
        cwd=settings.project_root,
        env=os.environ.copy(),
        expected_outputs=expected_outputs,
    )
