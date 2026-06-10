from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from app.pipelines.abmil_inference import build_abmil_command
from app.schemas.mil import ABMILInferenceRequest
import threading

from app.services.agent_loop import run_agent_full_background, _live_status_path
from app.services.diagnostic_report import (
    read_json,
    build_report_payload,
    generate_report_text,
)
from app.settings import settings
from app.workers.job_manager import job_manager

router = APIRouter()

NUCLEI_TYPE_NAMES = {
    "1": "Neoplastic",
    "2": "Inflammatory",
    "3": "Connective",
    "4": "Dead",
    "5": "Epithelial",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}


def run_root() -> Path:
    root = settings.output_dir / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_dir(slide_id: str) -> Path:
    return run_root() / slide_id


def run_json_path(slide_id: str) -> Path:
    return run_dir(slide_id) / "run.json"


def read_run(slide_id: str) -> dict:
    path = run_json_path(slide_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {slide_id}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_run(slide_id: str, payload: dict) -> None:
    path = run_json_path(slide_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def rel_output_url(path: Path | str | None) -> str | None:
    if path is None:
        return None
    if not Path(path).exists():
        return None
    try:
        rel = Path(path).resolve().relative_to(settings.output_dir.resolve())
    except ValueError:
        return None
    return f"/outputs/{rel.as_posix()}"


def normalize_nuclei_type(value: object) -> str:
    return NUCLEI_TYPE_NAMES.get(value, NUCLEI_TYPE_NAMES.get(str(value), str(value)))


def normalize_type_counts(counts: dict | None) -> dict:
    normalized: dict[str, int] = {}
    for key, value in (counts or {}).items():
        label = normalize_nuclei_type(key)
        normalized[label] = normalized.get(label, 0) + int(value)
    return normalized


def normalize_nuclei_summary(summary: dict) -> dict:
    summary["type_counts"] = normalize_type_counts(summary.get("type_counts"))
    for overlay in summary.get("overlays", []):
        overlay["type_counts"] = normalize_type_counts(overlay.get("type_counts"))
    return summary


def workflow_paths(slide_id: str, slide_filename: str | None = None) -> dict:
    base = run_dir(slide_id)
    trident_dir = base / "trident"
    mil_dir = base / "mil"
    nuclei_dir = base / "nuclei"
    report_dir = base / "report"
    slide_path = None
    if slide_filename:
        slide_path = base / "slide" / slide_filename
    else:
        slide_files = sorted((base / "slide").glob("*")) if (base / "slide").exists() else []
        slide_path = slide_files[0] if slide_files else None

    return {
        "run_dir": base,
        "slide_path": slide_path,
        "trident_dir": trident_dir,
        "feature_h5": trident_dir / "20x_256px_0px_overlap" / "features_uni_v1" / f"{slide_id}.h5",
        "coords_h5": trident_dir / "20x_256px_0px_overlap" / "patches" / f"{slide_id}_patches.h5",
        "mil_dir": mil_dir,
        "mil_result": mil_dir / "mil_result.json",
        "attention_geojson": mil_dir / "attention_heatmap_qupath.geojson",
        "attention_csv": mil_dir / "attention_scores.csv",
        "attention_thumbnail": mil_dir / "attention_heatmap_thumbnail.png",
        "wsi_thumbnail": mil_dir / "wsi_thumbnail.png",
        "topk_manifest": mil_dir / "topk" / "top25_patches.json",
        "nuclei_dir": nuclei_dir,
        "nuclei_summary": nuclei_dir / "nuclei_summary.json",
        "nuclei_geojson": nuclei_dir / "nuclei_instances.geojson",
        "nuclei_counts_csv": nuclei_dir / "cell_type_counts.csv",
        "nuclei_counts_json": nuclei_dir / "cell_type_counts.json",
        "nuclei_instances_json": nuclei_dir / "all_instances.json",
        "nuclei_instances_jsonl": nuclei_dir / "all_instances.jsonl",
        "patch_metrics_json": nuclei_dir / "patch_metrics.json",
        "patch_metrics_csv": nuclei_dir / "patch_metrics.csv",
        "metric_comparison": nuclei_dir / "metric_comparison.json",
        "report_dir": report_dir,
        "report_json": report_dir / "diagnostic_report.json",
        "report_markdown": report_dir / "diagnostic_report.md",
        "report_input": report_dir / "diagnostic_report_input.json",
    }


def job_state(job_id: str | None) -> str | None:
    if not job_id:
        return None
    job = job_manager.get(job_id)
    return job.status if job is not None else None


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def job_duration_seconds(job_id: str | None) -> float | None:
    if not job_id:
        return None
    job = job_manager.get(job_id)
    if job is None:
        return None
    started = parse_utc(job.started_at)
    if started is None:
        return None
    finished = parse_utc(job.finished_at) or datetime.now(timezone.utc)
    return max(0.0, (finished - started).total_seconds())


def persist_completed_duration(run: dict, stage: str, job_id: str | None) -> bool:
    key = f"{stage}_duration_seconds"
    if run.get(key) is not None:
        return False
    job = job_manager.get(job_id) if job_id else None
    if job is None or job.status != "COMPLETED":
        return False
    duration = job_duration_seconds(job_id)
    if duration is None:
        return False
    run[key] = duration
    return True


def stage_status(done: bool, job_id: str | None) -> str:
    if done:
        return "completed"
    state = job_state(job_id)
    if state in {"PENDING", "RUNNING"}:
        return "running"
    if state in {"FAILED", "CANCELLED"}:
        return state.lower()
    return "not_started"


def build_status(slide_id: str) -> dict:
    run = read_run(slide_id)
    paths = workflow_paths(slide_id, run.get("slide_filename"))
    copy_status = "completed" if paths["slide_path"] is not None and paths["slide_path"].exists() else "not_started"
    preprocess_status = stage_status(paths["feature_h5"].exists(), run.get("preprocess_job_id"))
    inference_status = stage_status(paths["mil_result"].exists(), run.get("inference_job_id"))
    nuclei_status = stage_status(paths["nuclei_summary"].exists(), run.get("nuclei_job_id"))
    report_status = "completed" if paths["report_json"].exists() else "not_started"

    changed = False
    changed |= persist_completed_duration(run, "preprocess", run.get("preprocess_job_id"))
    changed |= persist_completed_duration(run, "inference", run.get("inference_job_id"))
    changed |= persist_completed_duration(run, "nuclei", run.get("nuclei_job_id"))
    if changed:
        write_run(slide_id, run)

    durations = {
        "preprocess_seconds": run.get("preprocess_duration_seconds")
        or job_duration_seconds(run.get("preprocess_job_id")),
        "inference_seconds": run.get("inference_duration_seconds")
        or job_duration_seconds(run.get("inference_job_id")),
        "nuclei_seconds": run.get("nuclei_duration_seconds")
        or job_duration_seconds(run.get("nuclei_job_id")),
        "report_seconds": run.get("report_duration_seconds"),
    }

    result = None
    topk = []
    nuclei_summary = None
    metric_comparison = None
    patch_metrics = []
    llm_report = None
    if paths["mil_result"].exists():
        with paths["mil_result"].open("r", encoding="utf-8") as f:
            result = json.load(f)
        if paths["topk_manifest"].exists():
            with paths["topk_manifest"].open("r", encoding="utf-8") as f:
                topk = json.load(f)
            for patch in topk:
                patch["image_url"] = rel_output_url(patch.get("image_path"))
    if paths["nuclei_summary"].exists():
        with paths["nuclei_summary"].open("r", encoding="utf-8") as f:
            nuclei_summary = json.load(f)
        nuclei_summary = normalize_nuclei_summary(nuclei_summary)
        for overlay in nuclei_summary.get("overlays", []):
            overlay["image_url"] = rel_output_url(overlay.get("image_path"))
            overlay["overlay_url"] = rel_output_url(overlay.get("overlay_path"))
    if paths["metric_comparison"].exists():
        with paths["metric_comparison"].open("r", encoding="utf-8") as f:
            metric_comparison = json.load(f)
    if paths["patch_metrics_json"].exists():
        with paths["patch_metrics_json"].open("r", encoding="utf-8") as f:
            patch_metrics = json.load(f)
    if paths["report_json"].exists():
        with paths["report_json"].open("r", encoding="utf-8") as f:
            llm_report = json.load(f)

    return {
        "slide_id": slide_id,
        "slide_filename": run.get("slide_filename"),
        "run_dir": str(paths["run_dir"]),
        "slide_path": str(paths["slide_path"]) if paths["slide_path"] else None,
        "copy_status": copy_status,
        "preprocess_status": preprocess_status,
        "inference_status": inference_status,
        "nuclei_status": nuclei_status,
        "report_status": report_status,
        "durations": durations,
        "preprocess_job_id": run.get("preprocess_job_id"),
        "inference_job_id": run.get("inference_job_id"),
        "nuclei_job_id": run.get("nuclei_job_id"),
        "paths": {
            "feature_h5": str(paths["feature_h5"]),
            "coords_h5": str(paths["coords_h5"]),
            "mil_result": str(paths["mil_result"]),
            "attention_geojson": str(paths["attention_geojson"]),
            "attention_thumbnail": str(paths["attention_thumbnail"]),
            "topk_manifest": str(paths["topk_manifest"]),
            "nuclei_summary": str(paths["nuclei_summary"]),
            "nuclei_geojson": str(paths["nuclei_geojson"]),
            "nuclei_counts_csv": str(paths["nuclei_counts_csv"]),
            "nuclei_instances_json": str(paths["nuclei_instances_json"]),
            "patch_metrics_json": str(paths["patch_metrics_json"]),
            "patch_metrics_csv": str(paths["patch_metrics_csv"]),
            "metric_comparison": str(paths["metric_comparison"]),
            "report_json": str(paths["report_json"]),
            "report_markdown": str(paths["report_markdown"]),
        },
        "urls": {
            "attention_geojson": rel_output_url(paths["attention_geojson"]),
            "attention_csv": rel_output_url(paths["attention_csv"]),
            "attention_thumbnail": rel_output_url(paths["attention_thumbnail"]),
            "wsi_thumbnail": rel_output_url(paths["wsi_thumbnail"]),
            "topk_manifest": rel_output_url(paths["topk_manifest"]),
            "nuclei_summary": rel_output_url(paths["nuclei_summary"]),
            "nuclei_geojson": rel_output_url(paths["nuclei_geojson"]),
            "nuclei_counts_csv": rel_output_url(paths["nuclei_counts_csv"]),
            "nuclei_counts_json": rel_output_url(paths["nuclei_counts_json"]),
            "nuclei_instances_json": rel_output_url(paths["nuclei_instances_json"]),
            "nuclei_instances_jsonl": rel_output_url(paths["nuclei_instances_jsonl"]),
            "patch_metrics_json": rel_output_url(paths["patch_metrics_json"]),
            "patch_metrics_csv": rel_output_url(paths["patch_metrics_csv"]),
            "metric_comparison": rel_output_url(paths["metric_comparison"]),
            "report_json": rel_output_url(paths["report_json"]),
            "report_markdown": rel_output_url(paths["report_markdown"]),
        },
        "mil_result": result,
        "topk": topk,
        "nuclei_summary": nuclei_summary,
        "metric_comparison": metric_comparison,
        "patch_metrics": patch_metrics,
        "llm_report": llm_report,
    }


@router.post("/workflow/slides/upload")
def upload_slide(slide: UploadFile = File(...)) -> dict:
    if not slide.filename:
        raise HTTPException(status_code=400, detail="Slide filename is missing.")
    slide_id = Path(slide.filename).stem
    base = run_dir(slide_id)
    slide_dir = base / "slide"
    slide_dir.mkdir(parents=True, exist_ok=True)
    dest = slide_dir / Path(slide.filename).name

    with dest.open("wb") as f:
        shutil.copyfileobj(slide.file, f, length=1024 * 1024 * 16)

    run = {
        "slide_id": slide_id,
        "slide_filename": dest.name,
        "slide_path": str(dest),
        "mil_model": "abmil",
        "preprocess_job_id": None,
        "inference_job_id": None,
        "nuclei_job_id": None,
    }
    write_run(slide_id, run)
    return build_status(slide_id)


@router.get("/workflow/runs/{slide_id}")
def get_workflow_status(slide_id: str) -> dict:
    return build_status(slide_id)


@router.post("/workflow/runs/{slide_id}/preprocess")
def start_workflow_preprocess(slide_id: str, hf_token: str | None = Form(default=None)) -> dict:
    run = read_run(slide_id)
    paths = workflow_paths(slide_id, run.get("slide_filename"))
    if paths["slide_path"] is None or not paths["slide_path"].exists():
        raise HTTPException(status_code=400, detail="Copied slide is missing.")
    if paths["feature_h5"].exists():
        return build_status(slide_id)

    script_path = settings.project_root / "backend" / "scripts" / "run_trident_single_slide.py"
    if not script_path.exists():
        raise HTTPException(status_code=400, detail=f"TRIDENT workflow script not found: {script_path}")
    trident_python = settings.trident_python
    if not trident_python.exists():
        raise HTTPException(status_code=400, detail=f"TRIDENT python not found: {trident_python}")

    command = [
        str(trident_python),
        str(script_path),
        "--slide_path",
        str(paths["slide_path"]),
        "--job_dir",
        str(paths["trident_dir"]),
        "--trident_root",
        str(settings.trident_root),
        "--segmenter",
        "grandqc",
        "--remove_artifacts",
        "--mag",
        "20",
        "--patch_size",
        "256",
        "--overlap",
        "0",
        "--patch_encoder",
        "uni_v1",
        "--batch_size",
        "64",
        "--feat_batch_size",
        "512",
        "--device",
        "cuda:0",
    ]
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/wsi-tox-screening-mpl")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
        env["HUGGINGFACE_HUB_TOKEN"] = hf_token

    job = job_manager.start(
        command=command,
        cwd=settings.project_root,
        log_path=paths["trident_dir"] / "trident_pipeline.log",
        env=env,
        metadata={"kind": "workflow_preprocess", "slide_id": slide_id},
    )
    run["preprocess_job_id"] = job.job_id
    write_run(slide_id, run)
    return build_status(slide_id)


@router.post("/workflow/runs/{slide_id}/inference")
def start_workflow_inference(slide_id: str) -> dict:
    run = read_run(slide_id)
    paths = workflow_paths(slide_id, run.get("slide_filename"))
    if not paths["feature_h5"].exists():
        raise HTTPException(status_code=400, detail="Preprocess is not completed yet.")
    if paths["mil_result"].exists():
        return build_status(slide_id)

    request = ABMILInferenceRequest(
        slide_id=slide_id,
        slide_path=str(paths["slide_path"]),
        trident_job_dir=str(paths["trident_dir"]),
        output_dir=str(paths["mil_dir"]),
        top_k=25,
        device="cuda:0",
    )
    try:
        plan = build_abmil_command(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = job_manager.start(
        command=plan.command,
        cwd=plan.cwd,
        log_path=plan.log_path,
        env=plan.env,
        metadata={"kind": "workflow_abmil_inference", "slide_id": slide_id},
    )
    run["inference_job_id"] = job.job_id
    write_run(slide_id, run)
    return build_status(slide_id)


@router.post("/workflow/runs/{slide_id}/nuclei")
def start_workflow_nuclei(slide_id: str) -> dict:
    run = read_run(slide_id)
    paths = workflow_paths(slide_id, run.get("slide_filename"))
    if not paths["mil_result"].exists() or not paths["topk_manifest"].exists():
        raise HTTPException(status_code=400, detail="Model inference/top-k output is not completed yet.")
    if paths["nuclei_summary"].exists():
        return build_status(slide_id)

    script_path = settings.project_root / "backend" / "scripts" / "run_nulite_topk_inference.py"
    if not script_path.exists():
        raise HTTPException(status_code=400, detail=f"NuLite workflow script not found: {script_path}")
    if not settings.nulite_python.exists():
        raise HTTPException(status_code=400, detail=f"NuLite python not found: {settings.nulite_python}")
    if not settings.nulite_root.exists():
        raise HTTPException(status_code=400, detail=f"NuLite root not found: {settings.nulite_root}")
    if not settings.default_nulite_h_checkpoint.exists():
        raise HTTPException(
            status_code=400,
            detail=f"NuLite-H checkpoint not found: {settings.default_nulite_h_checkpoint}",
        )

    command = [
        str(settings.nulite_python),
        str(script_path),
        "--topk_manifest",
        str(paths["topk_manifest"]),
        "--mil_result",
        str(paths["mil_result"]),
        "--output_dir",
        str(paths["nuclei_dir"]),
        "--nulite_root",
        str(settings.nulite_root),
        "--checkpoint_path",
        str(settings.default_nulite_h_checkpoint),
        "--slide_id",
        slide_id,
        "--batch_size",
        "8",
        "--num_workers",
        "0",
        "--gpu",
        "0",
    ]
    if settings.matched_dataset_csv.exists() and settings.celltype_summary_csv.exists():
        command.extend(
            [
                "--matched_dataset_csv",
                str(settings.matched_dataset_csv),
                "--celltype_summary_csv",
                str(settings.celltype_summary_csv),
            ]
        )
    env = os.environ.copy()
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/wsi-tox-screening-numba-cache")
    env.setdefault("MPLCONFIGDIR", "/tmp/wsi-tox-screening-mpl")
    job = job_manager.start(
        command=command,
        cwd=settings.project_root,
        log_path=paths["nuclei_dir"] / "nulite_inference.log",
        env=env,
        metadata={"kind": "workflow_nulite_topk", "slide_id": slide_id},
    )
    run["nuclei_job_id"] = job.job_id
    write_run(slide_id, run)
    return build_status(slide_id)


class ReportRequest(BaseModel):
    reference_paper_ids: list[str] = []


@router.post("/workflow/runs/{slide_id}/report")
def generate_workflow_report(slide_id: str, req: ReportRequest | None = None) -> dict:
    run = read_run(slide_id)
    report_started = datetime.now(timezone.utc)
    paths = workflow_paths(slide_id, run.get("slide_filename"))
    if not paths["mil_result"].exists():
        raise HTTPException(status_code=400, detail="Model inference is not completed yet.")
    if not paths["nuclei_summary"].exists() or not paths["metric_comparison"].exists():
        raise HTTPException(status_code=400, detail="Nuclei analysis and metric comparison are required first.")

    mil_result = read_json(paths["mil_result"], {})
    topk = read_json(paths["topk_manifest"], [])
    nuclei_summary = read_json(paths["nuclei_summary"], {})
    metric_comparison = read_json(paths["metric_comparison"], {})
    patch_metrics = read_json(paths["patch_metrics_json"], [])

    payload = build_report_payload(
        slide_id=slide_id,
        mil_result=mil_result,
        topk=topk,
        nuclei_summary=nuclei_summary,
        metric_comparison=metric_comparison,
        patch_metrics=patch_metrics,
    )
    report_text, llm_metadata = generate_report_text(payload)
    report_duration_seconds = (datetime.now(timezone.utc) - report_started).total_seconds()

    paths["report_dir"].mkdir(parents=True, exist_ok=True)
    report = {
        "slide_id": slide_id,
        "report_text": report_text,
        "agent": llm_metadata,
        "generated_at": report_started.isoformat(),
        "duration_seconds": report_duration_seconds,
    }
    with paths["report_json"].open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with paths["report_markdown"].open("w", encoding="utf-8") as f:
        f.write(report_text.strip() + "\n")

    run["report_duration_seconds"] = report_duration_seconds
    write_run(slide_id, run)
    return build_status(slide_id)


# ── Agent Full-Run endpoints ───────────────────────────────────────────────────

class AgentFullRunRequest(BaseModel):
    user_instructions: str = ""


# Track running full-run threads per slide_id
_agent_full_threads: dict[str, threading.Thread] = {}


@router.post("/workflow/runs/{slide_id}/agent-full-run")
def start_agent_full_run(slide_id: str, req: AgentFullRunRequest | None = None) -> dict:
    """Start a fully autonomous pipeline+analysis+report run in the background."""
    instructions = req.user_instructions if req else ""

    # Prevent duplicate runs
    existing = _agent_full_threads.get(slide_id)
    if existing and existing.is_alive():
        # Check if already running
        live_path = _live_status_path(slide_id)
        if live_path.exists():
            try:
                live = json.loads(live_path.read_text(encoding="utf-8"))
                if live.get("state") == "running":
                    return {"status": "already_running", "message": "Agent full-run is already in progress."}
            except Exception:
                pass

    # Reset live status
    live_path = _live_status_path(slide_id)
    live_path.parent.mkdir(parents=True, exist_ok=True)
    live_path.write_text(json.dumps({
        "state": "running",
        "stage": "initializing",
        "iteration": 0,
        "pipeline": {"preprocess": "unknown", "inference": "unknown", "nuclei_topk": "unknown"},
        "tools_called": [],
        "log": [],
        "report": None,
        "error": None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    t = threading.Thread(
        target=run_agent_full_background,
        args=(slide_id, instructions or None, settings.output_dir),
        daemon=True,
    )
    _agent_full_threads[slide_id] = t
    t.start()
    return {"status": "started", "slide_id": slide_id}


@router.get("/workflow/runs/{slide_id}/wsi-thumbnail")
def get_wsi_thumbnail(slide_id: str, max_size: int = 6000):
    """Clean WSI thumbnail without heatmap overlay. On-the-fly fallback for old runs."""
    paths = workflow_paths(slide_id)
    thumb_path = paths["wsi_thumbnail"]

    # 이미 있으면 바로 반환
    if thumb_path.exists():
        return FileResponse(str(thumb_path), media_type="image/png")

    # 없으면 OpenSlide로 생성
    slide_path = paths["slide_path"]
    if slide_path is None or not slide_path.exists():
        raise HTTPException(status_code=404, detail="Slide file not found")

    try:
        import openslide
        from PIL import Image
        import io

        slide = openslide.OpenSlide(str(slide_path))
        w, h = slide.dimensions
        scale = min(max_size / max(w, h), 1.0)
        thumb = slide.get_thumbnail((max(1, int(w * scale)), max(1, int(h * scale)))).convert("RGB")
        slide.close()

        # 캐시
        thumb.save(str(thumb_path))

        buf = io.BytesIO()
        thumb.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Thumbnail generation failed: {e}")


@router.get("/workflow/runs/{slide_id}/agent-live-status")
def get_agent_live_status(slide_id: str) -> dict:
    """Poll live status of the agent full-run."""
    live_path = _live_status_path(slide_id)
    if not live_path.exists():
        return {"state": "not_started"}
    try:
        return json.loads(live_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"state": "error", "error": str(e)}
