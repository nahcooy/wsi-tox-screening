from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

from app.schemas.preprocess import TridentManifestSummary, TridentPreprocessRequest
from app.settings import settings


@dataclass(frozen=True)
class TridentCommandPlan:
    command: list[str]
    job_dir: Path
    manifest: TridentManifestSummary
    log_path: Path
    cwd: Path
    env: dict[str, str]


def default_job_dir() -> Path:
    return (settings.output_dir / "trident_runs" / "spass_2000").resolve()


def _slide_name_from_row(row: dict[str, str]) -> str:
    if row.get("wsi"):
        return os.path.basename(row["wsi"])
    if row.get("svs_filename"):
        return os.path.basename(row["svs_filename"])
    if row.get("absolute_path"):
        return os.path.basename(row["absolute_path"])
    raise ValueError("Dataset CSV must contain one of: wsi, svs_filename, absolute_path.")


def _read_dataset_rows(request: TridentPreprocessRequest) -> list[dict[str, str]]:
    if request.slide_path:
        slide_name = os.path.basename(request.slide_path)
        if not slide_name:
            raise ValueError(f"Invalid slide_path: {request.slide_path}")
        return [{"wsi": slide_name, "absolute_path": request.slide_path}]

    if not request.dataset_csv:
        raise ValueError("dataset_csv is required when slide_path is not provided.")

    csv_path = Path(request.dataset_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def prepare_trident_manifest(request: TridentPreprocessRequest, job_dir: Path) -> TridentManifestSummary:
    rows = _read_dataset_rows(request)
    wsi_dir = Path(request.wsi_dir)
    manifest_dir = job_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = manifest_dir / "trident_wsi_list.csv"
    missing_json = manifest_dir / "missing_slides.json"

    runnable_rows: list[dict[str, str]] = []
    missing: list[str] = []
    seen: set[str] = set()

    for row in rows:
        slide_name = _slide_name_from_row(row)
        if slide_name in seen:
            continue
        seen.add(slide_name)

        if not (wsi_dir / slide_name).exists():
            missing.append(slide_name)
            continue

        out_row = dict(row)
        out_row["wsi"] = slide_name
        runnable_rows.append(out_row)

    if missing and not request.allow_missing_slides:
        raise FileNotFoundError(
            f"{len(missing)} slide files are missing under {wsi_dir}. "
            f"Examples: {missing[:5]}"
        )

    if not runnable_rows:
        raise ValueError(f"No runnable slides found under {wsi_dir}.")

    fieldnames = ["wsi"]
    extra_fields = sorted({key for row in runnable_rows for key in row.keys()} - {"wsi"})
    fieldnames.extend(extra_fields)

    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(runnable_rows)

    with missing_json.open("w", encoding="utf-8") as f:
        json.dump({"missing": missing}, f, indent=2)

    return TridentManifestSummary(
        manifest_csv=str(manifest_csv),
        source_count=len(rows),
        runnable_count=len(runnable_rows),
        missing_count=len(missing),
        missing_examples=missing[:10],
    )


def build_trident_command(request: TridentPreprocessRequest) -> TridentCommandPlan:
    job_dir = Path(request.job_dir).resolve() if request.job_dir else default_job_dir()
    job_dir.mkdir(parents=True, exist_ok=True)
    manifest = prepare_trident_manifest(request, job_dir)

    trident_root = Path(request.trident_root)
    trident_script = trident_root / "run_batch_of_slides.py"
    if not trident_script.exists():
        raise FileNotFoundError(f"TRIDENT run_batch_of_slides.py not found: {trident_script}")

    trident_python = Path(request.trident_python)
    if not trident_python.exists():
        raise FileNotFoundError(f"TRIDENT python executable not found: {trident_python}")

    command = [
        str(trident_python),
        str(trident_script),
        "--task",
        request.task,
        "--gpu",
        str(request.gpu),
        "--wsi_dir",
        request.wsi_dir,
        "--custom_list_of_wsis",
        manifest.manifest_csv,
        "--job_dir",
        str(job_dir),
        "--segmenter",
        request.segmenter,
        "--mag",
        str(request.mag),
        "--patch_size",
        str(request.patch_size),
        "--overlap",
        str(request.overlap),
        "--min_tissue_proportion",
        str(request.min_tissue_proportion),
        "--patch_encoder",
        request.patch_encoder,
        "--batch_size",
        str(request.batch_size),
    ]

    if request.max_workers is not None:
        command.extend(["--max_workers", str(request.max_workers)])
    if request.seg_batch_size is not None:
        command.extend(["--seg_batch_size", str(request.seg_batch_size)])
    if request.feat_batch_size is not None:
        command.extend(["--feat_batch_size", str(request.feat_batch_size)])
    if request.patch_encoder_ckpt_path:
        command.extend(["--patch_encoder_ckpt_path", request.patch_encoder_ckpt_path])
    if request.remove_artifacts:
        command.append("--remove_artifacts")
    if request.remove_penmarks:
        command.append("--remove_penmarks")
    if request.skip_errors:
        command.append("--skip_errors")

    log_path = job_dir / "trident_pipeline.log"
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "wsi-tox-screening-mpl"))
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

    return TridentCommandPlan(
        command=command,
        job_dir=job_dir,
        manifest=manifest,
        log_path=log_path,
        cwd=trident_root,
        env=env,
    )

