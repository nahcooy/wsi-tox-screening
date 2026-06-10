from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.settings import settings


class TridentPreprocessRequest(BaseModel):
    dataset_csv: str | None = Field(
        default=str(settings.default_dataset_csv),
        description="CSV containing svs_filename, absolute_path, or wsi column.",
    )
    slide_path: str | None = Field(
        default=None,
        description="Optional single slide path. If provided, dataset_csv is ignored.",
    )
    wsi_dir: str = Field(default=str(settings.default_wsi_dir))
    job_dir: str | None = Field(
        default=None,
        description="TRIDENT output directory. Defaults to outputs/trident_runs/spass_2000.",
    )
    task: Literal["seg", "coords", "feat", "all"] = "all"
    segmenter: Literal["grandqc", "hest"] = "grandqc"
    remove_artifacts: bool = True
    remove_penmarks: bool = False
    mag: int = 20
    patch_size: int = 256
    overlap: int = 0
    min_tissue_proportion: float = 0.0
    patch_encoder: str = "uni_v1"
    patch_encoder_ckpt_path: str | None = None
    gpu: int = 0
    batch_size: int = 64
    seg_batch_size: int | None = None
    feat_batch_size: int | None = 512
    max_workers: int | None = 8
    skip_errors: bool = True
    allow_missing_slides: bool = False
    trident_python: str = Field(default=str(settings.trident_python))
    trident_root: str = Field(default=str(settings.trident_root))


class TridentManifestSummary(BaseModel):
    manifest_csv: str
    source_count: int
    runnable_count: int
    missing_count: int
    missing_examples: list[str] = []


class TridentPreprocessJobResponse(BaseModel):
    job_id: str
    status: str
    command: list[str]
    job_dir: str
    manifest: TridentManifestSummary
    log_path: str


class TridentPreprocessPlanResponse(BaseModel):
    command: list[str]
    job_dir: str
    manifest: TridentManifestSummary
    log_path: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    command: list[str]
    cwd: str
    log_path: str
    returncode: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    metadata: dict = {}
