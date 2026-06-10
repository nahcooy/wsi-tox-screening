from pydantic import BaseModel, Field

from app.settings import settings


class ABMILInferenceRequest(BaseModel):
    slide_id: str | None = None
    slide_path: str | None = None
    feature_h5: str | None = None
    coords_h5: str | None = None
    checkpoint_path: str = Field(
        default=str(settings.default_abmil_checkpoint),
        description="ABMIL checkpoint trained by grandqc_univ1_abmil_260310_ver3.py.",
    )
    output_dir: str | None = None
    trident_job_dir: str | None = None
    mil_lab_root: str = Field(default=str(settings.mil_lab_root))
    mil_python: str = Field(default=str(settings.mil_python))
    top_k: int = 25
    thumbnail_max_size: int = 6000
    device: str = "auto"


class ABMILInferencePlanResponse(BaseModel):
    command: list[str]
    output_dir: str
    feature_h5: str
    slide_path: str | None = None
    checkpoint_path: str
    expected_outputs: dict[str, str]


class ABMILInferenceJobResponse(ABMILInferencePlanResponse):
    job_id: str
    status: str
    log_path: str
