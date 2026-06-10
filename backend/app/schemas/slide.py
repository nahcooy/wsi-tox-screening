from pydantic import BaseModel, Field


class SlideRegisterRequest(BaseModel):
    slide_path: str = Field(..., min_length=1)
    slide_id: str | None = None
    species: str = "rat"
    organ: str = "liver"
    stain: str = "H&E"


class SlideMetadata(BaseModel):
    slide_id: str
    slide_path: str
    width: int
    height: int
    level_count: int
    level_dimensions: list[tuple[int, int]]
    mpp_x: float | None = None
    mpp_y: float | None = None
    objective_power: float | None = None
    stain: str = "H&E"
    species: str = "rat"
    organ: str = "liver"
    mock: bool = True

