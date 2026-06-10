from app.api.routes_mock import health, register_slide
from app.schemas.slide import SlideRegisterRequest


def test_health_endpoint() -> None:
    response = health()

    assert response["status"] == "ok"
    assert response["mock"] is True


def test_register_slide_mock_metadata() -> None:
    response = register_slide(
        SlideRegisterRequest(slide_path="/data/example.svs", species="rat", organ="liver")
    )

    body = response.model_dump()
    assert body["slide_id"] == "example"
    assert body["mock"] is True
    assert body["level_count"] == 4
