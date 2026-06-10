from app.pipelines.abmil_inference import build_abmil_command
from app.schemas.mil import ABMILInferenceRequest


def test_build_abmil_command_from_feature_h5(tmp_path) -> None:
    feature_h5 = tmp_path / "123.h5"
    feature_h5.write_bytes(b"placeholder")

    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"dummy")

    mil_python = tmp_path / "python"
    mil_python.write_text("", encoding="utf-8")

    request = ABMILInferenceRequest(
        feature_h5=str(feature_h5),
        checkpoint_path=str(checkpoint),
        output_dir=str(tmp_path / "out"),
        mil_python=str(mil_python),
    )

    plan = build_abmil_command(request)

    assert "--feature_h5" in plan.command
    assert str(feature_h5) in plan.command
    assert "--checkpoint_path" in plan.command
    assert str(checkpoint.resolve()) in plan.command
    assert plan.expected_outputs["qupath_geojson"].endswith("attention_heatmap_qupath.geojson")
