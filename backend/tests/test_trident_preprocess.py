import csv

from app.pipelines.trident_preprocess import build_trident_command, prepare_trident_manifest
from app.schemas.preprocess import TridentPreprocessRequest


def test_prepare_manifest_maps_svs_filename_to_local_wsi_dir(tmp_path) -> None:
    wsi_dir = tmp_path / "wsis"
    wsi_dir.mkdir()
    (wsi_dir / "123.svs").write_text("dummy", encoding="utf-8")

    dataset_csv = tmp_path / "dataset.csv"
    with dataset_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["svs_filename", "binary_label"])
        writer.writeheader()
        writer.writerow({"svs_filename": "123.svs", "binary_label": "1"})

    request = TridentPreprocessRequest(
        dataset_csv=str(dataset_csv),
        wsi_dir=str(wsi_dir),
        job_dir=str(tmp_path / "job"),
    )

    summary = prepare_trident_manifest(request, tmp_path / "job")

    assert summary.source_count == 1
    assert summary.runnable_count == 1
    assert summary.missing_count == 0
    assert "trident_wsi_list.csv" in summary.manifest_csv


def test_build_trident_command_uses_requested_pipeline(tmp_path) -> None:
    trident_root = tmp_path / "TRIDENT"
    trident_root.mkdir()
    (trident_root / "run_batch_of_slides.py").write_text("print('ok')", encoding="utf-8")

    trident_python = tmp_path / "python"
    trident_python.write_text("", encoding="utf-8")

    wsi_dir = tmp_path / "wsis"
    wsi_dir.mkdir()
    (wsi_dir / "123.svs").write_text("dummy", encoding="utf-8")

    request = TridentPreprocessRequest(
        slide_path="/original/path/123.svs",
        wsi_dir=str(wsi_dir),
        job_dir=str(tmp_path / "job"),
        trident_root=str(trident_root),
        trident_python=str(trident_python),
    )

    plan = build_trident_command(request)

    assert "--task" in plan.command
    assert "all" in plan.command
    assert "--segmenter" in plan.command
    assert "grandqc" in plan.command
    assert "--remove_artifacts" in plan.command
    assert "--patch_encoder" in plan.command
    assert "uni_v1" in plan.command
    assert plan.manifest.runnable_count == 1

