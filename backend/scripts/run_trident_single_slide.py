#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRIDENT preprocessing for one copied WSI.")
    parser.add_argument("--slide_path", required=True)
    parser.add_argument("--job_dir", required=True)
    parser.add_argument("--trident_root", default="/home/nahcooy/MIL/TRIDENT")
    parser.add_argument("--patch_encoder", default="uni_v1")
    parser.add_argument("--patch_encoder_ckpt_path", default=None)
    parser.add_argument("--mag", type=int, default=20)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--min_tissue_proportion", type=float, default=0.0)
    parser.add_argument("--segmenter", default="grandqc")
    parser.add_argument("--seg_conf_thresh", type=float, default=0.5)
    parser.add_argument("--remove_artifacts", action="store_true")
    parser.add_argument("--remove_penmarks", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--feat_batch_size", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(device_arg: str, gpu: int) -> str:
    if device_arg != "auto":
        if device_arg.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device_arg}, but CUDA is not available.")
        return device_arg
    return f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = parse_args()
    trident_root = Path(args.trident_root)
    sys.path.insert(0, str(trident_root))

    from trident import load_wsi
    from trident.patch_encoder_models import encoder_factory
    from trident.segmentation_models import segmentation_model_factory

    slide_path = Path(args.slide_path)
    job_dir = Path(args.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device, args.gpu)
    coords_dir = f"{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap"
    coords_root = job_dir / coords_dir

    config = {
        "slide_path": str(slide_path),
        "job_dir": str(job_dir),
        "device": device,
        "segmenter": args.segmenter,
        "remove_artifacts": args.remove_artifacts,
        "mag": args.mag,
        "patch_size": args.patch_size,
        "overlap": args.overlap,
        "patch_encoder": args.patch_encoder,
    }
    with (job_dir / "_config_workflow_single_slide.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"[TRIDENT-WORKFLOW] slide={slide_path}")
    print(f"[TRIDENT-WORKFLOW] job_dir={job_dir}")
    print(f"[TRIDENT-WORKFLOW] device={device}")

    slide = load_wsi(slide_path=str(slide_path), lazy_init=False)

    print("[PROGRESS] 0%")
    print("[1/3] GrandQC tissue segmentation")
    segmentation_model = segmentation_model_factory(
        args.segmenter,
        confidence_thresh=args.seg_conf_thresh,
    )
    slide.segment_tissue(
        segmentation_model=segmentation_model,
        target_mag=segmentation_model.target_mag,
        job_dir=str(job_dir),
        device=device,
        holes_are_tissue=True,
        batch_size=args.batch_size,
    )
    print("[PROGRESS] 25%")

    if args.remove_artifacts or args.remove_penmarks:
        print("[1b/3] GrandQC artifact segmentation")
        artifact_model = segmentation_model_factory(
            "grandqc_artifact",
            remove_penmarks_only=args.remove_penmarks and not args.remove_artifacts,
        )
        slide.segment_tissue(
            segmentation_model=artifact_model,
            target_mag=artifact_model.target_mag,
            holes_are_tissue=False,
            job_dir=str(job_dir),
            device=device,
            batch_size=args.batch_size,
        )

    print("[2/3] 20x 256x256 no-overlap patch coordinate extraction")
    coords_path = slide.extract_tissue_coords(
        target_mag=args.mag,
        patch_size=args.patch_size,
        save_coords=str(coords_root),
        overlap=args.overlap,
        min_tissue_proportion=args.min_tissue_proportion,
    )
    slide.visualize_coords(
        coords_path=coords_path,
        save_patch_viz=str(coords_root / "visualization"),
    )
    print("[PROGRESS] 50%")

    print("[3/3] UNI feature extraction")
    encoder = encoder_factory(args.patch_encoder, weights_path=args.patch_encoder_ckpt_path)
    encoder.eval()
    encoder.to(device)
    features_dir = coords_root / f"features_{args.patch_encoder}"
    slide.extract_patch_features(
        patch_encoder=encoder,
        coords_path=coords_path,
        save_features=str(features_dir),
        device=device,
        saveas="h5",
        batch_limit=args.feat_batch_size,
    )
    print("[PROGRESS] 75%")

    try:
        slide.release()
    except Exception:
        pass

    print("[PROGRESS] 100%")
    print("[DONE] TRIDENT single-slide preprocessing completed.")


if __name__ == "__main__":
    main()
