#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw
import torch


CLASS_ORDER = ["normal", "abnormal"]
DEFAULT_THUMBNAIL_MAX_SIZE = 6000
SPECTRAL_HIGH_RED = [
    (0.0, (94, 79, 162)),
    (0.1, (50, 136, 189)),
    (0.2, (102, 194, 165)),
    (0.3, (171, 221, 164)),
    (0.4, (230, 245, 152)),
    (0.5, (255, 255, 191)),
    (0.6, (254, 224, 139)),
    (0.7, (253, 174, 97)),
    (0.8, (244, 109, 67)),
    (0.9, (213, 62, 79)),
    (1.0, (158, 1, 66)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ABMIL inference and attention export.")
    parser.add_argument("--feature_h5", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mil_lab_root", default="/home/nahcooy/MIL/MIL_260527/MIL-Lab")
    parser.add_argument("--slide_path", default=None)
    parser.add_argument("--slide_id", default=None)
    parser.add_argument("--coords_h5", default=None)
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--thumbnail_max_size", type=int, default=DEFAULT_THUMBNAIL_MAX_SIZE)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_arg}, but CUDA is not available.")
    return torch.device(device_arg)


def load_features(feature_h5: Path, coords_h5: Path | None) -> tuple[np.ndarray, np.ndarray | None, dict]:
    with h5py.File(feature_h5, "r") as f:
        if "features" not in f:
            raise KeyError(f"features dataset not found in {feature_h5}")
        features = f["features"][:]
        coords = f["coords"][:] if "coords" in f else None
        attrs = dict(f["coords"].attrs) if "coords" in f else {}

    if coords is None and coords_h5 is not None:
        with h5py.File(coords_h5, "r") as f:
            coords = f["coords"][:]
            attrs = dict(f["coords"].attrs)

    if coords is not None and len(coords) != len(features):
        raise ValueError(
            f"coords/features length mismatch: coords={len(coords)} features={len(features)}"
        )
    return features, coords, attrs


def stable_softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    values = values - np.nanmax(values)
    exp = np.exp(values)
    denom = exp.sum()
    if denom <= 0 or not np.isfinite(denom):
        return np.ones_like(values, dtype=np.float64) / max(1, len(values))
    return exp / denom


def percentile_normalize(values: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float64)
    clipped = values.copy().astype(np.float64)
    p_lo, p_hi = np.percentile(clipped[finite], [lo, hi])
    if math.isclose(float(p_lo), float(p_hi)):
        return np.zeros_like(values, dtype=np.float64)
    clipped = np.clip(clipped, p_lo, p_hi)
    return (clipped - p_lo) / (p_hi - p_lo)


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def spectral_high_red(score: float) -> tuple[int, int, int]:
    score = clamp01(score)
    for index in range(1, len(SPECTRAL_HIGH_RED)):
        left_pos, left_rgb = SPECTRAL_HIGH_RED[index - 1]
        right_pos, right_rgb = SPECTRAL_HIGH_RED[index]
        if score <= right_pos:
            span = right_pos - left_pos
            t = 0.0 if span <= 0 else (score - left_pos) / span
            return tuple(
                int(round(left_rgb[channel] + t * (right_rgb[channel] - left_rgb[channel])))
                for channel in range(3)
            )
    return SPECTRAL_HIGH_RED[-1][1]


def load_model(mil_lab_root: Path, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(mil_lab_root))
    from src.builder import create_model

    model = create_model("abmil", num_classes=2).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def infer(
    model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    with torch.no_grad():
        results_dict, log_dict = model(tensor, loss_fn=None, label=None, return_attention=True)
    logits = results_dict["logits"].detach().cpu().numpy().squeeze(0)
    attention = log_dict["attention"].detach().cpu().numpy().squeeze()
    if attention.ndim > 1:
        attention = attention.reshape(-1, attention.shape[-1])[0]
    return logits, attention


def write_attention_csv(
    path: Path,
    coords: np.ndarray | None,
    attention_raw: np.ndarray,
    attention_softmax: np.ndarray,
    attention_norm: np.ndarray,
) -> list[dict]:
    order = np.argsort(-attention_norm)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)

    records: list[dict] = []
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "patch_id",
            "rank",
            "x",
            "y",
            "attention_raw",
            "attention_softmax",
            "attention_norm",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(attention_raw)):
            x, y = (None, None)
            if coords is not None:
                x, y = int(coords[i][0]), int(coords[i][1])
            rec = {
                "patch_id": i,
                "rank": int(ranks[i]),
                "x": x,
                "y": y,
                "attention_raw": float(attention_raw[i]),
                "attention_softmax": float(attention_softmax[i]),
                "attention_norm": float(attention_norm[i]),
            }
            writer.writerow(rec)
            records.append(rec)
    return records


def patch_size_level0(attrs: dict, default: int = 256) -> int:
    value = attrs.get("patch_size_level0", attrs.get("patch_size", default))
    try:
        return int(value)
    except Exception:
        return default


def write_geojson(path: Path, records: list[dict], patch_size: int) -> None:
    features = []
    for rec in records:
        if rec["x"] is None or rec["y"] is None:
            continue
        x = int(rec["x"])
        y = int(rec["y"])
        poly = [
            [x, y],
            [x + patch_size, y],
            [x + patch_size, y + patch_size],
            [x, y + patch_size],
            [x, y],
        ]
        score = clamp01(float(rec["attention_norm"]))
        color = spectral_high_red(score)
        color_bin = int(round(score * 10))
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "patch_id": rec["patch_id"],
                    "rank": rec["rank"],
                    "attention_raw": rec["attention_raw"],
                    "attention_softmax": rec["attention_softmax"],
                    "attention_norm": rec["attention_norm"],
                    "colormap": "Spectral_r_high_red",
                    "color_rgb": list(color),
                    "classification": {
                        "name": f"ABMIL attention {color_bin:02d}",
                        "color": list(color),
                    },
                },
                "geometry": {"type": "Polygon", "coordinates": [poly]},
            }
        )
    with path.open("w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)


def open_slide(slide_path: str | None):
    if not slide_path:
        return None
    try:
        import openslide

        return openslide.OpenSlide(slide_path)
    except Exception:
        return None


def make_thumbnail(slide, attrs: dict, max_size: int) -> tuple[Image.Image, tuple[int, int]]:
    if slide is not None:
        width, height = slide.dimensions
        scale = min(max_size / max(width, height), 1.0)
        thumb_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        thumb = slide.get_thumbnail(thumb_size).convert("RGB")
        return thumb, (width, height)

    width = int(attrs.get("level0_width", 4000))
    height = int(attrs.get("level0_height", 3000))
    scale = min(max_size / max(width, height), 1.0)
    thumb_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return Image.new("RGB", thumb_size, (245, 238, 235)), (width, height)


def write_heatmap_thumbnail(
    path: Path,
    slide,
    attrs: dict,
    records: list[dict],
    patch_size: int,
    max_size: int,
) -> None:
    thumb, (width, height) = make_thumbnail(slide, attrs, max_size)
    sx = thumb.width / max(1, width)
    sy = thumb.height / max(1, height)
    overlay = Image.new("RGBA", thumb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for rec in records:
        if rec["x"] is None or rec["y"] is None:
            continue
        score = clamp01(float(rec["attention_norm"]))
        red, green, blue = spectral_high_red(score)
        alpha = int(80 + 155 * score)
        x0 = int(rec["x"] * sx)
        y0 = int(rec["y"] * sy)
        x1 = int((rec["x"] + patch_size) * sx)
        y1 = int((rec["y"] + patch_size) * sy)
        rect = [x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)]
        draw.rectangle(rect, fill=(red, green, blue, alpha))
        draw.rectangle(rect, outline=(red, green, blue, min(255, alpha + 20)))
    Image.alpha_composite(thumb.convert("RGBA"), overlay).convert("RGB").save(path)


def crop_top_patches(
    out_dir: Path,
    slide,
    records: list[dict],
    patch_size: int,
    top_k: int,
) -> list[dict]:
    top_dir = out_dir / "topk"
    top_dir.mkdir(parents=True, exist_ok=True)
    top = sorted(records, key=lambda r: r["rank"])[:top_k]
    manifest = []

    for rec in top:
        rank = int(rec["rank"])
        image_name = f"rank_{rank:03d}_patch_{int(rec['patch_id']):06d}.png"
        image_path = top_dir / image_name
        if slide is not None and rec["x"] is not None and rec["y"] is not None:
            patch = slide.read_region((int(rec["x"]), int(rec["y"])), 0, (patch_size, patch_size))
            patch.convert("RGB").resize((256, 256)).save(image_path)
        else:
            img = Image.new("RGB", (256, 256), (245, 238, 235))
            draw = ImageDraw.Draw(img)
            draw.text((12, 12), f"rank {rank}", fill=(20, 20, 20))
            img.save(image_path)
        item = dict(rec)
        item["image_path"] = str(image_path)
        manifest.append(item)

    with (top_dir / "top25_patches.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    args = parse_args()
    feature_h5 = Path(args.feature_h5)
    checkpoint_path = Path(args.checkpoint_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slide_id = args.slide_id or feature_h5.stem
    coords_h5 = Path(args.coords_h5) if args.coords_h5 else None
    device = choose_device(args.device)

    print("[PROGRESS] 0%")
    features, coords, attrs = load_features(feature_h5, coords_h5)
    print(f"[PROGRESS] 10% — {features.shape[0]} patches loaded")

    model = load_model(Path(args.mil_lab_root), checkpoint_path, device)
    print("[PROGRESS] 20%")

    logits, attention_raw = infer(model, features, device)
    print("[PROGRESS] 40% — ABMIL inference done")

    cls_softmax = stable_softmax(logits)
    attention_softmax = stable_softmax(attention_raw)
    attention_norm = percentile_normalize(attention_raw)

    records = write_attention_csv(
        output_dir / "attention_scores.csv",
        coords,
        attention_raw,
        attention_softmax,
        attention_norm,
    )
    print("[PROGRESS] 60% — attention scores saved")

    patch_size = patch_size_level0(attrs)
    write_geojson(output_dir / "attention_heatmap_qupath.geojson", records, patch_size)

    slide = open_slide(args.slide_path)
    slide_width, slide_height = slide.dimensions if slide is not None else (None, None)

    # 깨끗한 WSI 썸네일 (히트맵 없이)
    wsi_thumb, _ = make_thumbnail(slide, attrs, args.thumbnail_max_size)
    wsi_thumb.save(output_dir / "wsi_thumbnail.png")

    write_heatmap_thumbnail(
        output_dir / "attention_heatmap_thumbnail.png",
        slide,
        attrs,
        records,
        patch_size,
        args.thumbnail_max_size,
    )
    print("[PROGRESS] 80% — heatmap thumbnail saved")

    topk = crop_top_patches(output_dir, slide, records, patch_size, args.top_k)
    if slide is not None:
        slide.close()
    print("[PROGRESS] 90% — top-k patches cropped")

    prediction_index = int(np.argmax(cls_softmax))
    result = {
        "slide_id": slide_id,
        "model_type": "abmil",
        "encoder": "uni_v1",
        "checkpoint_path": str(checkpoint_path),
        "feature_h5": str(feature_h5),
        "slide_path": args.slide_path,
        "class_order": CLASS_ORDER,
        "logits": [float(x) for x in logits.tolist()],
        "softmax": {label: float(cls_softmax[i]) for i, label in enumerate(CLASS_ORDER)},
        "prediction": CLASS_ORDER[prediction_index],
        "confidence_score": float(cls_softmax[prediction_index]),
        "abnormal_confidence_score": float(cls_softmax[1]),
        "num_patches": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "patch_size_level0": int(patch_size),
        "slide_width": int(slide_width) if slide_width is not None else None,
        "slide_height": int(slide_height) if slide_height is not None else None,
        "attention_colormap": "Spectral_r_high_red",
        "attention_normalization": {
            "method": "percentile_clip_minmax",
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
        },
        "thumbnail_max_size": int(args.thumbnail_max_size),
        "attention_outputs": {
            "csv": str(output_dir / "attention_scores.csv"),
            "geojson": str(output_dir / "attention_heatmap_qupath.geojson"),
            "thumbnail": str(output_dir / "attention_heatmap_thumbnail.png"),
        },
        "top_k": topk,
        "mock": False,
    }
    with (output_dir / "mil_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("[PROGRESS] 100% — results saved")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
