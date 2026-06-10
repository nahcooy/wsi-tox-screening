#!/usr/bin/env python3
"""
On-demand NuLite-H inference for agent-selected patches.
Reads coordinates from TRIDENT patches.h5, extracts images from WSI,
runs NuLite, and optionally computes morphometric metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/wsi-tox-screening-numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/wsi-tox-screening-mpl")

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

TYPE_COLORS = {
    "Neoplastic":  (255, 0,   0),
    "Inflammatory":(34,  221, 77),
    "Connective":  (35,  92,  236),
    "Dead":        (254, 255, 0),
    "Epithelial":  (255, 159, 68),
    "Unknown":     (180, 180, 180),
}
TYPE_NAMES = {
    1: "Neoplastic", 2: "Inflammatory", 3: "Connective", 4: "Dead", 5: "Epithelial",
    "1": "Neoplastic", "2": "Inflammatory", "3": "Connective", "4": "Dead", "5": "Epithelial",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="On-demand NuLite-H for agent-selected patches.")
    p.add_argument("--slide_id",           required=True)
    p.add_argument("--slide_path",         required=True)
    p.add_argument("--coords_h5",          required=True)
    p.add_argument("--patch_ids",          required=True, help="JSON list of patch IDs, e.g. '[40,41,45]'")
    p.add_argument("--patch_size_level0",  type=int, default=256)
    p.add_argument("--output_dir",         required=True)
    p.add_argument("--nulite_root",        required=True)
    p.add_argument("--checkpoint_path",    required=True)
    p.add_argument("--matched_dataset_csv",default=None)
    p.add_argument("--celltype_summary_csv", default=None)
    p.add_argument("--gpu",                type=int, default=0)
    p.add_argument("--batch_size",         type=int, default=8)
    return p.parse_args()


# ── helpers shared with run_nulite_topk_inference ───────────────────────────

def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if math.isfinite(result) else default


def normalize_cell_type(value: Any) -> str:
    if value in TYPE_NAMES:
        return TYPE_NAMES[value]
    try:
        numeric = int(value)
    except Exception:
        numeric = None
    if numeric in TYPE_NAMES:
        return TYPE_NAMES[numeric]
    return str(value or "Unknown")


def to_list(value: Any) -> list:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def contour_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i, pt in enumerate(points):
        nxt = points[(i + 1) % len(points)]
        area += float(pt[0]) * float(nxt[1]) - float(nxt[0]) * float(pt[1])
    return abs(area) / 2.0


def normalize_bbox(bbox: Any) -> list[list[float]]:
    flat = np.array(to_list(bbox), dtype=float).reshape(-1).tolist()
    if len(flat) < 4:
        return [[0.0, 0.0], [0.0, 0.0]]
    return [[flat[0], flat[1]], [flat[2], flat[3]]]


def build_geojson(instances: list[dict[str, Any]]) -> dict[str, Any]:
    features = []
    for inst in instances:
        contour = inst.get("contour_global") or []
        if len(contour) < 3:
            continue
        closed = contour + [contour[0]]
        cell_type = inst.get("type", "Unknown")
        color = TYPE_COLORS.get(cell_type, TYPE_COLORS["Unknown"])
        features.append({
            "type": "Feature",
            "properties": {
                "global_id": inst["global_id"],
                "patch_id": inst["patch_id"],
                "type": cell_type,
                "type_prob": inst["type_prob"],
                "area_px2": inst["area_px2"],
                "classification": {"name": cell_type, "color": list(color)},
            },
            "geometry": {"type": "Polygon", "coordinates": [closed]},
        })
    return {"type": "FeatureCollection", "features": features}


def draw_overlay(image_path: Path, overlay_path: Path, instances: list[dict[str, Any]]) -> None:
    image = Image.open(image_path).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for inst in instances:
        contour = inst.get("contour_local") or []
        if len(contour) < 3:
            continue
        cell_type = inst.get("type", "Unknown")
        color = TYPE_COLORS.get(cell_type, TYPE_COLORS["Unknown"])
        polygon = [(float(x), float(y)) for x, y in contour]
        draw.polygon(polygon, fill=(*color, 42), outline=(*color, 230))
    composed = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    composed.save(overlay_path)


def load_nulite_model(checkpoint_path: Path, device: torch.device):
    from models.nulite import NuLite
    from utils.tools import unflatten_dict
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    run_conf = unflatten_dict(checkpoint["config"], ".")
    model = NuLite(
        num_nuclei_classes=6,
        num_tissue_classes=19,
        vit_structure=run_conf["model"]["backbone"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    model.reparameterize_encoder()
    transform_settings = run_conf.get("transformations", {})
    normalize = transform_settings.get("normalize", {})
    mean = normalize.get("mean", (0.5, 0.5, 0.5))
    std  = normalize.get("std",  (0.5, 0.5, 0.5))
    transforms = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    return model, transforms


def get_cell_predictions(model, predictions: dict, magnification: int = 40) -> list[dict]:
    predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1)
    predictions["nuclei_type_map"]   = F.softmax(predictions["nuclei_type_map"],   dim=1)
    _, instance_types = model.calculate_instance_map(predictions, magnification=magnification)
    return instance_types


# ── patch extraction from WSI ────────────────────────────────────────────────

class OnDemandPatchDataset(Dataset):
    def __init__(self, patches: list[dict[str, Any]], transform: Any) -> None:
        self.patches   = patches
        self.transform = transform

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        patch = self.patches[index]
        image = Image.open(patch["image_path"]).convert("RGB")
        tensor = self.transform(image) if self.transform is not None else image
        metadata = {
            "index":       index,
            "image_path":  str(patch["image_path"]),
            "image_name":  Path(patch["image_path"]).name,
            "rank":        int(patch["patch_id"]),   # use patch_id as rank
            "patch_id":    int(patch["patch_id"]),
            "x":           int(patch["x"]),
            "y":           int(patch["y"]),
            "attention_norm": 0.0,
            "attention_raw":  0.0,
        }
        return tensor, metadata


def collate_batch(batch):
    tensors, metadata = zip(*batch)
    return torch.stack(list(tensors)), list(metadata)


def extract_patches_from_wsi(
    slide_path: Path,
    coords_h5: Path,
    patch_ids: list[int],
    patch_size_level0: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    import openslide

    with h5py.File(str(coords_h5), "r") as f:
        all_coords = f["coords"][:]   # (N, 2) — x, y at level 0

    slide  = openslide.OpenSlide(str(slide_path))
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    patches = []
    for pid in patch_ids:
        if pid < 0 or pid >= len(all_coords):
            print(f"[WARN] patch_id {pid} out of range (total {len(all_coords)}), skipping.")
            continue
        x, y = int(all_coords[pid][0]), int(all_coords[pid][1])
        region = slide.read_region((x, y), 0, (patch_size_level0, patch_size_level0))
        img = region.convert("RGB")
        if patch_size_level0 != 256:
            img = img.resize((256, 256))
        img_path = img_dir / f"patch_{pid:06d}.png"
        img.save(img_path)
        patches.append({
            "patch_id":   pid,
            "x":          x,
            "y":          y,
            "image_path": str(img_path),
        })
    slide.close()
    return patches


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args      = parse_args()
    patch_ids = json.loads(args.patch_ids)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, args.nulite_root)

    # 1. Extract patch images from WSI
    print(f"[1/4] Extracting {len(patch_ids)} patches from WSI...")
    patches = extract_patches_from_wsi(
        slide_path=Path(args.slide_path),
        coords_h5=Path(args.coords_h5),
        patch_ids=patch_ids,
        patch_size_level0=args.patch_size_level0,
        output_dir=output_dir,
    )
    if not patches:
        print("No valid patches extracted. Exiting.")
        sys.exit(1)

    # 2. Run NuLite inference
    print(f"[2/4] Running NuLite-H on {len(patches)} patches...")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, transforms = load_nulite_model(Path(args.checkpoint_path), device)
    dataset    = OnDemandPatchDataset(patches, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_batch)

    all_instances: list[dict[str, Any]] = []
    instances_by_patch: dict[int, list[dict[str, Any]]] = defaultdict(list)

    with torch.no_grad():
        for batch_tensors, batch_meta in dataloader:
            batch_tensors = batch_tensors.to(device)
            predictions   = model.forward(batch_tensors, retrieve_tokens=True)
            instance_batches = get_cell_predictions(model, predictions)

            for patch_instances, patch_meta in zip(instance_batches, batch_meta):
                image_path   = Path(patch_meta["image_path"])
                image_w, image_h = Image.open(image_path).size
                scale_x = args.patch_size_level0 / max(1, image_w)
                scale_y = args.patch_size_level0 / max(1, image_h)

                for inst_idx, cell in enumerate(patch_instances.values()):
                    cell_type = normalize_cell_type(cell.get("type", "Unknown"))
                    if cell_type == "Background":
                        continue

                    centroid_raw = np.array(to_list(cell.get("centroid")), dtype=float).reshape(-1)
                    if len(centroid_raw) < 2:
                        continue
                    centroid_local  = [finite_float(centroid_raw[0]), finite_float(centroid_raw[1])]
                    centroid_global = [
                        patch_meta["x"] + centroid_local[0] * scale_x,
                        patch_meta["y"] + centroid_local[1] * scale_y,
                    ]
                    contour_local = [
                        [finite_float(x), finite_float(y)]
                        for x, y in np.array(to_list(cell.get("contour")), dtype=float).reshape(-1, 2).tolist()
                    ]
                    contour_global = [
                        [patch_meta["x"] + x * scale_x, patch_meta["y"] + y * scale_y]
                        for x, y in contour_local
                    ]
                    bbox_local  = normalize_bbox(cell.get("bbox"))
                    bbox_global = [
                        [patch_meta["x"] + bbox_local[0][0] * scale_x, patch_meta["y"] + bbox_local[0][1] * scale_y],
                        [patch_meta["x"] + bbox_local[1][0] * scale_x, patch_meta["y"] + bbox_local[1][1] * scale_y],
                    ]
                    record = {
                        "global_id":      f"{args.slide_id}_demand_p{patch_meta['patch_id']:06d}_{inst_idx:05d}",
                        "slide_id":       args.slide_id,
                        "patch_rank":     patch_meta["patch_id"],
                        "patch_id":       patch_meta["patch_id"],
                        "patch_image":    patch_meta["image_name"],
                        "patch_x":        patch_meta["x"],
                        "patch_y":        patch_meta["y"],
                        "attention_norm": 0.0,
                        "attention_raw":  0.0,
                        "bbox_local":     bbox_local,
                        "bbox_global":    bbox_global,
                        "centroid_local":  centroid_local,
                        "centroid_global": centroid_global,
                        "contour_local":   contour_local,
                        "contour_global":  contour_global,
                        "area_px2":       contour_area(contour_global),
                        "type_prob":      finite_float(cell.get("type_prob")),
                        "type":           cell_type,
                    }
                    all_instances.append(record)
                    instances_by_patch[patch_meta["patch_id"]].append(record)

    # 3. Draw overlays
    print(f"[3/4] Drawing overlays for {len(patches)} patches...")
    overlays = []
    for patch in patches:
        pid        = patch["patch_id"]
        image_path = Path(patch["image_path"])
        overlay_path = output_dir / "overlays" / f"patch_{pid:06d}_nulite_overlay.png"
        draw_overlay(image_path, overlay_path, instances_by_patch.get(pid, []))
        overlays.append({
            "patch_id":     pid,
            "x":            patch["x"],
            "y":            patch["y"],
            "cell_count":   len(instances_by_patch.get(pid, [])),
            "type_counts":  dict(Counter(inst["type"] for inst in instances_by_patch.get(pid, []))),
            "image_path":   str(image_path),
            "overlay_path": str(overlay_path),
        })

    # 4. Save results
    print("[4/4] Saving results...")
    type_counts = Counter(inst["type"] for inst in all_instances)

    with (output_dir / "instances.jsonl").open("w", encoding="utf-8") as f:
        for inst in all_instances:
            f.write(json.dumps(inst) + "\n")

    instances_json_path = output_dir / "instances.json"
    with instances_json_path.open("w", encoding="utf-8") as f:
        json.dump(all_instances, f, indent=2)

    geojson_path = output_dir / "instances.geojson"
    with geojson_path.open("w", encoding="utf-8") as f:
        json.dump(build_geojson(all_instances), f)

    summary = {
        "slide_id":       args.slide_id,
        "patch_ids":      patch_ids,
        "num_patches":    len(patches),
        "total_nuclei":   len(all_instances),
        "type_counts":    dict(type_counts),
        "overlays":       overlays,
        "instances_json": str(instances_json_path),
        "geojson":        str(geojson_path),
    }

    # 5. Compute metrics if reference data is available
    if args.matched_dataset_csv and args.celltype_summary_csv:
        scripts_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(scripts_dir))
        from compute_nulite_patch_metrics import build_comparison, compute_patch_metrics, reference_distributions

        patch_metrics = compute_patch_metrics(all_instances)
        reference     = reference_distributions(Path(args.matched_dataset_csv), Path(args.celltype_summary_csv))
        comparison    = build_comparison(patch_metrics, reference)
        comparison.update({
            "slide_id": args.slide_id,
            "source":   "on_demand",
            "patch_ids": patch_ids,
        })
        metrics_json = output_dir / "patch_metrics.json"
        comparison_json = output_dir / "metric_comparison.json"
        with metrics_json.open("w", encoding="utf-8") as f:
            json.dump(patch_metrics, f, indent=2)
        with comparison_json.open("w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        summary["patch_metrics_json"] = str(metrics_json)
        summary["metric_comparison_json"] = str(comparison_json)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
