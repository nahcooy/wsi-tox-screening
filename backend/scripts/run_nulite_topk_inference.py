#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/wsi-tox-screening-numba-cache")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


TYPE_COLORS = {
    "Neoplastic": (255, 0, 0),
    "Inflammatory": (34, 221, 77),
    "Connective": (35, 92, 236),
    "Dead": (254, 255, 0),
    "Epithelial": (255, 159, 68),
    "Unknown": (180, 180, 180),
}
TYPE_NAMES = {
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
    "1": "Neoplastic",
    "2": "Inflammatory",
    "3": "Connective",
    "4": "Dead",
    "5": "Epithelial",
    "Background": "Background",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NuLite-H on ABMIL top-k patch crops.")
    parser.add_argument("--topk_manifest", required=True)
    parser.add_argument("--mil_result", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--nulite_root", default="/home/nahcooy/NK/NL/NuLite_patch_wise_inference")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--slide_id", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--matched_dataset_csv", default=None)
    parser.add_argument("--celltype_summary_csv", default=None)
    return parser.parse_args()


class TopKPatchDataset(Dataset):
    def __init__(self, patches: list[dict[str, Any]], transform: Any) -> None:
        self.patches = patches
        self.transform = transform

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        patch = self.patches[index]
        image_path = Path(patch["image_path"])
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image) if self.transform is not None else image
        metadata = {
            "index": index,
            "image_path": str(image_path),
            "image_name": image_path.name,
            "rank": int(patch.get("rank", index + 1)),
            "patch_id": int(patch.get("patch_id", index)),
            "x": int(patch.get("x") or 0),
            "y": int(patch.get("y") or 0),
            "attention_norm": float(patch.get("attention_norm") or 0.0),
            "attention_raw": float(patch.get("attention_raw") or 0.0),
        }
        return tensor, metadata


def collate_batch(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    tensors, metadata = zip(*batch)
    return torch.stack(list(tensors)), list(metadata)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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
    for index, point in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        area += float(point[0]) * float(nxt[1]) - float(nxt[0]) * float(point[1])
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
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "global_id": inst["global_id"],
                    "patch_rank": inst["patch_rank"],
                    "patch_id": inst["patch_id"],
                    "type": cell_type,
                    "type_prob": inst["type_prob"],
                    "area_px2": inst["area_px2"],
                    "classification": {
                        "name": cell_type,
                        "color": list(color),
                    },
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [closed],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


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
    std = normalize.get("std", (0.5, 0.5, 0.5))
    transforms = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    return model, transforms


def get_cell_predictions(model, predictions: dict, magnification: int = 40) -> list[dict]:
    predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1)
    predictions["nuclei_type_map"] = F.softmax(predictions["nuclei_type_map"], dim=1)
    _, instance_types = model.calculate_instance_map(predictions, magnification=magnification)
    return instance_types


def draw_overlay(
    image_path: Path,
    overlay_path: Path,
    instances: list[dict[str, Any]],
) -> None:
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


def main() -> None:
    args = parse_args()
    topk_manifest = Path(args.topk_manifest)
    mil_result_path = Path(args.mil_result)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, args.nulite_root)

    topk = load_json(topk_manifest)
    mil_result = load_json(mil_result_path)
    patch_size_level0 = int(mil_result.get("patch_size_level0") or 256)
    selected = [patch for patch in topk if patch.get("image_path") and Path(patch["image_path"]).exists()]
    if not selected:
        raise FileNotFoundError(f"No readable top-k patch images found in {topk_manifest}")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, inference_transforms = load_nulite_model(Path(args.checkpoint_path), device)
    dataset = TopKPatchDataset(selected, transform=inference_transforms)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    all_instances: list[dict[str, Any]] = []
    instances_by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    n_patches = len(selected)
    _progress_milestones = {max(1, int(n_patches * p)) for p in (0.25, 0.5, 0.75)} if n_patches > 0 else set()
    _processed = 0
    print(f"[PROGRESS] 0% — NuLite inference on {n_patches} patches")

    with torch.no_grad():
        for patches, metadata in dataloader:
            patches = patches.to(device)
            predictions = model.forward(patches, retrieve_tokens=True)
            instance_batches = get_cell_predictions(model, predictions)
            for patch_instances, patch_meta in zip(instance_batches, metadata):
                image_path = Path(patch_meta["image_path"])
                image_width, image_height = Image.open(image_path).size
                scale_x = patch_size_level0 / max(1, image_width)
                scale_y = patch_size_level0 / max(1, image_height)

                patch_records = []
                for instance_index, cell in enumerate(patch_instances.values()):
                    cell_type = normalize_cell_type(cell.get("type", "Unknown"))
                    if cell_type == "Background":
                        continue

                    centroid_local_raw = np.array(to_list(cell.get("centroid")), dtype=float).reshape(-1)
                    if len(centroid_local_raw) < 2:
                        continue
                    centroid_local = [
                        finite_float(centroid_local_raw[0]),
                        finite_float(centroid_local_raw[1]),
                    ]
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
                    bbox_local = normalize_bbox(cell.get("bbox"))
                    bbox_global = [
                        [patch_meta["x"] + bbox_local[0][0] * scale_x, patch_meta["y"] + bbox_local[0][1] * scale_y],
                        [patch_meta["x"] + bbox_local[1][0] * scale_x, patch_meta["y"] + bbox_local[1][1] * scale_y],
                    ]

                    record = {
                        "global_id": f"{args.slide_id}_rank{patch_meta['rank']:03d}_{instance_index:05d}",
                        "slide_id": args.slide_id,
                        "patch_rank": patch_meta["rank"],
                        "patch_id": patch_meta["patch_id"],
                        "patch_image": patch_meta["image_name"],
                        "patch_x": patch_meta["x"],
                        "patch_y": patch_meta["y"],
                        "attention_norm": patch_meta["attention_norm"],
                        "attention_raw": patch_meta["attention_raw"],
                        "bbox_local": bbox_local,
                        "bbox_global": bbox_global,
                        "centroid_local": centroid_local,
                        "centroid_global": centroid_global,
                        "contour_local": contour_local,
                        "contour_global": contour_global,
                        "area_px2": contour_area(contour_global),
                        "type_prob": finite_float(cell.get("type_prob")),
                        "type": cell_type,
                    }
                    all_instances.append(record)
                    patch_records.append(record)
                    instances_by_rank[patch_meta["rank"]].append(record)

                patch_json = output_dir / "per_patch" / f"rank_{patch_meta['rank']:03d}.json"
                save_json(patch_json, patch_records)

                _processed += 1
                if n_patches > 0 and _processed in _progress_milestones:
                    pct = round(_processed / n_patches * 50)  # inference = 0-50%
                    print(f"[PROGRESS] {pct}% — {_processed}/{n_patches} patches done")

    print(f"[PROGRESS] 50% — NuLite inference complete, drawing overlays")
    overlays = []
    _overlay_milestones = {max(1, int(n_patches * p)) for p in (0.25, 0.5, 0.75, 1.0)} if n_patches > 0 else set()
    for i, patch in enumerate(selected):
        rank = int(patch.get("rank", 0))
        image_path = Path(patch["image_path"])
        overlay_path = output_dir / "overlays" / f"rank_{rank:03d}_nulite_overlay.png"
        draw_overlay(image_path, overlay_path, instances_by_rank.get(rank, []))
        overlays.append(
            {
                "rank": rank,
                "patch_id": int(patch.get("patch_id", rank)),
                "attention_norm": float(patch.get("attention_norm") or 0.0),
                "image_path": str(image_path),
                "overlay_path": str(overlay_path),
                "cell_count": len(instances_by_rank.get(rank, [])),
                "type_counts": dict(Counter(inst["type"] for inst in instances_by_rank.get(rank, []))),
            }
        )
        if n_patches > 0 and (i + 1) in _overlay_milestones:
            pct = 50 + round((i + 1) / n_patches * 50)  # overlay = 50-100%
            print(f"[PROGRESS] {pct}% — overlay {i+1}/{n_patches} done")

    type_counts = Counter(inst["type"] for inst in all_instances)
    with (output_dir / "all_instances.jsonl").open("w", encoding="utf-8") as f:
        for inst in all_instances:
            f.write(json.dumps(inst) + "\n")
    save_json(output_dir / "all_instances.json", all_instances)
    save_json(output_dir / "nuclei_instances.geojson", build_geojson(all_instances))
    save_json(output_dir / "cell_type_counts.json", dict(type_counts))

    with (output_dir / "cell_type_counts.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "count"])
        writer.writeheader()
        for cell_type, count in sorted(type_counts.items()):
            writer.writerow({"type": cell_type, "count": count})

    summary = {
        "slide_id": args.slide_id,
        "model": "NuLite-H",
        "mock": False,
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "topk_manifest": str(topk_manifest.resolve()),
        "patch_size_level0": patch_size_level0,
        "num_patches": len(selected),
        "total_nuclei": len(all_instances),
        "type_counts": dict(type_counts),
        "overlays": overlays,
        "outputs": {
            "all_instances_json": str(output_dir / "all_instances.json"),
            "all_instances_jsonl": str(output_dir / "all_instances.jsonl"),
            "geojson": str(output_dir / "nuclei_instances.geojson"),
            "cell_type_counts_csv": str(output_dir / "cell_type_counts.csv"),
            "cell_type_counts_json": str(output_dir / "cell_type_counts.json"),
        },
    }
    save_json(output_dir / "nuclei_summary.json", summary)
    if args.matched_dataset_csv and args.celltype_summary_csv:
        scripts_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(scripts_dir))
        from compute_nulite_patch_metrics import build_comparison, compute_patch_metrics, reference_distributions

        patch_metrics = compute_patch_metrics(all_instances)
        reference = reference_distributions(Path(args.matched_dataset_csv), Path(args.celltype_summary_csv))
        comparison = build_comparison(patch_metrics, reference)
        comparison.update(
            {
                "slide_id": args.slide_id,
                "mapping": {
                    "Neoplastic": "Hep",
                    "Epithelial": "Hep if area_um2 >= 30 else NPC",
                    "Connective": "NPC",
                    "Inflammatory": "Imm",
                    "Dead": "excluded",
                    "Background": "excluded",
                },
                "reference": {
                    "matched_dataset_csv": str(Path(args.matched_dataset_csv).resolve()),
                    "celltype_summary_csv": str(Path(args.celltype_summary_csv).resolve()),
                    "case_n": int((reference["Label"] == 1).sum()),
                    "control_n": int((reference["Label"] == 0).sum()),
                },
            }
        )
        save_json(output_dir / "patch_metrics.json", patch_metrics)
        save_json(output_dir / "metric_comparison.json", comparison)
        with (output_dir / "patch_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({key for row in patch_metrics for key in row.keys()}))
            writer.writeheader()
            writer.writerows(patch_metrics)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
