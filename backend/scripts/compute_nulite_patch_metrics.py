#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy import stats

MPP = 0.496
PIXEL_TO_UM2 = MPP ** 2
MIN_AREA_UM2 = 12.0
MAX_AREA_UM2 = 160.0
HEPATOCYTE_MIN_AREA = 30.0

REQUESTED_METRICS = [
    "Hep_Area_Mean",
    "Hep_Area_Median",
    "Hep_Area_P90",
    "Hep_Solidity_Mean",
    "Hep_Circularity_Mean",
    "Hep_Convexity_Mean",
    "Hep_AspectRatio_Mean",
    "NPC_Area_Mean",
    "NPC_Circularity_Mean",
    "Imm_Area_Mean",
    "Imm_Circularity_Mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute patch-wise NuLite metrics and reference comparison.")
    parser.add_argument("--instances_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--matched_dataset_csv", required=True)
    parser.add_argument("--celltype_summary_csv", required=True)
    parser.add_argument("--slide_id", required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def safe_stats(values: list[float], prefix: str) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    out: dict[str, float | int | None] = {f"{prefix}_N": int(arr.size)}
    keys = ["Mean", "Median", "SD", "CV", "P10", "P25", "P75", "P90", "Skewness", "Kurtosis"]
    if arr.size < 4:
        for key in keys:
            out[f"{prefix}_{key}"] = None
        return out
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    out.update(
        {
            f"{prefix}_Mean": mean,
            f"{prefix}_Median": float(np.median(arr)),
            f"{prefix}_SD": sd,
            f"{prefix}_CV": sd / mean if mean != 0 else None,
            f"{prefix}_P10": float(np.percentile(arr, 10)),
            f"{prefix}_P25": float(np.percentile(arr, 25)),
            f"{prefix}_P75": float(np.percentile(arr, 75)),
            f"{prefix}_P90": float(np.percentile(arr, 90)),
            f"{prefix}_Skewness": float(stats.skew(arr, bias=False)),
            f"{prefix}_Kurtosis": float(stats.kurtosis(arr, fisher=False, bias=False)),
        }
    )
    return out


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def contour_metrics(points: list[list[float]]) -> dict[str, float] | None:
    if len(points) < 3:
        return None
    contour = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
    area_px2 = float(cv2.contourArea(contour))
    area_um2 = area_px2 * PIXEL_TO_UM2
    if not (MIN_AREA_UM2 <= area_um2 <= MAX_AREA_UM2):
        return None
    perim_px = float(cv2.arcLength(contour, True))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    hull_perim = float(cv2.arcLength(hull, True))
    solidity = area_px2 / hull_area if hull_area > 0 else None
    circularity = (4.0 * math.pi * area_px2) / (perim_px**2) if perim_px > 0 else None
    convexity = hull_perim / perim_px if perim_px > 0 else None
    if len(contour) >= 5:
        (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        long_axis = max(float(axis_a), float(axis_b))
        short_axis = min(float(axis_a), float(axis_b))
        aspect_ratio = short_axis / long_axis if long_axis > 0 else None
    else:
        (_, _), (w, h), _ = cv2.minAreaRect(contour)
        long_axis = max(float(w), float(h))
        short_axis = min(float(w), float(h))
        aspect_ratio = short_axis / long_axis if long_axis > 0 else None
    return {
        "Area": area_um2,
        "Solidity": solidity,
        "Circularity": circularity,
        "Convexity": convexity,
        "AspectRatio": aspect_ratio,
    }


def map_to_liver_group(cell_type: str, area_um2: float) -> str | None:
    if cell_type == "Neoplastic":
        return "Hep"
    if cell_type == "Epithelial":
        return "Hep" if area_um2 >= HEPATOCYTE_MIN_AREA else "NPC"
    if cell_type == "Connective":
        return "NPC"
    if cell_type == "Inflammatory":
        return "Imm"
    return None


def compute_patch_metrics(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_patch: dict[int, list[dict[str, Any]]] = {}
    for inst in instances:
        rank = int(inst.get("patch_rank", 0))
        by_patch.setdefault(rank, []).append(inst)

    rows = []
    for rank in sorted(by_patch):
        patch_instances = by_patch[rank]
        groups = {"Hep": {k: [] for k in ["Area", "Solidity", "Circularity", "Convexity", "AspectRatio"]},
                  "NPC": {k: [] for k in ["Area", "Solidity", "Circularity", "Convexity", "AspectRatio"]},
                  "Imm": {k: [] for k in ["Area", "Solidity", "Circularity", "Convexity", "AspectRatio"]}}
        first = patch_instances[0]
        row: dict[str, Any] = {
            "patch_rank": rank,
            "patch_id": first.get("patch_id"),
            "patch_x": first.get("patch_x"),
            "patch_y": first.get("patch_y"),
            "attention_norm": first.get("attention_norm"),
        }
        mapped_n = 0
        skipped_n = 0
        for inst in patch_instances:
            metrics = contour_metrics(inst.get("contour_local") or [])
            if metrics is None:
                skipped_n += 1
                continue
            group = map_to_liver_group(str(inst.get("type", "")), metrics["Area"])
            if group is None:
                skipped_n += 1
                continue
            mapped_n += 1
            for name, value in metrics.items():
                if value is not None and math.isfinite(value):
                    groups[group][name].append(float(value))
        row["Mapped_Nuclei"] = mapped_n
        row["Skipped_Nuclei"] = skipped_n
        for group, values_by_metric in groups.items():
            row[f"Count_{group}"] = len(values_by_metric["Area"])
            for feature, values in values_by_metric.items():
                row.update(safe_stats(values, f"{group}_{feature}"))
        rows.append(row)
    return rows


def reference_distributions(matched_path: Path, summary_path: Path) -> pd.DataFrame:
    matched = pd.read_csv(matched_path)
    summary = pd.read_csv(summary_path)
    matched["ImageID"] = pd.to_numeric(matched["ImageID"], errors="coerce")
    summary["ImageID"] = pd.to_numeric(summary["ImageID"], errors="coerce")
    cols = ["ImageID"] + [metric for metric in REQUESTED_METRICS if metric in summary.columns]
    merged = matched[["ImageID", "Label"]].merge(summary[cols], on="ImageID", how="inner")
    return merged


def percentile_rank(values: pd.Series, value: float) -> float | None:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0 or not math.isfinite(value):
        return None
    return float(100.0 * np.mean(arr <= value))


def summarize_reference(values: pd.Series) -> dict[str, float | int | None]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "sd": None, "median": None, "p10": None, "p90": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else None,
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def build_comparison(patch_rows: list[dict[str, Any]], reference: pd.DataFrame) -> dict[str, Any]:
    patch_df = pd.DataFrame(patch_rows)
    case_ref = reference[reference["Label"] == 1]
    control_ref = reference[reference["Label"] == 0]
    comparisons = []
    for metric in REQUESTED_METRICS:
        topk_values = pd.to_numeric(patch_df.get(metric), errors="coerce").dropna()
        topk_mean = float(topk_values.mean()) if len(topk_values) else None
        topk_median = float(topk_values.median()) if len(topk_values) else None
        ctrl = summarize_reference(control_ref[metric])
        case = summarize_reference(case_ref[metric])
        ctrl_mean = ctrl["mean"]
        ctrl_sd = ctrl["sd"]
        case_mean = case["mean"]
        z_vs_control = None
        if topk_mean is not None and ctrl_sd not in (None, 0):
            z_vs_control = float((topk_mean - float(ctrl_mean)) / float(ctrl_sd))
        closer_to = None
        if topk_mean is not None and ctrl_mean is not None and case_mean is not None:
            closer_to = "case" if abs(topk_mean - float(case_mean)) < abs(topk_mean - float(ctrl_mean)) else "control"
        comparisons.append(
            {
                "metric": metric,
                "topk_patch_n": int(len(topk_values)),
                "topk_mean": topk_mean,
                "topk_median": topk_median,
                "control": ctrl,
                "case": case,
                "z_vs_control": z_vs_control,
                "percentile_vs_control": percentile_rank(control_ref[metric], topk_mean) if topk_mean is not None else None,
                "percentile_vs_case": percentile_rank(case_ref[metric], topk_mean) if topk_mean is not None else None,
                "closer_to": closer_to,
                "topk_minus_control_mean": topk_mean - float(ctrl_mean) if topk_mean is not None and ctrl_mean is not None else None,
                "topk_minus_case_mean": topk_mean - float(case_mean) if topk_mean is not None and case_mean is not None else None,
            }
        )
    return {"metrics": comparisons}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    instances = load_json(Path(args.instances_json))
    patch_rows = compute_patch_metrics(instances)
    reference = reference_distributions(Path(args.matched_dataset_csv), Path(args.celltype_summary_csv))
    comparison = build_comparison(patch_rows, reference)
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
            "requested_metrics": REQUESTED_METRICS,
            "reference": {
                "matched_dataset_csv": str(Path(args.matched_dataset_csv).resolve()),
                "celltype_summary_csv": str(Path(args.celltype_summary_csv).resolve()),
                "case_n": int((reference["Label"] == 1).sum()),
                "control_n": int((reference["Label"] == 0).sum()),
            },
        }
    )
    patch_json = output_dir / "patch_metrics.json"
    patch_csv = output_dir / "patch_metrics.csv"
    comparison_json = output_dir / "metric_comparison.json"
    save_json(patch_json, patch_rows)
    save_json(comparison_json, comparison)
    with patch_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in patch_rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(patch_rows)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
