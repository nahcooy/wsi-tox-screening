#!/usr/bin/env python3
"""
Compute morphometric metrics for agent-selected patches.
Reads instances from an existing JSONL/JSON file (top-k or on-demand),
optionally filters by patch_ids, runs compute_patch_metrics + build_comparison,
and prints the comparison JSON to stdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute patch metrics for agent-selected patches.")
    p.add_argument("--instances_json",    required=True, help="Path to all_instances.json or on-demand instances.json")
    p.add_argument("--patch_ids",         default=None,  help="JSON list of patch_ids to filter (None = use all)")
    p.add_argument("--output_dir",        required=True)
    p.add_argument("--slide_id",          required=True)
    p.add_argument("--matched_dataset_csv", required=True)
    p.add_argument("--celltype_summary_csv", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    instances_path = Path(args.instances_json)
    if not instances_path.exists():
        print(json.dumps({"error": f"instances file not found: {args.instances_json}"}))
        sys.exit(1)

    with instances_path.open("r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        print(json.dumps({"error": "instances file is empty"}))
        sys.exit(1)

    # Support both JSON array and JSONL format
    if content.startswith("["):
        instances = json.loads(content)
    else:
        instances = [json.loads(line) for line in content.splitlines() if line.strip()]

    # Filter by patch_ids if specified
    if args.patch_ids:
        patch_id_set = set(json.loads(args.patch_ids))
        instances = [inst for inst in instances if inst.get("patch_id") in patch_id_set]

    if not instances:
        print(json.dumps({"error": "no instances after filtering", "patch_ids": args.patch_ids}))
        sys.exit(1)

    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    from compute_nulite_patch_metrics import build_comparison, compute_patch_metrics, reference_distributions

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_rows = compute_patch_metrics(instances)
    reference  = reference_distributions(Path(args.matched_dataset_csv), Path(args.celltype_summary_csv))
    comparison = build_comparison(patch_rows, reference)
    comparison.update({
        "slide_id":    args.slide_id,
        "source":      "on_demand",
        "num_patches": len(patch_rows),
        "num_instances": len(instances),
    })

    metrics_json     = output_dir / "patch_metrics.json"
    comparison_json  = output_dir / "metric_comparison.json"
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(patch_rows, f, indent=2)
    with comparison_json.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
