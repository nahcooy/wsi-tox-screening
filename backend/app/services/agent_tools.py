from __future__ import annotations

import base64
import csv
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from app.settings import settings


def _run_subprocess_with_progress(
    command: list[str],
    env: dict,
    timeout: int,
    log_path: Path,
    slide_id: str,
    stage_label: str,
) -> subprocess.CompletedProcess:
    """
    subprocess를 Popen으로 실행하며 stdout에서 [PROGRESS] N% 라인을 실시간으로
    live_status.json에 기록한다. 완료 후 CompletedProcess를 반환한다.
    """
    from datetime import datetime, timezone

    def _ts() -> str:
        return datetime.now(timezone.utc).strftime("%H:%M:%S")

    def _append_log(entry: dict) -> None:
        live_path = settings.output_dir / "runs" / slide_id / "agent_run" / "live_status.json"
        if not live_path.exists():
            return
        try:
            live = json.loads(live_path.read_text(encoding="utf-8"))
            live.setdefault("log", []).append(entry)
            live_path.write_text(json.dumps(live, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=str(settings.project_root),
        )

        def _drain_stderr():
            for line in proc.stderr:
                stderr_lines.append(line)
                log_f.write(line)

        t = threading.Thread(target=_drain_stderr, daemon=True)
        t.start()

        last_pct = -1
        for line in proc.stdout:
            stdout_lines.append(line)
            log_f.write(line)
            stripped = line.strip()
            if stripped.startswith("[PROGRESS]"):
                try:
                    pct_str = stripped.split()[1].rstrip("%")
                    pct = int(pct_str)
                    if pct != last_pct:
                        last_pct = pct
                        note = stripped[len(f"[PROGRESS] {pct_str}%"):].strip(" —-")
                        msg = f"{stage_label} {pct}%" + (f" — {note}" if note else "")
                        _append_log({"ts": _ts(), "type": "info", "content": msg})
                except (IndexError, ValueError):
                    pass

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise
        t.join(timeout=5)

    return subprocess.CompletedProcess(
        args=command,
        returncode=proc.returncode,
        stdout="".join(stdout_lines),
        stderr="".join(stderr_lines),
    )


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _image_to_base64(path: Path) -> str | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _run_dir(slide_id: str) -> Path:
    return settings.output_dir / "runs" / slide_id


# ── Tool 1 ────────────────────────────────────────────────────────────────────

def get_mil_summary(slide_id: str) -> dict[str, Any]:
    """ABMIL 슬라이드 레벨 예측 결과."""
    result = _read_json(_run_dir(slide_id) / "mil" / "mil_result.json", {})
    return {
        "prediction": result.get("prediction"),
        "logits": result.get("logits"),
        "softmax": result.get("softmax"),
        "confidence_score": result.get("confidence_score"),
        "abnormal_confidence_score": result.get("abnormal_confidence_score"),
        "num_patches": result.get("num_patches"),
        "attention_normalization": result.get("attention_normalization"),
        "class_order": result.get("class_order"),
    }


# ── Tool 2 ────────────────────────────────────────────────────────────────────

def get_attention_heatmap(slide_id: str) -> dict[str, Any]:
    """
    어텐션 점수 분포 및 공간 통계 (데이터 기반, 이미지 아님).
    전체 패치의 attention 분포, top-25 좌표, 사분면별 집중도를 반환.
    """
    csv_path = _run_dir(slide_id) / "mil" / "attention_scores.csv"
    if not csv_path.exists():
        return {"error": "attention_scores.csv not found — run inference first"}

    rows: list[dict] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "patch_id":      int(row.get("patch_id", 0)),
                "rank":          int(row["rank"]) if row.get("rank") else None,
                "attention_norm": float(row.get("attention_norm", 0)),
                "attention_raw":  float(row.get("attention_raw",  0)),
                "x": int(row.get("x", 0)),
                "y": int(row.get("y", 0)),
            })

    rows.sort(key=lambda r: r["attention_norm"], reverse=True)
    attn = [r["attention_norm"] for r in rows]

    mil_result = _read_json(_run_dir(slide_id) / "mil" / "mil_result.json", {})
    slide_w = mil_result.get("slide_width")  or 1
    slide_h = mil_result.get("slide_height") or 1

    # 사분면별 high-attention 패치 수 (상위 25%)
    top_n = max(1, len(rows) // 4)
    top_rows = rows[:top_n]
    quadrants: dict[str, int] = {"TL": 0, "TR": 0, "BL": 0, "BR": 0}
    for r in top_rows:
        qh = "T" if r["y"] < slide_h / 2 else "B"
        qw = "L" if r["x"] < slide_w / 2 else "R"
        quadrants[qh + qw] += 1

    return {
        "total_patches": len(rows),
        "attention_statistics": {
            "mean":       round(float(sum(attn) / len(attn)), 4) if attn else 0,
            "max":        round(float(max(attn)), 4) if attn else 0,
            "min":        round(float(min(attn)), 4) if attn else 0,
            "top10_mean": round(float(sum(attn[:10]) / min(10, len(attn))), 4) if attn else 0,
            "top25_mean": round(float(sum(attn[:25]) / min(25, len(attn))), 4) if attn else 0,
            "p25":        round(float(sorted(attn)[len(attn) // 4]), 4) if attn else 0,
            "p75":        round(float(sorted(attn)[3 * len(attn) // 4]), 4) if attn else 0,
        },
        "top25_patches": [
            {"rank": r["rank"], "patch_id": r["patch_id"],
             "attention_norm": r["attention_norm"], "x": r["x"], "y": r["y"]}
            for r in rows[:25]
        ],
        "spatial_distribution_top25pct": {
            "quadrant_counts": quadrants,
            "dominant_quadrant": max(quadrants, key=quadrants.get),
            "note": "TL=top-left, TR=top-right, BL=bottom-left, BR=bottom-right of slide",
        },
    }


# ── Tool 3 ────────────────────────────────────────────────────────────────────

def get_topk_patches(slide_id: str, ranks: list[int]) -> list[dict[str, Any]]:
    """특정 rank의 H&E 패치 이미지 + 어텐션 점수."""
    topk = _read_json(_run_dir(slide_id) / "mil" / "topk" / "top25_patches.json", [])
    rank_set = set(ranks)
    results = []
    for patch in topk:
        if patch.get("rank") not in rank_set:
            continue
        img_path = patch.get("image_path")
        b64 = _image_to_base64(Path(img_path)) if img_path else None
        results.append({
            "rank": patch.get("rank"),
            "attention_norm": patch.get("attention_norm"),
            "attention_raw": patch.get("attention_raw"),
            "x": patch.get("x"),
            "y": patch.get("y"),
            "image_b64": b64,
        })
    return results


# ── Tool 4 ────────────────────────────────────────────────────────────────────

def get_nulite_overlays(slide_id: str, ranks: list[int]) -> list[dict[str, Any]]:
    """특정 rank 패치의 NuLite-H 핵 분할 오버레이 이미지 + 세포유형 카운트."""
    summary = _read_json(_run_dir(slide_id) / "nuclei" / "nuclei_summary.json", {})
    rank_set = set(ranks)
    results = []
    for overlay in summary.get("overlays", []):
        if overlay.get("rank") not in rank_set:
            continue
        overlay_path = overlay.get("overlay_path")
        b64 = _image_to_base64(Path(overlay_path)) if overlay_path else None
        results.append({
            "rank": overlay.get("rank"),
            "cell_count": overlay.get("cell_count"),
            "type_counts": overlay.get("type_counts"),
            "attention_norm": overlay.get("attention_norm"),
            "image_b64": b64,
        })
    return results


# ── Tool 5 ────────────────────────────────────────────────────────────────────

def get_nuclei_summary(slide_id: str) -> dict[str, Any]:
    """전체 top-k 패치의 핵 분석 요약 (총 핵 수, 세포유형별 카운트)."""
    summary = _read_json(_run_dir(slide_id) / "nuclei" / "nuclei_summary.json", {})
    return {
        "model": summary.get("model"),
        "num_patches": summary.get("num_patches"),
        "total_nuclei": summary.get("total_nuclei"),
        "type_counts": summary.get("type_counts"),
    }


# ── Tool 6 ────────────────────────────────────────────────────────────────────

def get_patch_metrics(slide_id: str, ranks: list[int]) -> list[dict[str, Any]]:
    """특정 rank 패치의 Hep/NPC/Imm 형태 메트릭 (Area, Solidity, Circularity 등)."""
    patch_metrics = _read_json(_run_dir(slide_id) / "nuclei" / "patch_metrics.json", [])
    rank_set = set(ranks)
    return [row for row in patch_metrics if row.get("patch_rank") in rank_set]


# ── Tool 7 ────────────────────────────────────────────────────────────────────

def get_metric_comparison(slide_id: str) -> dict[str, Any]:
    """11개 Hep/NPC/Imm 메트릭의 case-control 비교 (z-score, percentile, closer_to)."""
    comparison = _read_json(_run_dir(slide_id) / "nuclei" / "metric_comparison.json", {})
    rows = []
    for row in comparison.get("metrics", []):
        rows.append({
            "metric": row.get("metric"),
            "topk_mean": row.get("topk_mean"),
            "topk_median": row.get("topk_median"),
            "control_mean": row.get("control", {}).get("mean"),
            "control_sd": row.get("control", {}).get("sd"),
            "control_p10": row.get("control", {}).get("p10"),
            "control_p90": row.get("control", {}).get("p90"),
            "case_mean": row.get("case", {}).get("mean"),
            "case_sd": row.get("case", {}).get("sd"),
            "z_vs_control": row.get("z_vs_control"),
            "percentile_vs_control": row.get("percentile_vs_control"),
            "percentile_vs_case": row.get("percentile_vs_case"),
            "closer_to": row.get("closer_to"),
        })
    return {
        "reference": comparison.get("reference"),
        "metrics": rows,
    }


# ── ACTIVE TOOLS (agent-driven inference) ────────────────────────────────────

# ── Tool 8 ────────────────────────────────────────────────────────────────────

def get_all_patch_attention(slide_id: str) -> dict[str, Any]:
    """전체 패치의 어텐션 점수 목록 (내림차순). top-25 이외 패치 탐색용."""
    csv_path = _run_dir(slide_id) / "mil" / "attention_scores.csv"
    if not csv_path.exists():
        # Fallback to top25 if full CSV is missing
        topk = _read_json(_run_dir(slide_id) / "mil" / "topk" / "top25_patches.json", [])
        return {
            "source": "top25_fallback",
            "total_patches": len(topk),
            "patches": [
                {
                    "patch_id": p.get("patch_id"),
                    "rank": p.get("rank"),
                    "attention_norm": p.get("attention_norm"),
                    "attention_raw": p.get("attention_raw"),
                    "x": p.get("x"),
                    "y": p.get("y"),
                }
                for p in topk
            ],
        }
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "patch_id": int(row.get("patch_id", row.get("index", 0))),
                "rank": int(row.get("rank", 0)) if row.get("rank") else None,
                "attention_norm": float(row.get("attention_norm", 0)),
                "attention_raw": float(row.get("attention_raw", 0)),
                "x": int(row.get("x", 0)),
                "y": int(row.get("y", 0)),
            })
    # Already sorted desc in the CSV, but ensure it
    rows.sort(key=lambda r: r["attention_norm"], reverse=True)
    return {
        "source": "attention_scores.csv",
        "total_patches": len(rows),
        "patches": rows,
    }


# ── Tool 9 ────────────────────────────────────────────────────────────────────

def extract_patch_image(slide_id: str, patch_ids: list[int]) -> list[dict[str, Any]]:
    """
    지정한 patch_id의 WSI 패치 이미지를 직접 추출 (openslide).
    top-25 외의 패치도 조회 가능. 최대 10개.
    """
    patch_ids = patch_ids[:10]  # hard cap to avoid huge payloads

    mil_result = _read_json(_run_dir(slide_id) / "mil" / "mil_result.json", {})
    patch_size_level0 = int(mil_result.get("patch_size_level0", 256))

    coords_h5 = (
        _run_dir(slide_id)
        / "trident"
        / "20x_256px_0px_overlap"
        / "patches"
        / f"{slide_id}_patches.h5"
    )
    if not coords_h5.exists():
        return [{"error": f"coords h5 not found: {coords_h5}"}]

    # Locate WSI file
    slide_path = _find_slide(slide_id)
    if slide_path is None:
        return [{"error": "WSI file not found"}]

    try:
        import h5py
        import openslide
        from PIL import Image
        import io
    except ImportError as e:
        return [{"error": f"missing dependency: {e}"}]

    with h5py.File(str(coords_h5), "r") as hf:
        all_coords = hf["coords"][:]

    slide = openslide.OpenSlide(str(slide_path))
    results = []
    for pid in patch_ids:
        if pid < 0 or pid >= len(all_coords):
            results.append({"patch_id": pid, "error": f"out of range (total={len(all_coords)})"})
            continue
        x, y = int(all_coords[pid][0]), int(all_coords[pid][1])
        region = slide.read_region((x, y), 0, (patch_size_level0, patch_size_level0))
        img = region.convert("RGB")
        if patch_size_level0 != 256:
            img = img.resize((256, 256))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        results.append({
            "patch_id": pid,
            "x": x,
            "y": y,
            "image_b64": b64,
        })
    slide.close()
    return results


# ── Tool 10 ───────────────────────────────────────────────────────────────────

def run_nulite_on_patches(slide_id: str, patch_ids: list[int]) -> dict[str, Any]:
    """
    지정 patch_id에 NuLite-H 핵 분할을 새로 실행 (cv 환경 subprocess).
    top-25 이외 패치에 대한 능동적 핵 분석 가능.
    결과: 핵 카운트, 세포유형 분포, 오버레이 이미지.
    """
    patch_ids = patch_ids[:20]  # cap

    mil_result = _read_json(_run_dir(slide_id) / "mil" / "mil_result.json", {})
    patch_size_level0 = int(mil_result.get("patch_size_level0", 256))

    coords_h5 = (
        _run_dir(slide_id)
        / "trident"
        / "20x_256px_0px_overlap"
        / "patches"
        / f"{slide_id}_patches.h5"
    )
    if not coords_h5.exists():
        return {"error": f"coords h5 not found: {coords_h5}"}

    slide_path = _find_slide(slide_id)
    if slide_path is None:
        return {"error": "WSI file not found"}

    timestamp = int(time.time())
    output_dir = _run_dir(slide_id) / "nuclei" / "on_demand" / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "run_nulite_on_demand.py"
    )
    cmd = [
        str(settings.nulite_python),
        str(script),
        "--slide_id",          slide_id,
        "--slide_path",        str(slide_path),
        "--coords_h5",         str(coords_h5),
        "--patch_ids",         json.dumps(patch_ids),
        "--patch_size_level0", str(patch_size_level0),
        "--output_dir",        str(output_dir),
        "--nulite_root",       str(settings.nulite_root),
        "--checkpoint_path",   str(settings.default_nulite_h_checkpoint),
        "--gpu",               "0",
        "--batch_size",        "4",
    ]
    if settings.matched_dataset_csv.exists():
        cmd += ["--matched_dataset_csv", str(settings.matched_dataset_csv)]
    if settings.celltype_summary_csv.exists():
        cmd += ["--celltype_summary_csv", str(settings.celltype_summary_csv)]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {"error": "NuLite on-demand inference timed out (300s)"}
    except Exception as e:
        return {"error": f"subprocess error: {e}"}

    if proc.returncode != 0:
        return {
            "error": "NuLite on-demand inference failed",
            "stderr": proc.stderr[-2000:],
        }

    try:
        summary = json.loads(proc.stdout.strip().split("\n")[-1])
    except Exception:
        summary = _read_json(output_dir / "summary.json", {})

    # Attach overlay images (cap at 5)
    overlays_with_images = []
    for ov in summary.get("overlays", [])[:5]:
        ov_path = ov.get("overlay_path")
        b64 = _image_to_base64(Path(ov_path)) if ov_path else None
        overlays_with_images.append({
            "patch_id":   ov.get("patch_id"),
            "x":          ov.get("x"),
            "y":          ov.get("y"),
            "cell_count": ov.get("cell_count"),
            "type_counts": ov.get("type_counts"),
            "image_b64":  b64,
        })

    return {
        "slide_id":      slide_id,
        "patch_ids":     patch_ids,
        "total_nuclei":  summary.get("total_nuclei"),
        "type_counts":   summary.get("type_counts"),
        "output_dir":    str(output_dir),
        "overlays":      overlays_with_images,
        "metrics_available": (output_dir / "patch_metrics.json").exists(),
    }


# ── Tool 11 ───────────────────────────────────────────────────────────────────

def compute_metrics_for_patches(slide_id: str, patch_ids: list[int]) -> dict[str, Any]:
    """
    지정 patch_id의 Hep/NPC/Imm 형태 메트릭 계산.
    on-demand NuLite 결과가 있으면 우선 사용; 없으면 top-k 전체에서 필터.
    """
    run_dir = _run_dir(slide_id)

    # Find most recent on-demand instances for these patch_ids
    on_demand_dir = run_dir / "nuclei" / "on_demand"
    instances_json = None
    if on_demand_dir.exists():
        runs = sorted(on_demand_dir.iterdir(), reverse=True)
        for r in runs:
            candidate = r / "instances.json"
            if candidate.exists():
                # Check it contains at least some of our requested patch_ids
                sample = _read_json(candidate, [])
                found_ids = {inst.get("patch_id") for inst in sample[:20]}
                if found_ids & set(patch_ids):
                    instances_json = candidate
                    break

    if instances_json is None:
        # Fall back to top-k instances
        topk_instances = run_dir / "nuclei" / "all_instances.json"
        if topk_instances.exists():
            instances_json = topk_instances

    if instances_json is None:
        return {"error": "no NuLite instances found — run run_nulite_on_patches first"}

    if not settings.matched_dataset_csv.exists() or not settings.celltype_summary_csv.exists():
        return {"error": "reference CSV files not found — cannot run comparison"}

    timestamp = int(time.time())
    output_dir = run_dir / "nuclei" / "on_demand" / f"metrics_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "compute_metrics_on_demand.py"
    )
    cmd = [
        str(settings.nulite_python),
        str(script),
        "--instances_json",      str(instances_json),
        "--patch_ids",           json.dumps(patch_ids),
        "--output_dir",          str(output_dir),
        "--slide_id",            slide_id,
        "--matched_dataset_csv", str(settings.matched_dataset_csv),
        "--celltype_summary_csv", str(settings.celltype_summary_csv),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"error": "metrics computation timed out (120s)"}
    except Exception as e:
        return {"error": f"subprocess error: {e}"}

    if proc.returncode != 0:
        return {
            "error": "metrics computation failed — do NOT retry this tool. Use get_metric_comparison instead.",
            "stderr": proc.stderr[-500:],
        }

    try:
        comparison = json.loads(proc.stdout.strip().split("\n")[-1])
    except Exception:
        comparison = _read_json(output_dir / "metric_comparison.json", {})

    return comparison


# ── Tool 12: TTA ──────────────────────────────────────────────────────────────

def run_tta_inference(slide_id: str, n_trials: int = 10, sample_ratio: float = 0.8) -> dict[str, Any]:
    """
    Test-Time Augmentation: 패치를 sample_ratio 비율로 무작위 샘플링해 ABMIL을 n_trials회 반복 추론.
    confidence가 50-80% 사이일 때 사용. 결과: 다수결, mean prob, std, 95% CI.
    """
    n_trials    = min(max(1, n_trials), 30)
    sample_ratio = min(max(0.3, sample_ratio), 0.95)

    base = _run_dir(slide_id)
    feature_h5 = (
        base / "trident" / "20x_256px_0px_overlap" / "features_uni_v1" / f"{slide_id}.h5"
    )
    if not feature_h5.exists():
        return {"error": "feature_h5 not found — preprocess first"}

    mil_result = _read_json(base / "mil" / "mil_result.json", {})
    checkpoint_path = mil_result.get("checkpoint_path")
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return {"error": f"checkpoint not found: {checkpoint_path}"}

    script = settings.project_root / "backend" / "scripts" / "run_abmil_tta.py"
    if not script.exists():
        return {"error": f"TTA script not found: {script}"}

    mil_lab_root = mil_result.get("checkpoint_path", "")
    # Derive mil_lab_root from checkpoint path (2 levels up from checkpoints dir)
    ckpt_path = Path(checkpoint_path)
    mil_lab_root_path = ckpt_path.parents[2] if len(ckpt_path.parts) > 2 else Path("/home/nahcooy/MIL/MIL_260527/MIL-Lab")

    try:
        import sys as _sys
        python = _sys.executable  # same env as FastAPI
        proc = subprocess.run(
            [
                python, str(script),
                "--feature_h5",      str(feature_h5),
                "--checkpoint_path", checkpoint_path,
                "--mil_lab_root",    str(mil_lab_root_path),
                "--n_trials",        str(n_trials),
                "--sample_ratio",    str(sample_ratio),
                "--slide_id",        slide_id,
                "--device",          "cuda:0",
            ],
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {"error": "TTA inference timed out (300s)"}
    except Exception as e:
        return {"error": f"subprocess error: {e}"}

    if proc.returncode != 0:
        return {"error": "TTA failed", "stderr": proc.stderr[-1000:]}

    try:
        result = json.loads(proc.stdout.strip().split("\n")[-1])
    except Exception as e:
        return {"error": f"TTA output parse failed: {e}", "stdout": proc.stdout[-500:]}

    # Add comparison with original single-pass result
    orig_pred    = mil_result.get("prediction")
    orig_conf    = mil_result.get("confidence_score")
    tta_majority = result.get("majority_vote")
    tta_mean     = result.get("mean_softmax", {})

    result["comparison_with_single_pass"] = {
        "single_pass_prediction": orig_pred,
        "single_pass_confidence": orig_conf,
        "tta_majority_vote":      tta_majority,
        "tta_mean_abnormal":      tta_mean.get("abnormal"),
        "agreement":              orig_pred == tta_majority,
        "interpretation": (
            "일치 — 단일 추론 신뢰도 높음" if orig_pred == tta_majority
            else "불일치 — 경계 케이스, 병리학자 검토 권고"
        ),
    }
    return result


# ── Pipeline Tools (agent-full-run 전용) ──────────────────────────────────────

def _slide_in_run_dir(slide_id: str) -> Path | None:
    """업로드된 슬라이드를 run_dir/slide/ 에서 찾는다."""
    slide_dir = _run_dir(slide_id) / "slide"
    if slide_dir.exists():
        for ext in (".svs", ".ndpi", ".tiff", ".tif", ".scn", ".mrxs", ".vms", ".vmu"):
            for candidate in slide_dir.iterdir():
                if candidate.suffix.lower() == ext:
                    return candidate
    return None


def get_pipeline_status(slide_id: str) -> dict:
    """현재 파이프라인 각 단계 완료 여부를 확인한다. agent 시작 시 반드시 먼저 호출."""
    base = _run_dir(slide_id)
    feature_h5   = base / "trident" / "20x_256px_0px_overlap" / "features_uni_v1" / f"{slide_id}.h5"
    mil_result   = base / "mil" / "mil_result.json"
    topk_manifest= base / "mil" / "topk" / "top25_patches.json"
    nuclei_sum   = base / "nuclei" / "nuclei_summary.json"
    slide_path   = _slide_in_run_dir(slide_id)
    return {
        "slide_found":   slide_path is not None,
        "slide_path":    str(slide_path) if slide_path else None,
        "preprocess":    "completed" if feature_h5.exists()    else "not_started",
        "inference":     "completed" if mil_result.exists()    else "not_started",
        "nuclei_topk":   "completed" if nuclei_sum.exists()    else "not_started",
        "note": "Run each missing stage in order: preprocess → inference → nuclei_topk",
    }


def run_preprocess_pipeline(slide_id: str) -> dict:
    """
    TRIDENT 전처리를 실행한다 (tissue segmentation + UNI feature extraction).
    이미 완료된 경우 바로 반환. 미완료 시 blocking subprocess (~20-30 min).
    """
    base = _run_dir(slide_id)
    feature_h5 = base / "trident" / "20x_256px_0px_overlap" / "features_uni_v1" / f"{slide_id}.h5"
    if feature_h5.exists():
        return {"status": "already_completed", "feature_h5": str(feature_h5)}

    slide_path = _slide_in_run_dir(slide_id)
    if slide_path is None:
        return {"status": "failed", "error": "슬라이드 파일을 run_dir/slide/ 에서 찾을 수 없습니다."}

    script = settings.project_root / "backend" / "scripts" / "run_trident_single_slide.py"
    if not script.exists():
        return {"status": "failed", "error": f"Script not found: {script}"}
    if not settings.trident_python.exists():
        return {"status": "failed", "error": f"TRIDENT python not found: {settings.trident_python}"}

    trident_dir = base / "trident"
    command = [
        str(settings.trident_python), str(script),
        "--slide_path",   str(slide_path),
        "--job_dir",      str(trident_dir),
        "--trident_root", str(settings.trident_root),
        "--segmenter",    "grandqc",
        "--remove_artifacts",
        "--mag",          "20",
        "--patch_size",   "256",
        "--overlap",      "0",
        "--patch_encoder","uni_v1",
        "--batch_size",   "64",
        "--feat_batch_size", "512",
        "--device",       "cuda:0",
    ]
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/wsi-tox-screening-mpl")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

    log_path = trident_dir / "trident_pipeline.log"
    try:
        proc = _run_subprocess_with_progress(
            command, env, timeout=7200, log_path=log_path,
            slide_id=slide_id, stage_label="Preprocess",
        )
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": "TRIDENT 전처리 타임아웃 (7200s)"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

    if proc.returncode != 0:
        return {"status": "failed", "error": "TRIDENT 실패", "stderr": proc.stderr[-2000:]}
    if not feature_h5.exists():
        return {"status": "failed", "error": f"feature_h5 생성 실패: {feature_h5}"}
    return {"status": "completed", "feature_h5": str(feature_h5)}


def run_inference_pipeline(slide_id: str) -> dict:
    """
    ABMIL 추론을 실행한다 (attention MIL + top-25 패치 추출).
    이미 완료된 경우 바로 반환. 미완료 시 blocking subprocess (~3-8 min).
    """
    import importlib
    base = _run_dir(slide_id)
    mil_result_path = base / "mil" / "mil_result.json"
    if mil_result_path.exists():
        return {"status": "already_completed", "mil_result": str(mil_result_path)}

    feature_h5 = base / "trident" / "20x_256px_0px_overlap" / "features_uni_v1" / f"{slide_id}.h5"
    if not feature_h5.exists():
        return {"status": "failed", "error": "Preprocess가 완료되지 않았습니다. run_preprocess_pipeline을 먼저 실행하세요."}

    slide_path = _slide_in_run_dir(slide_id)

    try:
        from app.pipelines.abmil_inference import build_abmil_command
        from app.schemas.mil import ABMILInferenceRequest
        req = ABMILInferenceRequest(
            slide_id=slide_id,
            slide_path=str(slide_path) if slide_path else None,
            trident_job_dir=str(base / "trident"),
            output_dir=str(base / "mil"),
            top_k=25,
            device="cuda:0",
        )
        plan = build_abmil_command(req)
    except Exception as e:
        return {"status": "failed", "error": f"Command build 실패: {e}"}

    try:
        proc = _run_subprocess_with_progress(
            plan.command, plan.env, timeout=1800, log_path=plan.log_path,
            slide_id=slide_id, stage_label="Inference",
        )
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": "ABMIL 추론 타임아웃 (1800s)"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

    if proc.returncode != 0:
        return {"status": "failed", "error": "ABMIL 실패", "stderr": proc.stderr[-2000:]}
    if not mil_result_path.exists():
        return {"status": "failed", "error": f"mil_result.json 생성 실패"}
    return {"status": "completed", "mil_result": str(mil_result_path)}


def run_nulite_topk_pipeline(slide_id: str) -> dict:
    """
    NuLite-H를 top-25 어텐션 패치에 실행한다.
    이미 완료된 경우 바로 반환. 미완료 시 blocking subprocess (~5-15 min).
    """
    base = _run_dir(slide_id)
    nuclei_summary_path = base / "nuclei" / "nuclei_summary.json"
    if nuclei_summary_path.exists():
        return {"status": "already_completed", "nuclei_summary": str(nuclei_summary_path)}

    topk_manifest = base / "mil" / "topk" / "top25_patches.json"
    mil_result    = base / "mil" / "mil_result.json"
    if not topk_manifest.exists() or not mil_result.exists():
        return {"status": "failed", "error": "Inference가 완료되지 않았습니다. run_inference_pipeline을 먼저 실행하세요."}

    script = settings.project_root / "backend" / "scripts" / "run_nulite_topk_inference.py"
    if not script.exists():
        return {"status": "failed", "error": f"NuLite script not found: {script}"}
    if not settings.nulite_python.exists():
        return {"status": "failed", "error": f"NuLite python not found: {settings.nulite_python}"}

    nuclei_dir = base / "nuclei"
    command = [
        str(settings.nulite_python), str(script),
        "--topk_manifest",   str(topk_manifest),
        "--mil_result",      str(mil_result),
        "--output_dir",      str(nuclei_dir),
        "--nulite_root",     str(settings.nulite_root),
        "--checkpoint_path", str(settings.default_nulite_h_checkpoint),
        "--slide_id",        slide_id,
        "--batch_size",      "8",
        "--num_workers",     "0",
        "--gpu",             "0",
    ]
    if settings.matched_dataset_csv.exists() and settings.celltype_summary_csv.exists():
        command += ["--matched_dataset_csv", str(settings.matched_dataset_csv),
                    "--celltype_summary_csv", str(settings.celltype_summary_csv)]

    env = os.environ.copy()
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/wsi-tox-screening-numba-cache")
    env.setdefault("MPLCONFIGDIR", "/tmp/wsi-tox-screening-mpl")

    log_path = nuclei_dir / "nulite_inference.log"
    try:
        proc = _run_subprocess_with_progress(
            command, env, timeout=2400, log_path=log_path,
            slide_id=slide_id, stage_label="NuLite",
        )
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": "NuLite top-k 타임아웃 (2400s)"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

    if proc.returncode != 0:
        return {"status": "failed", "error": "NuLite top-k 실패", "stderr": proc.stderr[-2000:]}
    if not nuclei_summary_path.exists():
        return {"status": "failed", "error": "nuclei_summary.json 생성 실패"}
    return {"status": "completed", "nuclei_summary": str(nuclei_summary_path)}


# ── Helper: locate WSI file ───────────────────────────────────────────────────

def _find_slide(slide_id: str) -> Path | None:
    """Search for WSI file in default_wsi_dir and run_dir."""
    for ext in (".svs", ".ndpi", ".tiff", ".tif", ".scn", ".mrxs"):
        candidate = settings.default_wsi_dir / f"{slide_id}{ext}"
        if candidate.exists():
            return candidate
    # Also check if stored in run metadata
    meta = _read_json(_run_dir(slide_id) / "mil" / "mil_result.json", {})
    if slide_path := meta.get("slide_path"):
        p = Path(slide_path)
        if p.exists():
            return p
    return None
