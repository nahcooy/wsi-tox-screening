from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import HTTPException

from app.settings import settings


DOMAIN_KNOWLEDGE = [
    {
        "topic": "Rodent liver terminology",
        "summary": (
            "Use standardized toxicologic pathology terms for rodent liver. Distribution and "
            "severity should be interpreted against concurrent controls where possible."
        ),
        "source": "https://ntp.niehs.nih.gov/atlas/nnl/hepatobiliary-system",
    },
    {
        "topic": "Hepatocyte hypertrophy",
        "summary": (
            "Hepatocyte hypertrophy is an observable increase in hepatocyte size versus "
            "concurrent controls. It is often centrilobular and can be associated with xenobiotic "
            "microsomal enzyme induction; marked cases may show karyomegaly or multinucleation."
        ),
        "source": "https://ntp.niehs.nih.gov/sites/default/files/nnl/hepatobiliary/liver/hhypertr/liver_hepatocyte-hypertrophy_508.pdf",
    },
    {
        "topic": "Inflammation",
        "summary": (
            "Focal inflammatory aggregates are common background findings in rodent liver, but "
            "xenobiotic exposure may increase lesion number or severity. Inflammatory aggregates "
            "can accompany hepatocellular necrosis."
        ),
        "source": "https://ntp.niehs.nih.gov/atlas/nnl/hepatobiliary-system/liver/Inflammation",
    },
    {
        "topic": "Apoptosis and necrosis",
        "summary": (
            "Single-cell hepatocyte death usually occurs by apoptosis, while necrosis usually "
            "involves groups, regions, or zones of cells and is typically accompanied by inflammation."
        ),
        "source": "https://ntp.niehs.nih.gov/atlas/nnl/hepatobiliary-system/liver/Hepatocyte-Apoptosis",
    },
    {
        "topic": "Severity grading",
        "summary": (
            "NTP nonneoplastic lesion grading commonly uses four levels: minimal, mild, moderate, "
            "and marked. The report should avoid overcalling severity from model-derived features alone."
        ),
        "source": "https://ntp.niehs.nih.gov/atlas/nnl/guide",
    },
]


REPORT_METRICS = [
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


def read_json(path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compact_topk(topk: list[dict[str, Any]], limit: int = 25) -> list[dict[str, Any]]:
    rows = []
    for patch in topk[:limit]:
        rows.append(
            {
                "rank": patch.get("rank"),
                "patch_id": patch.get("patch_id"),
                "x": patch.get("x"),
                "y": patch.get("y"),
                "attention_raw": patch.get("attention_raw"),
                "attention_softmax": patch.get("attention_softmax"),
                "attention_norm": patch.get("attention_norm"),
            }
        )
    return rows


def compact_metric_comparison(comparison: dict[str, Any] | None) -> dict[str, Any] | None:
    if not comparison:
        return None
    rows = []
    for row in comparison.get("metrics", []):
        if row.get("metric") not in REPORT_METRICS:
            continue
        rows.append(
            {
                "metric": row.get("metric"),
                "topk_patch_n": row.get("topk_patch_n"),
                "topk_mean": row.get("topk_mean"),
                "topk_median": row.get("topk_median"),
                "control_mean": row.get("control", {}).get("mean"),
                "control_sd": row.get("control", {}).get("sd"),
                "control_p10": row.get("control", {}).get("p10"),
                "control_p90": row.get("control", {}).get("p90"),
                "case_mean": row.get("case", {}).get("mean"),
                "case_sd": row.get("case", {}).get("sd"),
                "case_p10": row.get("case", {}).get("p10"),
                "case_p90": row.get("case", {}).get("p90"),
                "z_vs_control": row.get("z_vs_control"),
                "percentile_vs_control": row.get("percentile_vs_control"),
                "percentile_vs_case": row.get("percentile_vs_case"),
                "closer_to": row.get("closer_to"),
            }
        )
    return {"reference": comparison.get("reference"), "metrics": rows}


def compact_nuclei_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not summary:
        return None
    overlays = []
    for overlay in summary.get("overlays", [])[:25]:
        overlays.append(
            {
                "rank": overlay.get("rank"),
                "patch_id": overlay.get("patch_id"),
                "attention_norm": overlay.get("attention_norm"),
                "cell_count": overlay.get("cell_count"),
                "type_counts": overlay.get("type_counts"),
            }
        )
    return {
        "model": summary.get("model"),
        "num_patches": summary.get("num_patches"),
        "total_nuclei": summary.get("total_nuclei"),
        "type_counts": summary.get("type_counts"),
        "overlays": overlays,
        "cell_group_mapping": {
            "Neoplastic": "Hep",
            "Epithelial": "Hep if area_um2 >= 30, otherwise NPC",
            "Connective": "NPC",
            "Inflammatory": "Imm",
            "Dead": "excluded",
            "Background": "excluded",
        },
    }


def compact_patch_metrics(patch_metrics: list[dict[str, Any]], limit: int = 25) -> list[dict[str, Any]]:
    rows = []
    for row in patch_metrics[:limit]:
        compact = {"patch_rank": row.get("patch_rank"), "patch_id": row.get("patch_id")}
        for metric in REPORT_METRICS:
            compact[metric] = row.get(metric)
        rows.append(compact)
    return rows


def build_report_payload(
    slide_id: str,
    mil_result: dict[str, Any],
    topk: list[dict[str, Any]],
    nuclei_summary: dict[str, Any],
    metric_comparison: dict[str, Any],
    patch_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "slide_id": slide_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "assay_context": "Rat liver WSI toxicity screening, ABMIL slide-level classifier plus NuLite-H top-k nuclei analysis.",
        "mil_model": {
            "model_type": mil_result.get("model_type"),
            "encoder": mil_result.get("encoder"),
            "class_order": mil_result.get("class_order"),
            "prediction": mil_result.get("prediction"),
            "logits": mil_result.get("logits"),
            "softmax": mil_result.get("softmax"),
            "confidence_score": mil_result.get("confidence_score"),
            "abnormal_confidence_score": mil_result.get("abnormal_confidence_score"),
            "num_patches": mil_result.get("num_patches"),
            "attention_normalization": mil_result.get("attention_normalization"),
        },
        "attention_topk": compact_topk(topk),
        "nulite": compact_nuclei_summary(nuclei_summary),
        "case_control_statistics": compact_metric_comparison(metric_comparison),
        "patch_metrics": compact_patch_metrics(patch_metrics),
        "domain_knowledge": DOMAIN_KNOWLEDGE,
    }


def report_instructions() -> str:
    return (
        "You are a board-certified toxicologic pathologist writing a structured screening report "
        "for rat liver H&E WSI. You will receive a JSON object with pipeline results.\n\n"

        "## STRICT RULES\n"
        "1. Write ENTIRELY in Korean using NTP/INHAND histopathology terminology.\n"
        "2. Every claim must cite a specific number from the data (confidence, Z-score, cell count, etc.).\n"
        "3. NEVER mention absent data. If a section has nothing real to say, DELETE it — do not write '없음' or 'N/A'.\n"
        "4. COHERENCE: Report conclusion MUST match ABMIL prediction. If normal (>90% confidence), do NOT describe active lesions.\n"
        "5. Do NOT fabricate dose, sex, gross findings, serum chemistry, or anything not in the data.\n"
        "6. Write like a real pathology report — not a summary of the JSON. Interpret, don't just transcribe.\n"
        "7. Write in full detail. Do not truncate. A thorough, complete report is expected.\n\n"

        "## REPORT STRUCTURE\n\n"

        "## (1) 핵심 판정\n"
        "One or two sentences: ABMIL 예측(normal/abnormal), 신뢰도 수치, 최종 판정.\n\n"

        "## (2) 정량적 근거\n"
        "**ABMIL 모델**: prediction, softmax probabilities, num_patches.\n"
        "**어텐션 분포**: 고주목 패치 집중 영역, top-25 attention 범위. spatial_distribution이 있으면 우세 사분면 언급.\n"
        "**NuLite-H 핵 분석**: 총 핵 수, 세포유형별 개수/비율. 이상 세포(Neoplastic, Inflammatory)가 있으면 비율과 함께 강조.\n"
        "**Case-Control 비교**: |Z| > 2인 메트릭만 나열. 각 지표의 Z-score와 percentile_vs_control 인용. "
        "|Z| < 2인 메트릭은 '정상 범위 내'로 일괄 처리하고 개별 나열 금지.\n\n"

        "## (3) 병리학적 해석\n"
        "데이터에서 관찰된 소견을 독성학적으로 해석:\n"
        "- 형태계측 이상(Z-score 이상치)이 나타내는 세포 변화 (예: 간세포 비대, 핵 이상, 세포질 변화)\n"
        "- Neoplastic/Inflammatory 세포 비율의 의미\n"
        "- 독성 등급 추정: minimal / mild / moderate / severe (수치 근거 제시)\n"
        "- 어텐션 집중 영역과 형태 변화의 공간적 연관성\n\n"

        "## (4) 한계 및 추가 확인 권고\n"
        "Boilerplate 금지. 이 슬라이드 데이터에서 실제로 불확실한 부분만 기재.\n"
        "예: null 메트릭, 어텐션이 특정 영역에만 과도하게 집중된 경우, 추가 패치 검토 권고 등.\n\n"

        "---\n"
        "이 보고서는 의사결정 지원용입니다. 정식 진단을 위해서는 병리사의 WSI 직접 검토가 필요합니다."
    )


def extract_response_text(response_json: dict[str, Any]) -> str:
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: list[str] = []
    for item in response_json.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    if parts:
        return "\n".join(parts).strip()
    raise HTTPException(status_code=502, detail="OpenAI response did not contain report text.")


def generate_report_text(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Qwen (vLLM) 로컬 서버로 보고서 생성."""
    request_body = {
        "model": settings.agent_model,
        "messages": [
            {"role": "system", "content": report_instructions()},
            {"role": "user",   "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ],
        "temperature": 0.2,
        "max_tokens": 4096,
    }
    try:
        with httpx.Client(timeout=settings.agent_timeout_seconds) as client:
            response = client.post(
                f"{settings.agent_base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.agent_api_key}",
                    "Content-Type": "application/json",
                },
                json=request_body,
            )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Agent LLM unreachable: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Agent LLM error: {response.text[:500]}")

    response_json = response.json()
    text = response_json["choices"][0]["message"].get("content", "").strip()
    if not text:
        raise HTTPException(status_code=502, detail="Agent LLM returned empty report.")
    return text, {
        "model": settings.agent_model,
        "usage": response_json.get("usage"),
    }
