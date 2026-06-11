from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException

from app.settings import settings
from app.services.rag.feedback_store import FeedbackStore
from app.services.rag.literature_store import LiteratureStore
from app.services.agent_tools import (
    compute_metrics_for_patches,
    extract_patch_image,
    get_all_patch_attention,
    get_attention_heatmap,
    get_metric_comparison,
    get_mil_summary,
    get_nulite_overlays,
    get_nuclei_summary,
    get_patch_metrics,
    get_topk_patches,
    run_nulite_on_patches,
    run_tta_inference,
    # Pipeline tools (agent-full-run)
    get_pipeline_status,
    run_preprocess_pipeline,
    run_inference_pipeline,
    run_nulite_topk_pipeline,
)

SYSTEM_PROMPT = (
    "You are a board-certified toxicologic pathologist analyzing rat liver H&E WSI for toxicity screening.\n\n"

    "## MANDATORY RULES — NEVER VIOLATE\n"
    "1. You MUST call tools before writing the final report. NEVER write a report without tool evidence.\n"
    "2. MINIMUM required tool calls before report: get_mil_summary → get_attention_heatmap → "
    "get_topk_patches → get_nuclei_summary → get_metric_comparison (5 calls minimum).\n"
    "3. Every claim in the report MUST cite actual numbers from tool results "
    "(e.g., prediction confidence, nuclei counts, z-scores). NO fabrication.\n"
    "4. If ABMIL prediction is 'normal' with high confidence (>95%), the report must acknowledge this "
    "as primary finding. Do NOT invent pathological findings contradicting the model output.\n"
    "5. Write the FINAL REPORT in Korean using NTP/INHAND-style terminology.\n\n"

    "## TOOLS\n"
    "READ-ONLY (call these first, in order):\n"
    "1. get_mil_summary — ABMIL prediction, logits, confidence, num_patches\n"
    "2. get_attention_heatmap — spatial attention map image\n"
    "3. get_topk_patches(ranks) — H&E patch images for selected ranks (1-25). "
    "Look at the heatmap first, then request only suspicious ranks.\n"
    "4. get_nulite_overlays(ranks) — nuclei overlay images + per-cell-type counts for top-25 patches\n"
    "5. get_nuclei_summary — global cell-type distribution across all top-25 patches\n"
    "6. get_patch_metrics(ranks) — Hep/NPC/Imm Area/Circularity/Solidity metrics\n"
    "7. get_metric_comparison — 11-metric case vs control z-scores and percentiles\n\n"
    "ACTIVE (trigger new inference — use only when top-25 is insufficient):\n"
    "8. get_all_patch_attention — full attention list beyond top-25\n"
    "9. extract_patch_image(patch_ids) — H&E images for any patch_ids (max 10)\n"
    "10. run_nulite_on_patches(patch_ids) — NuLite-H on chosen patches (max 20, ~2-5 min)\n"
    "11. compute_metrics_for_patches(patch_ids) — morphometrics + case-control for chosen patches\n\n"

    "## WORKFLOW\n"
    "Step 1: Call get_mil_summary. Read the prediction and confidence score carefully.\n"
    "Step 2: Call get_attention_heatmap. Note which regions have high attention.\n"
    "Step 3: Call get_topk_patches with 3-5 most suspicious ranks from the heatmap.\n"
    "Step 4: Call get_nulite_overlays for ranks where you see abnormal morphology.\n"
    "Step 5: Call get_nuclei_summary and get_metric_comparison.\n"
    "Step 6: If evidence warrants, explore beyond top-25 using tools 8-11.\n"
    "Step 7: Write the final report based ONLY on what the tools actually returned.\n\n"

    "## REPORT FORMAT (Korean, NTP/INHAND terminology)\n"
    "(1) 핵심 판정 — ABMIL 예측 결과와 신뢰도를 반드시 명시. 수치 포함.\n"
    "(2) 근거 — 각 tool에서 얻은 실제 수치(핵 개수, z-score, confidence 등)를 인용.\n"
    "(3) 병리학적 해석 — 관찰된 소견의 독성학적 의미.\n"
    "(4) 한계 및 확인 필요 항목\n"
    "(5) 참고 문헌 URL\n\n"
    "This report is decision-support only. Formal diagnosis requires pathologist WSI review."
)

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_mil_summary",
            "description": "ABMIL 슬라이드 레벨 예측 결과 조회 (prediction, logits, softmax, confidence, num_patches)",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_attention_heatmap",
            "description": (
                "어텐션 점수 데이터 조회 (이미지 아님). "
                "전체 패치의 attention 통계(mean/max/p25/p75), top-25 패치 좌표, "
                "슬라이드 사분면별 고주목 패치 분포를 반환. "
                "spatial_distribution으로 어느 영역이 모델 주목도가 높은지 파악할 것."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_topk_patches",
            "description": "특정 rank의 H&E 패치 원본 이미지와 어텐션 점수 조회. 히트맵 관찰 결과를 바탕으로 의심 rank만 선택할 것.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ranks": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1, "maximum": 25},
                        "description": "조회할 패치 rank 목록 (1-25). 필요한 rank만 선택.",
                    }
                },
                "required": ["ranks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_nulite_overlays",
            "description": "특정 rank 패치의 NuLite-H 핵 분할 오버레이 이미지와 세포유형 카운트 조회. 시각 이상 소견이 확인된 rank만 요청할 것.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ranks": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1, "maximum": 25},
                        "description": "조회할 패치 rank 목록",
                    }
                },
                "required": ["ranks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_nuclei_summary",
            "description": "전체 top-k 패치의 핵 분석 요약 조회 (총 핵 수, 세포유형별 전체 카운트)",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_patch_metrics",
            "description": "특정 rank 패치의 Hep/NPC/Imm 형태 메트릭 조회 (Area_Mean, Circularity, Solidity 등). 이미 시각 확인한 rank에 대해 정량 검증 시 사용.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ranks": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1, "maximum": 25},
                        "description": "조회할 패치 rank 목록",
                    }
                },
                "required": ["ranks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_metric_comparison",
            "description": "11개 Hep/NPC/Imm 메트릭의 case-control 비교 통계 조회 (z-score, percentile_vs_control, closer_to). 최종 판정 종합 시 호출.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # ── Active Tools ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_all_patch_attention",
            "description": (
                "전체 패치의 어텐션 점수 목록 조회 (내림차순). "
                "top-25 이외의 고주목 패치를 탐색할 때 사용. "
                "결과에는 patch_id, attention_norm, attention_raw, x, y 포함."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_patch_image",
            "description": (
                "지정 patch_id의 WSI H&E 패치 이미지를 직접 추출 (openslide). "
                "top-25 외 패치 시각 확인 시 사용. 최대 10개."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patch_ids": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "description": "추출할 patch_id 목록 (최대 10개). get_all_patch_attention 결과의 patch_id 사용.",
                        "maxItems": 10,
                    }
                },
                "required": ["patch_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_nulite_on_patches",
            "description": (
                "지정 patch_id에 NuLite-H 핵 분할을 새로 실행. "
                "top-25 이외 패치에 대한 능동적 핵 분석. "
                "실행 시간 약 2-5분. 최대 20개. "
                "결과: 총 핵 수, 세포유형 카운트, 오버레이 이미지."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patch_ids": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "description": "NuLite를 실행할 patch_id 목록 (최대 20개). 시각 확인 후 가장 이상 소견이 의심되는 패치만 선택.",
                        "maxItems": 20,
                    }
                },
                "required": ["patch_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_metrics_for_patches",
            "description": (
                "지정 patch_id의 Hep/NPC/Imm 형태 메트릭 계산 및 case-control 비교. "
                "on-demand NuLite 결과 또는 top-k 전체에서 필터하여 계산. "
                "run_nulite_on_patches 실행 후 호출하면 가장 정확."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patch_ids": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "description": "메트릭을 계산할 patch_id 목록.",
                    }
                },
                "required": ["patch_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tta_inference",
            "description": (
                "Test-Time Augmentation: 패치를 무작위 샘플링해 ABMIL을 N회 반복 추론. "
                "단일 추론 confidence가 50-80% 사이일 때 호출해 의사결정 신뢰도를 높여라. "
                "결과: majority_vote, mean_softmax, std_softmax, 95% CI, trial별 softmax."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n_trials": {
                        "type": "integer",
                        "minimum": 5,
                        "maximum": 30,
                        "description": "반복 추론 횟수 (기본 10).",
                    },
                    "sample_ratio": {
                        "type": "number",
                        "minimum": 0.3,
                        "maximum": 0.95,
                        "description": "각 trial에서 사용할 패치 비율 (기본 0.8).",
                    },
                },
                "required": [],
            },
        },
    },
]

_TOOL_DISPATCH = {
    # Read-only
    "get_mil_summary":        lambda slide_id, args: get_mil_summary(slide_id),
    "get_attention_heatmap":  lambda slide_id, args: get_attention_heatmap(slide_id),
    "get_topk_patches":       lambda slide_id, args: get_topk_patches(slide_id, args["ranks"]),
    "get_nulite_overlays":    lambda slide_id, args: get_nulite_overlays(slide_id, args["ranks"]),
    "get_nuclei_summary":     lambda slide_id, args: get_nuclei_summary(slide_id),
    "get_patch_metrics":      lambda slide_id, args: get_patch_metrics(slide_id, args["ranks"]),
    "get_metric_comparison":  lambda slide_id, args: get_metric_comparison(slide_id),
    # Active
    "get_all_patch_attention":    lambda slide_id, args: get_all_patch_attention(slide_id),
    "extract_patch_image":        lambda slide_id, args: extract_patch_image(slide_id, args["patch_ids"]),
    "run_nulite_on_patches":      lambda slide_id, args: run_nulite_on_patches(slide_id, args["patch_ids"]),
    "compute_metrics_for_patches": lambda slide_id, args: compute_metrics_for_patches(slide_id, args["patch_ids"]),
    "run_tta_inference":          lambda slide_id, args: run_tta_inference(
        slide_id,
        n_trials=args.get("n_trials", 10),
        sample_ratio=args.get("sample_ratio", 0.8),
    ),
}

# tools that return base64 images (get_attention_heatmap 제거 — 이제 데이터 기반)
_IMAGE_TOOLS = {
    "get_topk_patches",
    "get_nulite_overlays",
    "extract_patch_image",
    "run_nulite_on_patches",
}


def _split_images(fn_name: str, result: Any) -> tuple[Any, list[tuple[str, str]]]:
    """Separate image bytes from metadata. Returns (text_result, [(caption, b64), ...])."""
    images: list[tuple[str, str]] = []

    if fn_name in ("get_topk_patches", "get_nulite_overlays"):
        text_items = []
        for item in result:
            b64 = item.pop("image_b64", None)
            caption = f"rank={item.get('rank')} 이미지:"
            if b64:
                images.append((caption, b64))
            text_items.append(item)
        return text_items, images

    if fn_name == "extract_patch_image":
        text_items = []
        for item in result:
            b64 = item.pop("image_b64", None)
            caption = f"patch_id={item.get('patch_id')} H&E 이미지:"
            if b64:
                images.append((caption, b64))
            text_items.append(item)
        return text_items, images

    if fn_name == "run_nulite_on_patches":
        # result is a dict with "overlays" list — each overlay has image_b64
        overlays = result.get("overlays", [])
        clean_overlays = []
        for ov in overlays:
            b64 = ov.pop("image_b64", None)
            caption = f"patch_id={ov.get('patch_id')} NuLite 오버레이 이미지:"
            if b64:
                images.append((caption, b64))
            clean_overlays.append(ov)
        result["overlays"] = clean_overlays
        return result, images

    return result, images


_VLLM_MAX_MODEL_LEN = 16384
_VLLM_MAX_IMAGES_PER_PROMPT = 10


def _trim_messages(messages: list[dict], reserve_output: int = 2048) -> list[dict]:
    """Context window 초과 방지 + 누적 이미지 수 10장 이하 유지."""
    # ── 1. 이미지 누적 제거 ───────────────────────────────────────
    # 오래된 user 메시지의 image_url을 제거해 총 이미지 수를 10 이하로 유지
    total_images = 0
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            total_images += sum(1 for p in m["content"] if p.get("type") == "image_url")

    if total_images > _VLLM_MAX_IMAGES_PER_PROMPT:
        # 뒤에서부터 이미지를 살리고 앞쪽 이미지는 제거
        keep_budget = _VLLM_MAX_IMAGES_PER_PROMPT
        result: list[dict] = []
        for m in reversed(messages):
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                imgs_here = [p for p in m["content"] if p.get("type") == "image_url"]
                if keep_budget <= 0:
                    # 이 메시지의 이미지 전부 제거, 텍스트만 유지
                    text_only = [p for p in m["content"] if p.get("type") != "image_url"]
                    if text_only:
                        result.append({**m, "content": text_only})
                elif len(imgs_here) <= keep_budget:
                    keep_budget -= len(imgs_here)
                    result.append(m)
                else:
                    # 일부만 유지
                    kept_imgs = imgs_here[:keep_budget]
                    kept_set = set(id(p) for p in kept_imgs)
                    new_content = [p for p in m["content"] if p.get("type") != "image_url" or id(p) in kept_set]
                    keep_budget = 0
                    result.append({**m, "content": new_content})
            else:
                result.append(m)
        messages = list(reversed(result))

    # ── 2. Context 길이 축약 ──────────────────────────────────────
    budget = _VLLM_MAX_MODEL_LEN - reserve_output - 200
    total_chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
    if total_chars / 3.5 <= budget:
        return messages

    head = messages[:2]
    tail = messages[2:]
    trimmed: list[dict] = []
    for m in tail:
        if m.get("role") == "tool":
            content = m.get("content", "")
            if len(content) > 400:
                m = {**m, "content": content[:400] + " ...[trimmed]"}
        trimmed.append(m)

    return head + trimmed


def _call_llm(
    messages: list[dict],
    tools: list[dict] | None,
    *,
    tool_choice: str = "auto",
    max_tokens: int = 2048,
    max_retries: int = 3,
) -> dict:
    request_body: dict[str, Any] = {
        "model": settings.agent_model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    if tools:
        request_body["tools"] = tools
        request_body["tool_choice"] = tool_choice

    last_exc: Exception | None = None
    for attempt in range(max_retries):
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

            # 4xx는 재시도 불필요 (잘못된 요청 / 인증 오류)
            if response.status_code < 500:
                if response.status_code >= 400:
                    raise HTTPException(status_code=502, detail=f"Agent LLM error: {response.text[:500]}")
                return response.json()

            # 5xx: 서버 일시 오류 → 재시도
            last_exc = Exception(f"HTTP {response.status_code}: {response.text[:200]}")

        except HTTPException:
            raise
        except httpx.RequestError as exc:
            last_exc = exc

        wait = 2 ** attempt  # 1s, 2s, 4s
        time.sleep(wait)

    raise HTTPException(status_code=502, detail=f"Agent LLM unreachable after {max_retries} retries: {last_exc}")


def _build_rag_context(slide_id: str, reference_paper_ids: list[str] | None = None) -> str:
    """
    Retrieve RAG context to prepend to the agent system prompt:
    1) Similar past feedback from the feedback store
    2) Relevant literature chunks (semantic search) + explicitly cited papers
    """
    sections: list[str] = []

    # Feedback RAG
    try:
        fb_store = FeedbackStore()
        query    = f"rat liver H&E toxicity slide_id={slide_id}"
        fb_ctx   = fb_store.build_rag_context(query, k=3)
        if fb_ctx:
            sections.append(fb_ctx)
    except Exception:
        pass

    # Literature RAG — explicit papers first
    try:
        lit_store = LiteratureStore()
        if reference_paper_ids:
            explicit_ctx = lit_store.build_explicit_context(reference_paper_ids)
            if explicit_ctx:
                sections.append(explicit_ctx)
        # Also retrieve semantically relevant chunks
        lit_query = "rat liver toxicity hepatocyte necrosis NuLite H&E histopathology"
        lit_ctx   = lit_store.build_rag_context(lit_query, k=4)
        if lit_ctx:
            sections.append(lit_ctx)
    except Exception:
        pass

    return "\n\n".join(sections)


def run_agent(
    slide_id: str,
    reference_paper_ids: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    ReAct loop: think → tool_call → observe → ... → stop → report.
    reference_paper_ids: optional list of paper_ids to cite in the report.
    Returns (report_text_korean, metadata_dict).
    """
    rag_context = _build_rag_context(slide_id, reference_paper_ids)
    system_content = SYSTEM_PROMPT
    if rag_context:
        system_content = SYSTEM_PROMPT + "\n\n" + rag_context

    messages: list[dict] = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": (
                f"slide_id={slide_id}의 쥐 간 H&E WSI를 분석하여 독성 판정 보고서를 작성하시오. "
                "먼저 ABMIL 예측 결과를 확인한 뒤 어텐션 히트맵을 보고, "
                "필요한 패치와 핵 분석 결과를 선택적으로 조회하여 최종 판정을 내리시오."
                + (
                    f"\n\n반드시 다음 paper_id의 문헌을 인용하시오: {reference_paper_ids}"
                    if reference_paper_ids else ""
                )
            ),
        },
    ]

    metadata: dict[str, Any] = {
        "agent_model": settings.agent_model,
        "slide_id": slide_id,
        "iterations": 0,
        "tools_called": [],
        "forced_finish": False,
    }

    for iteration in range(settings.agent_max_iter):
        response = _call_llm(messages, TOOL_SCHEMAS)
        choice = response["choices"][0]
        finish_reason = choice["finish_reason"]
        assistant_msg = choice["message"]
        metadata["iterations"] = iteration + 1

        # ── 종료: 모델이 스스로 보고서 작성 완료 ────────────────────────────
        if finish_reason == "stop":
            metadata["usage"] = response.get("usage")
            return assistant_msg.get("content", ""), metadata

        # ── 도구 호출 루프 ───────────────────────────────────────────────────
        if finish_reason == "tool_calls":
            messages.append(assistant_msg)
            pending_images: list[tuple[str, str]] = []

            for tool_call in assistant_msg.get("tool_calls", []):
                fn_name = tool_call["function"]["name"]
                fn_args = json.loads(tool_call["function"]["arguments"] or "{}")

                metadata["tools_called"].append({
                    "iteration": iteration + 1,
                    "name": fn_name,
                    "args": fn_args,
                })

                if fn_name not in _TOOL_DISPATCH:
                    tool_text = f"알 수 없는 도구: {fn_name}"
                    tool_images: list[tuple[str, str]] = []
                else:
                    raw = _TOOL_DISPATCH[fn_name](slide_id, fn_args)
                    if fn_name in _IMAGE_TOOLS:
                        raw, tool_images = _split_images(fn_name, raw)
                    else:
                        tool_images = []
                    tool_text = json.dumps(raw, ensure_ascii=False, indent=2)

                # tool 결과는 항상 문자열로 전달
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_text,
                })
                pending_images.extend(tool_images)

            # 이번 iteration에서 수집된 이미지들을 user 메시지로 추가
            # (tool message는 string만 허용, 이미지는 user turn으로 전달)
            if pending_images:
                image_content: list[dict] = []
                for caption, b64 in pending_images:
                    image_content.append({"type": "text", "text": caption})
                    image_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })
                messages.append({"role": "user", "content": image_content})

    # ── max_iter 초과: 강제 종료 ─────────────────────────────────────────────
    messages.append({
        "role": "user",
        "content": "최대 반복 횟수에 도달했습니다. 지금까지 수집한 정보로 최종 보고서를 작성하시오.",
    })
    response = _call_llm(messages, tools=None)
    metadata["usage"] = response.get("usage")
    metadata["forced_finish"] = True
    return response["choices"][0]["message"].get("content", ""), metadata


# ── Agent Full-Run (pipeline → analysis → report) ─────────────────────────────

AGENT_FULL_SYSTEM_PROMPT = (
    "You are an autonomous toxicologic pathology AI agent analyzing rat liver H&E WSI for toxicity screening.\n\n"

    "## CRITICAL RULES — NEVER VIOLATE\n"
    "1. NEVER write Python code. NEVER write step-by-step plans as text. "
    "The ONLY valid actions are: call a tool, OR write the final report.\n"
    "2. Do NOT output any text before your first tool call. Call get_pipeline_status immediately.\n"
    "3. Do NOT summarize what you plan to do. Just do it — call the tool.\n"
    "4. Every claim in the final report MUST be grounded in actual tool outputs (numbers, counts, scores).\n\n"

    "## TOOLS\n"
    "PIPELINE (run in order if not completed):\n"
    "P1. get_pipeline_status — call this FIRST, always\n"
    "P2. run_preprocess_pipeline — TRIDENT tissue segmentation + UNI feature extraction\n"
    "P3. run_inference_pipeline — ABMIL attention MIL inference + top-25 patch extraction\n"
    "P4. run_nulite_topk_pipeline — NuLite-H nuclei segmentation on top-25 patches\n\n"

    "ANALYSIS (after pipeline complete):\n"
    "A1. get_mil_summary — ABMIL prediction, confidence, logits\n"
    "A2. get_attention_heatmap — attention spatial stats + top-25 coords (DATA not image)\n"
    "A3. get_topk_patches(ranks) — H&E images for suspicious ranks\n"
    "A4. get_nulite_overlays(ranks) — nuclei overlay images + cell counts\n"
    "A5. get_nuclei_summary — global cell-type distribution\n"
    "A6. get_patch_metrics(ranks) — morphometrics per patch\n"
    "A7. get_metric_comparison — case-control Z-scores for 11 metrics\n\n"

    "ACTIVE (trigger new inference — when top-25 is insufficient):\n"
    "A8. get_all_patch_attention — full patch attention list\n"
    "A9. extract_patch_image(patch_ids) — H&E for any patch (max 10)\n"
    "A10. run_nulite_on_patches(patch_ids) — on-demand NuLite (max 20)\n"
    "A11. compute_metrics_for_patches(patch_ids) — morphometrics for any patches\n"
    "A12. run_tta_inference — TTA: repeat ABMIL N times. "
    "CALL THIS when confidence is 50-80%.\n\n"

    "## EXECUTION ORDER\n"
    "1. get_pipeline_status → run missing stages (P2/P3/P4) in sequence\n"
    "2. get_mil_summary + get_attention_heatmap\n"
    "3. If confidence 50-80%: run_tta_inference\n"
    "4. get_topk_patches for top suspicious ranks (read heatmap spatial data first)\n"
    "5. get_nulite_overlays for ranks with abnormal morphology → visually inspect\n"
    "6. get_nuclei_summary + get_metric_comparison\n"
    "7. If warranted: explore A8-A11 for deeper analysis\n"
    "8. Write final report in Korean\n\n"

    "## REPORT FORMAT (Korean, NTP/INHAND style)\n"
    "You are writing a SUPERIOR report compared to a simple data-summary tool.\n"
    "You have DIRECTLY VIEWED patch images and NuLite overlays — your report MUST reflect this.\n\n"

    "ABSOLUTE RULES:\n"
    "- NEVER mention tools not called or data not retrieved. If TTA was not run, do not mention it at all.\n"
    "- NEVER include empty sections. If a section has nothing real to say, DELETE it entirely — do not write '없음', 'N/A', or placeholder text.\n"
    "- COHERENCE: The report conclusion MUST be consistent with the ABMIL prediction. "
    "If ABMIL says normal (>90% confidence), do NOT describe active pathological lesions or disease. "
    "Incidental findings may be noted but must not contradict the primary prediction.\n"
    "- Write like a pathologist describing slides, NOT like a data analyst summarizing a JSON.\n"
    "- Every numerical claim must cite a specific value from tool results.\n"
    "- Write in full detail — do not truncate or summarize prematurely. A thorough report is expected.\n\n"

    "## (1) 핵심 판정\n"
    "ABMIL 예측(normal/abnormal), 신뢰도, [TTA 실행한 경우에만: majority vote + 95% CI], 최종 판정 한 문장.\n\n"

    "## (2) 시각적 병리 소견 ★ 에이전트 독점 섹션\n"
    "get_topk_patches와 get_nulite_overlays로 직접 관찰한 패치별 형태학적 소견을 기술하라.\n"
    "각 rank별로: 세포 크기/모양 이상, 핵의 변화(다형성, 과염색, 비정상 유사분열), 세포질 변화,\n"
    "염증세포 침윤, 괴사 소견 등을 NTP/INHAND 용어로 구체적으로 기술하라.\n"
    "어텐션이 높은 패치일수록 왜 모델이 주목했는지 시각적 근거와 연결하라.\n"
    "NuLite 오버레이에서 관찰된 세포유형 분포(Neoplastic/Inflammatory 비율)를 패치별로 언급하라.\n\n"

    "## (3) 정량적 근거\n"
    "- 어텐션 분포: 고주목 패치 공간 집중 영역, dominant quadrant (spatial_distribution 데이터 인용)\n"
    "- NuLite 전체 통계: 총 핵 수, 세포유형 비율, 이상 세포 절대수와 비율\n"
    "- Case-control 비교: |Z| > 2인 메트릭만 기재. Z-score + percentile_vs_control 인용.\n"
    "  |Z| ≤ 2인 메트릭은 '나머지 지표는 대조군 정상 범위 내'로 한 줄 처리.\n\n"

    "## (4) 병리학적 종합 해석\n"
    "- 시각 소견 + 정량 지표를 종합한 독성학적 해석\n"
    "- 독성 등급: minimal / mild / moderate / severe (수치와 시각 소견 모두 근거로)\n"
    "- 어텐션 집중 영역의 공간적 의미 (예: 소엽 중심성 vs 문맥 주변 분포)\n\n"

    "## (5) 한계 및 추가 확인 권고\n"
    "Boilerplate 금지. 이 케이스에서 실제로 불확실한 부분만. 추가 검토가 필요한 구체적 영역 명시.\n\n"

    "## (6) 참고 문헌 (RAG에서 실제 제공된 경우에만. 없으면 섹션 전체 생략)\n\n"
    "---\n"
    "이 보고서는 의사결정 지원용입니다. 정식 진단을 위해서는 병리사의 WSI 직접 검토가 필요합니다."
)


PIPELINE_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_pipeline_status",
            "description": "파이프라인 각 단계 완료 여부 확인. 항상 가장 먼저 호출.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_preprocess_pipeline",
            "description": "TRIDENT 전처리 실행 (tissue segmentation + UNI feature extraction). 약 20-30분.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_inference_pipeline",
            "description": "ABMIL 추론 실행 (attention MIL + top-25 patch 추출). 약 3-8분.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_nulite_topk_pipeline",
            "description": "NuLite-H 핵 분할을 top-25 패치에 실행. 약 5-15분.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

ALL_FULL_RUN_SCHEMAS = PIPELINE_TOOL_SCHEMAS + TOOL_SCHEMAS

_PIPELINE_DISPATCH = {
    "get_pipeline_status":    lambda slide_id, args: get_pipeline_status(slide_id),
    "run_preprocess_pipeline": lambda slide_id, args: run_preprocess_pipeline(slide_id),
    "run_inference_pipeline":  lambda slide_id, args: run_inference_pipeline(slide_id),
    "run_nulite_topk_pipeline": lambda slide_id, args: run_nulite_topk_pipeline(slide_id),
}

_FULL_DISPATCH = {**_PIPELINE_DISPATCH, **_TOOL_DISPATCH}


def _live_status_path(slide_id: str) -> Path:
    return settings.output_dir / "runs" / slide_id / "agent_run" / "live_status.json"


def _write_live_status(slide_id: str, status: dict) -> None:
    p = _live_status_path(slide_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def _ts() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _log_entry(type_: str, **kwargs) -> dict:
    return {"ts": _ts(), "type": type_, **kwargs}


def run_agent_full(
    slide_id: str,
    user_instructions: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Fully autonomous pipeline + analysis + report generation.
    Writes live_status.json at each step for frontend polling.
    Should be called in a background thread.
    """
    from app.services.agent_tools import get_pipeline_status as _get_pipeline_status

    # 파일시스템에서 실제 pipeline 상태 초기화
    _init_pl = _get_pipeline_status(slide_id)
    live: dict[str, Any] = {
        "state": "running",
        "stage": "initializing",
        "iteration": 0,
        "pipeline": {
            "preprocess":  _init_pl.get("preprocess",  "unknown"),
            "inference":   _init_pl.get("inference",   "unknown"),
            "nuclei_topk": _init_pl.get("nuclei_topk", "unknown"),
        },
        "tools_called": [],
        "log": [_log_entry("info", content=f"에이전트 시작 — slide_id={slide_id}")],
        "report": None,
        "error": None,
    }
    _write_live_status(slide_id, live)

    rag_context = _build_rag_context(slide_id)
    system_content = AGENT_FULL_SYSTEM_PROMPT
    if rag_context:
        system_content = AGENT_FULL_SYSTEM_PROMPT + "\n\n" + rag_context

    user_content = (
        f"slide_id={slide_id}. Call get_pipeline_status now."
    )
    if user_instructions and user_instructions.strip():
        user_content += f"\n\n[사용자 지시사항]\n{user_instructions.strip()}"

    messages: list[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    metadata: dict[str, Any] = {
        "agent_model": settings.agent_model,
        "slide_id": slide_id,
        "mode": "full_run",
        "iterations": 0,
        "tools_called": [],
        "forced_finish": False,
    }

    # 분석 도구 최소 호출 횟수 달성 여부 추적
    _REQUIRED_ANALYSIS_TOOLS = {"get_mil_summary", "get_attention_heatmap", "get_nuclei_summary", "get_metric_comparison"}
    _called_tools: set[str] = set()

    for iteration in range(settings.agent_max_iter + 10):  # extra headroom for pipeline steps
        live["iteration"] = iteration + 1
        live["stage"] = "thinking"
        _write_live_status(slide_id, live)

        # 파이프라인 미완료 또는 핵심 분석 도구 미호출 시 tool_choice="required"
        pl = live["pipeline"]
        pipeline_done = all(
            pl.get(k) in ("completed", "already_completed", "failed")
            for k in ("preprocess", "inference", "nuclei_topk")
        )
        analysis_done = _REQUIRED_ANALYSIS_TOOLS.issubset(_called_tools)
        cur_tool_choice = "auto" if (pipeline_done and analysis_done) else "required"

        try:
            response = _call_llm(
                _trim_messages(messages),
                ALL_FULL_RUN_SCHEMAS,
                tool_choice=cur_tool_choice,
            )
        except HTTPException as e:
            live["state"] = "failed"
            live["error"] = str(e.detail)
            _write_live_status(slide_id, live)
            raise

        choice = response["choices"][0]
        finish_reason = choice["finish_reason"]
        assistant_msg = choice["message"]
        metadata["iterations"] = iteration + 1

        # 모델의 reasoning text 캡처 (tool call 전 사고 과정)
        reasoning_text = (assistant_msg.get("content") or "").strip()
        if reasoning_text:
            live["log"].append(_log_entry(
                "thinking",
                iter=iteration + 1,
                content=reasoning_text[:600],  # 최대 600자
            ))
            _write_live_status(slide_id, live)

        if finish_reason == "stop":
            # tool_choice="required" 상태에서 stop → 모델이 텍스트만 출력(Python 코드 등)
            # 보고서가 아닌 계획 텍스트라면 tool call을 강제 재요청
            if cur_tool_choice == "required":
                live["log"].append(_log_entry(
                    "warn",
                    content="모델이 tool call 대신 텍스트를 출력했습니다. 다시 tool을 호출하도록 요청합니다.",
                ))
                _write_live_status(slide_id, live)
                messages.append(assistant_msg)
                messages.append({
                    "role": "user",
                    "content": (
                        "Do NOT write text. Do NOT write Python code. "
                        "You MUST call a tool right now. "
                        "If pipeline is incomplete, call the next pipeline tool. "
                        "If pipeline is done, call get_mil_summary."
                    ),
                })
                continue

            report_text = assistant_msg.get("content", "")
            metadata["usage"] = response.get("usage")
            live["state"] = "completed"
            live["stage"] = "report_done"
            live["report"] = report_text
            live["log"].append(_log_entry("complete", content="최종 보고서 생성 완료"))
            _write_live_status(slide_id, live)
            return report_text, metadata

        if finish_reason == "tool_calls":
            messages.append(assistant_msg)
            pending_images: list[tuple[str, str]] = []

            for tool_call in assistant_msg.get("tool_calls", []):
                fn_name = tool_call["function"]["name"]
                fn_args = json.loads(tool_call["function"]["arguments"] or "{}")

                stage_map = {
                    "get_pipeline_status": "checking_pipeline_status",
                    "run_preprocess_pipeline": "running_preprocess",
                    "run_inference_pipeline": "running_inference",
                    "run_nulite_topk_pipeline": "running_nuclei_topk",
                }
                live["stage"] = stage_map.get(fn_name, f"tool:{fn_name}")

                live["log"].append(_log_entry(
                    "tool_call",
                    iter=iteration + 1,
                    tool=fn_name,
                    args=fn_args,
                ))
                _write_live_status(slide_id, live)

                _called_tools.add(fn_name)
                tool_entry = {"iteration": iteration + 1, "name": fn_name, "args": fn_args}
                metadata["tools_called"].append(tool_entry)
                live["tools_called"] = metadata["tools_called"]

                t_start = time.time()
                if fn_name not in _FULL_DISPATCH:
                    tool_text = f"알 수 없는 도구: {fn_name}"
                    tool_images = []
                    result_preview = tool_text
                    is_multimodal = False
                else:
                    raw = _FULL_DISPATCH[fn_name](slide_id, fn_args)

                    # Sync pipeline status
                    if fn_name == "get_pipeline_status" and isinstance(raw, dict):
                        for key in ("preprocess", "inference", "nuclei_topk"):
                            if key in raw:
                                live["pipeline"][key] = raw[key]
                    elif fn_name == "run_preprocess_pipeline":
                        live["pipeline"]["preprocess"] = raw.get("status", "unknown")
                    elif fn_name == "run_inference_pipeline":
                        live["pipeline"]["inference"] = raw.get("status", "unknown")
                    elif fn_name == "run_nulite_topk_pipeline":
                        live["pipeline"]["nuclei_topk"] = raw.get("status", "unknown")

                    if fn_name in _IMAGE_TOOLS:
                        raw, tool_images = _split_images(fn_name, raw)
                        is_multimodal = len(tool_images) > 0
                    else:
                        tool_images = []
                        is_multimodal = False
                    tool_text = json.dumps(raw, ensure_ascii=False, indent=2)
                    result_preview = tool_text[:400]

                duration_ms = int((time.time() - t_start) * 1000)
                tool_entry["result_summary"] = result_preview
                tool_entry["duration_ms"] = duration_ms

                live["log"].append(_log_entry(
                    "tool_result",
                    iter=iteration + 1,
                    tool=fn_name,
                    result_preview=result_preview,
                    multimodal=is_multimodal,
                    image_count=len(tool_images),
                    duration_ms=duration_ms,
                ))
                _write_live_status(slide_id, live)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_text,
                })
                pending_images.extend(tool_images)

            if pending_images:
                # vLLM 이미지 10장 제한
                capped = pending_images[:_VLLM_MAX_IMAGES_PER_PROMPT]
                image_content: list[dict] = []
                for caption, b64 in capped:
                    image_content.append({"type": "text", "text": caption})
                    image_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })
                if len(pending_images) > _VLLM_MAX_IMAGES_PER_PROMPT:
                    image_content.append({
                        "type": "text",
                        "text": f"[나머지 {len(pending_images) - _VLLM_MAX_IMAGES_PER_PROMPT}장 이미지는 한도 초과로 생략됨]",
                    })
                messages.append({"role": "user", "content": image_content})

    # Force finish — max iterations reached
    live["log"].append(_log_entry("warn", content="⚠ 최대 반복 횟수 도달 — 강제 보고서 생성"))
    live["stage"] = "report_generating"
    _write_live_status(slide_id, live)

    messages.append({
        "role": "user",
        "content": "최대 반복 횟수에 도달했습니다. 지금까지 수집한 정보로 최종 보고서를 작성하시오.",
    })
    response = _call_llm(messages, tools=None, max_tokens=4096)
    report_text = response["choices"][0]["message"].get("content", "")
    metadata["usage"] = response.get("usage")
    metadata["forced_finish"] = True
    live["state"] = "completed"
    live["stage"] = "report_done_forced"
    live["report"] = report_text
    live["log"].append(_log_entry("complete", content="강제 보고서 생성 완료"))
    _write_live_status(slide_id, live)
    return report_text, metadata


def run_agent_full_background(
    slide_id: str,
    user_instructions: str | None,
    output_dir: Path,
) -> None:
    """
    Background thread entry: run_agent_full() and persist report to disk.
    All errors are caught and written to live_status.json.
    """
    try:
        report_text, metadata = run_agent_full(slide_id, user_instructions)
        run_dir = output_dir / "runs" / slide_id
        agent_run_dir = run_dir / "agent_run"
        agent_run_dir.mkdir(parents=True, exist_ok=True)
        (agent_run_dir / "report.md").write_text(report_text, encoding="utf-8")
        import json as _json
        from datetime import datetime, timezone
        (agent_run_dir / "metadata.json").write_text(
            _json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # fetchStatus()가 인식할 수 있도록 report/ 에도 동일하게 저장
        report_dir = run_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_payload = {
            "slide_id": slide_id,
            "report_text": report_text,
            "agent": metadata,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "agent_full_run",
        }
        (report_dir / "diagnostic_report.json").write_text(
            _json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (report_dir / "diagnostic_report.md").write_text(
            report_text.strip() + "\n", encoding="utf-8"
        )
    except Exception as exc:
        p = _live_status_path(slide_id)
        try:
            existing = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except Exception:
            existing = {}
        existing["state"] = "failed"
        existing["error"] = str(exc)
        _write_live_status(slide_id, existing)


def run_feedback_agent(
    slide_id: str,
    feedback_text: str,
    prior_report: str,
) -> tuple[str, dict[str, Any]]:
    """
    Level 1 피드백 재추론:
    병리학자 피드백을 받아 이전 보고서를 context로 제공하고,
    agent가 추가 tool 호출 후 수정 보고서를 생성.
    """
    rag_context = _build_rag_context(slide_id)
    system_content = SYSTEM_PROMPT
    if rag_context:
        system_content = SYSTEM_PROMPT + "\n\n" + rag_context

    messages: list[dict] = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": (
                f"[slide_id={slide_id}] 이전 AI 분석 보고서와 병리학자 피드백이 있습니다.\n\n"
                f"=== 이전 보고서 ===\n{prior_report}\n\n"
                f"=== 병리학자 피드백 ===\n{feedback_text}\n\n"
                "피드백에서 지적한 부분을 재조사하시오. 필요하면 tool을 호출하여 "
                "추가 패치 또는 핵 분석을 수행한 뒤 수정된 보고서를 작성하시오."
            ),
        },
    ]

    metadata: dict[str, Any] = {
        "agent_model":    settings.agent_model,
        "slide_id":       slide_id,
        "mode":           "feedback_revision",
        "iterations":     0,
        "tools_called":   [],
        "forced_finish":  False,
    }

    for iteration in range(settings.agent_max_iter):
        response = _call_llm(messages, TOOL_SCHEMAS)
        choice         = response["choices"][0]
        finish_reason  = choice["finish_reason"]
        assistant_msg  = choice["message"]
        metadata["iterations"] = iteration + 1

        if finish_reason == "stop":
            metadata["usage"] = response.get("usage")
            return assistant_msg.get("content", ""), metadata

        if finish_reason == "tool_calls":
            messages.append(assistant_msg)
            pending_images: list[tuple[str, str]] = []

            for tool_call in assistant_msg.get("tool_calls", []):
                fn_name = tool_call["function"]["name"]
                fn_args = json.loads(tool_call["function"]["arguments"] or "{}")
                metadata["tools_called"].append({
                    "iteration": iteration + 1,
                    "name":      fn_name,
                    "args":      fn_args,
                })
                if fn_name not in _TOOL_DISPATCH:
                    tool_text = f"알 수 없는 도구: {fn_name}"
                    tool_images: list[tuple[str, str]] = []
                else:
                    raw = _TOOL_DISPATCH[fn_name](slide_id, fn_args)
                    if fn_name in _IMAGE_TOOLS:
                        raw, tool_images = _split_images(fn_name, raw)
                    else:
                        tool_images = []
                    tool_text = json.dumps(raw, ensure_ascii=False, indent=2)
                messages.append({
                    "role":        "tool",
                    "tool_call_id": tool_call["id"],
                    "content":     tool_text,
                })
                pending_images.extend(tool_images)

            if pending_images:
                image_content: list[dict] = []
                for caption, b64 in pending_images:
                    image_content.append({"type": "text", "text": caption})
                    image_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })
                messages.append({"role": "user", "content": image_content})

    messages.append({
        "role": "user",
        "content": "최대 반복 횟수에 도달했습니다. 지금까지의 재분석 결과로 수정 보고서를 작성하시오.",
    })
    response = _call_llm(messages, tools=None)
    metadata["usage"]         = response.get("usage")
    metadata["forced_finish"] = True
    return response["choices"][0]["message"].get("content", ""), metadata
