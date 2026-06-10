# WSI Toxicologic Pathology Screening Workbench

Local FastAPI web application for explainable rat liver H&E WSI toxicity screening.  
Connects ABMIL slide-level prediction, attention heatmap, NuLite-H nuclei analysis, case-control morphometrics, and a Qwen2.5-VL multimodal agent that autonomously runs the full pipeline and generates a Korean pathology report.

---

## Pipeline

```
WSI
 └─ TRIDENT  ──── patch segmentation + UNI-v1 feature extraction
     └─ ABMIL ─── slide-level normal/abnormal prediction + attention scores
         └─ NuLite-H ── nuclei segmentation & cell-type classification (top-k patches)
             └─ Morphometrics ── Hep/NPC/Imm metrics vs. case-control reference
                 └─ Qwen2.5-VL Agent ── multimodal tool-calling loop → Korean pathology report
```

---

## Features

| Category | Detail |
|---|---|
| **Pipeline** | One-click: preprocess → inference → nuclei → report |
| **Agent** | Qwen2.5-VL-72B via vLLM; tool-call loop with `tool_choice="required"` until all analysis tools are called |
| **Multimodal** | Agent directly views top-attention H&E patches and NuLite overlay images |
| **Live log** | Structured real-time log (thinking / tool_call / tool_result / complete) streamed to UI |
| **Attention** | Heatmap thumbnail, QuPath GeoJSON export, spatial quadrant analysis |
| **Nuclei** | NuLite-H cell type counts, patch overlay gallery, NuLite GeoJSON export |
| **Metrics** | Patch-wise Hep/NPC/Imm morphometrics vs. matched case/control Z-scores |
| **Report** | Manual (Qwen direct) and Agent (autonomous loop) reports, saved as Markdown |
| **Exports** | QuPath GeoJSON, Top-25 manifest, Cell type CSV, Patch metrics CSV, Metric comparison JSON |

---

## Quick Start

### 1. Requirements

- Python 3.11+
- [TRIDENT](https://github.com/mahmoodlab/TRIDENT) for preprocessing/feature extraction
- [NuLite](https://github.com/A-Haider13/NuLite) for nuclei inference
- [vLLM](https://github.com/vllm-project/vllm) serving Qwen2.5-VL-72B with Hermes tool-call parser

### 2. Install

```bash
cd backend
pip install -e .
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your paths
```

Key variables:

```env
TRIDENT_ROOT=/path/to/TRIDENT
TRIDENT_PYTHON=/path/to/envs/trident/bin/python

MIL_LAB_ROOT=/path/to/MIL-Lab
MIL_PYTHON=/path/to/envs/mil/bin/python
DEFAULT_ABMIL_CHECKPOINT=models/mil/best_grandqc_univ1_abmil_h5_new_label.pth

NULITE_ROOT=/path/to/NuLite_patch_wise_inference
NULITE_PYTHON=/path/to/envs/nulite/bin/python
DEFAULT_NULITE_H_CHECKPOINT=models/nulite/NuLite-H-Weights.pth

MATCHED_DATASET_CSV=/path/to/case_control_matched.csv
CELLTYPE_SUMMARY_CSV=/path/to/summary_celltype.csv

# Qwen2.5-VL via vLLM
AGENT_BASE_URL=http://localhost:8080
AGENT_MODEL=/path/to/Qwen2.5-VL-72B-Instruct-AWQ
AGENT_API_KEY=EMPTY
```

### 4. Start vLLM (Qwen2.5-VL-72B)

```bash
vllm serve /path/to/Qwen2.5-VL-72B-Instruct-AWQ \
  --port 8080 \
  --tool-call-parser hermes \
  --enable-auto-tool-choice \
  --limit-mm-per-prompt image=10 \
  --max-model-len 16384
```

### 5. Start the workbench

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000`

---

## Agent Mode

Click **Run Agent** to let the agent autonomously:

1. Check pipeline status → run any missing steps (preprocess / inference / nuclei)
2. Call `get_mil_summary`, `get_attention_heatmap`, `get_nuclei_summary`, `get_metric_comparison`
3. View top-attention H&E patches and NuLite overlays directly as images
4. Generate a Korean NTP/INHAND-terminology pathology report

The agent enforces `tool_choice="required"` until all four analysis tools have been called, preventing the model from skipping to text generation early.

---

## Output Layout

```
outputs/runs/{slide_id}/
  run.json
  slide/
  trident/                    # TRIDENT features (.h5)
  mil/
    mil_result.json
    attention_scores.csv
    attention_heatmap_thumbnail.png
    attention_heatmap_qupath.geojson
    topk/
      top25_patches.json
      rank_*.png
  nuclei/
    nuclei_summary.json
    nuclei_instances.geojson
    cell_type_counts.csv
    patch_metrics.json
    metric_comparison.json
    overlays/
      rank_*_nulite_overlay.png
  report/
    diagnostic_report.md
    diagnostic_report.json
  agent_run/
    live_status.json          # real-time agent progress
    agent_report.md
    agent_report.json
```

---

## Cell Type Mapping

NuLite-H was trained on PanNuke (pan-cancer human tissue). Labels are remapped for rat liver:

| NuLite label | Rat liver interpretation | Morphometric group |
|---|---|---|
| Neoplastic | Large hepatocytes (expected majority in normal liver) | Hep |
| Epithelial (area ≥ 30 µm²) | Hepatocytes | Hep |
| Epithelial (area < 30 µm²) | Non-parenchymal cells | NPC |
| Connective | Portal fibroblasts, endothelium | NPC |
| Inflammatory | Lymphocytes, Kupffer cells | Imm |
| Dead / Background | Excluded | — |

> A high "Neoplastic" ratio in a normal liver prediction is expected and does **not** indicate true neoplasia.

---

## Model Weights

Model weights are **not** included in this repository.

```
models/mil/   ← place ABMIL checkpoint here
models/nulite/ ← place NuLite-H weights here
```

---

## Tests

```bash
cd backend
pytest tests/
```

---

## References

- Thoolen et al. (2010). Proliferative and nonproliferative lesions of the rat and mouse hepatobiliary system. *Toxicol Pathol* 38(7 Suppl):5S–81S.
- [NTP Nonneoplastic Lesion Atlas](https://ntp.niehs.nih.gov/nnl)
- Chen et al. (2024). [TRIDENT](https://github.com/mahmoodlab/TRIDENT) — Universal WSI preprocessing.
- [NuLite](https://github.com/A-Haider13/NuLite) — Lightweight nuclei instance segmentation.
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) — Multimodal language model.
