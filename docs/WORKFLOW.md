# Workflow And Implementation

## Architecture

The active application is served by FastAPI.

```text
backend/app/main.py
  -> mounts /static from backend/app/static
  -> mounts /outputs from outputs/
  -> mounts API routers under /api

backend/app/static/index.html
backend/app/static/styles.css
backend/app/static/main.js
  -> active browser GUI

backend/app/api/routes_workflow.py
  -> end-to-end workflow API

backend/scripts/
  -> TRIDENT, ABMIL, NuLite-H, and patch metric scripts
```

The `frontend/` React/Vite scaffold is not the current GUI. The current UI is intentionally lightweight static HTML/CSS/JavaScript served by FastAPI.

## API Endpoints

Primary workflow endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Backend health check |
| `POST` | `/api/workflow/slides/upload` | Copy selected WSI into the run directory |
| `GET` | `/api/workflow/runs/{slide_id}` | Load run status and existing artifacts |
| `POST` | `/api/workflow/runs/{slide_id}/preprocess` | Run TRIDENT preprocessing |
| `POST` | `/api/workflow/runs/{slide_id}/inference` | Run ABMIL slide-level inference |
| `POST` | `/api/workflow/runs/{slide_id}/nuclei` | Run NuLite-H top-k patch nuclei analysis |
| `POST` | `/api/workflow/runs/{slide_id}/report` | Generate OpenAI API diagnostic report |

Static routes:

| Route | Purpose |
|---|---|
| `/` | GUI entrypoint |
| `/static/*` | HTML/CSS/JS assets |
| `/outputs/*` | Generated run artifacts |
| `/docs` | FastAPI OpenAPI UI |

## UI Pages

The center viewer has four pages.

| Page | Name | Contents |
|---|---|---|
| `1` | Attention | Attention heatmap thumbnail and top-25 patch gallery |
| `2` | NuLite | Nuclei summary, cell type counts, patch overlays, clickable contour viewer |
| `3` | Metrics | Patch-wise Hep/NPC/Imm metrics and case-control distribution comparison |
| `4` | LLM | Markdown-rendered pathology-support report and report generation button |

The right sidebar shows:

- Slide ID.
- Copy/preprocess/inference/nuclei/LLM status.
- Stage elapsed seconds to one decimal place for long-running stages.
- Model confidence and logits/softmax.
- Download links for major exports.
- Status details.

## Stage 1: Slide Copy

When a user selects a slide, the UI derives:

```text
slide_id = Path(slide_filename).stem
```

If `outputs/runs/{slide_id}/run.json` already exists, the UI loads the existing run immediately.

Otherwise `POST /api/workflow/slides/upload` copies the WSI into:

```text
outputs/runs/{slide_id}/slide/{slide_filename}
```

and creates:

```text
outputs/runs/{slide_id}/run.json
```

## Stage 2: TRIDENT Preprocessing

Endpoint:

```text
POST /api/workflow/runs/{slide_id}/preprocess
```

Script:

```text
backend/scripts/run_trident_single_slide.py
```

Default workflow:

- GrandQC tissue segmentation.
- GrandQC artifact removal.
- 20x magnification.
- 256 px patches.
- 0 px overlap.
- UNI v1 feature extraction.

Important outputs:

```text
outputs/runs/{slide_id}/trident/20x_256px_0px_overlap/features_uni_v1/{slide_id}.h5
outputs/runs/{slide_id}/trident/20x_256px_0px_overlap/patches/{slide_id}_patches.h5
```

## Stage 3: ABMIL Inference

Endpoint:

```text
POST /api/workflow/runs/{slide_id}/inference
```

Script:

```text
backend/scripts/run_abmil_inference.py
```

Model context:

- MIL model: ABMIL.
- Encoder feature source: UNI v1.
- Class order: `normal`, `abnormal`.

Important outputs:

```text
mil/mil_result.json
mil/attention_scores.csv
mil/attention_heatmap_qupath.geojson
mil/attention_heatmap_thumbnail.png
mil/topk/top25_patches.json
mil/topk/rank_*.png
```

Attention normalization currently recorded in `mil_result.json`:

```json
{
  "method": "percentile_clip_minmax",
  "lower_percentile": 1.0,
  "upper_percentile": 99.0
}
```

Heatmap coloring:

```text
Spectral_r_high_red
```

Higher normalized attention is redder.

## Stage 4: NuLite-H Nuclei Analysis

Endpoint:

```text
POST /api/workflow/runs/{slide_id}/nuclei
```

Script:

```text
backend/scripts/run_nulite_topk_inference.py
```

NuLite source:

```text
/home/nahcooy/NK/NL/NuLite_patch_wise_inference
```

Default checkpoint:

```text
models/nulite/NuLite-H-Weights.pth
```

NuLite type labels:

| Type | Label |
|---|---|
| `0` | Background |
| `1` | Neoplastic |
| `2` | Inflammatory |
| `3` | Connective |
| `4` | Dead |
| `5` | Epithelial |

Important outputs:

```text
nuclei/nuclei_summary.json
nuclei/all_instances.json
nuclei/all_instances.jsonl
nuclei/nuclei_instances.geojson
nuclei/cell_type_counts.csv
nuclei/cell_type_counts.json
nuclei/overlays/rank_*_nulite_overlay.png
```

The UI renders cell contours from `all_instances.json` as SVG over the selected patch image. Contour fill can be toggled on/off.

## Stage 5: Patch Metrics And Case-Control Comparison

Script:

```text
backend/scripts/compute_nulite_patch_metrics.py
```

Reference inputs:

```text
/home/nahcooy/MIL/MIL_260527/tggates_1to1_matched_dataset (2).csv
/home/nahcooy/MIL/MIL_260527/summary_celltype_final_260509.csv
```

Cell-group mapping:

| NuLite output | Metric group |
|---|---|
| `Neoplastic` | `Hep` |
| `Epithelial` | `Hep` if `area_um2 >= 30`, otherwise `NPC` |
| `Connective` | `NPC` |
| `Inflammatory` | `Imm` |
| `Dead`, `Background` | excluded |

Metrics shown in page 3:

```text
Hep_Area_Mean
Hep_Area_Median
Hep_Area_P90
Hep_Solidity_Mean
Hep_Circularity_Mean
Hep_Convexity_Mean
Hep_AspectRatio_Mean
NPC_Area_Mean
NPC_Circularity_Mean
Imm_Area_Mean
Imm_Circularity_Mean
```

Important outputs:

```text
nuclei/patch_metrics.json
nuclei/patch_metrics.csv
nuclei/metric_comparison.json
```

## Stage 6: LLM Diagnostic Report

Endpoint:

```text
POST /api/workflow/runs/{slide_id}/report
```

Service:

```text
backend/app/services/diagnostic_report.py
```

OpenAI API:

```text
POST https://api.openai.com/v1/responses
```

Configured model:

```text
OPENAI_MODEL
```

The default in `.env.example` is:

```text
gpt-4.1
```

Report inputs:

- ABMIL logits/softmax/prediction/confidence.
- Attention normalization and top-k attention patch scores.
- NuLite-H nuclei counts and per-patch cell type counts.
- Patch-wise Hep/NPC/Imm metrics.
- Case-control comparison statistics.
- Built-in rat liver toxicologic pathology domain knowledge sources.

Important outputs:

```text
report/diagnostic_report.json
report/diagnostic_report.md
report/diagnostic_report_input.json
```

The UI renders the markdown report as formatted HTML in page 4.

The generated report is decision-support only and should not be treated as a final toxicologic pathology diagnosis without pathologist review.
