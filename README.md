# WSI Toxicity Screening Workbench

Rat liver H&E whole-slide image toxicity screening GUI that connects slide-level MIL prediction, attention heatmap review, NuLite-H nuclei analysis, case-control morphometric comparison, and an OpenAI API pathology-support report.

## Project Summary

This project is a local FastAPI web application for explainable rat liver toxicity screening.

Pipeline:

```text
Input WSI
-> TRIDENT preprocessing and UNI v1 feature extraction
-> ABMIL slide-level normal/abnormal inference
-> attention heatmap and top-k patch extraction
-> NuLite-H nuclei segmentation/type inference on top-k patches
-> patch-wise Hep/NPC/Imm morphometric metrics
-> case-control reference comparison
-> OpenAI API Korean pathology decision-support report
```

The current product UI is the backend-served static workbench in:

```text
backend/app/static/
```

The older React/Vite scaffold remains in `frontend/`, but it is not the active GUI path.

## Current Features

- Select and copy a WSI slide into a per-slide run directory.
- Auto-detect existing run artifacts when the same slide is selected again.
- Run TRIDENT single-slide preprocessing.
- Run ABMIL inference with logits, softmax, prediction, confidence, top-k patches, and attention outputs.
- Render a high-resolution attention heatmap thumbnail.
- Export QuPath-compatible attention GeoJSON colored with `Spectral_r_high_red`.
- Run NuLite-H on top-k high-attention patch crops.
- Show NuLite-H cell type counts, patch overlay gallery, and clickable full-size patch contour view.
- Toggle transparent contour fill on/off.
- Compute patch-wise liver cell-group metrics using the local Hep/NPC/Imm mapping.
- Compare top-k patch metrics against matched case/control reference statistics.
- Generate, save, reload, and render a markdown LLM diagnostic report in page 4.
- Show stage status and elapsed seconds for preprocessing, inference, NuLite, and LLM report generation.

## Quick Start

```bash
cd /home/nahcooy/MIL/MIL_260527/wsi-tox-screening
./.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend
```

Open:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```bash
curl -s http://127.0.0.1:8000/api/health
```

## Main Workflow

1. Select a WSI file.
2. If an existing run is found under `outputs/runs/{slide_id}`, the UI loads it automatically.
3. Otherwise click `Copy Slide To Run Dir`.
4. Click `Run Preprocess`.
5. Click `Run Model Inference`.
6. Review page `1 Attention`.
7. Click `Run Nuclei Level Analysis`.
8. Review page `2 NuLite` and page `3 Metrics`.
9. Click `Generate Diagnostic Report` on page `4 LLM`.

## Output Layout

Outputs are organized per slide:

```text
outputs/runs/{slide_id}/
  run.json
  slide/
  trident/
  mil/
  nuclei/
  report/
```

Key files:

```text
mil/mil_result.json
mil/attention_scores.csv
mil/attention_heatmap_qupath.geojson
mil/attention_heatmap_thumbnail.png
mil/topk/top25_patches.json
mil/topk/rank_*.png

nuclei/nuclei_summary.json
nuclei/all_instances.json
nuclei/all_instances.jsonl
nuclei/nuclei_instances.geojson
nuclei/cell_type_counts.csv
nuclei/patch_metrics.json
nuclei/patch_metrics.csv
nuclei/metric_comparison.json
nuclei/overlays/rank_*_nulite_overlay.png

report/diagnostic_report.json
report/diagnostic_report.md
report/diagnostic_report_input.json
```

## Data And Model Context

- Dataset context: TG-GATEs rat liver H&E WSI.
- Slide-level model: ABMIL with UNI v1 features.
- Preprocessing/feature extraction: TRIDENT.
- Nuclei model: NuLite-H.
- Nuclei classes used in the UI:
  - `1 Neoplastic`
  - `2 Inflammatory`
  - `3 Connective`
  - `4 Dead`
  - `5 Epithelial`
- Cell-group mapping for patch metrics:
  - `Neoplastic -> Hep`
  - `Epithelial -> Hep` if `area_um2 >= 30`, otherwise `NPC`
  - `Connective -> NPC`
  - `Inflammatory -> Imm`
  - `Dead/Background -> excluded`

## Configuration

Runtime configuration is read from environment variables and `.env`.

Copy or edit:

```text
.env.example
```

Important variables:

```text
TRIDENT_ROOT
TRIDENT_PYTHON
MIL_LAB_ROOT
MIL_PYTHON
DEFAULT_ABMIL_CHECKPOINT
NULITE_ROOT
NULITE_PYTHON
DEFAULT_NULITE_H_CHECKPOINT
MATCHED_DATASET_CSV
CELLTYPE_SUMMARY_CSV
OPENAI_API_KEY
OPENAI_MODEL
```

Do not commit `.env`; it may contain secrets. The repository `.gitignore` excludes it.

## Documentation

Current docs:

```text
docs/README.md
docs/WORKFLOW.md
docs/OPERATIONS.md
```

## Tests

```bash
cd /home/nahcooy/MIL/MIL_260527/wsi-tox-screening
./.venv/bin/pytest backend/tests
```

Last verified test suite:

```text
5 passed
```
