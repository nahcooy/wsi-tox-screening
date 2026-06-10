# Operations

## Start The Server

```bash
cd /home/nahcooy/MIL/MIL_260527/wsi-tox-screening
./.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend
```

Open:

```text
http://127.0.0.1:8000
```

## Start In Background

```bash
cd /home/nahcooy/MIL/MIL_260527/wsi-tox-screening
nohup ./.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend > /tmp/wsi_gui_8000.log 2>&1 &
```

Check log:

```bash
tail -f /tmp/wsi_gui_8000.log
```

## Stop The Server

Find the process:

```bash
ps -ef | grep "uvicorn app.main:app" | grep -v grep
```

Stop it:

```bash
kill <PID>
```

## Health Check

```bash
curl -s http://127.0.0.1:8000/api/health
```

Expected response includes:

```json
{
  "status": "ok",
  "app": "WSI Toxicity Screening Workbench"
}
```

## Configuration

The app reads configuration from environment variables and a local `.env` file.

Template:

```text
.env.example
```

Local secret/config file:

```text
.env
```

`.env` is ignored by git and should never be committed.

Recommended permissions:

```bash
chmod 600 .env
```

## Important Environment Variables

```text
OUTPUT_DIR=./outputs

TRIDENT_ROOT=/home/nahcooy/MIL/TRIDENT
TRIDENT_PYTHON=/home/nahcooy/miniconda3/envs/gg/bin/python

MIL_LAB_ROOT=/home/nahcooy/MIL/MIL_260527/MIL-Lab
MIL_PYTHON=/home/nahcooy/miniconda3/envs/gg/bin/python
DEFAULT_ABMIL_CHECKPOINT=./models/mil/best_grandqc_univ1_abmil_h5_new_label.pth

NULITE_ROOT=/home/nahcooy/NK/NL/NuLite_patch_wise_inference
NULITE_PYTHON=/home/nahcooy/miniconda3/envs/cv/bin/python
DEFAULT_NULITE_H_CHECKPOINT=./models/nulite/NuLite-H-Weights.pth

MATCHED_DATASET_CSV=../tggates_1to1_matched_dataset (2).csv
CELLTYPE_SUMMARY_CSV=../summary_celltype_final_260509.csv

OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1
OPENAI_TIMEOUT_SECONDS=90
```

## OpenAI API Key

The OpenAI key is read only by the backend.

Do not put the key in frontend files, generated reports, screenshots, or markdown docs.

`.env` example:

```text
OPENAI_API_KEY=<your_key_here>
OPENAI_MODEL=gpt-4.1
OPENAI_TIMEOUT_SECONDS=90
```

After changing `.env`, restart the server.

## Browser Cache

If only `backend/app/static/main.js` or `backend/app/static/styles.css` changed, the server usually does not need to restart.

Use browser hard refresh:

```text
Ctrl + Shift + R
```

If the UI still looks stale, restart the server.

## Existing Run Auto-Load

When a slide file is selected, the UI uses the filename stem as `slide_id`.

Example:

```text
51903.svs -> slide_id 51903
```

If this path exists:

```text
outputs/runs/51903/run.json
```

then existing outputs are loaded automatically into the status panel and center pages.

## Common Troubleshooting

### `OPENAI_API_KEY is not set in the backend server environment`

Cause:

- `.env` does not contain `OPENAI_API_KEY`, or
- server was started before `.env` was updated.

Fix:

```bash
cd /home/nahcooy/MIL/MIL_260527/wsi-tox-screening
chmod 600 .env
ps -ef | grep "uvicorn app.main:app" | grep -v grep
kill <PID>
./.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend
```

### `Not Found` when clicking a workflow button

Likely causes:

- stale browser JavaScript,
- server is not running the latest code,
- wrong port.

Fix:

```text
Ctrl + Shift + R
```

Then verify:

```bash
curl -s http://127.0.0.1:8000/api/health
```

### Existing completed run shows no elapsed seconds

Elapsed seconds are persisted for jobs completed while the current backend process can still see the in-memory job record.

Older runs completed before this feature or before a backend restart may show blank duration fields.

### LLM report markdown looks raw or broken

Use browser hard refresh:

```text
Ctrl + Shift + R
```

The current UI renders markdown headings, lists, links, bold text, inline code, and markdown tables.

## Verification

Run backend tests:

```bash
cd /home/nahcooy/MIL/MIL_260527/wsi-tox-screening
./.venv/bin/pytest backend/tests
```

Compile modified Python files:

```bash
cd /home/nahcooy/MIL/MIL_260527/wsi-tox-screening
./.venv/bin/python -m py_compile backend/app/api/routes_workflow.py backend/app/settings.py backend/app/services/diagnostic_report.py
```

Check static files are served:

```bash
curl -s http://127.0.0.1:8000/static/main.js
curl -s http://127.0.0.1:8000/static/styles.css
```
