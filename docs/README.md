# Documentation Index

This directory contains the current documentation for the WSI Toxicity Screening Workbench.

## Documents

- [`../README.md`](../README.md): project overview, quick start, output layout, and configuration summary.
- [`WORKFLOW.md`](WORKFLOW.md): implementation workflow, API endpoints, data flow, output files, and UI pages.
- [`OPERATIONS.md`](OPERATIONS.md): server start/stop, environment variables, OpenAI key handling, troubleshooting, and verification commands.

## Current Implementation Snapshot

The active application is a FastAPI backend with a backend-served static HTML/CSS/JavaScript GUI.

The implemented workflow is:

```text
WSI slide
-> TRIDENT preprocessing
-> ABMIL slide-level inference
-> attention heatmap/top-k patches
-> NuLite-H top-k nuclei analysis
-> Hep/NPC/Imm patch metrics
-> case-control comparison
-> OpenAI API pathology-support report
```

The legacy specification/review documents were removed because they described the earlier scaffold and no longer matched the implemented project.
