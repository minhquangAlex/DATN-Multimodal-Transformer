---
name: kaggle-ml-experiment
description: Prepare, validate, run, recover, and package this retail-demand project's Kaggle experiments, especially embedding extraction and baseline or multimodal training.
metadata:
  short-description: Kaggle workflow for retail forecasting experiments
---

# Kaggle ML Experiment

Use this skill for requests to prepare, troubleshoot, run, recover, or package a Kaggle experiment in this project. It applies to `notebooks/02_MultiModal_Vector.ipynb`, `notebooks/03_baselines.ipynb`, their source under `src/`, and their artifacts under `processed_data/v2`, `embeddings/v2`, and `results/v2`.

## Preserve experiment validity

- Keep Stage 1 v2 data, Stage 2 embeddings, and Stage 3 results paired by their metadata and hashes. Do not bypass validation by copying a metadata file from a different selection.
- Treat `/kaggle/input` as read-only and place generated artifacts in `/kaggle/working`.
- Before a costly run, verify the selected-article CSV, image/text embedding files, `metadata.json`, accelerator, device, and output path.
- Use the notebook's path resolvers or a clearly named override. Never silently choose among multiple matching Kaggle inputs.
- Do not claim an experiment completed until its `run_config.json` has `status: complete`.

## Choose the appropriate run mode

- **Smoke test:** one seed, short epoch budget, selected learning models only. Use it to check paths, GPU operation, dimensions, and output packaging.
- **Presentation baseline:** use the fixed split, `L=12`, `H=2`, `log1p`, early stopping, and the final seed configuration selected by the user. Keep the full metric table and run metadata.
- **Missing-modality evaluation:** distinguish same-checkpoint masking from retraining under a missing-modality condition. State which protocol is used in every result table.

## GPU and runtime

- Detect and print the actual device before training. If GPU use is required but CUDA is unavailable, stop before starting a long CPU run.
- During exploratory runs, reduce seeds and disable retrained ablation before reducing the dataset or changing the temporal split.
- Retain the full validation set for a final reported run. A validation subset is only appropriate for explicitly labelled profiling.

## Artifact delivery

- Archive the complete output directory, including result CSVs, run configuration, checkpoint summaries, and selected prediction files.
- For Kaggle download failures, prefer saving the notebook version/output and retrieving the archived output; do not rely on a transient `FileLink` to a large `.pt` file.
- Report exact input dataset slugs or local paths the user must attach, but do not publish a notebook or dataset unless explicitly authorized.
