---
name: retail-forecast-evaluation
description: Evaluate and compare retail-demand forecasting runs in this project, including seed stability, sparse-demand metrics, and missing-modality robustness.
metadata:
  short-description: Evaluate retail forecasting and missing-modality results
---

# Retail Forecast Evaluation

Use this skill when reviewing results from `results/v2`, comparing models, selecting a presentation result, or interpreting missing-modality experiments.

## Source of truth

- Read `run_config.json`, `baseline_results.csv`, `seed_summary.csv`, and `robustness_comparison.csv` together. A result table without a completed run configuration is incomplete evidence.
- Compare only runs with the same Stage 1 hashes, split, `seq_len`, `pred_len`, target transform, and evaluation protocol unless the comparison is explicitly labelled exploratory.
- Do not count duplicated files or identical result hashes as independent experiments.

## Metrics for sparse retail demand

- Lead with MAE, RMSE, and wMAPE in original sales units.
- Report mean and standard deviation across seeds for learned models.
- Treat sMAPE as supplementary: zeros make it unstable because any positive prediction for an actual zero can contribute close to 200 percent. Explain this when sMAPE disagrees with MAE or wMAPE.
- Do not infer statistical significance from small numerical differences when they are comparable with seed variation.

## Missing-modality protocol

- Label each row as `same_checkpoint` or `retrained`; they answer different questions and must not be pooled.
- Quantify degradation against the full-modality result from the same model and seed.
- State whether masking uses zero vectors, learnable mask tokens, or modality dropout. A model trained only on full modalities is a robustness baseline, not a learned missing-modality solution.

## Deliverables

- Produce a compact main table, a seed-stability table where relevant, and a missing-modality degradation table.
- State the strongest supported conclusion, the limitation, and the next ablation that would reduce uncertainty.
- Create figures from saved CSVs and record the run identifier in captions. Never fabricate a numeric result for a planned model.
