---
name: retail-code-verify
description: Review, validate, and repair generated Python and notebook code in this retail-demand forecasting project without launching costly training or altering experiment artifacts.
metadata:
  short-description: Find and fix verified code defects in the retail project
---

# Retail Code Verify

Use this skill when the user asks to double-check, review, validate, debug, or repair generated code in this retail-demand forecasting project. It covers `src/`, `tests/`, `scripts/`, and notebooks `01_eda.ipynb`, `02_MultiModal_Vector.ipynb`, and `03_baselines.ipynb`.

## Review before changing code

- Read the affected implementation, its callers, the related test, and the matching notebook cell before proposing a fix.
- Compare changed code with the project invariants: temporal split without future leakage, selected-article/embedding provenance, non-negative sales forecasts after inverse transform, and an explicit missing-modality protocol.
- Treat an error shown by a traceback, a failing test, inconsistent output, or a reproducible static defect as evidence. Do not rewrite working components for style alone.
- Check the existing diff and preserve unrelated user changes.

## Validation ladder

Start with the least costly meaningful validation and report what was run:

1. Check Python syntax and imports in the affected source.
2. Run the smallest relevant test file under `tests/`.
3. Run the complete `pytest -q` suite after a cross-cutting change or before declaring the repair complete.
4. For notebook edits, validate JSON structure and code/import/path consistency. Execute cells only when the user explicitly requests it or a lightweight cell can reproduce the fault without downloading models, extracting embeddings, or training.
5. For data or embedding code, validate metadata, IDs, shapes, and availability rather than regenerating embeddings.

Do not start a full training run, download pretrained models, rebuild Stage 1, overwrite `embeddings/v2`, or overwrite `results/v2` merely to verify a code fix. Ask before any action that would consume a material Kaggle quota, replace existing experimental evidence, or alter external Kaggle state.

## Repair rules

- Make the smallest change that resolves the evidenced defect and preserves public function signatures unless a signature change is necessary.
- Add or update a focused regression test when the bug can recur and the test is cheap and meaningful.
- Keep Local and Kaggle paths explicit. Avoid hard-coded machine-specific paths in reusable source.
- Retain validation errors that protect data/version compatibility; improve their message only when it helps the user diagnose the issue.
- Do not silence exceptions, skip tests, weaken data checks, or replace missing inputs with fabricated artifacts to obtain a passing run.

## Completion report

State the defect, its impact, files changed, validation performed, and any remaining limitation. When no defect is found, say so and identify what was checked instead of inventing an improvement.
