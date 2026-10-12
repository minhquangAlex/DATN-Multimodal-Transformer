---
name: thesis-experiment-report
description: Turn verified artifacts from this multimodal retail-demand thesis into traceable report sections, figures, captions, and defense-slide content.
metadata:
  short-description: Build traceable thesis figures, tables, and slide content
---

# Thesis Experiment Report

Use this skill when preparing the written thesis, defense slides, figures, tables, or captions from this project's real artifacts.

## Evidence discipline

- Separate implemented work, completed experimental evidence, and planned Stage 4–5 work. Do not present a proposed Multimodal Transformer, mask token, modality dropout, or alignment loss as implemented until its run artifacts exist.
- Trace every reported numerical claim to a run identifier and source CSV. For Stage 1 and Stage 2 claims, use the corresponding summary, audit, validation report, and metadata.
- State the fixed split and forecast task: 8,000 selected articles, 104 complete weeks, train/validation/test temporal split, `L=12`, and `H=2`, only when those remain true for the cited run.

## Recommended narrative

1. Describe data construction and leakage prevention.
2. Describe text and image representation provenance and coverage.
3. Present sales-only baselines before multimodal results.
4. Present missing-modality results with their protocol and seed variability.
5. Relate limitations directly to the next experiment, especially fusion quality and learned handling of absent modalities.

## Visuals and outputs

- Prefer reproducible charts generated from versioned CSVs for measured results.
- Use Mermaid or code-based diagrams for model/dataflow architecture. Use generative images only for non-quantitative illustration.
- Make captions self-contained: what is compared, which metric/protocol, the data split, and the run identifier when applicable.
- For Word, PDF, or PowerPoint deliverables, use the appropriate document, PDF, or presentation workflow and visually inspect rendered output.
