# ChessEvolve Report Guide

The final submission report is [`reports/final_report.md`](../reports/final_report.md).
Use this file as a checklist when regenerating or extending it.

## Required sections

1. Research question and project scope.
2. Dataset provenance, source-game split, filtering, and fixed seeds.
3. Model architecture and baseline.
4. Fair comparison table with top-1, regret, and efficiency metrics.
5. Self-learning and distillation results, including negative results.
6. Failure analysis and explicit limitations.
7. Reproduction commands and artifact paths.

## Current authoritative experiment

- Dataset: `data/processed/dag12yi_phase35_labeled/`
- Manifest seed: `20261026`
- Records: 87,993 train, 19,101 development, 19,324 test generated;
  87,962, 19,101, and 19,318 labeled.
- Final evaluation: 19,316 test positions after two invalid-FEN exclusions.
- Recommended checkpoint: `experiments/results/v2_mixed_B/best.pt`
- Current tests: 204 Python tests.

Do not replace these values with the earlier smoke dataset. If a new
experiment is run, record its manifest, seed, checkpoint, split, and result
JSON separately before updating the report.
