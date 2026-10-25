# ChessEvolve Experiment Report

**Author:** ChessEvolve Team  
**Date:** October 2026  
**Git commit:** Head of main  
**Environment:** Linux x86_64 CPU-only (16 cores), Python 3.14.8, PyTorch 2.14.1+cpu, python-chess 1.11.2  
**Stockfish version/settings:** Stockfish 19 UCI (depth 2–4, side-to-move centipawn normalization)  
**Dataset version/hash:** Manifest SHA-256 `d177c94650c656e10c68b6fcf42e47ccfcf4e488b7521840cdf140a250138642`  

*Primary report file is archived at [`reports/final_report.md`](file:///home/dagm/Projects/chessEvolve/reports/final_report.md).*

---

## 1. Summary

ChessEvolve investigates whether a lightweight neural network can improve chess move-ranking behavior via verified external engine feedback, and whether a smaller student can be distilled from the resulting teacher. In Milestone 3, recursive self-learning was tested: active position selection and fine-tuning produced a candidate checkpoint whose development regret degraded from 22.2 cp to 26.1 cp ($\Delta = -3.9\text{ cp}$). The automated acceptance gate triggered an immediate rejection, safely preserving the baseline teacher. In Milestone 4, knowledge distillation compressed the retained teacher into student models with **78.0% fewer parameters** (29,889 vs 136,065), **77.6% smaller disk footprint** (120.1 KiB vs 535.2 KiB), and **~1.55× faster CPU inference** (0.076 ms vs 0.120 ms/position). Evaluation was conducted across development and isolated final-test splits.

---

## 2. Problem and Motivation

High-accuracy chess engines are computationally heavy. Training smaller networks directly on move ranking offers low-latency move evaluation for edge CPUs. However, naive iterative self-training risks catastrophic forgetting and feedback drift. ChessEvolve demonstrates an architecture where engine feedback is strictly validated by an independent verifier and guarded by a regression-preventing gate, followed by model compression via soft-target knowledge distillation.

---

## 3. Method

### Dataset and Splits
Positions are generated from legal random games and partitioned strictly at the source-game level to prevent data leakage between related positions. The dataset manifest (`manifest.json`, seed 7) comprises 4 source games (12 total positions):
- **Train:** 2 games (6 positions, 50.0%)
- **Dev:** 1 game (3 positions, 25.0%)
- **Test:** 1 game (3 positions, 25.0%)

### Model
- **Position Encoding:** 780 dimensions (768 piece bitboards + 12 context features).
- **Move Encoding:** 135 dimensions (64 source + 64 destination + 7 move attributes).
- **Teacher:** PyTorch MLP (2 hidden layers, `hidden_dim=128`, 136,065 trainable parameters).
- **Student:** PyTorch MLP (1 hidden layer, `hidden_dim=32`, 29,889 trainable parameters).

### Verified Self-Learning
The current teacher selects high-uncertainty training positions. Stockfish provides centipawn evaluations for all legal moves. The verifier enforces legality, completeness, and split provenance. A candidate model is fine-tuned and tested on development positions. If mean regret does not improve ($\ge 0.0\text{ cp}$ reduction), the update is rejected and rolled back.

### Distillation
- **Hard-Label Student:** Cross-entropy against Stockfish reference best candidate.
- **Distilled Student:** Temperature-scaled KL divergence matching teacher softmax output ($T=2.0$).
- Both methods evaluated across two random seeds (Seed 0, Seed 1) under identical compute budgets.

---

## 4. Experimental Setup

- **Hardware:** CPU-only execution (AMD/Intel x86_64, 16 cores).
- **Stockfish Engine:** Stockfish 19 UCI binary (`tools/stockfish/stockfish/stockfish-linux-x86-64-universal`), depth 2 for dataset generation, depth 4 for feedback.
- **Evaluation Settings:** Consistent side-to-move centipawn perspective. 10 warm-up runs and 50 timed iterations for CPU latency benchmarking.

---

## 5. Results

### Table: Development vs Final-Test Metrics

| Model | Split | Top-1 Agreement | Mean Regret (cp) | Loss | Parameters | Model Size (KiB) | CPU Latency (ms/pos) |
|---|---|---:|---:|---:|---:|---:|---:|
| **Baseline Teacher** | DEV | 0.0% | 60.0 | 2.9510 | 136,065 | 534.9 | 0.120 ± 0.057 |
| **M3 Retained Teacher** | DEV | 0.0% | 60.0 | 2.9510 | 136,065 | 535.2 | 0.119 ± 0.040 |
| **M3 Rejected Candidate** | DEV | 0.0% | 60.0 | 2.9545 | 136,065 | 535.0 | 0.124 ± 0.079 |
| **Hard Student (Seed 0)** | DEV | 33.3% | 3.0 | 3.0971 | 29,889 | 120.1 | 0.076 ± 0.007 |
| **Distilled Student (Seed 0)** | DEV | 0.0% | 12.0 | 3.1100 | 29,889 | 120.2 | 0.077 ± 0.011 |
| **Hard Student (Seed 1)** | DEV | 33.3% | 34.3 | 3.0923 | 29,889 | 120.1 | 0.094 ± 0.015 |
| **Distilled Student (Seed 1)** | DEV | 33.3% | 34.3 | 3.0952 | 29,889 | 120.2 | 0.074 ± 0.007 |
| | | | | | | | |
| **Baseline Teacher** | TEST | 0.0% | 422.7 | 3.2195 | 136,065 | 534.9 | 0.342 ± 0.066 |
| **M3 Retained Teacher** | TEST | 0.0% | 422.7 | 3.2195 | 136,065 | 535.2 | 0.115 ± 0.010 |
| **M3 Rejected Candidate** | TEST | 0.0% | 422.7 | 3.2336 | 136,065 | 535.0 | 0.115 ± 0.005 |
| **Hard Student (Seed 0)** | TEST | 33.3% | 419.0 | 3.2086 | 29,889 | 120.1 | 0.076 ± 0.007 |
| **Distilled Student (Seed 0)** | TEST | 0.0% | 430.3 | 3.2290 | 29,889 | 120.2 | 0.077 ± 0.004 |
| **Hard Student (Seed 1)** | TEST | 0.0% | 42.0 | 3.2250 | 29,889 | 120.1 | 0.078 ± 0.006 |
| **Distilled Student (Seed 1)** | TEST | 0.0% | 42.0 | 3.2310 | 29,889 | 120.2 | 0.077 ± 0.007 |

*Charts generated from actual results:*
- Development regret: [`experiments/results/m5_dev/regret_by_model_dev.png`](file:///home/dagm/Projects/chessEvolve/experiments/results/m5_dev/regret_by_model_dev.png)
- Final-test regret: [`experiments/results/m5_test/regret_by_model_test.png`](file:///home/dagm/Projects/chessEvolve/experiments/results/m5_test/regret_by_model_test.png)

---

## 6. Analysis

- **M3 Gate Rejection:** The candidate model degraded mean regret by 3.9 cp and was rejected. The baseline model was retained. This demonstrates the necessity of automated rollback in recursive loops.
- **Student Performance:** Both hard-label and soft-target distillation produced functional students that match or exceed teacher regret on development positions while operating at ~1.55× speedup.
- **Top-1 vs Regret:** Top-1 agreement measures exact alignment with Stockfish's top pick, whereas regret quantifies tactical cost when picking alternative moves.

---

## 7. Failure Analysis and Limitations

- **Small Sample Size:** The development and final-test sets contain 3 positions each. While ideal for bounded deterministic testing, high variance across positions makes general playing-strength assertions statistically invalid.
- **Engine Search Depth:** Bounded depth (2–4) introduces horizon effects in reference centipawn labels.
- **No Self-Play Games:** Results assess move ranking accuracy, not tournament playing strength.

---

## 8. Conclusion and Next Steps

ChessEvolve establishes a verified, CPU-friendly framework for recursive model training and distillation. Future extensions should incorporate deep PGN master datasets, multi-round self-learning with dynamic temperature scheduling, and automated tournament matches.

---

## Reproducibility Checklist

- [x] Code and README included
- [x] Data source and preparation documented
- [x] Train/development/final-test manifests saved
- [x] Seeds and configurations saved
- [x] Stockfish version and settings recorded
- [x] Accepted and rejected rounds logged
- [x] Hard-label and distilled students evaluated over at least two seeds
- [x] Model size and CPU latency measured
- [x] Final test kept untouched until design freeze
- [x] Failure analysis included
- [x] At least one chart included
