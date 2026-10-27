# ChessEvolve Submission Report

## Summary

ChessEvolve is a CPU-friendly chess move-ranking system. `python-chess`
generates legal moves and a small PyTorch model ranks them using
Stockfish-generated centipawn labels. The project evaluates chess-aware
features, centipawn-aware ranking losses, verified self-learning, knowledge
distillation, and shallow neural-guided search. It is an engineering and
research study, not a replacement for Stockfish.

The strongest validated model is the mixed classification-plus-ranking B
model. On the untouched final test split it achieved 27.09% top-1 agreement
and 171.4 cp non-mate regret. The original B teacher achieved 26.70% and
200.0 cp respectively. The improvement is modest but reproducible and is
reported alongside the cases that did not help.

## Dataset and reproducibility

The reproducible dataset is
`data/processed/dag12yi_phase35_labeled/`. Its manifest records seed
`20261026`, source-game-level splitting, and the following sizes:

| Split | Source games | Generated records | Labeled records |
|---|---:|---:|---:|
| Train | 3,357 | 87,993 | 87,962 |
| Development | 720 | 19,101 | 19,101 |
| Final test | 719 | 19,324 | 19,318 |

The final evaluation excludes two invalid FEN records from the 19,318 labeled
test records and evaluates 19,316 positions. The 31 train and 6 test records
that failed labeling are recorded in `labeling_summary.json`; they are not
silently treated as successful labels. No source game is shared between
splits. Training experiments use fixed seeds 0 and 1, Adam, five epochs, batch
size 128, learning rate 0.001, and two 128-unit hidden layers unless otherwise
stated. The complete setup, commands, and dependency versions are documented
in the README and `docs/experimental_protocol.md`.

## Models and fair comparisons

Feature groups were trained under the same split and budget:

| Group | Main change | Test top-1 | Non-mate regret (cp) | Parameters | Median latency (ms) |
|---|---|---:|---:|---:|---:|
| A | Legacy representation | 26.02% | 201.3 | 136,065 | 0.120 |
| B | A + tactical move features | 26.51% | 201.2 | 136,577 | 0.136 |
| C | B + chess-aware position features | 26.37% | 201.9 | 238,849 | 0.194 |
| D | C + attack maps | 25.35% | 223.2 | 255,233 | 0.204 |

The primary final comparison used the same B architecture and held-out test
positions:

| Model | Top-1 | Mean regret (cp) | Median regret (cp) | Non-mate regret (cp) |
|---|---:|---:|---:|---:|
| B teacher | 26.70% | 1,971.2 | 71 | 200.0 |
| B pairwise-ranked | 21.22% | 1,688.8 | 78 | 175.1 |
| B mixed loss | **27.09%** | 1,798.9 | **63** | **171.4** |

The mixed model is the recommended model because it improves both top-1
agreement over the teacher and the primary non-mate regret metric. Mean
regret is also reported because mate scores can dominate it.

## What failed and why

The verified self-learning round selected uncertain training positions,
obtained fresh Stockfish feedback, and applied a development acceptance gate.
The candidate was rejected because development non-mate regret worsened from
200.81 to 201.00 cp. This is a valid negative result: the gate prevented a
degraded checkpoint from replacing the teacher.

Distillation also did not improve the B teacher. Hard and soft students were
smaller (30,017 vs. 136,577 parameters) and faster (about 0.046 vs. 0.117 ms
per position), but their final-test non-mate regret was approximately
227–230 cp. The likely causes are the small student capacity and the mismatch
between soft teacher targets and the difficult move-ranking distribution.

Adding attack-map features (group D) increased computation and worsened
non-mate regret. This demonstrates that more features are not automatically
better. Pairwise ranking reduced regret but lowered top-1 agreement, showing
that optimizing the metrics involves a real trade-off.

## Tactical search and demo

The backend guarantees immediate checkmates before neural inference. Depth-2
neural-guided search retains the model's quiet-move candidates while adding
all legal checks, captures, and promotions to the search shortlist. Tactical
regression tests cover mate, promotion, captures, forcing candidates, and
search legality. The browser demo supports human-vs-model and model-vs-model
play.

## Limitations

The labels come from shallow bounded Stockfish searches, so they contain
horizon effects and are not an absolute chess-strength oracle. The model
ranks legal candidates rather than performing unrestricted engine search.
Evaluation measures agreement and regret, not Elo or tournament performance.
The dataset is substantially larger than the initial smoke experiment, but it
still comes from a limited set of source-game collections and should not be
treated as a general chess benchmark. Search depth 3 is available but slower;
depth 2 is the practical default on the target CPU.

## Reproduction and artifacts

Run:

```bash
source .venv/bin/activate
python -m pytest
cd frontend && npm ci && npm run build
```

The machine-readable artifacts are in:

- `experiments/results/v2_features/comparison.json`
- `experiments/results/v2_mixed_B/test.json`
- `experiments/results/v2_ranked_B/test.json`
- `experiments/results/v2_distillation_B/final_test.json`
- `experiments/results/v2_self_learning_B/round1/decision.json`

The chart in `reports/current_results.png` is generated by
`scripts/plot_submission_results.py`. The accompanying 10-slide presentation
is in [`reports/presentation.md`](presentation.md).
