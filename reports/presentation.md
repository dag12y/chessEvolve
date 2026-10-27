# ChessEvolve: Submission Presentation

## Slide 1 — Problem

- Can a small CPU-only neural model rank chess moves using engine supervision?
- Can self-learning improve it safely?
- Can distillation reduce cost without hiding failures?
- Scope: move ranking, not a Stockfish replacement.

## Slide 2 — System

`legal positions -> game-level split -> Stockfish labels -> MLP ranker ->
development gate -> final test -> browser demo`

- `python-chess` supplies legal moves.
- The model scores candidates; it never creates illegal moves.
- The backend adds immediate-mate handling and shallow search.

## Slide 3 — Data and protocol

- Phase 3.5 dataset, fixed manifest seed `20261026`.
- Train: 3,357 games / 87,993 records.
- Development: 720 games / 19,101 records.
- Final test: 719 games / 19,324 declared records.
- Splitting is by source game to prevent position-level leakage.
- Final test was not used for checkpoint selection.

## Slide 4 — Baselines and metrics

- A: legacy features.
- B: legacy plus tactical move features.
- C: B plus chess-aware position features.
- D: C plus attack maps.
- Metrics: top-1 agreement, mean/median regret, non-mate regret, latency.
- Lower regret and higher top-1 are better.

## Slide 5 — Feature comparison

| Group | Top-1 | Non-mate regret | Latency |
|---|---:|---:|---:|
| A | 26.02% | 201.3 cp | 0.120 ms |
| B | 26.51% | 201.2 cp | 0.136 ms |
| C | 26.37% | 201.9 cp | 0.194 ms |
| D | 25.35% | 223.2 cp | 0.204 ms |

Finding: attack maps increased cost and failed to improve quality.

## Slide 6 — Loss comparison

| B model | Top-1 | Median regret | Non-mate regret |
|---|---:|---:|---:|
| Teacher | 26.70% | 71 cp | 200.0 cp |
| Pairwise ranking | 21.22% | 78 cp | 175.1 cp |
| Mixed loss | **27.09%** | **63 cp** | **171.4 cp** |

The mixed classification-plus-ranking objective is the final selected model.
Results are from the fixed final test split.

## Slide 7 — Self-learning result

- Select uncertain training positions.
- Query Stockfish and verify legal, complete feedback.
- Fine-tune a candidate.
- Promote only if development regret does not worsen.
- Candidate changed development regret from 200.81 to 201.00 cp.
- Decision: **reject and roll back**.

This negative result is evidence that the safety gate works.

## Slide 8 — Distillation result

- Teacher: 136,577 parameters.
- Student: 30,017 parameters, about 78% fewer.
- Student latency: about 0.046 ms versus 0.117 ms.
- Student final-test non-mate regret: about 227–230 cp.

Conclusion: compression improved efficiency but did not improve move quality.
Both hard-label and soft-target results are retained and reported.

## Slide 9 — Demo and tactical protection

- Human-vs-model and model-vs-model browser modes.
- Immediate checkmates are selected before model inference.
- Search depth 2 is the practical default on the target CPU.
- Forcing checks, captures, and promotions are retained in search candidates.
- Tactical regression tests cover mates, promotions, captures, and legality.

## Slide 10 — Conclusions and limitations

- Mixed loss produced the best validated B-model result.
- Self-learning and distillation did not automatically improve quality.
- Game-level splits, fixed seeds, machine-readable artifacts, and rollback make
  the comparisons reproducible.
- Shallow Stockfish labels, limited source collections, and move-ranking
  evaluation limit claims about general playing strength.
- See `reports/final_report.md` and `reports/current_results.png`.
