# ChessEvolve — Implementation Roadmap

## Working rules

- Finish and review documentation before coding.
- Implement one milestone at a time.
- Ask Codex to work only on the current milestone.
- Run tests and inspect the diff before committing.
- Keep the system CPU-friendly.
- Never claim an experiment succeeded before running it.

## Milestone 0 — Environment and repository

Tasks:
- Create the repository and virtual environment.
- Install PyTorch CPU, python-chess, NumPy, pandas, matplotlib, and pytest.
- Install Stockfish.
- Verify that Python can start Stockfish through UCI.
- Record software versions.

Acceptance criteria:
- Imports work.
- Stockfish responds to a simple analysis request.
- Tests run.
- No GPU is required.

## Milestone 1 — Data and Stockfish adapter

Tasks:
- Define the position record.
- Implement legal candidate generation.
- Implement bounded Stockfish analysis.
- Normalize score perspective.
- Split data by source game.
- Save manifests and metadata.

Tests:
- Candidate moves are legal.
- Scores have a consistent perspective.
- Source-game IDs do not overlap between splits.
- Same seed produces reproducible split manifests.
- Engine failures are handled safely.

## Milestone 2 — Features and baseline teacher

Tasks:
- Implement board and candidate-move encoding.
- Implement a small PyTorch MLP ranker.
- Train using engine-derived reference labels.
- Save checkpoints and configurations.
- Measure development metrics.

Tests:
- Feature dimensions and dtypes are stable.
- Model output matches candidate count.
- Loss and gradients remain finite.
- The model can overfit a tiny batch as a sanity check.

## Milestone 3 — Verified self-learning loop

Tasks:
- Select candidate training positions.
- Generate bounded Stockfish feedback.
- Verify generated examples.
- Fine-tune a candidate checkpoint.
- Evaluate on development positions.
- Implement accept/reject and rollback.
- Save structured logs.

Tests:
- A candidate that passes the gate is accepted.
- A candidate that fails is rejected.
- Rejection preserves the prior checkpoint.
- Invalid metrics and engine errors fail safely.
- Final-test data cannot enter candidate generation.

## Milestone 4 — Distillation

Tasks:
- Freeze the selected teacher.
- Train a hard-label student.
- Train a soft-target distilled student.
- Use comparable student architectures and budgets.
- Run at least two random seeds per method.

Tests:
- Teacher probabilities are normalized.
- Temperature is positive.
- Distillation loss is finite.
- Student has fewer parameters than teacher.

## Milestone 5 — Evaluation

Tasks:
- Freeze configurations and checkpoints.
- Run final evaluation.
- Save results as CSV or JSON.
- Create at least one chart.
- Measure CPU latency consistently.
- Analyze failed and rejected updates.

Acceptance criteria:
- Every reported metric is traceable to a saved run.
- All planned seeds and failed runs are reported.
- No unsupported claim of superiority over Stockfish.

## Milestone 6 — Submission

Tasks:
- Complete README and reproduction instructions.
- Write the report.
- Prepare 8–12 slides or a concise demo.
- Re-run tests.
- Verify the repository works from a clean environment.

## Seven-day schedule

| Day | Focus |
|---|---|
| 1 | Documentation, repository, environment, design review |
| 2 | Dataset, Stockfish adapter, baseline teacher |
| 3 | Verification, self-learning loop, rollback |
| 4 | Hard-label and distilled students |
| 5 | Multi-seed experiments and metrics |
| 6 | Final evaluation, charts, failure analysis |
| 7 | Report, presentation, reproducibility check |

If the schedule slips, prioritize a defensible primary self-learning loop and a smaller but valid distillation comparison over a frontend or expensive tournament.