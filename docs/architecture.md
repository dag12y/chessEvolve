# ChessEvolve — Architecture

## 1. System overview

The system has five major stages:

1. Prepare chess positions and generate reference evaluations.
2. Train and evaluate a baseline teacher.
3. Run a bounded verified self-learning loop.
4. Train hard-label and distilled students.
5. Evaluate all frozen models and produce reports.

## 2. Data pipeline

Input sources may include public PGN games or reproducibly generated legal games.

The pipeline:
- Reads games and extracts positions.
- Preserves source-game identifiers.
- Splits games into training, development, and final-test sets.
- Enumerates legal candidate moves.
- Queries Stockfish using fixed settings.
- Stores candidate scores, reference labels, and metadata.

The split must occur before related positions are extracted to avoid leakage.

## 3. Stockfish adapter

The adapter communicates with Stockfish through the UCI interface using `python-chess`.

Responsibilities:
- Check engine availability.
- Record the engine version and settings.
- Use a bounded search budget.
- Evaluate candidate moves consistently.
- Normalize scores to a single perspective.
- Handle missing results and engine failures.
- Cache completed analyses where appropriate.

Stockfish evaluations are reference signals, not guaranteed correct labels.

## 4. Position and move encoding

The initial representation should be small and documented.

Potential board features:
- Piece placement.
- Side to move.
- Castling rights.
- En-passant availability, if included.

Potential candidate-move features:
- Source square.
- Destination square.
- Moving piece.
- Captured piece.
- Promotion type.

Start with a simple, deterministic representation. Expand only after the baseline is working and tested.

## 5. Teacher model

A PyTorch MLP receives position features and candidate-move features. It returns one logit for each candidate supplied for the current position.

The model does not generate legal moves. `python-chess` provides those candidates, and the model ranks them.

## 6. Candidate generator

The candidate generator selects training positions using a predefined policy, such as:
- Low teacher confidence.
- Disagreement between teacher ranking and Stockfish ranking.

It must use training positions only and enforce a fixed maximum number of candidates per round.

## 7. Verifier

The verifier checks:
- The move is legal.
- The candidate belongs to the supplied candidate set.
- The engine returned a valid score.
- Score perspective is consistent.
- Data belongs to the training split.
- Any predefined score-quality or confidence conditions are satisfied.

The verifier filters unreliable examples; it does not prove that a move is objectively best.

## 8. Update trainer

The update trainer:
1. Loads the latest accepted checkpoint.
2. Creates a separate candidate model.
3. Fine-tunes it on verified examples.
4. Saves it as a candidate checkpoint.
5. Sends it to the acceptance gate.

The previous accepted checkpoint remains unchanged until the decision is complete.

## 9. Acceptance gate

The acceptance gate evaluates the previous and candidate models on the same development positions.

Suggested primary metric: mean engine-score regret of the model-selected candidate relative to the best candidate in the evaluated set.

Suggested secondary metric: top-1 agreement with the Stockfish reference best candidate.

The gate applies thresholds defined before the experiment. If the candidate fails, the system rejects it and retains the previous checkpoint. Failed runs, non-finite metrics, and exceptions must not silently pass.

## 10. Distillation

After selecting a teacher using development data, freeze it.

Train:
- A hard-label student using reference best-move labels.
- A distilled student using soft teacher targets.

Use comparable student capacity and training budgets. The distilled student must have fewer parameters than the teacher.

## 11. Experiment logger

Record:
- Git commit.
- Configuration and seed.
- Dataset and split hashes.
- Software and Stockfish versions.
- Round number and parent checkpoint.
- Candidate counts and verifier decisions.
- Pre-update and post-update metrics.
- Accept/reject decision and reason.
- Model parameter count and size.
- Runtime and inference latency.

## 12. Failure containment

- Train candidate updates on copies of accepted checkpoints.
- Reject candidates with invalid metrics or failed runs.
- Preserve the last accepted checkpoint.
- Do not expose final-test positions to training or update selection.
- Keep all rejection logs for analysis.

## 13. Architecture principle

The model proposes; Stockfish supplies reference feedback; the verifier checks data quality; the development gate decides whether an update is retained; and the final test measures performance only after the design is frozen.