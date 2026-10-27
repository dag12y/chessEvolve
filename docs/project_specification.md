# ChessEvolve — Project Specification

## 1. Project identity

**Project:** ChessEvolve  
**Assignment:** Emerging AI Paradigms  
**Primary track:** Verified recursive self-learning  
**Connected track:** Knowledge distillation  
**Application domain:** Chess move ranking  
**Runtime target:** CPU  
**Project state:** Complete

## 2. Problem statement

Neural networks can learn to rank chess moves from examples, but their predictions may be weak, uncertain, or inconsistent. Recursive self-learning provides a way to generate candidate training examples and improve a model through bounded feedback. Knowledge distillation provides a way to transfer a teacher's behavior to a smaller student.

ChessEvolve combines these ideas in a controlled experiment. Stockfish provides external move evaluations, a verifier checks candidate examples, and a separate acceptance gate determines whether a proposed model update should replace the current checkpoint. A teacher model is then used to train a smaller student with soft targets.

Stockfish is an approximate reference engine, not a perfect source of ground truth.

## 3. Goals

### Primary goal: recursive self-learning

Implement a bounded improvement loop that includes:

- Candidate position selection.
- External feedback from Stockfish.
- Verification of candidate examples.
- Candidate checkpoint training.
- Development-set evaluation.
- Acceptance, rejection, and rollback.
- Per-round logging.

### Connected goal: knowledge distillation

Train and compare:

1. A teacher neural network.
2. A smaller student trained on hard reference labels.
3. A similarly sized student trained on soft teacher targets.

### Supporting goals

- Keep the experiment CPU-friendly.
- Make runs reproducible.
- Avoid train/test leakage.
- Measure model quality and computational cost.
- Explain limitations and failure cases.
- Understand the implementation rather than treating it as a black box.

## 4. Research questions

**RQ1:** Does verified self-learning improve held-out move-ranking quality?

**RQ2:** How many updates are accepted and rejected, and what causes rejection?

**RQ3:** Does soft-target distillation perform differently from hard-label student training?

**RQ4:** Can the student reduce model size and inference latency while retaining useful performance?

These are research questions, not guaranteed outcomes. A negative result is acceptable if the experiment is correctly designed and honestly reported.

## 5. Dataset

The dataset will contain chess positions and bounded sets of legal candidate moves.

For each position, the pipeline will record:

- A stable position identifier.
- Source game identifier or generation seed.
- FEN representation.
- Legal candidate moves.
- Stockfish evaluation for each candidate.
- Reference best move.
- Optional normalized probability targets.
- Engine version and analysis settings.
- Dataset split.

The data source, its terms of use, sampling procedure, and final dataset size must be documented before the formal experiment.

## 6. Data splitting

Split by source game before extracting related positions.

Use three separate sets:

- **Training:** model fitting.
- **Development:** threshold selection, checkpoint selection, and update acceptance.
- **Final test:** final comparison after the design is frozen.

Never use final-test results to choose training settings, thresholds, or checkpoints.

## 7. Model interface

The model will rank a bounded list of legal candidate moves.

Inputs:
- Encoded chess position.
- Encoded candidate move features.

Output:
- One score or logit per supplied candidate.

Legal moves will be generated using `python-chess`. The model will rank only the supplied legal candidates, rather than attempting to generate arbitrary UCI moves.

## 8. Recursive self-learning procedure

1. Train a baseline teacher.
2. Select candidate positions from the training split.
3. Identify uncertain predictions or disagreements using a predefined rule.
4. Query Stockfish for bounded candidate evaluations.
5. Verify the resulting examples.
6. Fine-tune a copy of the current teacher.
7. Evaluate the candidate checkpoint on a fixed development set.
8. Accept it only if it passes the predefined acceptance criteria.
9. Otherwise, reject it and restore the previous checkpoint.
10. Save all decisions and metrics.
11. Repeat for a small, fixed number of rounds.

The verifier checks examples. The acceptance gate evaluates model updates. These are separate responsibilities.

## 9. Knowledge distillation

The selected teacher checkpoint will be frozen before student training.

**Hard-label student:** Trains against the reference best-move label.

**Distilled student:** Trains against a temperature-scaled teacher probability distribution, optionally combined with hard-label loss.

Both students should use comparable architectures, data exposure, and optimization budgets. The distilled student must be smaller than the teacher by parameter count.

## 10. Evaluation

Measure:

- Top-1 agreement with the reference best candidate.
- Mean engine-score regret.
- Accepted and rejected model updates.
- Rejection reasons by round.
- Parameter count.
- Serialized checkpoint size.
- CPU inference latency.
- Variation across at least two student-training seeds.

If controlled matches are feasible, report them separately from reference-move agreement.

## 11. Completion criteria

The project is complete when:

- The baseline teacher trains and evaluates.
- The self-learning loop includes verification, acceptance, rejection, and rollback.
- The loop records accepted and rejected updates.
- The hard-label and distilled students are compared over at least two seeds.
- The final test is kept separate from model selection.
- Results are saved in machine-readable form.
- At least one chart and a failure analysis are produced.
- A reproducible README, short report, and 8–12 slide presentation are ready.

## 12. Out of scope

- Building a grandmaster-level engine.
- Training Stockfish from scratch.
- Unbounded self-play.
- Expensive reinforcement learning.
- Claiming the neural network is stronger than Stockfish without rigorous match testing.
- Building a polished frontend before the experiments work.

## 13. Deliverables

- Source code and tests.
- Data preparation scripts and split manifests.
- Experiment configurations and seed records.
- Teacher and student training scripts.
- Self-learning round logs.
- Results table and chart.
- Failure analysis.
- Written report.
- Presentation or concise demonstration.

## 14. Final decisions and implementation notes

The implementation uses deterministic legal random-game positions and
source-game-level train/dev/test splitting. The teacher uses a 780-dimensional
position representation and a 135-dimensional move representation. The
baseline is a two-hidden-layer MLP; students use a one-hidden-layer,
32-unit MLP. Stockfish search depth and acceptance thresholds are recorded in
the saved run configurations and reports.

The browser demo is an optional presentation layer, not part of model
training. It exposes only configured checkpoints and keeps final-test data
outside inference and game-play state.