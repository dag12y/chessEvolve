# ChessEvolve — Experimental Protocol

## 1. Principles

- Define metrics and acceptance rules before the final test.
- Split data by source game.
- Use fixed and recorded random seeds.
- Compare models under comparable conditions.
- Keep the final-test set untouched until the design is frozen.
- Report every planned run, including negative results and failures.

## 2. Dataset protocol

Begin with a modest dataset that can be analyzed on CPU within the project schedule.

Record:
- Data source and terms of use.
- Retrieval or generation date.
- Number of source games and positions.
- Filtering and sampling rules.
- Position and candidate limits.
- Dataset hashes.
- Stockfish version and settings.

A starting split could be 70% training, 15% development, and 15% final test by source game. Finalize the split before model selection.

## 3. Reference labels

For each position:
1. Enumerate a bounded set of legal candidate moves.
2. Analyze candidates using fixed Stockfish settings.
3. Normalize scores to a consistent perspective.
4. Select the highest-scoring candidate as the hard label.
5. Convert candidate scores or teacher logits into a documented soft-target distribution where required.

Define how mate scores are handled. Do not mix White-relative and side-to-move-relative scores.

## 4. Baseline teacher

Train a small neural network to rank candidate moves.

Record:
- Training loss.
- Development top-1 reference agreement.
- Development mean engine-score regret.
- Parameter count.
- Checkpoint size.
- CPU inference latency.
- Training time.

Top-1 reference agreement measures agreement with the selected engine configuration. It is not, by itself, a measurement of playing strength.

## 5. Verified self-learning

Run a small, fixed number of rounds, such as two or three. The recursive
runner carries only each round's `current_best.pt` into the next round:
accepted candidates become the next teacher, while rejected candidates are
never reused. Previously labeled records are cached by position and engine
settings, so repeated selection does not silently relabel identical inputs.

For each round:
1. Select training positions using the predeclared policy.
2. Generate Stockfish feedback.
3. Verify examples and record rejection reasons.
4. Fine-tune a candidate copy of the current teacher.
5. Evaluate current and candidate checkpoints on the same development positions.
6. Apply the frozen acceptance rule.
7. Accept or reject the candidate.
8. Save metrics, configuration, and checkpoint hashes.

The independent verifier also checks that feedback refers to the selected
training record exactly (position, source game, FEN, candidate ordering, and
split). Development records may be evaluated from their frozen labels; the
final-test records are not inputs to the recursive runner.

### Suggested primary metric

Mean engine-score regret:

For each position, compare the Stockfish score of the best candidate in the bounded set with the score of the candidate selected by the model. Define the score perspective and mate-score handling consistently.

Lower regret is better.

### Suggested secondary metric

Top-1 agreement with the Stockfish reference best candidate. Higher agreement is better.

### Acceptance rule

Before the first run, define a minimum meaningful reduction in development regret and any permitted secondary-metric degradation. Accept only if the candidate meets those criteria. Otherwise, reject and roll back.

Do not change the acceptance rule after viewing final-test results.

Example using the labeled real-data pilot:

```bash
.venv/bin/python scripts/run_recursive_self_learning.py \
  --dataset-dir data/processed/dag12yi_labeled \
  --manifest data/processed/dag12yi_labeled/manifest.json \
  --baseline-checkpoint experiments/results/real_baseline_seed0/best.pt \
  --baseline-checkpoint experiments/results/real_baseline_seed1/best.pt \
  --output-dir experiments/results/phase3_recursive \
  --seed 0 --seed 1 --iterations 2 --engine-depth 4
```

This command writes one directory per seed, per-round decisions, accepted or
retained checkpoints, development metrics, and a final-test report generated
after the recursive decisions are complete. A rejected update is a valid
result; it is not evidence of improvement.

## 6. Distillation experiment

Freeze the teacher selected on development data.

Train two students:
- **Hard-label student:** cross-entropy against the reference best-candidate label.
- **Distilled student:** temperature-scaled KL-divergence loss against teacher soft targets, optionally combined with hard-label loss.

Use the same student architecture, training data, and comparable optimization budgets. Run at least two seeds for each student method.

Report per seed and summary statistics:
- Top-1 reference agreement.
- Mean engine-score regret.
- Parameter count.
- Checkpoint size.
- CPU inference latency.

## 7. Latency measurement

- Use the same CPU and software environment.
- Warm up the model before measurement.
- Measure the same input and batch-size conditions.
- Repeat the measurement.
- Report the median and measurement procedure.
- Separate model inference latency from Stockfish search latency.

## 8. Optional match evaluation

If time permits, run controlled games against fixed opponents. Use consistent time controls and settings, and swap colors across paired games.

Treat small match samples as exploratory. Do not infer strong playing-strength claims from a few games.

## 9. Final test

After architecture, thresholds, and checkpoints are frozen:
- Evaluate final models on the final-test set.
- Do not tune any settings based on those results.
- Save the final metrics once.
- Document any post-test changes and avoid presenting reused test results as a fresh independent test.

## 10. Results table

| Model | Seed | Top-1 agreement | Mean score regret | Parameters | Model size | CPU latency |
|---|---:|---:|---:|---:|---:|---:|
| Baseline teacher | | | | | | |
| Evolved teacher | | | | | | |
| Hard-label student | | | | | | |
| Distilled student | | | | | | |

## 11. Interpretation questions

- Did the self-learning loop improve the primary metric?
- How many updates were accepted and rejected?
- What were the main rejection reasons?
- Did soft-target distillation outperform hard-label training?
- Was the student smaller or faster?
- Were results consistent across seeds?
- Could leakage, engine bias, or different compute budgets explain the results?
- What failed, and what would be improved in a follow-up experiment?