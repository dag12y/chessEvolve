# ChessEvolve — Concepts Guide

## 1. Teacher model

The teacher is a neural network that ranks a supplied set of legal candidate moves. It learns from reference labels or scores generated using Stockfish under documented settings.

The teacher is not automatically strong because it is larger. Its performance must be measured.

## 2. Recursive self-learning

The model proposes candidate cases, receives external feedback, trains a candidate update, and checks whether the update improves held-out performance. If the candidate passes the acceptance gate, it becomes the new current model. Otherwise, the previous checkpoint is retained.

The loop must be bounded and evaluated. Repeating training on model-generated examples without checking quality can reinforce mistakes.

## 3. Verifier and acceptance gate

The **verifier** checks the quality and validity of candidate examples before training.

The **acceptance gate** checks whether a newly trained checkpoint should replace the existing checkpoint.

These are different tasks. A valid example does not guarantee that training on it will improve the model.

## 4. Pseudo-labels

Pseudo-labels are targets produced by an automated model or process. In ChessEvolve, Stockfish evaluations are used to label candidate moves.

Pseudo-labels may be noisy or wrong. Stockfish is a powerful reference engine, but its scores depend on version, search settings, and computational budget.

## 5. Hard labels

A hard label selects one candidate move as the reference best move. The student is trained using cross-entropy against that target.

Hard labels are simple but discard information about how the other candidate moves compare.

## 6. Soft targets

Soft targets assign probabilities to multiple candidate moves. They can represent relative preference rather than only one selected answer.

For logits z and temperature T:

`p_i = exp(z_i / T) / sum_j exp(z_j / T)`

Higher temperature generally makes the probability distribution softer.

## 7. Knowledge distillation

Knowledge distillation transfers behavior from a teacher to a smaller student.

ChessEvolve compares:
- A student trained on hard reference labels.
- A student trained on soft teacher targets, optionally combined with hard-label loss.

The goal is to measure whether the smaller student retains useful behavior with fewer parameters, smaller storage, or lower latency. Distillation does not guarantee an improvement.

## 8. Cross-entropy and KL divergence

Cross-entropy measures the mismatch between a target distribution and a predicted distribution. It is commonly used for hard-label classification.

Kullback–Leibler divergence measures how one probability distribution differs from another. In distillation, it can measure the difference between teacher and student distributions.

When temperature-scaled distillation uses KL divergence, implementations often multiply the KL term by T² to keep gradient scales more comparable. The chosen loss must be documented.

## 9. Metrics

**Top-1 reference agreement:** Fraction of positions where the model's top candidate matches the reference best candidate.

**Engine-score regret:** Difference between the reference score of the best candidate and the score of the model-selected candidate. Lower is better if the score convention is defined consistently.

**Parameter count:** Number of trainable parameters.

**Checkpoint size:** Serialized model size on disk.

**CPU inference latency:** Time required for the model to rank candidates, measured with a documented and repeated procedure.

**Playing strength:** Performance against fixed opponents in controlled games. This is not interchangeable with reference agreement.

## 10. Data leakage

Leakage occurs when evaluation information influences training or model selection. Related positions from the same game can be very similar. Split by source game before extracting positions, and never use final-test results to select checkpoints or thresholds.

## 11. Interpreting results

A valid conclusion may be that the method helped, did not help, or was inconclusive. The important requirements are fair comparisons, reproducibility, clear metrics, and honest limitations.