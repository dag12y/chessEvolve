# ChessEvolve

**Verified Self-Learning and Knowledge Distillation for Chess Move Ranking**

ChessEvolve is an educational research project for the iCog Labs *Emerging AI Paradigms* assignment. It investigates whether a small neural network can improve its chess move-ranking behavior through independently evaluated feedback, and whether a smaller student network can retain useful behavior through knowledge distillation.

ChessEvolve is **not intended to rival Stockfish**. It is a bounded experiment designed to demonstrate recursive self-improvement, independent verification, knowledge distillation, reproducible evaluation, and honest analysis.

## Research questions

1. Does a bounded, Stockfish-guided self-learning loop improve a teacher model's move-ranking quality on held-out positions?
2. How many candidate updates are accepted or rejected, and why?
3. Does a student trained using teacher soft targets perform differently from a student trained using hard labels?
4. What trade-offs exist between model quality, parameter count, storage, and CPU inference latency?

## Technology stack

- Python
- PyTorch for neural networks and training
- Stockfish for external chess evaluation
- `python-chess` for legal moves, board representations, PGN processing, and UCI communication
- NumPy and pandas for data processing
- matplotlib for visualizations
- pytest for automated testing

The experiment must run on a CPU. GPU support is optional and is not a requirement.

## Main components

### 1. Teacher model

A neural network that ranks a bounded list of legal candidate moves for a chess position.

### 2. Verified self-learning

The current model identifies candidate positions. Stockfish supplies reference evaluations. A verifier checks the generated examples, and a candidate model is trained. The candidate is accepted only if it passes predefined evaluation criteria; otherwise, the previous checkpoint is restored.

### 3. Hard-label student

A smaller neural network trained using reference best-move labels.

### 4. Distilled student

A student with the same architecture as the hard-label student, trained using soft probability targets from a frozen teacher, optionally combined with hard-label supervision.

### 5. Evaluation

The system measures reference-move agreement, engine-score regret, accepted and rejected updates, parameter count, model size, and CPU inference latency.

## Repository structure

```text
ChessEvolve/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
├── src/chessevolve/
├── tests/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── experiments/
│   ├── configs/
│   └── results/
├── models/
└── reports/
```

## Reproducibility

- Fix and record random seeds.
- Record software versions and Stockfish settings.
- Split data by source game to reduce leakage from related positions.
- Keep the final test set untouched until model selection and thresholds are frozen.
- Save configuration files and results for every run.
- Record rejected updates and failed experiments.
- Do not claim an improvement unless the measurements support it.

## Current status

This repository begins with project documentation and experimental planning. Implementation and results are not yet complete. The project must be built and tested milestone by milestone.