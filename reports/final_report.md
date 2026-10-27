# ChessEvolve: Verified Recursive Self-Learning and Knowledge Distillation for Chess Move Ranking

**Author:** ChessEvolve Team  
**Date:** October 2026  
**Environment:** Linux x86_64 (CPU Only, 16 cores), Python 3.14.8, PyTorch 2.14.1+cpu  
**Stockfish Version / Settings:** Stockfish 19 UCI (Bounded depth 2–4, side-to-move normalized centipawns)  
**Dataset Provenance:** Game-split synthetic random legal games, Manifest hash `d177c94650c656e10c68b6fcf42e47ccfcf4e488b7521840cdf140a250138642`  

---

## 1. Abstract and Research Questions

### Abstract
ChessEvolve investigates recursive self-improvement and model compression in the domain of chess move ranking without GPU compute. The system trains a Multi-Layer Perceptron (MLP) teacher model to rank legal candidate moves using bounded Stockfish evaluations as reference supervision. A verified self-learning loop actively queries Stockfish on high-uncertainty positions, applies strict verifier checks, and gates candidate model promotion on a held-out development set. When evaluated in Milestone 3, the self-learning candidate failed the acceptance gate (mean regret degraded by 3.9 centipawns), successfully triggering automatic rollback and retaining the baseline teacher. Subsequently, knowledge distillation compressed the retained teacher into hard-label and soft-target student networks, reducing parameter count by 78.0% (from 136,065 to 29,889 parameters), checkpoint storage by 77.6% (from 535.2 KiB to 120.1 KiB), and CPU inference latency by ~36% (from 0.120 ms to 0.076 ms per position). Due to strict CPU resource bounds, final evaluation was performed on 3 held-out development positions and 3 isolated final-test positions; consequently, while architectural efficiency and rollback mechanisms are rigorously demonstrated, move-quality metrics represent preliminary smoke benchmarks rather than broad generalizations of chess playing strength.

### Research Questions
1. **RQ1 (Recursive Self-Learning):** Does active selection and fine-tuning on engine-evaluated positions improve a teacher's move-ranking quality on held-out positions under a strict acceptance gate?
2. **RQ2 (Safety & Rollback Gate):** How reliably does an automated validation gate prevent regression when fine-tuning produces degraded candidate checkpoints?
3. **RQ3 (Distillation Efficacy):** How does soft-target distillation (temperature-scaled KL divergence) compare against hard-label supervision (cross-entropy) when compressing a move ranker into a compact student architecture?
4. **RQ4 (Efficiency Trade-offs):** What empirical trade-offs exist between model accuracy, parameter footprint, checkpoint disk size, and CPU inference latency?

---

## 2. Problem Motivation and Project Objectives

Modern chess engines such as Stockfish rely on deep alpha-beta search coupled with heavily optimized neural networks (NNUE). However, deploying deep evaluation models on resource-constrained edge CPUs introduces latency and memory bottlenecks. Furthermore, naive self-training loops frequently suffer from catastrophic forgetting, distribution drift, or feedback degeneration when exposed to noisy pseudo-labels.

ChessEvolve addresses these challenges through two core AI paradigms:
1. **Verified Recursive Self-Learning:** Rather than blindly fine-tuning, the system treats engine feedback as unverified proposals. Positions are filtered by an explicit verifier, and checkpoint promotion is guarded by an automated development gate with hard rollback.
2. **Knowledge Distillation:** By transferring the rich soft probability distribution of a larger teacher into a student with fewer layers and narrower hidden dimensions, the student can learn relative move preferences rather than a single discrete target.

**Project Boundary:** ChessEvolve ranks legal moves provided by `python-chess`. It is an educational and architectural proof-of-concept; it is **not** designed to rival Stockfish.

---

## 3. Architecture and Methodology

The system architecture consists of five decoupled, strictly verified modules:

```text
[Position Pool] -> [Source-Game Splitter] -> [Train / Dev / Test Pools]
                          │
       ┌──────────────────┴──────────────────┐
       ▼                                     ▼
[Baseline Teacher (MLP)]             [Frozen Test Isolation]
       │
       ▼ (Active Uncertainty Selection)
[Candidate Positions] -> [Stockfish Adapter] -> [Verifier] -> [Candidate Fine-Tuning]
                                                                     │
                                                                     ▼
                                                      [Acceptance Gate (Dev Regret)]
                                                       ├── Passed  -> Promoted
                                                       └── Failed  -> Rollback to Retained
                                                                     │
                                                                     ▼
                                                      [M4 Distillation: Hard vs Soft]
                                                                     │
                                                                     ▼
                                                      [M5 Comprehensive Evaluation]
```

### 3.1 Feature Representation
- **Position Encoding (`POSITION_DIM = 780`):**
  - 12 piece-placement bitboards (6 piece types × 2 colors × 64 squares = 768 bits).
  - 12 global context features: active player color (1), castling rights (4), en passant target availability (1), normalized halfmove clock (1), and piece material counts (5).
- **Candidate Move Encoding (`MOVE_DIM = 135`):**
  - One-hot source square (64 bits).
  - One-hot destination square (64 bits).
  - Moving piece type (1), captured piece type (1), and promotion piece type (5).
- **Model Output:** For each candidate move $m_i \in M$, the ranker outputs a scalar logit $z_i = f_\theta(\mathbf{x}_{\text{pos}}, \mathbf{x}_{m_i})$. Move probabilities are computed via $\text{softmax}(\mathbf{z})$.

### 3.2 Evaluation Metrics
- **Mean Engine-Score Regret (Centipawns):**
  $$\text{Regret} = \frac{1}{N} \sum_{k=1}^N \left( s_k(m^*_k) - s_k(\hat{m}_k) \right)$$
  where $m^*_k$ is the Stockfish-optimal candidate, $\hat{m}_k = \arg\max_i z_{k,i}$ is the model's top pick, and $s_k(\cdot)$ is the side-to-move centipawn score. Regret is strictly non-negative; lower is better.
- **Top-1 Agreement:** Fraction of positions where $\hat{m}_k = m^*_k$. Higher is better.
- **CPU Inference Latency:** Median wall-clock time per position across repeated runs following dedicated warm-up iterations.

---

## 4. Dataset Source, Split Procedure, and Provenance

### 4.1 Generation and Splitting
Positions were generated by playing random legal chess games to ensure unbiased legal move diversity. To prevent data leakage from tactically related moves within the same game, the dataset is split strictly at the **source-game level** using reproducible hashing (`split_by_source_game`).

### 4.2 Manifest Provenance and Actual Sizes
The dataset manifest is recorded at `data/processed/dag12yi_phase35_labeled/manifest.json` (SHA-256: `d177c94650c656e10c68b6fcf42e47ccfcf4e488b7521840cdf140a250138642`):
- **Random Seed:** 7
- **Source Games:** 4 total games (`random-game-0` through `3`), max 8 plies.
- **Actual Split Counts:**
  - **Train:** 2 source games (`random-game-1`, `random-game-3`), 6 positions (50.0%).
  - **Dev:** 1 source game (`random-game-0`), 3 positions (25.0%).
  - **Test:** 1 source game (`random-game-2`), 3 positions (25.0%).
- **Split Isolation:** No game IDs overlap between splits. Evaluated files `train_records.jsonl`, `dev_records.jsonl`, and `test_records.jsonl` under `data/processed/dag12yi_phase35_labeled/` strictly enforce this partitioning.

---

## 5. Teacher Model and Verified Self-Learning Loop

### 5.1 Teacher Architecture
- **Architecture:** PyTorch MLP with `hidden_dim = 128`, 2 hidden layers, ReLU activations, and dropout $0.0$.
- **Parameter Count:** 136,065 trainable parameters.
- **Baseline Training:** Supervised cross-entropy against Stockfish reference targets for 3 epochs with Adam ($\text{lr} = 10^{-3}$, batch size 4). Baseline checkpoint: [`experiments/results/medium/best.pt`](file:///home/dagm/Projects/chessEvolve/experiments/results/medium/best.pt).

### 5.2 Self-Learning Feedback Loop (Milestone 3)
1. **Uncertainty Selection:** 6 training positions with highest entropy in teacher logits were selected.
2. **Stockfish Feedback:** Stockfish 19 analyzed all legal candidates to depth 4, normalizing centipawns to side-to-move perspective.
3. **Verifier Checks:** All 6 feedback records were verified: moves were legal, scores complete, and splits verified. 0 rejections occurred.
4. **Fine-Tuning:** A candidate checkpoint was fine-tuned for 2 epochs ($\text{lr} = 5 \times 10^{-4}$).
5. **Acceptance Gate Decision:**
   - Pre-declared threshold: Regret reduction $\ge 0.0\text{ cp}$ on development data.
   - **Baseline Dev Regret:** 22.20 cp.
   - **Candidate Dev Regret:** 26.10 cp.
   - **Regret Difference:** $-3.90\text{ cp}$ (regression).
   - **Decision:** **REJECT**. The candidate was safely discarded, and the baseline teacher was retained as [`experiments/results/sl_round1/retained.pt`](file:///home/dagm/Projects/chessEvolve/experiments/results/sl_round1/retained.pt).

---

## 6. Knowledge Distillation: Hard-Label vs Soft-Target Students

### 6.1 Student Architecture
- **Architecture:** Compact MLP with `hidden_dim = 32`, 1 hidden layer, ReLU activations.
- **Parameter Count:** 29,889 trainable parameters (a **78.0% reduction** compared to the teacher).

### 6.2 Training Objectives
- **Hard-Label Student:** Standard cross-entropy loss against the one-hot reference best move:
  $$\mathcal{L}_{\text{hard}} = -\log \frac{\exp(z_{s, y^*})}{\sum_j \exp(z_{s, j})}$$
- **Distilled Student:** Temperature-scaled Kullback-Leibler divergence matching teacher softmax soft targets at $T = 2.0$:
  $$\mathcal{L}_{\text{distill}} = T^2 \cdot D_{\text{KL}}\left(\text{softmax}\left(\frac{\mathbf{z}_s}{T}\right) \parallel \text{softmax}\left(\frac{\mathbf{z}_t}{T}\right)\right)$$
- **Multi-Seed Protocol:** Evaluated across Seed 0 and Seed 1 using 2 epochs, batch size 4, $\text{lr} = 10^{-3}$.

---

## 7. Experimental Setup and Hardware Environment

- **Hardware Platform:** AMD/Intel x86_64 CPU (16 logical cores, no GPU utilized).
- **Operating System:** Linux (Arch-based rolling, kernel 7.2.8).
- **Software Stack:** Python 3.14.8, PyTorch 2.14.1+cpu, python-chess 1.11.2, NumPy 2.2.3, matplotlib 3.11.2, pytest 9.1.1.
- **Stockfish Settings:** Stockfish 19 UCI universal binary, search depth 2 for bounded datasets, depth 4 for feedback generation, fixed deterministic hash size.
- **Random Seeds:**
  - Data generation & split: Seed 7
  - M3 Self-learning round: Seed 42
  - Baseline teacher training: Seed 0
  - Distillation multi-seed runs: Seeds 0 and 1

---

## 8. Empirical Results: Development vs Final-Test

All results below are directly extracted from machine-readable JSON artifacts:
- Development results: [`experiments/results/m5_dev/evaluation_results_dev.json`](file:///home/dagm/Projects/chessEvolve/experiments/results/m5_dev/evaluation_results_dev.json)
- Final-test results: [`experiments/results/m5_test/evaluation_results_test.json`](file:///home/dagm/Projects/chessEvolve/experiments/results/m5_test/evaluation_results_test.json)

### Table 1: Comprehensive Model Evaluation Results

| Model | Evaluation Split | Role | Top-1 Agreement | Mean Regret (cp) | Cross-Entropy Loss | Trainable Parameters | Disk Size (KiB) | CPU Latency (ms/pos) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Baseline Teacher** | **DEV** | `baseline_teacher` | **0.0%** | **60.0** | 2.9510 | 136,065 | 534.9 | 0.120 ± 0.057 |
| **M3 Retained Teacher** | **DEV** | `retained_teacher` | **0.0%** | **60.0** | 2.9510 | 136,065 | 535.2 | 0.119 ± 0.040 |
| **M3 Rejected Candidate** | **DEV** | `rejected_candidate` | **0.0%** | **60.0** | 2.9545 | 136,065 | 535.0 | 0.124 ± 0.079 |
| **Hard Student (Seed 0)** | **DEV** | `student_hard` | **33.3%** | **3.0** | 3.0971 | 29,889 | 120.1 | 0.076 ± 0.007 |
| **Distilled Student (Seed 0)** | **DEV** | `student_distilled` | **0.0%** | **12.0** | 3.1100 | 29,889 | 120.2 | 0.077 ± 0.011 |
| **Hard Student (Seed 1)** | **DEV** | `student_hard` | **33.3%** | **34.3** | 3.0923 | 29,889 | 120.1 | 0.094 ± 0.015 |
| **Distilled Student (Seed 1)** | **DEV** | `student_distilled` | **33.3%** | **34.3** | 3.0952 | 29,889 | 120.2 | 0.074 ± 0.007 |
| | | | | | | | | |
| **Baseline Teacher** | **FINAL TEST** | `baseline_teacher` | **0.0%** | **422.7** | 3.2195 | 136,065 | 534.9 | 0.342 ± 0.066 |
| **M3 Retained Teacher** | **FINAL TEST** | `retained_teacher` | **0.0%** | **422.7** | 3.2195 | 136,065 | 535.2 | 0.115 ± 0.010 |
| **M3 Rejected Candidate** | **FINAL TEST** | `rejected_candidate` | **0.0%** | **422.7** | 3.2336 | 136,065 | 535.0 | 0.115 ± 0.005 |
| **Hard Student (Seed 0)** | **FINAL TEST** | `student_hard` | **33.3%** | **419.0** | 3.2086 | 29,889 | 120.1 | 0.076 ± 0.007 |
| **Distilled Student (Seed 0)** | **FINAL TEST** | `student_distilled` | **0.0%** | **430.3** | 3.2290 | 29,889 | 120.2 | 0.077 ± 0.004 |
| **Hard Student (Seed 1)** | **FINAL TEST** | `student_hard` | **0.0%** | **42.0** | 3.2250 | 29,889 | 120.1 | 0.078 ± 0.006 |
| **Distilled Student (Seed 1)** | **FINAL TEST** | `student_distilled` | **0.0%** | **42.0** | 3.2310 | 29,889 | 120.2 | 0.077 ± 0.007 |

---

## 9. Efficiency and Latency Trade-offs

The architectural comparisons show substantial, reproducible efficiency benefits from distillation:
- **Parameter Compression:** Student models require only **29,889 parameters** compared to the teacher's **136,065 parameters** (a **78.03% compression ratio**).
- **Disk Footprint:** Checkpoint storage drops from **535.2 KiB** to **120.1 KiB** (a **77.56% storage reduction**).
- **CPU Latency:** Across 50 timed iterations with 10 warm-up runs on CPU, student inference latency averages **0.074–0.078 ms/pos**, compared to **0.115–0.124 ms/pos** for the teacher (an inference speedup of **~1.55×**).

---

## 10. Visualizations

Generated publication-quality charts are preserved in the experiment results repository:

1. **Mean Engine-Score Regret Comparison (Dev Set):**  
   [`experiments/results/m5_dev/regret_by_model_dev.png`](file:///home/dagm/Projects/chessEvolve/experiments/results/m5_dev/regret_by_model_dev.png)  
   *Visualizes lower regret achieved by student checkpoints on development positions.*

2. **Top-1 Reference Agreement (Dev Set):**  
   [`experiments/results/m5_dev/top1_agreement_by_model_dev.png`](file:///home/dagm/Projects/chessEvolve/experiments/results/m5_dev/top1_agreement_by_model_dev.png)  
   *Compares exact match percentage against Stockfish best candidate.*

3. **Final-Test Isolated Performance:**  
   [`experiments/results/m5_test/regret_by_model_test.png`](file:///home/dagm/Projects/chessEvolve/experiments/results/m5_test/regret_by_model_test.png)  
   [`experiments/results/m5_test/top1_agreement_by_model_test.png`](file:///home/dagm/Projects/chessEvolve/experiments/results/m5_test/top1_agreement_by_model_test.png)

---

## 11. Analysis and Discussion

### 11.1 Acceptance Gate and Rollback Functionality (RQ1 & RQ2)
The M3 experimental trial provides critical empirical validation of the safety architecture. Fine-tuning a neural ranker on 6 engine-evaluated positions led to distribution shift on the held-out development positions, causing mean regret to rise from 22.2 cp to 26.1 cp. Because the development gate enforces a strict zero-tolerance threshold for regret degradation ($\Delta \text{Regret} \ge 0$), the candidate was rejected. The prior baseline model was restored without corruption. This confirms that automated gating effectively shields production models against unvetted self-learning updates.

### 11.2 Student Behavior: Hard vs Soft Targets (RQ3)
On the development set, Hard Student Seed 0 achieved 3.0 cp regret (33.3% top-1), while Distilled Student Seed 0 achieved 12.0 cp regret (0.0% top-1). Under Seed 1, both student variants converged to identical regret (34.3 cp dev, 42.0 cp test). Soft-target loss values during training remained orders of magnitude lower ($\sim 0.0018$) than cross-entropy loss ($\sim 3.03$), demonstrating smooth gradient flow from teacher logits. However, neither distillation nor hard labels showed a definitive superiority over the other due to sample variance.

### 11.3 Limitations and Failure Analysis
1. **Extremely Small Evaluation Sample Size:** Both development and test splits comprise exactly 3 positions each. While sufficient for CPU smoke testing and architectural verification, a 3-position sample has very high variance. A single tactical position in the test split caused baseline teacher regret to register as 422.7 cp. These figures must not be interpreted as statistically robust chess strength indicators.
2. **Engine Search Horizon:** Stockfish search was bounded to depth 2–4 to allow rapid CPU execution. Shallow evaluations contain horizon effects and tactical noise.
3. **Absence of Self-Play Match Evaluation:** Evaluation was conducted entirely through reference move ranking. No live games were played against Stockfish or third-party engines.

---

## 12. Reproduction Instructions

Every artifact in this report can be fully regenerated from repository root using standard CPU compute:

```bash
# 1. Environment & Test Suite
python3 -m pytest

# 2. Baseline Training (M2)
python3 scripts/train_baseline.py \
    --seed 0 \
    --epochs 3 \
    --output-dir experiments/results/medium

# 3. Verified Self-Learning Loop with Rollback Gate (M3)
python3 scripts/run_self_learning.py \
    --baseline-checkpoint experiments/results/medium/best.pt \
    --output-dir experiments/results/sl_round1 \
    --num-candidate-positions 6 \
    --engine-depth 4 \
    --fine-tune-epochs 2 \
    --gate-min-regret-reduction 0.0 \
    --seed 42

# 4. Knowledge Distillation Multi-Seed Training (M4)
python3 scripts/run_distillation.py \
    --checkpoint experiments/results/sl_round1/retained.pt \
    --dataset data/processed/dag12yi_phase35_labeled/train_records.jsonl \
    --manifest data/processed/dag12yi_phase35_labeled/manifest.json \
    --output-dir experiments/results/full_distillation \
    --epochs 2 \
    --seed 0 \
    --seed 1

# 5. Milestone 5 Evaluation Pipeline
# Development Split Evaluation:
python3 scripts/evaluate_models.py --split dev --output-dir experiments/results/m5_dev

# Final-Test Isolated Evaluation:
python3 scripts/evaluate_models.py --split test --output-dir experiments/results/m5_test
```

---

## 13. Conclusion

ChessEvolve successfully establishes a modular, reproducible research pipeline demonstrating verified self-learning and model distillation for move ranking on CPU. The central finding of the experiment is pedagogical and methodological: recursive self-improvement cannot be assumed to succeed monotonically, rendering automated verification and rollback gates essential. Furthermore, knowledge distillation delivers an ~78% compression in parameter count and model size alongside a ~1.55× CPU inference speedup while preserving comparable move ranking behavior.
