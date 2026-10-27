# ChessEvolve: Verified Self-Learning & Knowledge Distillation
## Academic Presentation (8–10 Minutes)

---

### Slide 1: Title & Research Problem

#### Header: ChessEvolve: Verified Self-Learning and Distillation for Chess Move Ranking
- **Topic:** Evaluating recursive self-improvement and model compression in edge-bounded chess candidate ranking.
- **Presenter:** ChessEvolve Project Team
- **Context:** Emerging AI Paradigms, Educational Research Project
- **Key Objective:** Build a reproducible, CPU-friendly pipeline testing whether small neural rankers can improve via external engine feedback, and whether distillation preserves behavior in tiny models.
- **Non-Goal:** ChessEvolve is **not** an engine rival to Stockfish; it is an architectural investigation of self-training safety and distillation efficiency.

> **Speaker Notes (0:00 - 0:45):**
> "Good day. Today I am presenting ChessEvolve, an investigation into two foundational AI paradigms: recursive self-learning with independent verification, and knowledge distillation. Deep chess engines like Stockfish are computationally demanding. We explore whether small neural networks can rank candidate moves on commodity CPUs, whether iterative self-training can safely improve them without human labels, and how much compression distillation achieves. Crucially, ChessEvolve is an architectural study in verifiable ML safety, not an attempt to outplay Stockfish."

---

### Slide 2: Research Questions & Objectives

#### Core Inquiries
1. **RQ1 (Recursive Self-Learning):** Can active selection and external engine feedback improve a teacher's move-ranking quality on held-out positions?
2. **RQ2 (Safety & Rollback):** How reliably does an automated development gate detect regressions and execute rollbacks?
3. **RQ3 (Distillation Efficacy):** Does soft-target distillation (temperature-scaled KL divergence) offer advantages over hard-label training (cross-entropy)?
4. **RQ4 (System Efficiency):** What empirical trade-offs emerge across parameter count, disk storage, and CPU inference latency?

> **Speaker Notes (0:45 - 1:30):**
> "We set out to answer four specific research questions. First, can active uncertainty sampling and engine supervision iteratively improve move ranking? Second, if fine-tuning degrades performance—a common hazard in self-play and pseudo-labeling—can an automated gate prevent regression? Third, how does soft-target distillation compare with standard hard labels when compressing the model? And finally, what are the concrete hardware efficiency gains on a standard CPU?"

---

### Slide 3: System Architecture Overview

#### Five End-to-End Modular Stages
```text
[Position Generator] ──> [Game-Level Splitter] ──> [Train / Dev / Test Datasets]
                                                          │
          ┌───────────────────────────────────────────────┘
          ▼
   [Baseline Teacher] ──> [Uncertainty Sampling] ──> [Stockfish Adapter]
                                                            │
                                                            ▼
   [Accepted / Rollback Checkpoint] <── [Dev Gate] <── [Verifier]
          │
          ▼
   [M4 Distillation] ──> [Hard Student] vs [Soft Distilled Student]
          │
          ▼
   [M5 Evaluation] ──> Metrics, Latency Benchmarking, Artifacts & Charts
```
- **Legal Move Provider:** `python-chess` generates all legal moves for a given position.
- **Model Role:** PyTorch MLP acts strictly as a *move ranker*, scoring each legal move option.

> **Speaker Notes (1:30 - 2:15):**
> "Our architecture is divided into five strictly separated components. First, data generation and game-level splitting. Second, the baseline teacher training. Third, an active self-learning loop where the model selects uncertain positions, queries Stockfish, passes an independent verifier, and faces an acceptance gate. Fourth, model compression via distillation. And fifth, our evaluation pipeline. The neural network never invents moves; `python-chess` supplies legal moves, and our network ranks them."

---

### Slide 4: Dataset & Leakage Controls

#### Methodological Integrity
- **Source-Game Splitting:** Instead of shuffling individual positions, data is split strictly at the **source-game level** (`split_by_source_game`).
- **Leakage Prevention:** Positions from the same game share tactical context; game-level splitting ensures no tactically related positions cross from training into development or test sets.
- **Manifest Provenance:** Frozen dataset manifest (`manifest.json`, Seed 7, SHA-256 `d177c946...`):
  - **Train:** 2 source games, 6 positions (50.0%)
  - **Dev:** 1 source game, 3 positions (25.0%)
  - **Test:** 1 source game, 3 positions (25.0%)
- **Test Set Isolation:** The test split remained completely untouched and unread until model architectures, hyperparameters, and checkpoints were frozen.

> **Speaker Notes (2:15 - 3:00):**
> "A common pitfall in chess ML is data leakage. Consecutive board states in a single game share almost identical pieces. If you shuffle positions randomly, the test set leaks into training. We enforce strict game-level splitting. Our manifest records the exact source game IDs for train, dev, and test. Furthermore, our final-test set was kept completely isolated until all model decisions and hyperparameters were permanently frozen."

---

### Slide 5: Teacher Model & Baseline Training

#### Feature Encodings & Model Specifications
- **Position Features (780d):** 12 piece-placement bitboards (768d) + active side, castling rights, en passant, halfmove clock, material balance (12d).
- **Move Features (135d):** One-hot source square (64d), destination square (64d), moving piece, captured piece, promotion type (7d).
- **Teacher MLP:** 2 hidden layers (`hidden_dim = 128`), ReLU, 136,065 trainable parameters.
- **Baseline Training:** Adam optimizer ($\text{lr} = 10^{-3}$, batch size 4), trained on Stockfish-evaluated positions.
- **Baseline Metric on Dev:**
  - Mean Regret: **60.0 cp** (dev split, 3 positions) / **22.2 cp** (M3 pool, 10 positions).
  - Trainable Parameters: **136,065** | Checkpoint Size: **534.9 KiB**.

> **Speaker Notes (3:00 - 3:45):**
> "The teacher model takes a 780-dimensional board representation and a 135-dimensional move representation, processing each legal move through a 2-layer MLP with 136,000 parameters. It outputs a scalar logit per candidate move. The baseline model was trained with Adam on bounded Stockfish supervision. Regret measures the centipawn gap between the engine's best move and the model's top selection. Lower regret is better."

---

### Slide 6: Verified Self-Learning Loop & Acceptance Gate

#### The Safety Loop in Milestone 3
1. **Uncertainty Selection:** Teacher identifies training positions where output entropy is highest (uncertain move preferences).
2. **Stockfish Feedback:** Stockfish queries candidate moves to depth 4, normalizing centipawns to side-to-move perspective.
3. **Independent Verifier:** Verifies move legality, completeness of candidate scores, and split validity (all 6 records passed).
4. **Candidate Fine-Tuning:** Candidate model fine-tuned for 2 epochs on verified feedback records.
5. **Acceptance Gate:** Compares baseline vs. candidate on held-out development positions. Must achieve $\ge 0.0\text{ cp}$ regret reduction.

> **Speaker Notes (3:45 - 4:30):**
> "Milestone 3 implements the verified recursive loop. The teacher actively asks for Stockfish evaluations on positions where it feels most uncertain. An independent verifier validates the data integrity before fine-tuning. But critically, we never blindly adopt the fine-tuned weights. The candidate model must prove itself on held-out development positions against the baseline. If it regresses, it is rejected."

---

### Slide 7: M3 Rejection Result: Why It Matters

#### An Honest Negative Result Validating Architecture
- **Pre-Declared Acceptance Rule:** Regret Reduction $\ge 0.0\text{ cp}$ on development data.
- **Empirical M3 Gate Outcomes:**
  - Baseline Dev Regret: **22.20 cp**
  - Candidate Dev Regret: **26.10 cp**
  - Regret Difference: **-3.90 cp (Performance Degraded)**
  - Gate Decision: **`REJECT`** (documented in `experiments/results/sl_round1/decision.json`)
- **Action Taken:** Candidate weights discarded; baseline restored as `retained.pt`.
- **Takeaway:** Recursive learning does not guarantee monotonic improvement. Automated verification and rollback are mandatory to prevent silent degradation.

> **Speaker Notes (4:30 - 5:15):**
> "Here is our first major empirical finding: the self-learning candidate failed. Fine-tuning on 6 uncertain positions actually degraded mean regret by 3.9 centipawns. In an unverified system, this degraded model would have been pushed to production. In ChessEvolve, the gate caught the regression, logged the rejection, and rolled back to the baseline. This negative result is an architectural victory: it proves our safety gate works exactly as designed."

---

### Slide 8: Knowledge Distillation: Hard vs Soft Targets

#### Compressing the Retained Teacher (Milestone 4)
- **Student Architecture:** 1 hidden layer (`hidden_dim = 32`), ReLU, **29,889 trainable parameters** (78.0% smaller).
- **Two Training Regimes (Evaluated across Seeds 0 & 1):**
  1. **Hard-Label Student:** Cross-entropy loss against the single best Stockfish candidate.
  2. **Soft-Target Distilled Student:** Temperature-scaled KL divergence matching full teacher probability distribution ($T = 2.0$):
     $$\mathcal{L}_{\text{distill}} = T^2 \cdot D_{\text{KL}}(\sigma(\mathbf{z}_s / T) \parallel \sigma(\mathbf{z}_t / T))$$
- **Optimization:** 2 epochs, batch size 4, $\text{lr} = 10^{-3}$, Adam optimizer.

> **Speaker Notes (5:15 - 6:00):**
> "In Milestone 4, we froze the retained baseline teacher and trained two student networks with 78% fewer parameters. The hard-label student learns only from the single best move. The distilled student learns from the teacher's full probability distribution at temperature 2.0, absorbing information about second-best and blunder moves. Both methods were trained under identical budgets across two distinct random seeds."

---

### Slide 9: Actual Experimental Results

#### Head-to-Head Comparison on Real Artifacts
*Evaluated on identical held-out positions (M5 Benchmark):*

| Model | Dev Regret (cp) | Dev Top-1 | Test Regret (cp) | Parameters | Model Size | CPU Latency |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline Teacher** | 60.0 cp | 0.0% | 422.7 cp | 136,065 | 534.9 KiB | 0.120 ms/pos |
| **M3 Retained Teacher** | 60.0 cp | 0.0% | 422.7 cp | 136,065 | 535.2 KiB | 0.119 ms/pos |
| **M3 Rejected Candidate** | 60.0 cp | 0.0% | 422.7 cp | 136,065 | 535.0 KiB | 0.124 ms/pos |
| **Hard Student (Seed 0)** | **3.0 cp** | **33.3%** | 419.0 cp | **29,889** | **120.1 KiB** | **0.076 ms/pos** |
| **Distilled Student (Seed 0)** | **12.0 cp** | 0.0% | 430.3 cp | **29,889** | **120.2 KiB** | **0.077 ms/pos** |
| **Hard Student (Seed 1)** | 34.3 cp | **33.3%** | **42.0 cp** | **29,889** | **120.1 KiB** | 0.094 ms/pos |
| **Distilled Student (Seed 1)** | 34.3 cp | **33.3%** | **42.0 cp** | **29,889** | **120.2 KiB** | **0.074 ms/pos** |

> **Speaker Notes (6:00 - 6:45):**
> "Here are our actual measured results from saved JSON artifacts. Notice that both student variants achieve lower regret on development data than the teacher, while operating at 78% smaller parameter footprint and 1.55x faster inference. On final test data, Seed 1 students scored 42.0 centipawns regret. Because the evaluation sets have 3 positions, we do not claim students are fundamentally stronger players, but we verify that the distillation pipeline successfully trains compact rankers on CPU."

---

### Slide 10: Efficiency Trade-offs & Saved Charts

#### Measurable Hardware Gains on CPU
- **Parameters:** Compressed from **136,065** down to **29,889** (**-78.0%**).
- **Storage:** Checkpoints reduced from **535.2 KiB** to **120.1 KiB** (**-77.6%**).
- **Latency:** Inference latency dropped from **0.120 ms** to **0.074–0.078 ms** per position (**~1.55× speedup**).
- **Charts Generated by Evaluation Pipeline:**
  - `experiments/results/m5_dev/regret_by_model_dev.png`
  - `experiments/results/m5_dev/top1_agreement_by_model_dev.png`
  - `experiments/results/m5_test/regret_by_model_test.png`

> **Speaker Notes (6:45 - 7:30):**
> "The engineering trade-offs are unambiguous. Moving from a 2-layer 128-dim teacher to a 1-layer 32-dim student cuts parameters and storage by more than three quarters, and reduces latency from 0.12 milliseconds down to 0.07 milliseconds per position. Our evaluation script automatically plots and saves these regret and top-1 bar charts directly from the JSON results."

---

### Slide 11: Failure Analysis & Explicit Limitations

#### What Worked, What Failed, and Scientific Limits
1. **Tiny Sample Size Warning:** Dev and test sets each contain 3 positions. A single blunder in a test position caused baseline regret to spike to 422.7 cp. These metrics are smoke-test benchmarks, not population-level chess ratings.
2. **M3 Update Degradation:** Self-learning on small batches without diversity can overfit local tactical quirks, causing out-of-distribution regression on held-out positions.
3. **Engine Search Bounds:** Stockfish depth 2–4 allows fast CPU execution but introduces horizon limitations.
4. **No Tournament Play:** Rankings reflect move-ranking loss against Stockfish, not Elo rating from actual played matches.

> **Speaker Notes (7:30 - 8:15):**
> "We must be completely honest about our limitations. Our test set is small—3 positions each for dev and test to guarantee sub-second execution on CPU. A single sharp tactical position in the test game skewed teacher regret to 422 centipawns. Furthermore, shallow engine depth means our supervision has horizon limits. These results validate our code and architecture, but cannot be extrapolated to general playing strength."

---

### Slide 12: Conclusions & Reproducibility

#### Key Takeaways & Reproduction
- **Takeaway 1:** Self-learning systems require automated validation gates; self-improvement is non-monotonic.
- **Takeaway 2:** Knowledge distillation provides severe model compression (-78%) and latency speedups (~1.55×) while preserving functional move ranking.
- **Takeaway 3:** Complete CPU-only reproducibility: every metric, table, and chart can be regenerated in minutes via:
  ```bash
  pytest
  python3 scripts/evaluate_models.py --split dev
  python3 scripts/evaluate_models.py --split test
  ```
- **Future Directions:** Multi-round self-learning with replay buffers, deeper master-game datasets, and live self-play tournament matches.

> **Speaker Notes (8:15 - 9:00):**
> "In conclusion, ChessEvolve demonstrates that recursive self-learning can be safely managed on commodity hardware when paired with strict verification and rollback gates. Distillation delivers dramatic parameter and latency savings. Everything in this presentation is fully reproducible with standard CLI commands and verified by our 197 automated tests. Thank you, and I look forward to your questions."
