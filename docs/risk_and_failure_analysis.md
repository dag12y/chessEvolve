# ChessEvolve — Risk and Failure Analysis

## 1. Main risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| Stockfish bias | The model may learn to imitate one engine configuration rather than chess quality in general. | Record version and settings; describe engine agreement as a proxy. |
| Noisy pseudo-labels | Bad examples may reinforce mistakes. | Validate legality, score perspective, and data quality; use held-out acceptance gates. |
| Development overfitting | Repeated checkpoint selection can overfit the development set. | Bound rounds and preserve an untouched final test. |
| Data leakage | Related positions may appear across splits. | Split by source game before extracting positions. |
| Score-perspective errors | White-relative and side-to-move scores can reverse preferences. | Normalize scores in one tested function. |
| Mate-score handling | Mate scores cannot be treated as ordinary centipawn scores without care. | Define conversion rules or exclude mate positions from regret calculations. |
| Slow engine analysis | Large searches may exceed the time budget. | Limit positions, candidates, and search depth/time. Cache results. |
| Weak baseline | Poor labels or features can make the entire comparison misleading. | Test features, labels, loss, and tiny-batch overfitting. |
| Distillation underperforms | Soft targets may not help this task. | Keep the hard-label baseline and analyze temperature/loss. |
| Unfair comparisons | Different budgets may confound results. | Match student size, data, optimization budget, and seeds. |
| Misleading latency | Single measurements are noisy. | Warm up, repeat, and report median latency. |
| Scope creep | UI work can consume the schedule. | Prioritize working experiments and reproducible results. |

## 2. Self-learning failure questions

- Did the verifier accept ambiguous or poor examples?
- Did training metrics improve while development metrics worsened?
- Were updates rejected because they were worse, or because the acceptance threshold was too strict?
- Did repeated development-set use encourage overfitting?
- Did the metric measure move quality or merely reference agreement?

## 3. Distillation failure questions

- Were soft targets too sharp or too uniform?
- Was temperature chosen without using final-test results?
- Was the teacher informative enough?
- Did the student have sufficient capacity?
- Were hard-label and distilled students trained under comparable conditions?
- Did the student actually become smaller or faster?

## 4. Data and engineering questions

- Are source games disjoint across splits?
- Are engine scores consistently oriented?
- Are seeds and configurations recorded?
- Can each chart be regenerated from saved result files?
- Are failed runs included in the final analysis?

## 5. Claims to avoid

Do not claim:
- The model taught itself chess from scratch.
- The verifier guarantees a move is correct.
- The neural network is stronger than Stockfish without a fair playing-strength evaluation.
- Distillation worked based only on training loss or a single seed.

Prefer precise statements, such as: “The candidate checkpoint reduced mean engine-score regret on the development set under the selected Stockfish settings.”

## 6. Fallback plan

If time or CPU performance becomes a problem:
1. Finish a valid baseline teacher.
2. Implement a small verified update/rollback loop.
3. Run the hard-label versus distilled-student comparison on a smaller fixed corpus.
4. Report the limitations and negative results clearly.

A smaller, reproducible experiment is preferable to an ambitious but unverified system.