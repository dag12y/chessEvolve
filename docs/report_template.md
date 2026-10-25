# ChessEvolve Experiment Report

**Author:**  
**Date:**  
**Git commit:**  
**Environment:**  
**Stockfish version/settings:**  
**Dataset version/hash:**  

## 1. Summary

Summarize the research questions, what was implemented, and the most important measured results. Do not claim improvements that were not measured.

## 2. Problem and motivation

Explain verified self-learning and knowledge distillation in the context of chess move ranking.

## 3. Method

### Dataset and splits

Describe the data source, filtering, number of games and positions, grouping strategy, split sizes, and leakage controls.

### Model

Describe the feature representation, candidate-move ranking interface, teacher and student architectures, loss functions, optimizer, and training budget.

### Verified self-learning

Describe candidate generation, verifier checks, acceptance thresholds, rollback, number of rounds, and rejection reasons.

### Distillation

Describe hard labels, soft targets, temperature, loss function, student size, seeds, and comparable training budgets.

## 4. Experimental setup

Record hardware, software versions, Stockfish settings, seeds, runtime limits, and evaluation procedure.

## 5. Results

Include a table with all planned configurations and seeds. Add at least one chart. Report multi-seed results with a mean and a measure of variation.

## 6. Analysis

Answer each research question. Explain accepted and rejected updates and whether distillation helped. Separate reference agreement from playing strength.

## 7. Failure analysis and limitations

Discuss engine bias, pseudo-label quality, development-set reuse, data size, model capacity, runtime, and failed runs.

## 8. Conclusion and next steps

State what the evidence supports, what remains uncertain, and one or two follow-up experiments.

## Reproducibility checklist

- [ ] Code and README included
- [ ] Data source and preparation documented
- [ ] Train/development/final-test manifests saved
- [ ] Seeds and configurations saved
- [ ] Stockfish version and settings recorded
- [ ] Accepted and rejected rounds logged
- [ ] Hard-label and distilled students evaluated over at least two seeds
- [ ] Model size and CPU latency measured
- [ ] Final test kept untouched until design freeze
- [ ] Failure analysis included
- [ ] At least one chart included