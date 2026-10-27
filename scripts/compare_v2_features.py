#!/usr/bin/env python3
"""Compare A-D checkpoints on the dev split only."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chessevolve.distillation import _record_to_example  # noqa: E402
from chessevolve.real_train import load_records_from_jsonl  # noqa: E402
from chessevolve.train import load_checkpoint  # noqa: E402


def measure(model, examples, repeats: int) -> float:
    sample = examples[0]
    position = torch.from_numpy(sample.position_features).float()
    moves = torch.from_numpy(sample.move_features).float()
    torch.set_num_threads(1)
    with torch.no_grad():
        model(position, moves)
        durations = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            model(position, moves)
            durations.append((time.perf_counter_ns() - started) / 1_000_000)
    return float(statistics.median(durations))


def evaluate(model, examples):
    rows = []
    with torch.no_grad():
        for ex in examples:
            logits = model(
                torch.from_numpy(ex.position_features).float(),
                torch.from_numpy(ex.move_features).float(),
            )
            predicted = int(torch.argmax(logits).item())
            regret = ex.candidate_scores[ex.target_index] - ex.candidate_scores[predicted]
            rows.append(
                (
                    predicted == ex.target_index,
                    regret,
                    not any(abs(score) >= 90000 for score in ex.candidate_scores),
                )
            )
    regrets = [row[1] for row in rows]
    non_mate = [row[1] for row in rows if row[2]]
    return {
        "top1": sum(row[0] for row in rows) / len(rows),
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(statistics.median(regrets)),
        "non_mate_regret": float(np.mean(non_mate)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    dev_records = load_records_from_jsonl(args.dataset_dir / "dev_records.jsonl")
    rows = []
    for group in ("A", "B", "C", "D"):
        for seed in (0, 1):
            if group == "A":
                checkpoint = args.results_dir.parent / f"full_baseline_seed{seed}" / "final.pt"
                encoding_group = "A"
            else:
                checkpoint = args.results_dir / f"{group}_seed{seed}" / "best.pt"
                encoding_group = group
            model, checkpoint_data = load_checkpoint(checkpoint)
            examples = [_record_to_example(record, encoding_group) for record in dev_records]
            metrics = evaluate(model, examples)
            rows.append(
                {
                    "variant": group,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "input_dim": model.config.position_dim + model.config.move_dim,
                    "parameters": model.parameter_count(),
                    "median_latency_ms": measure(model, examples, args.repeats),
                    **metrics,
                }
            )

    for group in ("A", "B", "C", "D"):
        group_rows = [row for row in rows if row["variant"] == group]
        rows.append(
            {
                "variant": group,
                "seed": "mean",
                "checkpoint": "",
                "input_dim": group_rows[0]["input_dim"],
                "parameters": group_rows[0]["parameters"],
                **{
                    key: float(np.mean([row[key] for row in group_rows]))
                    for key in (
                        "median_latency_ms",
                        "top1",
                        "mean_regret",
                        "median_regret",
                        "non_mate_regret",
                    )
                },
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print("| Variant | Seed | Input | Params | Top-1 | Mean regret | Median regret | Non-mate regret | Latency ms |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['variant']} | {row['seed']} | {row['input_dim']} | "
            f"{row['parameters']} | {row['top1']:.4f} | {row['mean_regret']:.2f} | "
            f"{row['median_regret']:.2f} | {row['non_mate_regret']:.2f} | "
            f"{row['median_latency_ms']:.4f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
