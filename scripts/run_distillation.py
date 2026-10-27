#!/usr/bin/env python3
"""Run comparable hard-label and soft-target M4 student experiments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from chessevolve.distillation import (  # noqa: E402
    _record_to_example,
    DistillationConfig,
    load_records,
    run_student,
    validate_records_against_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Frozen split manifest used to validate train/dev source-game membership.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--hard-loss-weight", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-hidden-layers", type=int, default=1)
    parser.add_argument("--feature-group", choices=("A", "B", "C", "D"), default="D")
    args = parser.parse_args()
    seeds = args.seed or [0, 1]
    records = validate_records_against_manifest(
        load_records(args.dataset), args.manifest, "train"
    )
    valid_records = []
    invalid_train_ids = []
    for record in records:
        try:
            _record_to_example(record)
        except ValueError as exc:
            if "board is not a valid position" not in str(exc):
                raise
            invalid_train_ids.append(record.position_id)
        else:
            valid_records.append(record)
    dataset_dir = args.dataset if args.dataset.is_dir() else args.dataset.parent
    dev_records = validate_records_against_manifest(
        load_records(dataset_dir / "dev_records.jsonl"), args.manifest, "dev"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for seed in seeds:
        config = DistillationConfig(
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            temperature=args.temperature,
            hard_loss_weight=args.hard_loss_weight,
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.num_hidden_layers,
            feature_group=args.feature_group,
        )
        for method in ("hard", "distilled"):
            all_results.append(
                run_student(
                    args.checkpoint,
                    valid_records,
                    args.output_dir,
                    config,
                    method,
                    dev_records=dev_records,
                )
            )
    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "manifest": str(args.manifest) if args.manifest else None,
        "invalid_train_records_excluded": invalid_train_ids,
        "results": all_results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
