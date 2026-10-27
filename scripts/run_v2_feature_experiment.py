#!/usr/bin/env python3
"""Run one controlled A/B/C/D feature-group training job.

Only train_records.jsonl and dev_records.jsonl are read. The test split is
intentionally not loaded by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chessevolve.real_train import (  # noqa: E402
    RealTrainConfig,
    load_records_from_jsonl,
    run_training_on_records,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-group", choices=("B", "C", "D"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--loss-mode", choices=("classification", "pairwise"), default="classification")
    parser.add_argument("--ranking-margin-scale", type=float, default=100.0)
    args = parser.parse_args()

    manifest = args.dataset_dir / "manifest.json"
    train_records = load_records_from_jsonl(args.dataset_dir / "train_records.jsonl")
    dev_records = load_records_from_jsonl(args.dataset_dir / "dev_records.jsonl")
    print(f"Loaded train={len(train_records)} dev={len(dev_records)} records")
    if not train_records or not dev_records:
        raise SystemExit("train and dev records are required")

    metadata = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "library_versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "numpy", "python-chess")
        },
    }
    config = RealTrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=2,
        dropout=0.0,
        feature_group=args.feature_group,
        loss_mode=args.loss_mode,
        ranking_margin_scale=args.ranking_margin_scale,
    )
    run_training_on_records(
        train_records,
        dev_records,
        config,
        args.output_dir,
        metadata=metadata,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
