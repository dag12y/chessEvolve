#!/usr/bin/env python3
"""CLI entry point for baseline training."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly from the repository root without setting
# PYTHONPATH manually.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import argparse

from chessevolve.train import TrainConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-games", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=12)
    parser.add_argument("--engine-depth", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="experiments/results/smoke")
    args = parser.parse_args()

    config = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_games=args.num_games,
        max_plies=args.max_plies,
        engine_depth=args.engine_depth,
    )
    run_training(config, Path(args.output_dir))


if __name__ == "__main__":
    main()
