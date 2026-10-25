#!/usr/bin/env python3
"""CLI to label an existing JSONL dataset with Stockfish evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path for direct invocation.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from chessevolve.labeling import LabelingConfig, label_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Label an existing JSONL dataset with Stockfish evaluations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing the unlabeled {train,dev,test}_records.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write labeled JSONL files and updated manifest.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Stockfish search depth.",
    )
    parser.add_argument(
        "--stockfish",
        type=str,
        default="tools/stockfish/stockfish/stockfish-linux-x86-64-universal",
        help="Path to the Stockfish executable.",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Maximum number of positions to label per split (useful for smoke tests).",
    )

    args = parser.parse_args()

    config = LabelingConfig(
        engine_depth=args.depth,
        stockfish_path=args.stockfish,
        max_positions=args.max_positions,
    )

    print(f"Labeling dataset at {args.dataset_dir} → {args.output_dir}")
    print(f"  Stockfish: {args.stockfish}")
    print(f"  Depth:     {args.depth}")
    if args.max_positions is not None:
        print(f"  Max positions per split: {args.max_positions}")

    try:
        summary = label_dataset(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            config=config,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nLabeling complete. Engine: {summary['engine_version']}")
    total_labeled = sum(v["labeled"] for v in summary["splits"].values())
    total_skipped = sum(v["skipped"] for v in summary["splits"].values())
    for split, info in summary["splits"].items():
        print(
            f"  {split:5s}: {info['input_records']:4d} input → "
            f"{info['labeled']:4d} labeled, {info['skipped']:3d} skipped"
        )
    print(f"\n  Total labeled:  {total_labeled}")
    print(f"  Total skipped:  {total_skipped}")
    print(f"\nArtifacts saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
