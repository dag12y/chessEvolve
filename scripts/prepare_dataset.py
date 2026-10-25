#!/usr/bin/env python3
"""CLI script to prepare reproducible train/dev/test datasets from a PGN file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path for direct invocation
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from chessevolve.pgn import PGNExtractionConfig, prepare_dataset_from_pgn


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract and split positions from a PGN file into reproducible datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pgn",
        type=Path,
        required=True,
        help="Path to the input PGN file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/default"),
        help="Directory to save train/dev/test JSONL records and manifest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic source-game splitting.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Target proportion for training split.",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.15,
        help="Target proportion for development split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Target proportion for test split.",
    )
    parser.add_argument(
        "--min-ply",
        type=int,
        default=4,
        help="Minimum ply number to begin extracting positions (skips early opening).",
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=80,
        help="Maximum plies per game to inspect.",
    )
    parser.add_argument(
        "--sample-frequency",
        type=int,
        default=2,
        help="Extract a position every N plies.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum number of valid games to process.",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Maximum total positions to extract across all games.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Enable bounded CPU smoke mode (max 5 games, min-ply 2, max-plies 20).",
    )

    args = parser.parse_args()

    config = PGNExtractionConfig(
        min_ply=args.min_ply,
        max_plies_per_game=args.max_plies,
        sample_frequency=args.sample_frequency,
        max_games=args.max_games,
        max_positions=args.max_positions,
        smoke=args.smoke,
    )

    proportions = (args.train_ratio, args.dev_ratio, args.test_ratio)

    print(f"Extracting positions from {args.pgn} (smoke={args.smoke})...")
    try:
        summary = prepare_dataset_from_pgn(
            pgn_path=args.pgn,
            output_dir=args.output_dir,
            proportions=proportions,
            seed=args.seed,
            config=config,
        )
    except Exception as exc:
        print(f"Error preparing dataset: {exc}", file=sys.stderr)
        return 1

    counts = summary["counts"]
    stats = summary["stats"]
    print("\nDataset Preparation Successful:")
    print(f"  Total games read:        {stats['total_games_read']}")
    print(f"  Valid games processed:   {stats['valid_games']}")
    print(f"  Rejected games:          {len(stats['rejected_games'])}")
    print(f"  Total positions:         {counts['total_records']}")
    print(f"  Train:                   {counts['train']['records']} records ({counts['train']['source_games']} games)")
    print(f"  Dev:                     {counts['dev']['records']} records ({counts['dev']['source_games']} games)")
    print(f"  Test:                    {counts['test']['records']} records ({counts['test']['source_games']} games)")
    print(f"\nArtifacts saved to {args.output_dir}:")
    print(f"  - train_records.jsonl")
    print(f"  - dev_records.jsonl")
    print(f"  - test_records.jsonl")
    print(f"  - manifest.json")
    print(f"  - summary.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
