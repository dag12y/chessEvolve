#!/usr/bin/env python3
"""Benchmark CPU search depths on a small tactical regression suite."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import chess

# Allow direct execution from the repository root or from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app import ModelService


POSITIONS = {
    "mate_in_one": "7k/5K2/6Q1/8/8/8/8/8 w - - 0 1",
    "queen_endgame": "7k/8/6Q1/5K2/8/8/8/7q w - - 0 1",
    "promotion": "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "capture": "4k3/8/8/3r4/8/2B5/8/4K3 w - - 0 1",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="v2_mixed_teacher_seed1")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    service = ModelService()
    print("| Position | Depth | Median seconds | Chosen move |")
    print("|---|---:|---:|---|")
    for name, fen in POSITIONS.items():
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError(f"invalid tactical FEN: {name}")
        for depth in (0, 1, 2):
            durations = []
            chosen = ""
            for _ in range(args.repeats):
                started = time.perf_counter()
                result = service.rank(board.copy(), args.model_id, 0.0, depth)
                durations.append(time.perf_counter() - started)
                chosen = result["chosen"]["uci"]
            print(
                f"| {name} | {depth} | {statistics.median(durations):.3f} | {chosen} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
