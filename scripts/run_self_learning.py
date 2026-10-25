#!/usr/bin/env python3
"""CLI entry point for one round of the verified self-learning loop.

Usage example (from repository root):

    python scripts/run_self_learning.py \\
        --baseline-checkpoint experiments/results/medium/best.pt \\
        --output-dir experiments/results/sl_round1 \\
        --num-candidate-positions 8 \\
        --engine-depth 4 \\
        --fine-tune-epochs 2 \\
        --gate-min-regret-reduction 0 \\
        --round 1 \\
        --seed 42

Artifacts written to --output-dir:
    config.json            – all parameters for this run
    decision.json          – ACCEPT / REJECT with metrics and thresholds
    round_summary.json     – human-readable summary
    feedback_records.jsonl – one line per generated feedback record
    rejections.jsonl       – rejected feedback items with reasons
    rejection_summary.json – rejection counts per reason
    metrics_baseline.json  – baseline dev evaluation
    metrics_candidate.json – candidate dev evaluation
    retained.pt            – copy of baseline (safe even on REJECT)
    candidate.pt           – fine-tuned candidate (always saved for audit)
    accepted.pt            – candidate promoted (only on ACCEPT)
    current_best.pt        – checkpoint to carry forward to the next round
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repository root without setting PYTHONPATH manually.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from chessevolve.data import PositionRecord
from chessevolve.dataset import generate_random_positions, split_records
from chessevolve.self_learning import (
    GateConfig,
    SelfLearningRoundConfig,
    VerifierConfig,
    InsufficientVerifiedExamplesError,
    run_self_learning_round,
)
from chessevolve.train import load_checkpoint


def _load_records_jsonl(path: Path) -> list[PositionRecord]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            records.append(
                PositionRecord(
                    position_id=payload["position_id"],
                    source_game_id=payload["source_game_id"],
                    fen=payload["fen"],
                    candidate_moves=payload["candidate_moves"],
                    split=payload["split"],
                    engine_version=payload.get("engine_version"),
                    engine_settings=payload.get("engine_settings"),
                    reference_best_move=payload.get("reference_best_move"),
                    candidate_scores_cp=payload.get("candidate_scores_cp"),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid PositionRecord at {path}:{line_number}: {exc}") from exc
    return records


def _load_manifest_pools(manifest_path: Path) -> tuple[list[PositionRecord], list[PositionRecord]]:
    """Load frozen M2 train/dev records without re-splitting them."""
    manifest_path = manifest_path.resolve()
    if manifest_path.is_dir():
        dataset_dir = manifest_path
        manifest_path = dataset_dir / "manifest.json"
    else:
        dataset_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if isinstance(manifest, list):
        records = [
            PositionRecord(
                position_id=item["position_id"],
                source_game_id=item["source_game_id"],
                fen=item["fen"],
                candidate_moves=item["candidate_moves"],
                split=item["split"],
                engine_version=item.get("engine_version"),
                engine_settings=item.get("engine_settings"),
                reference_best_move=item.get("reference_best_move"),
                candidate_scores_cp=item.get("candidate_scores_cp"),
            )
            for item in manifest
        ]
    elif isinstance(manifest, dict) and {"source_game_ids", "counts"} <= set(manifest):
        train_path = dataset_dir / "train_records.jsonl"
        dev_path = dataset_dir / "dev_records.jsonl"
        if not train_path.is_file() or not dev_path.is_file():
            raise ValueError(
                f"Manifest {manifest_path} requires sibling train_records.jsonl and dev_records.jsonl"
            )
        records = _load_records_jsonl(train_path) + _load_records_jsonl(dev_path)
        declared = manifest["source_game_ids"]
        for record in records:
            if record.split not in {"train", "dev"}:
                raise ValueError(f"Non-train/dev record in M2 dataset: {record.position_id}")
            if record.source_game_id not in declared.get(record.split, []):
                raise ValueError(
                    f"Record {record.position_id} has source game {record.source_game_id!r}, "
                    f"not declared in manifest {record.split!r} split"
                )
    else:
        raise ValueError(
            "Manifest must be a PositionRecord list or an M2 manifest with "
            "'source_game_ids' and 'counts'"
        )

    train_pool = [record for record in records if record.split == "train"]
    dev_pool = [record for record in records if record.split == "dev"]
    if not train_pool or not dev_pool:
        raise ValueError("Frozen manifest must contain non-empty train and dev records")
    return train_pool, dev_pool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one round of the ChessEvolve verified self-learning loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Checkpoint paths
    parser.add_argument(
        "--baseline-checkpoint",
        required=True,
        help="Path to the M2 baseline checkpoint (.pt) to start from.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where all round artifacts will be written.",
    )

    # Data generation (simple random positions; replace with real PGN data later)
    parser.add_argument("--num-games", type=int, default=10,
                        help="Number of random games to generate for train/dev pool.")
    parser.add_argument("--max-plies", type=int, default=16,
                        help="Maximum plies per random game.")
    parser.add_argument("--data-seed", type=int, default=0,
                        help="Seed for random position generation and splitting.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Frozen M2 manifest or dataset directory; loads sibling train/dev JSONL without re-splitting.",
    )

    # Position selection
    parser.add_argument("--num-candidate-positions", type=int, default=8,
                        help="Number of training positions to select for feedback.")
    parser.add_argument("--selection-policy", default="uncertainty",
                        choices=["random", "uncertainty", "disagreement"],
                        help="Position selection policy.")

    # Engine
    parser.add_argument("--engine-depth", type=int, default=4,
                        help="Stockfish search depth for feedback generation.")
    parser.add_argument(
        "--stockfish-path",
        default="tools/stockfish/stockfish/stockfish-linux-x86-64-universal",
    )

    # Fine-tuning
    parser.add_argument("--fine-tune-epochs", type=int, default=2)
    parser.add_argument("--fine-tune-lr", type=float, default=5e-4)
    parser.add_argument("--fine-tune-batch-size", type=int, default=4)

    # Gate thresholds (must be declared before running)
    parser.add_argument(
        "--gate-min-regret-reduction",
        type=float,
        default=0.0,
        help="Minimum centipawn regret reduction for the candidate to be accepted.",
    )
    parser.add_argument(
        "--gate-max-agreement-drop",
        type=float,
        default=0.05,
        help="Maximum permitted drop in top-1 agreement for the candidate.",
    )

    # Verifier
    parser.add_argument("--min-score-margin", type=int, default=None,
                        help="Minimum centipawn margin between top 2 candidates.")
    parser.add_argument("--min-verified-examples", type=int, default=2,
                        help="Minimum verified examples required to proceed.")

    # Misc
    parser.add_argument("--round", type=int, default=1, dest="round_number",
                        help="Round number (for logging).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master seed for this round (selection + fine-tuning).")

    args = parser.parse_args()

    baseline_path = Path(args.baseline_checkpoint)
    output_dir = Path(args.output_dir)

    if not baseline_path.exists():
        print(f"ERROR: baseline checkpoint not found: {baseline_path}", file=sys.stderr)
        sys.exit(1)

    # Build train/dev pool from a frozen manifest or, explicitly, synthetic data.
    if args.manifest:
        print(f"Loading PositionRecord data from manifest: {args.manifest}...")
        train_pool, dev_pool = _load_manifest_pools(Path(args.manifest))
        print(f"  Train records: {len(train_pool)}  Dev records: {len(dev_pool)}")
    else:
        print(f"Generating {args.num_games} random games (seed={args.data_seed})...")
        records = generate_random_positions(args.num_games, args.max_plies, args.data_seed)
        split_res = split_records(records, seed=args.data_seed)
        train_pool = list(split_res["train"])
        dev_pool = list(split_res["dev"])
        print(f"  Train records: {len(train_pool)}  Dev records: {len(dev_pool)}")

    gate_cfg = GateConfig(
        min_regret_reduction_cp=args.gate_min_regret_reduction,
        max_agreement_drop=args.gate_max_agreement_drop,
    )
    verifier_cfg = VerifierConfig(
        allowed_split="train",
        min_score_margin_cp=args.min_score_margin,
    )

    cfg = SelfLearningRoundConfig(
        seed=args.seed,
        round_number=args.round_number,
        num_candidate_positions=args.num_candidate_positions,
        selection_policy=args.selection_policy,
        engine_depth=args.engine_depth,
        stockfish_path=args.stockfish_path,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_lr=args.fine_tune_lr,
        fine_tune_batch_size=args.fine_tune_batch_size,
        min_verified_examples=args.min_verified_examples,
        gate=gate_cfg,
        verifier=verifier_cfg,
    )

    print(f"\nRunning self-learning round {args.round_number}...")
    print(f"  Baseline:            {baseline_path}")
    print(f"  Output dir:          {output_dir}")
    print(f"  Selection policy:    {args.selection_policy}")
    print(f"  Candidate positions: {args.num_candidate_positions}")
    print(f"  Engine depth:        {args.engine_depth}")
    print(f"  Fine-tune epochs:    {args.fine_tune_epochs}")
    print(f"  Gate thresholds:     regret_reduction >= {args.gate_min_regret_reduction} cp,"
          f" agreement_drop <= {args.gate_max_agreement_drop}")

    try:
        result = run_self_learning_round(
            baseline_path, train_pool, dev_pool, output_dir, cfg
        )
    except InsufficientVerifiedExamplesError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise

    s = result.summary
    print(f"\n{'='*60}")
    print(f"Round {s['round_number']} complete")
    print(f"  Decision:            {'ACCEPT' if s['accepted'] else 'REJECT'}")
    print(f"  Reason:              {s['decision_reason']}")
    print(f"  Positions selected:  {s['num_candidate_positions']}")
    print(f"  Feedback generated:  {s['num_feedback_generated']}")
    print(f"  Verified examples:   {s['num_verified_examples']}")
    print(f"  Rejected examples:   {s['num_rejected_examples']}")
    bm = s["baseline_metrics"]
    cm = s["candidate_metrics"]
    print(f"  Baseline  regret={bm['mean_regret_cp']:.1f} cp  top1={bm['top1_agreement']:.3f}")
    print(f"  Candidate regret={cm['mean_regret_cp']:.1f} cp  top1={cm['top1_agreement']:.3f}")
    print(f"  Regret improvement:  {s['regret_diff_cp']:+.1f} cp")
    print(f"{'='*60}")
    if result.decision.accepted:
        print(f"\nAccepted checkpoint: {output_dir / 'accepted.pt'}")
        print(f"Current best:        {output_dir / 'current_best.pt'}")
    else:
        print(f"\nBaseline retained:   {output_dir / 'retained.pt'}")
        print(f"Current best:        {output_dir / 'current_best.pt'}")
    print(f"Full artifacts:      {output_dir}")


if __name__ == "__main__":
    main()
