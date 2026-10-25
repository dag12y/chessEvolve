#!/usr/bin/env python3
"""CLI to train real-data baseline models from a labeled JSONL dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path for direct invocation.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from chessevolve.real_train import RealTrainConfig, load_records_from_jsonl, run_training_on_records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train real-data baseline MLPRanker models from a labeled JSONL dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--labeled-dataset",
        type=Path,
        required=True,
        help="Path to a labeled dataset directory containing train/dev/test_records.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Base directory for results (each seed gets its own sub-directory).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        dest="seeds",
        default=None,
        help="Random seed(s). May be repeated. Default: [0, 1].",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size.",
    )

    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else [0, 1]

    # Load records once; they are shared across seeds.
    labeled_dir = args.labeled_dataset
    print(f"Loading records from {labeled_dir} …")
    train_records = load_records_from_jsonl(labeled_dir / "train_records.jsonl")
    dev_records = load_records_from_jsonl(labeled_dir / "dev_records.jsonl")
    print(f"  Train records: {len(train_records)}")
    print(f"  Dev records:   {len(dev_records)}")

    if not train_records:
        print("ERROR: No train records found. Cannot train.", file=sys.stderr)
        return 1

    all_results = {}
    for seed in seeds:
        run_dir = args.output_dir / f"real_baseline_seed{seed}"
        print(f"\n--- Seed {seed} → {run_dir} ---")
        config = RealTrainConfig(
            seed=seed,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
        try:
            result = run_training_on_records(
                train_records=train_records,
                dev_records=dev_records,
                config=config,
                output_dir=run_dir,
            )
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            all_results[seed] = {"error": str(exc)}
            continue

        # Print final metrics.
        history = result["history"]
        split_info = result["split_info"]
        if history:
            last = history[-1]
            print(
                f"  Final: dev_loss={last.get('loss', '?'):.4f}  "
                f"top1={last.get('top1_agreement', '?'):.3f}  "
                f"regret={last.get('mean_regret_cp', '?'):.1f} cp"
            )
        all_results[seed] = {
            "split_info": split_info,
            "final_metrics": history[-1] if history else {},
            "output_dir": str(run_dir),
        }

    # Write aggregate summary.
    summary_path = args.output_dir / "real_baseline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"seeds": seeds, "results": {str(k): v for k, v in all_results.items()}}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nSummary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
