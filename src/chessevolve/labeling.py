"""Stockfish labeling pipeline for PositionRecord datasets.

This module provides functions to enrich unlabeled PositionRecords with
Stockfish evaluations, writing labeled JSONL files ready for supervised
training.

Score semantics
---------------
- Scores are expressed from the **side-to-move** perspective: a positive
  value means the player whose turn it is has an advantage.
- Mate-in-N is stored as ±MATE_SCORE (100 000) via
  ``PovScore.score(mate_score=MATE_SCORE)``.
- **Tie-break rule**: when multiple moves share the maximum score the
  *first-occurring* index wins (argmax with first-wins tie-break).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import chess

from .data import PositionRecord
from .engine import EngineSettings, StockfishAdapter


@dataclass
class LabelingConfig:
    """Configuration for the Stockfish labeling step."""

    engine_depth: int = 4
    stockfish_path: str = "tools/stockfish/stockfish/stockfish-linux-x86-64-universal"
    max_positions: Optional[int] = None


def label_record(
    record: PositionRecord,
    adapter: StockfishAdapter,
    settings: EngineSettings,
) -> Tuple[PositionRecord, Optional[str]]:
    """Label a single PositionRecord with Stockfish scores.

    The adapter must already be open (used inside a ``with`` block).

    Steps
    -----
    1. Reconstruct the board from ``record.fen``.
    2. Validate that **all** ``record.candidate_moves`` are legal for that
       board; skip the record (with a reason string) if any is illegal.
    3. Call ``adapter.evaluate_candidates`` to get centipawn scores.
    4. If any returned score is ``None``, skip the record.
    5. Select ``reference_best_move`` as the candidate at ``argmax(scores)``
       (first-wins tie-break).
    6. Return an updated PositionRecord (using ``dataclasses.replace``) with
       ``candidate_scores_cp``, ``reference_best_move``, ``engine_version``,
       and ``engine_settings`` populated.

    On ``RuntimeError`` from the adapter the record is returned unchanged with
    a skip reason.

    Parameters
    ----------
    record:
        Unlabeled position record.
    adapter:
        An already-open StockfishAdapter context manager instance.
    settings:
        Engine search settings (depth / time).

    Returns
    -------
    (labeled_record, skip_reason)
        ``skip_reason`` is ``None`` on success, otherwise a human-readable
        string describing why the record was skipped.
    """
    board = chess.Board(record.fen)

    # Validate all candidates are legal.
    chess_moves: List[chess.Move] = []
    for uci in record.candidate_moves:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return record, f"invalid UCI string: {uci!r}"
        if move not in board.legal_moves:
            return record, f"illegal candidate move {uci!r} in position {record.position_id}"
        chess_moves.append(move)

    # Evaluate candidates.
    try:
        scores = adapter.evaluate_candidates(board, chess_moves, settings)
    except RuntimeError as exc:
        return record, f"engine error for {record.position_id}: {exc}"

    # Ensure all scores are present.
    if any(s is None for s in scores):
        return record, f"missing score for {record.position_id}"

    # Best move: first-wins tie-break argmax.
    best_index = max(range(len(scores)), key=lambda i: scores[i])  # type: ignore[arg-type]
    reference_best_move = record.candidate_moves[best_index]

    # Engine provenance.
    try:
        engine_version = adapter.version()
    except Exception:
        engine_version = "unknown"

    engine_settings_dict = {
        "depth": settings.depth,
        "time_seconds": settings.time_seconds,
    }

    labeled = dataclasses.replace(
        record,
        candidate_scores_cp=[int(s) for s in scores],  # type: ignore[arg-type]
        reference_best_move=reference_best_move,
        engine_version=engine_version,
        engine_settings=engine_settings_dict,
    )
    return labeled, None


def label_records(
    records: List[PositionRecord],
    adapter: StockfishAdapter,
    settings: EngineSettings,
    max_positions: Optional[int] = None,
) -> Tuple[List[PositionRecord], List[dict]]:
    """Label a list of PositionRecords.

    Parameters
    ----------
    records:
        Input records (any split).
    adapter:
        An already-open StockfishAdapter.
    settings:
        Engine search settings.
    max_positions:
        If given, only the first ``max_positions`` records are processed.

    Returns
    -------
    (labeled_records, skipped_with_reasons)
        ``skipped_with_reasons`` is a list of dicts with keys
        ``position_id`` and ``reason``.
    """
    labeled: List[PositionRecord] = []
    skipped: List[dict] = []

    input_records = records if max_positions is None else records[:max_positions]

    for record in input_records:
        result, reason = label_record(record, adapter, settings)
        if reason is not None:
            skipped.append({"position_id": record.position_id, "reason": reason})
        else:
            labeled.append(result)

    return labeled, skipped


def label_dataset(
    dataset_dir: Path,
    output_dir: Path,
    config: LabelingConfig,
) -> dict:
    """Label an entire dataset directory (train / dev / test splits).

    Reads ``{train,dev,test}_records.jsonl`` from ``dataset_dir``, labels
    each split independently using a single Stockfish process, and writes
    labeled JSONL files to ``output_dir``.  The original ``manifest.json``
    is copied and updated with labeling provenance.

    Empty JSONL files (zero records) are handled gracefully.

    Parameters
    ----------
    dataset_dir:
        Directory containing the unlabeled JSONL files and ``manifest.json``.
    output_dir:
        Destination directory for labeled files.
    config:
        Labeling configuration (depth, Stockfish path, max_positions).

    Returns
    -------
    Summary dict with labeled/skipped counts per split plus engine info.
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = EngineSettings(depth=config.engine_depth)

    splits = ["train", "dev", "test"]
    summary: dict = {
        "engine_version": None,
        "engine_settings": {"depth": config.engine_depth, "time_seconds": None},
        "splits": {},
    }

    with StockfishAdapter(Path(config.stockfish_path)) as adapter:
        engine_version = adapter.version()
        summary["engine_version"] = engine_version

        for split in splits:
            src_path = dataset_dir / f"{split}_records.jsonl"
            dst_path = output_dir / f"{split}_records.jsonl"

            # Load records (handle missing or empty files gracefully).
            records: List[PositionRecord] = []
            if src_path.exists():
                with src_path.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            records.append(PositionRecord(**json.loads(line)))

            # Determine how many to process for this split.
            # max_positions is applied globally across all splits combined if
            # you prefer; here we apply it per-split as a ceiling for safety.
            # The CLI allows passing it as a quick "limit total" flag.
            max_pos = config.max_positions

            # Reset engine between splits to avoid stale transposition tables.
            adapter.reset()

            labeled, skipped = label_records(records, adapter, settings, max_positions=max_pos)

            # Write labeled records.
            with dst_path.open("w", encoding="utf-8") as fh:
                for rec in labeled:
                    fh.write(json.dumps(dataclasses.asdict(rec)) + "\n")

            summary["splits"][split] = {
                "input_records": len(records),
                "labeled": len(labeled),
                "skipped": len(skipped),
                "skipped_reasons": skipped,
            }

    # Copy and update manifest.
    manifest_src = dataset_dir / "manifest.json"
    manifest_dst = output_dir / "manifest.json"
    if manifest_src.exists():
        manifest = json.loads(manifest_src.read_text(encoding="utf-8"))
    else:
        manifest = {}

    manifest["labeling_provenance"] = {
        "engine_version": summary["engine_version"],
        "engine_settings": summary["engine_settings"],
        "labeled_at": datetime.now(tz=timezone.utc).isoformat(),
        "counts": {
            split: {
                "input": summary["splits"][split]["input_records"],
                "labeled": summary["splits"][split]["labeled"],
                "skipped": summary["splits"][split]["skipped"],
            }
            for split in splits
            if split in summary["splits"]
        },
    }
    manifest_dst.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Copy summary.json if present.
    summary_src = dataset_dir / "summary.json"
    if summary_src.exists():
        shutil.copy2(summary_src, output_dir / "summary.json")

    # Write labeling summary.
    (output_dir / "labeling_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    return summary
