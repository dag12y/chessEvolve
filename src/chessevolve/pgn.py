"""PGN parsing and reproducible dataset extraction pipeline for ChessEvolve.

Extracts position records and legal candidate moves from real PGN game files,
enforcing source-game tracking, strict validation, deterministic splitting,
and no premature label assignment.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import chess
import chess.pgn

from .data import PositionRecord
from .splitting import split_by_source_game, write_split_manifest


@dataclass
class PGNExtractionConfig:
    """Configuration for extracting positions from PGN games."""

    min_ply: int = 4
    max_plies_per_game: int = 80
    sample_frequency: int = 2
    max_games: Optional[int] = None
    max_positions: Optional[int] = None
    smoke: bool = False

    def __post_init__(self) -> None:
        if self.min_ply < 1:
            raise ValueError(f"min_ply must be >= 1, got {self.min_ply}")
        if self.sample_frequency < 1:
            raise ValueError(f"sample_frequency must be >= 1, got {self.sample_frequency}")
        if self.max_plies_per_game < self.min_ply:
            raise ValueError("max_plies_per_game cannot be less than min_ply")

        # Apply smoke test limits if smoke is requested
        if self.smoke:
            if self.max_games is None:
                self.max_games = 5
            if self.max_plies_per_game == 80:
                self.max_plies_per_game = 20
            if self.min_ply == 4:
                self.min_ply = 2

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtractionStats:
    """Summary statistics for a PGN extraction run."""

    total_games_read: int = 0
    valid_games: int = 0
    rejected_games: List[dict] = field(default_factory=list)
    total_positions_extracted: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extract_positions_from_game(
    game: chess.pgn.Game,
    game_id: str,
    config: PGNExtractionConfig,
) -> Tuple[List[PositionRecord], Optional[str]]:
    """Extract valid PositionRecords from a single chess game.

    Checks move legality at every step. If an illegal move or parse corruption
    is detected, rejects the game and returns an explanatory reason.
    """
    if game.errors:
        return [], f"PGN syntax/parser errors: {game.errors}"

    board = game.board()
    records: List[PositionRecord] = []

    for ply, move in enumerate(game.mainline_moves(), start=1):
        # Validate move legality before pushing
        if move not in board.legal_moves:
            return [], f"Illegal move {move.uci()!r} played at ply {ply}"

        # Sample position if eligible
        if ply >= config.min_ply and (ply % config.sample_frequency == 0):
            if not board.is_game_over() and board.legal_moves:
                candidates = [m.uci() for m in board.legal_moves]
                records.append(
                    PositionRecord(
                        position_id=f"{game_id}-ply{ply}",
                        source_game_id=game_id,
                        fen=board.fen(),
                        candidate_moves=candidates,
                        split="train",  # placeholder, reassigned during splitting
                        engine_version=None,
                        engine_settings=None,
                        reference_best_move=None,
                        candidate_scores_cp=None,
                    )
                )

        board.push(move)

        if config.max_plies_per_game and ply >= config.max_plies_per_game:
            break

    if not records:
        return [], f"Game had no eligible positions between ply {config.min_ply} and {config.max_plies_per_game}"

    return records, None


def extract_positions_from_pgn_stream(
    stream: io.TextIOBase,
    config: Optional[PGNExtractionConfig] = None,
) -> Tuple[List[PositionRecord], ExtractionStats]:
    """Parse games from a text PGN stream and extract valid PositionRecords."""
    if config is None:
        config = PGNExtractionConfig()

    stats = ExtractionStats()
    all_records: List[PositionRecord] = []
    game_idx = 0

    while True:
        try:
            game = chess.pgn.read_game(stream)
        except Exception as exc:
            stats.rejected_games.append({
                "game_id": f"game-{game_idx:04d}",
                "reason": f"Exception reading PGN game: {exc}",
            })
            continue

        if game is None:
            break

        stats.total_games_read += 1
        game_id = f"game-{game_idx:04d}"
        game_idx += 1

        records, rejection = extract_positions_from_game(game, game_id, config)
        if rejection is not None:
            stats.rejected_games.append({
                "game_id": game_id,
                "reason": rejection,
            })
            continue

        stats.valid_games += 1
        all_records.extend(records)

        if config.max_games and stats.valid_games >= config.max_games:
            break
        if config.max_positions and len(all_records) >= config.max_positions:
            all_records = all_records[: config.max_positions]
            break

    stats.total_positions_extracted = len(all_records)
    return all_records, stats


def extract_positions_from_pgn_file(
    pgn_path: Path,
    config: Optional[PGNExtractionConfig] = None,
) -> Tuple[List[PositionRecord], ExtractionStats]:
    """Extract PositionRecords from a PGN file on disk."""
    p = Path(pgn_path)
    if not p.is_file():
        raise FileNotFoundError(f"PGN file not found: {pgn_path}")

    with p.open(encoding="utf-8", errors="replace") as f:
        return extract_positions_from_pgn_stream(f, config=config)


def save_records_jsonl(path: Path, records: Iterable[PositionRecord]) -> None:
    """Save PositionRecords as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")


def prepare_dataset_from_pgn(
    pgn_path: Path,
    output_dir: Path,
    proportions: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    config: Optional[PGNExtractionConfig] = None,
) -> dict:
    """End-to-end dataset preparation pipeline.

    1. Parses PGN and validates moves/positions.
    2. Extracts candidate moves without assuming played moves are labels.
    3. Splits strictly by source game using reproducible hashing.
    4. Writes train/dev/test JSONL records, manifest.json, and summary.json.
    """
    p_pgn = Path(pgn_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records, stats = extract_positions_from_pgn_file(p_pgn, config=config)
    if not records:
        raise ValueError(
            f"No valid positions could be extracted from {pgn_path}. "
            f"Games read: {stats.total_games_read}, rejected: {len(stats.rejected_games)}"
        )

    # Split records strictly by source game
    props_dict = {"train": proportions[0], "dev": proportions[1], "test": proportions[2]}
    split_result = split_by_source_game(records, proportions=props_dict, seed=seed)

    train_records = split_result["train"]
    dev_records = split_result["dev"]
    test_records = split_result["test"]

    # Verify game-level split isolation
    train_games = {r.source_game_id for r in train_records}
    dev_games = {r.source_game_id for r in dev_records}
    test_games = {r.source_game_id for r in test_records}

    if train_games & dev_games or train_games & test_games or dev_games & test_games:
        raise RuntimeError("Source game overlap detected across splits!")

    # Write split JSONL files
    train_path = out / "train_records.jsonl"
    dev_path = out / "dev_records.jsonl"
    test_path = out / "test_records.jsonl"
    manifest_path = out / "manifest.json"
    summary_path = out / "summary.json"

    save_records_jsonl(train_path, train_records)
    save_records_jsonl(dev_path, dev_records)
    save_records_jsonl(test_path, test_records)

    # Write frozen manifest
    write_split_manifest(
        manifest_path,
        split_result,
        seed=seed,
        proportions=props_dict,
    )

    # Compute hashes
    pgn_sha256 = compute_file_sha256(p_pgn)
    train_sha256 = compute_file_sha256(train_path)
    dev_sha256 = compute_file_sha256(dev_path)
    test_sha256 = compute_file_sha256(test_path)
    manifest_sha256 = compute_file_sha256(manifest_path)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_pgn_path": str(p_pgn),
        "input_pgn_sha256": pgn_sha256,
        "seed": seed,
        "proportions": props_dict,
        "config": (config or PGNExtractionConfig()).to_dict(),
        "stats": stats.to_dict(),
        "counts": {
            "train": {"records": len(train_records), "source_games": len(train_games)},
            "dev": {"records": len(dev_records), "source_games": len(dev_games)},
            "test": {"records": len(test_records), "source_games": len(test_games)},
            "total_records": len(records),
            "total_games": len(train_games | dev_games | test_games),
        },
        "file_hashes": {
            "train_records_sha256": train_sha256,
            "dev_records_sha256": dev_sha256,
            "test_records_sha256": test_sha256,
            "manifest_sha256": manifest_sha256,
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
