"""Dataset construction helpers for Stockfish-supervised baseline training."""

from __future__ import annotations

import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import chess

from .data import PositionRecord
from .encoding import encode_position, encode_move
from .engine import EngineSettings, StockfishAdapter
from .splitting import split_by_source_game


@dataclass
class TrainingExample:
    """One training example: a position with encoded candidates and a target."""

    position_features: "numpy.ndarray"  # (POSITION_DIM,)
    move_features: "numpy.ndarray"      # (N, MOVE_DIM)
    candidate_scores: List[Optional[int]]  # Stockfish cp scores, side-to-move perspective
    target_index: int  # index of the Stockfish-best candidate
    position_id: str
    fen: str
    record: Optional[PositionRecord] = None


def generate_random_positions(
    num_games: int,
    max_plies: int = 20,
    seed: int = 0,
) -> List[PositionRecord]:
    """Generate positions by playing random legal games.

    Each game contributes one position record per ply (up to ``max_plies``).
    Source-game IDs are unique per game so source-game splitting works.
    """
    rng = random.Random(seed)
    records: List[PositionRecord] = []
    for game_idx in range(num_games):
        board = chess.Board()
        source_game_id = f"random-game-{game_idx}"
        ply = 0
        while ply < max_plies and not board.is_game_over():
            moves = list(board.legal_moves)
            if not moves:
                break
            # Save a position record every few plies to keep count manageable.
            if ply % 2 == 0 and ply > 0:
                candidates = [m.uci() for m in moves]
                records.append(
                    PositionRecord(
                        position_id=f"{source_game_id}-ply{ply}",
                        source_game_id=source_game_id,
                        fen=board.fen(),
                        candidate_moves=candidates,
                        split="train",  # placeholder; reassigned by splitting
                    )
                )
            move = rng.choice(moves)
            board.push(move)
            ply += 1
        if not records or records[-1].source_game_id != source_game_id:
            pass
    return records


def build_examples(
    records: List[PositionRecord],
    adapter: StockfishAdapter,
    settings: EngineSettings,
) -> Tuple[List[TrainingExample], List[str]]:
    """Evaluate records and build aligned training examples.

    Returns (examples, skipped_reasons).  A record is skipped if any candidate
    lacks a score or the engine fails.  Labels are aligned with candidate order.
    """
    examples: List[TrainingExample] = []
    skipped: List[str] = []
    for rec in records:
        board = chess.Board(rec.fen)
        candidates = [chess.Move.from_uci(u) for u in rec.candidate_moves]
        # Validate legality; skip illegal records.
        if any(m not in board.legal_moves for m in candidates):
            skipped.append(f"{rec.position_id}: illegal candidate")
            continue
        try:
            scores = adapter.evaluate_candidates(board, candidates, settings)
        except Exception as exc:  # engine failure
            skipped.append(f"{rec.position_id}: engine error {exc}")
            continue
        if any(s is None for s in scores):
            skipped.append(f"{rec.position_id}: missing score")
            continue
        best_index = max(range(len(scores)), key=lambda i: scores[i])  # type: ignore[arg-type]
        pos_feat = encode_position(board)
        move_feat = np.stack([encode_move(board, m) for m in candidates], axis=0)
        try:
            engine_ver = adapter.version()
        except Exception:
            engine_ver = "unknown"
        evaluated_rec = PositionRecord(
            position_id=rec.position_id,
            source_game_id=rec.source_game_id,
            fen=rec.fen,
            candidate_moves=rec.candidate_moves,
            split=rec.split,
            engine_version=engine_ver,
            engine_settings={"depth": settings.depth, "time_seconds": settings.time_seconds},
            reference_best_move=candidates[best_index].uci(),
            candidate_scores_cp=[int(s) for s in scores],  # type: ignore[arg-type]
        )
        examples.append(
            TrainingExample(
                position_features=pos_feat,
                move_features=move_feat,
                candidate_scores=[int(s) for s in scores],  # type: ignore[arg-type]
                target_index=best_index,
                position_id=rec.position_id,
                fen=rec.fen,
                record=evaluated_rec,
            )
        )
    return examples, skipped


def split_records(
    records: List[PositionRecord],
    proportions: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict:
    """Split records by source game using the existing splitter."""
    from .splitting import SplitProportions
    props = {"train": proportions[0], "dev": proportions[1], "test": proportions[2]}
    result = split_by_source_game(records, proportions=props, seed=seed)
    return result
