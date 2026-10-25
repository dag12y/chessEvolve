"""Data structures for ChessEvolve positions and labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class PositionRecord:
    """A single chess position with candidate moves and metadata.

    The record is intentionally small and serializable. It does not contain
    model tensors; encoding happens later in Milestone 2.
    """

    position_id: str
    source_game_id: str
    fen: str
    candidate_moves: List[str]
    split: str  # "train", "dev", or "test"
    engine_version: Optional[str] = None
    engine_settings: Optional[dict] = None
    reference_best_move: Optional[str] = None
    candidate_scores_cp: Optional[List[Optional[int]]] = None

    def __post_init__(self) -> None:
        if self.split not in {"train", "dev", "test"}:
            raise ValueError(f"Unknown split: {self.split!r}")
        if not self.candidate_moves:
            raise ValueError("candidate_moves must not be empty")
        if self.candidate_scores_cp is not None and len(self.candidate_scores_cp) != len(
            self.candidate_moves
        ):
            raise ValueError("candidate_scores_cp length must match candidate_moves")

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)
