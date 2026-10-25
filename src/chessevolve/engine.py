"""Stockfish adapter for bounded, perspective-normalized chess analysis."""

from __future__ import annotations

import chess
import chess.engine
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


DEFAULT_STOCKFISH_PATH = Path("tools/stockfish/stockfish/stockfish-linux-x86-64-universal")
MATE_SCORE = 100_000


@dataclass(frozen=True)
class EngineSettings:
    depth: Optional[int] = 10
    time_seconds: Optional[float] = None

    def to_limit(self) -> chess.engine.Limit:
        if self.depth is None and self.time_seconds is None:
            raise ValueError("Either depth or time_seconds must be set")
        return chess.engine.Limit(depth=self.depth, time=self.time_seconds)


class StockfishAdapter:
    """Thin wrapper around a single Stockfish UCI process.

    One process is reused for all candidate moves in a position, which avoids
    the cost of launching Stockfish repeatedly.
    """

    def __init__(self, executable: str | Path = DEFAULT_STOCKFISH_PATH) -> None:
        self.executable = Path(executable)
        if not self.executable.exists():
            raise FileNotFoundError(f"Stockfish executable not found: {self.executable}")
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def __enter__(self) -> "StockfishAdapter":
        self._engine = chess.engine.SimpleEngine.popen_uci(str(self.executable))
        return self

    def __exit__(self, *exc: object) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    @property
    def engine(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            raise RuntimeError("StockfishAdapter must be used as a context manager")
        return self._engine

    def version(self) -> str:
        return self.engine.id.get("name", "unknown")

    def legal_candidates(self, board: chess.Board) -> List[chess.Move]:
        """Return all legal moves for the position."""
        return list(board.legal_moves)

    def evaluate_candidates(
        self,
        board: chess.Board,
        candidates: List[chess.Move],
        settings: EngineSettings,
    ) -> List[Optional[int]]:
        """Evaluate each candidate and return centipawn scores.

        Scores are normalized to the side-to-move perspective: a positive
        value means the player to move has an advantage. Mate scores are
        converted to large signed centipawn values (+/- MATE_SCORE).
        """
        if not candidates:
            raise ValueError("candidates must not be empty")
        for move in candidates:
            if move not in board.legal_moves:
                raise ValueError(f"Illegal candidate move: {move.uci()}")
        limit = settings.to_limit()
        scores: List[Optional[int]] = []
        for move in candidates:
            try:
                info = self.engine.analyse(board, limit, root_moves=[move])
            except chess.engine.EngineError as exc:
                raise RuntimeError(f"Stockfish failed for move {move.uci()}: {exc}") from exc
            if "score" not in info:
                raise RuntimeError(f"Stockfish returned no score for move {move.uci()}")
            pov_score = info["score"].pov(board.turn)
            if hasattr(pov_score, "relative"):
                numeric = pov_score.relative.score(mate_score=MATE_SCORE)
            else:
                numeric = pov_score.score(mate_score=MATE_SCORE)
            scores.append(numeric)
        return scores

    def best_move(self, board: chess.Board, settings: EngineSettings) -> Optional[chess.Move]:
        limit = settings.to_limit()
        try:
            result = self.engine.play(board, limit)
        except chess.engine.EngineError as exc:
            raise RuntimeError(f"Stockfish play failed: {exc}") from exc
        return result.move
