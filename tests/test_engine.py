"""Tests for position records and the Stockfish adapter."""

from __future__ import annotations

import chess

from chessevolve.data import PositionRecord
from chessevolve.engine import EngineSettings, StockfishAdapter


STOCKFISH = "tools/stockfish/stockfish/stockfish-linux-x86-64-universal"


def test_position_record_validation() -> None:
    rec = PositionRecord(
        position_id="p1",
        source_game_id="g1",
        fen=chess.STARTING_FEN,
        candidate_moves=["e2e4", "d2d4"],
        split="train",
    )
    assert rec.split == "train"
    assert len(rec.candidate_moves) == 2


def test_position_record_rejects_bad_split() -> None:
    try:
        PositionRecord(
            position_id="p1",
            source_game_id="g1",
            fen=chess.STARTING_FEN,
            candidate_moves=["e2e4"],
            split="validation",
        )
    except ValueError:
        return
    raise AssertionError("Expected ValueError for unknown split")


def test_legal_candidates_are_legal() -> None:
    board = chess.Board()
    with StockfishAdapter(STOCKFISH) as adapter:
        cands = adapter.legal_candidates(board)
    assert len(cands) == 20
    for move in cands:
        assert move in board.legal_moves


def test_candidate_evaluation_length_and_perspective() -> None:
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    # Black to move; Stockfish should see White slightly better, so from
    # Black's perspective the best move score should be negative-ish.
    with StockfishAdapter(STOCKFISH) as adapter:
        cands = [chess.Move.from_uci(m) for m in ["e7e5", "c7c5", "d7d5"]]
        scores = adapter.evaluate_candidates(board, cands, EngineSettings(depth=8))
    assert len(scores) == len(cands)
    assert all(isinstance(s, int) for s in scores)
    # All candidate scores are from Black (side to move) perspective.
    # The position is better for White, so scores should be <= 0 for reasonable moves.
    assert max(scores) <= 50


def test_illegal_candidate_raises() -> None:
    board = chess.Board()
    with StockfishAdapter(STOCKFISH) as adapter:
        try:
            adapter.evaluate_candidates(board, [chess.Move.from_uci("e2e5")], EngineSettings(depth=5))
        except ValueError:
            return
    raise AssertionError("Expected ValueError for illegal candidate")


def test_missing_executable_raises() -> None:
    try:
        StockfishAdapter("/no/such/stockfish")
    except FileNotFoundError:
        return
    raise AssertionError("Expected FileNotFoundError")


def test_reproducibility() -> None:
    board = chess.Board()
    cands = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    with StockfishAdapter(STOCKFISH) as adapter:
        s1 = adapter.evaluate_candidates(board, cands, EngineSettings(depth=6))
    with StockfishAdapter(STOCKFISH) as adapter:
        s2 = adapter.evaluate_candidates(board, cands, EngineSettings(depth=6))
    assert s1 == s2


def test_stockfish_reset_outside_context_manager() -> None:
    adapter = StockfishAdapter(STOCKFISH)
    try:
        adapter.reset()
    except RuntimeError:
        return
    raise AssertionError("Expected RuntimeError when reset() is called outside context manager")


def test_stockfish_reset_inside_context_manager() -> None:
    board = chess.Board()
    cands = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    with StockfishAdapter(STOCKFISH) as adapter:
        s1 = adapter.evaluate_candidates(board, cands, EngineSettings(depth=6))
        adapter.reset()
        s2 = adapter.evaluate_candidates(board, cands, EngineSettings(depth=6))
    assert s1 == s2


def test_stockfish_independent_sessions() -> None:
    board = chess.Board()
    cands = [chess.Move.from_uci("e2e4")]
    with StockfishAdapter(STOCKFISH) as a1:
        s1 = a1.evaluate_candidates(board, cands, EngineSettings(depth=5))
    with StockfishAdapter(STOCKFISH) as a2:
        s2 = a2.evaluate_candidates(board, cands, EngineSettings(depth=5))
    assert s1 == s2

