"""Tests for src/chessevolve/labeling.py — ALL OFFLINE (no real Stockfish)."""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import chess

# ---------------------------------------------------------------------------
# Make sure the project is importable when running from repo root.
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessevolve.data import PositionRecord
from chessevolve.engine import EngineSettings
from chessevolve.labeling import LabelingConfig, label_dataset, label_record, label_records
from chessevolve.real_train import RealTrainConfig, run_training_on_records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _make_board_and_moves(fen: str = STARTPOS_FEN, n: int = 3):
    """Return (board, first_n_legal_moves_as_uci_strings)."""
    board = chess.Board(fen)
    moves = [m.uci() for m in list(board.legal_moves)[:n]]
    return board, moves


def _make_record(
    fen: str = STARTPOS_FEN,
    n_candidates: int = 3,
    split: str = "train",
    position_id: str = "pos-001",
    source_game_id: str = "game-001",
) -> PositionRecord:
    _, moves = _make_board_and_moves(fen, n_candidates)
    return PositionRecord(
        position_id=position_id,
        source_game_id=source_game_id,
        fen=fen,
        candidate_moves=moves,
        split=split,
    )


def _make_adapter(scores, version_str: str = "Stockfish_test") -> MagicMock:
    """Return a mock StockfishAdapter with evaluate_candidates → scores."""
    adapter = MagicMock()
    adapter.evaluate_candidates.return_value = scores
    adapter.version.return_value = version_str
    adapter.reset.return_value = None
    return adapter


# ---------------------------------------------------------------------------
# Test 1: label_record — valid path
# ---------------------------------------------------------------------------

class TestLabelRecordValid(unittest.TestCase):
    def test_label_record_valid(self):
        """Mock adapter returns scores → PositionRecord updated correctly."""
        record = _make_record(n_candidates=3)
        scores = [50, 30, 70]
        adapter = _make_adapter(scores, version_str="Stockfish_test")
        settings = EngineSettings(depth=4)

        labeled, reason = label_record(record, adapter, settings)

        self.assertIsNone(reason)
        self.assertEqual(labeled.candidate_scores_cp, scores)
        self.assertEqual(labeled.engine_version, "Stockfish_test")
        self.assertIsNotNone(labeled.engine_settings)
        self.assertEqual(labeled.engine_settings["depth"], 4)
        # reference_best_move is the move at index 2 (score 70)
        self.assertEqual(labeled.reference_best_move, record.candidate_moves[2])


# ---------------------------------------------------------------------------
# Test 2: Best move selection
# ---------------------------------------------------------------------------

class TestLabelRecordBestMoveSelection(unittest.TestCase):
    def test_best_move_selection(self):
        """argmax picks correct reference_best_move."""
        record = _make_record(n_candidates=3)
        # Highest score is at index 1
        scores = [10, 99, 5]
        adapter = _make_adapter(scores)
        settings = EngineSettings(depth=4)

        labeled, reason = label_record(record, adapter, settings)

        self.assertIsNone(reason)
        self.assertEqual(labeled.reference_best_move, record.candidate_moves[1])


# ---------------------------------------------------------------------------
# Test 3: Tie handling (first-index wins)
# ---------------------------------------------------------------------------

class TestLabelRecordTieHandling(unittest.TestCase):
    def test_tie_first_index_wins(self):
        """Equal top scores → first-index wins (documented tie-break)."""
        record = _make_record(n_candidates=3)
        # All scores equal → index 0 should win.
        scores = [42, 42, 42]
        adapter = _make_adapter(scores)
        settings = EngineSettings(depth=4)

        labeled, reason = label_record(record, adapter, settings)

        self.assertIsNone(reason)
        self.assertEqual(labeled.reference_best_move, record.candidate_moves[0])

    def test_tie_first_among_top_wins(self):
        """Two moves share the max score → the one with the lower index wins."""
        record = _make_record(n_candidates=3)
        # Indices 0 and 2 tie; index 0 should win.
        scores = [100, 50, 100]
        adapter = _make_adapter(scores)
        settings = EngineSettings(depth=4)

        labeled, reason = label_record(record, adapter, settings)

        self.assertIsNone(reason)
        self.assertEqual(labeled.reference_best_move, record.candidate_moves[0])


# ---------------------------------------------------------------------------
# Test 4: Illegal candidate move → record skipped
# ---------------------------------------------------------------------------

class TestLabelRecordIllegalCandidate(unittest.TestCase):
    def test_illegal_candidate_skipped(self):
        """Illegal move in candidate_moves → record skipped with reason."""
        # Standard starting position; "e1e8" is illegal for white king there.
        board = chess.Board(STARTPOS_FEN)
        legal_uci = [m.uci() for m in list(board.legal_moves)[:2]]
        # Add a clearly illegal move.
        illegal_uci = "e1e8"
        # Make sure it is actually illegal.
        self.assertNotIn(chess.Move.from_uci(illegal_uci), board.legal_moves)

        record = PositionRecord(
            position_id="pos-illegal",
            source_game_id="game-001",
            fen=STARTPOS_FEN,
            candidate_moves=legal_uci + [illegal_uci],
            split="train",
        )

        adapter = _make_adapter([10, 20, 30])
        settings = EngineSettings(depth=4)

        labeled, reason = label_record(record, adapter, settings)

        self.assertIsNotNone(reason)
        self.assertIn("illegal", reason.lower())
        # adapter.evaluate_candidates should NOT have been called.
        adapter.evaluate_candidates.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5: Missing score (None in list) → record skipped
# ---------------------------------------------------------------------------

class TestLabelRecordMissingScore(unittest.TestCase):
    def test_missing_score_skipped(self):
        """Engine returns list with None → record skipped with reason."""
        record = _make_record(n_candidates=3)
        adapter = _make_adapter([50, None, 30])
        settings = EngineSettings(depth=4)

        labeled, reason = label_record(record, adapter, settings)

        self.assertIsNotNone(reason)
        self.assertIn("missing score", reason.lower())


# ---------------------------------------------------------------------------
# Test 6: Engine failure (RuntimeError) → skipped with reason
# ---------------------------------------------------------------------------

class TestLabelRecordEngineFailure(unittest.TestCase):
    def test_engine_failure_skipped(self):
        """Adapter raises RuntimeError → record skipped with reason."""
        record = _make_record(n_candidates=3)
        adapter = MagicMock()
        adapter.evaluate_candidates.side_effect = RuntimeError("Stockfish crashed")
        adapter.version.return_value = "Stockfish_test"

        settings = EngineSettings(depth=4)

        labeled, reason = label_record(record, adapter, settings)

        self.assertIsNotNone(reason)
        self.assertIn("engine error", reason.lower())
        # The returned record should be the original (unchanged).
        self.assertIsNone(labeled.candidate_scores_cp)


# ---------------------------------------------------------------------------
# Test 7: Metadata preservation
# ---------------------------------------------------------------------------

class TestLabelRecordPreservesMetadata(unittest.TestCase):
    def test_metadata_preserved(self):
        """position_id, source_game_id, split, fen, candidate_moves are unchanged."""
        record = _make_record(
            n_candidates=3,
            split="dev",
            position_id="pos-xyz",
            source_game_id="game-abc",
        )
        scores = [5, 10, 3]
        adapter = _make_adapter(scores)
        settings = EngineSettings(depth=4)

        labeled, reason = label_record(record, adapter, settings)

        self.assertIsNone(reason)
        self.assertEqual(labeled.position_id, "pos-xyz")
        self.assertEqual(labeled.source_game_id, "game-abc")
        self.assertEqual(labeled.split, "dev")
        self.assertEqual(labeled.fen, record.fen)
        self.assertEqual(labeled.candidate_moves, record.candidate_moves)


# ---------------------------------------------------------------------------
# Test 8: label_records respects max_positions
# ---------------------------------------------------------------------------

class TestLabelRecordsMaxPositions(unittest.TestCase):
    def test_max_positions_limits_output(self):
        """max_positions limits total records processed."""
        records = [_make_record(position_id=f"pos-{i}") for i in range(5)]
        adapter = _make_adapter([10, 20, 30])
        settings = EngineSettings(depth=4)

        labeled, skipped = label_records(records, adapter, settings, max_positions=2)

        # Only the first 2 should have been processed.
        self.assertEqual(len(labeled), 2)
        self.assertEqual(len(skipped), 0)
        self.assertEqual(adapter.evaluate_candidates.call_count, 2)


# ---------------------------------------------------------------------------
# Test 9: run_training_on_records raises ValueError on dev records in train
# ---------------------------------------------------------------------------

class TestNoDevInTrainRecords(unittest.TestCase):
    def test_raises_on_dev_record_in_train(self):
        """run_training_on_records raises ValueError if a dev record is in train_records."""
        # Create a minimal labeled record with split="dev".
        board = chess.Board(STARTPOS_FEN)
        moves = [m.uci() for m in list(board.legal_moves)[:3]]
        dev_record = PositionRecord(
            position_id="dev-pos",
            source_game_id="game-x",
            fen=STARTPOS_FEN,
            candidate_moves=moves,
            split="dev",
            candidate_scores_cp=[10, 20, 30],
            reference_best_move=moves[1],
            engine_version="Stockfish_test",
            engine_settings={"depth": 4, "time_seconds": None},
        )

        config = RealTrainConfig(seed=0, epochs=1)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                run_training_on_records(
                    train_records=[dev_record],  # dev record passed as train!
                    dev_records=[],
                    config=config,
                    output_dir=Path(tmp) / "out",
                )
            self.assertIn("train", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# Test 10: label_dataset handles empty JSONL without crashing
# ---------------------------------------------------------------------------

class TestLabelDatasetEmptySplit(unittest.TestCase):
    def test_empty_split_no_crash(self):
        """label_dataset handles empty JSONL files without crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            output_dir = Path(tmpdir) / "labeled"
            dataset_dir.mkdir()

            # Create empty JSONL files for all splits.
            for split in ["train", "dev", "test"]:
                (dataset_dir / f"{split}_records.jsonl").write_text("", encoding="utf-8")

            # Write a minimal manifest.
            (dataset_dir / "manifest.json").write_text(
                json.dumps({"manifest_schema_version": "1.0"}), encoding="utf-8"
            )

            stockfish_exe = "tools/stockfish/stockfish/stockfish-linux-x86-64-universal"

            # We need to patch StockfishAdapter because there's no real engine in CI.
            mock_adapter_instance = MagicMock()
            mock_adapter_instance.__enter__ = MagicMock(return_value=mock_adapter_instance)
            mock_adapter_instance.__exit__ = MagicMock(return_value=False)
            mock_adapter_instance.version.return_value = "Stockfish_test"
            mock_adapter_instance.reset.return_value = None
            mock_adapter_instance.evaluate_candidates.return_value = []

            with patch(
                "chessevolve.labeling.StockfishAdapter",
                return_value=mock_adapter_instance,
            ):
                config = LabelingConfig(
                    engine_depth=4,
                    stockfish_path=stockfish_exe,
                )
                # Should not raise.
                summary = label_dataset(dataset_dir, output_dir, config)

            # Verify outputs exist.
            self.assertTrue((output_dir / "train_records.jsonl").exists())
            self.assertTrue((output_dir / "dev_records.jsonl").exists())
            self.assertTrue((output_dir / "test_records.jsonl").exists())
            self.assertTrue((output_dir / "manifest.json").exists())

            # All splits should have 0 records.
            for split, info in summary["splits"].items():
                self.assertEqual(info["input_records"], 0)
                self.assertEqual(info["labeled"], 0)
                self.assertEqual(info["skipped"], 0)


if __name__ == "__main__":
    unittest.main()
