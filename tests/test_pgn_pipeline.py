"""Tests for PGN parsing and dataset preparation pipeline."""

from __future__ import annotations

import io
import json
from pathlib import Path

import chess
import pytest

from chessevolve.pgn import (
    ExtractionStats,
    PGNExtractionConfig,
    extract_positions_from_game,
    extract_positions_from_pgn_stream,
    prepare_dataset_from_pgn,
)


SAMPLE_PGN_GAME_1 = """[Event "Test Game 1"]
[Site "ChessEvolve"]
[Date "2026.01.01"]
[Round "1"]
[White "PlayerA"]
[Black "PlayerB"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0
"""

SAMPLE_PGN_GAME_2 = """[Event "Test Game 2"]
[Site "ChessEvolve"]
[Date "2026.01.02"]
[Round "2"]
[White "PlayerC"]
[Black "PlayerD"]
[Result "0-1"]

1. d4 d5 2. c4 c6 3. Nf3 Nf6 4. Nc3 e6 5. e3 Nbd7 6. Bd3 dxc4 7. Bxc4 b5 8. Bd3 Bb7 9. O-O a6 10. e4 c5 0-1
"""

SAMPLE_PGN_GAME_3 = """[Event "Test Game 3"]
[Site "ChessEvolve"]
[Date "2026.01.03"]
[Round "3"]
[White "PlayerE"]
[Black "PlayerF"]
[Result "1/2-1/2"]

1. c4 e5 2. Nc3 Nf6 3. g3 d5 4. cxd5 Nxd5 5. Bg2 Nb6 6. Nf3 Nc6 7. O-O Be7 8. a3 O-O 9. b4 Be6 10. d3 f6 1/2-1/2
"""

SAMPLE_PGN_ILLEGAL = """[Event "Illegal Game"]
[Site "ChessEvolve"]
[Date "2026.01.04"]
[Round "4"]
[White "BadPlayer1"]
[Black "BadPlayer2"]
[Result "*"]

1. e4 e5 2. Nf3 Ke7 3. Ke8 *
"""


def test_extract_positions_valid_game():
    game = chess.pgn.read_game(io.StringIO(SAMPLE_PGN_GAME_1))
    assert game is not None

    config = PGNExtractionConfig(min_ply=4, sample_frequency=2, max_plies_per_game=20)
    records, rejection = extract_positions_from_game(game, "game-0001", config)

    assert rejection is None
    assert len(records) > 0

    for rec in records:
        assert rec.source_game_id == "game-0001"
        assert rec.position_id.startswith("game-0001-ply")
        # Ensure candidate moves are legal in this FEN
        board = chess.Board(rec.fen)
        legal_ucis = {m.uci() for m in board.legal_moves}
        assert set(rec.candidate_moves) == legal_ucis
        # Crucial: move played in PGN is NOT assigned as reference label
        assert rec.candidate_scores_cp is None
        assert rec.reference_best_move is None
        assert rec.engine_version is None


def test_extract_positions_rejects_illegal_moves():
    game = chess.pgn.read_game(io.StringIO(SAMPLE_PGN_ILLEGAL))
    assert game is not None

    config = PGNExtractionConfig(min_ply=2, sample_frequency=2)
    records, rejection = extract_positions_from_game(game, "game-illegal", config)

    assert len(records) == 0
    assert rejection is not None
    assert "illegal" in rejection.lower()


def test_smoke_configuration():
    config = PGNExtractionConfig(smoke=True)
    assert config.max_games == 5
    assert config.min_ply == 2
    assert config.max_plies_per_game == 20


def test_extract_positions_from_pgn_stream():
    combined_pgn = "\n\n".join([SAMPLE_PGN_GAME_1, SAMPLE_PGN_ILLEGAL, SAMPLE_PGN_GAME_2])
    stream = io.StringIO(combined_pgn)

    config = PGNExtractionConfig(min_ply=4, sample_frequency=2)
    records, stats = extract_positions_from_pgn_stream(stream, config)

    assert stats.total_games_read == 3
    assert stats.valid_games == 2
    assert len(stats.rejected_games) == 1
    assert stats.rejected_games[0]["game_id"] == "game-0001"  # 2nd game (0-indexed 1)
    assert stats.total_positions_extracted == len(records)


def test_prepare_dataset_from_pgn_full_pipeline(tmp_path: Path):
    combined_pgn = "\n\n".join([
        SAMPLE_PGN_GAME_1,
        SAMPLE_PGN_GAME_2,
        SAMPLE_PGN_GAME_3,
    ])
    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(combined_pgn, encoding="utf-8")

    out_dir = tmp_path / "dataset"
    summary = prepare_dataset_from_pgn(
        pgn_path=pgn_path,
        output_dir=out_dir,
        proportions=(0.6, 0.2, 0.2),
        seed=42,
        config=PGNExtractionConfig(min_ply=2, sample_frequency=2),
    )

    # Verify expected files exist
    assert (out_dir / "train_records.jsonl").is_file()
    assert (out_dir / "dev_records.jsonl").is_file()
    assert (out_dir / "test_records.jsonl").is_file()
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "summary.json").is_file()

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "source_game_ids" in manifest
    declared = manifest["source_game_ids"]

    # Verify zero game overlap between splits
    train_games = set(declared.get("train", []))
    dev_games = set(declared.get("dev", []))
    test_games = set(declared.get("test", []))

    assert len(train_games & dev_games) == 0
    assert len(train_games & test_games) == 0
    assert len(dev_games & test_games) == 0

    # Verify counts match summary
    assert summary["counts"]["total_records"] > 0
    assert summary["counts"]["total_games"] == 3


def test_empty_pgn_raises_error(tmp_path: Path):
    empty_pgn = tmp_path / "empty.pgn"
    empty_pgn.write_text("", encoding="utf-8")

    out_dir = tmp_path / "dataset"
    with pytest.raises(ValueError, match="No valid positions could be extracted"):
        prepare_dataset_from_pgn(
            pgn_path=empty_pgn,
            output_dir=out_dir,
        )
