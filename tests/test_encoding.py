"""Tests for chessevolve.encoding."""
from __future__ import annotations

import chess
import numpy as np
import pytest

from chessevolve.encoding import (
    POSITION_DIM,
    MOVE_DIM,
    NUM_PIECE_PLANES,
    SQUARES_PER_PLANE,
    GLOBAL_FEATURES,
    encode_position,
    encode_move,
    encode_position_from_fen,
    encode_position_and_move,
    piece_plane,
)


def _plane_index(piece):
    return piece_plane(piece)


def test_dimensions_and_dtypes():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    p = encode_position(board)
    m = encode_move(board, move)
    assert p.shape == (POSITION_DIM,)
    assert m.shape == (MOVE_DIM,)
    assert p.dtype == np.float32
    assert m.dtype == np.float32
    assert POSITION_DIM == 774
    assert MOVE_DIM == 158


def test_constants_consistent():
    assert NUM_PIECE_PLANES * SQUARES_PER_PLANE + GLOBAL_FEATURES == POSITION_DIM


def test_determinism():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    p1 = encode_position(board)
    p2 = encode_position(board)
    m1 = encode_move(board, move)
    m2 = encode_move(board, move)
    assert np.array_equal(p1, p2)
    assert np.array_equal(m1, m2)


def test_starting_position_piece_planes():
    board = chess.Board()
    p = encode_position(board)
    # 32 pieces -> 32 ones across planes
    assert int((p[:768] == 1.0).sum()) == 32
    # each piece on its correct plane
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            plane = piece_plane(piece)
            assert p[plane * 64 + square] == 1.0
    # all other squares zero
    for square in chess.SQUARES:
        for plane in range(NUM_PIECE_PLANES):
            piece = board.piece_at(square)
            if piece is None or piece_plane(piece) != plane:
                assert p[plane * 64 + square] == 0.0
    # global features
    assert p[768] == 0.0  # white to move
    assert p[769] == 1.0 and p[770] == 1.0 and p[771] == 1.0 and p[772] == 1.0
    assert p[773] == -1.0  # no ep


def test_side_to_move_and_castling():
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    p = encode_position(board)
    assert p[768] == 1.0  # black to move
    assert p[769] == 1.0 and p[770] == 1.0 and p[771] == 1.0 and p[772] == 1.0
    board2 = chess.Board()
    board2.push(chess.Move.from_uci("e2e4"))
    board2.push(chess.Move.from_uci("e7e5"))
    p2 = encode_position(board2)
    assert p2[773] == 4.0  # ep file e
    assert p2[768] == 0.0  # white to move


def test_castling_flags_lost():
    # After white king moves, white castling rights gone
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("e7e5"))
    board.push(chess.Move.from_uci("e1e2"))  # king moves -> rights lost
    p = encode_position(board)
    assert p[769] == 0.0 and p[770] == 0.0
    assert p[771] == 1.0 and p[772] == 1.0
    # After black rook moves, black queenside right lost (depending); test black king move
    board2 = chess.Board()
    board2.push(chess.Move.from_uci("e2e4"))
    board2.push(chess.Move.from_uci("e7e5"))
    board2.push(chess.Move.from_uci("g1f3"))
    board2.push(chess.Move.from_uci("e8e7"))  # black king moves
    p2 = encode_position(board2)
    assert p2[771] == 0.0 and p2[772] == 0.0


def test_white_castling_encoding():
    # Position where white can castle kingside; verify flags
    board = chess.Board()
    # clear path: move knights/bishops already? Use standard start, flags all on
    p = encode_position(board)
    assert p[769] == 1.0  # white kingside
    assert p[770] == 1.0  # white queenside
    # After O-O, rights gone
    board2 = chess.Board()
    # play a quick game allowing white O-O
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"]:
        board2.push(chess.Move.from_uci(uci))
    assert board2.is_valid()
    p2 = encode_position(board2)
    assert p2[769] == 0.0 and p2[770] == 0.0


def test_black_castling_encoding():
    board = chess.Board()
    p = encode_position(board)
    assert p[771] == 1.0 and p[772] == 1.0
    board2 = chess.Board()
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1", "e8g8"]:
        board2.push(chess.Move.from_uci(uci))
    assert board2.is_valid()
    p2 = encode_position(board2)
    assert p2[771] == 0.0 and p2[772] == 0.0


def test_move_one_hot_and_piece():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    m = encode_move(board, move)
    assert m[chess.E2] == 1.0
    assert m[64 + chess.E4] == 1.0
    # moving piece white pawn -> plane 0 -> index 128+0
    assert m[128 + 0] == 1.0
    assert m[128:140].sum() == 1.0
    # no capture
    assert m[140] == 1.0
    assert m[140:153].sum() == 1.0
    # no promotion
    assert m[153] == 1.0


def test_black_move_and_capture():
    # black pawn capture
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("d7d5"))
    # black to move? no, after d7d5 it's white. Let's make black capture: white e4, black d5 -> white plays, then black captures exd4? 
    # Instead: position where black captures a white piece
    board2 = chess.Board()
    board2.push(chess.Move.from_uci("e2e4"))
    board2.push(chess.Move.from_uci("d7d5"))
    board2.push(chess.Move.from_uci("e4d5"))  # white captures
    # now black to move; black can capture back? pawn on c6? Let's use a simpler black capture scenario
    board3 = chess.Board()
    board3.push(chess.Move.from_uci("d2d4"))
    board3.push(chess.Move.from_uci("d7d5"))
    board3.push(chess.Move.from_uci("c2c4"))  # white
    board3.push(chess.Move.from_uci("d5c4"))  # black captures pawn on c4
    assert board3.is_valid()
    move = chess.Move.from_uci("d5c4")
    # already played; reconstruct position before move
    board4 = chess.Board()
    board4.push(chess.Move.from_uci("d2d4"))
    board4.push(chess.Move.from_uci("d7d5"))
    board4.push(chess.Move.from_uci("c2c4"))
    assert move in board4.legal_moves
    m = encode_move(board4, move)
    # moving piece black pawn -> plane 6 -> index 128+6
    assert m[128 + 6] == 1.0
    # captured white pawn -> plane 0 -> captured_index 1 -> offset 140+1
    assert m[140 + 1] == 1.0
    assert m[140] == 0.0


def test_non_pawn_capture():
    # white knight captures black pawn
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("d7d5"))
    board.push(chess.Move.from_uci("e4d5"))  # white pawn captures
    # position: white pawn d5, black to move. Make black move a knight then white knight captures
    # Simpler: set up white knight capturing black rook
    board2 = chess.Board("r3k3/8/8/8/8/8/8/2N1K3 w - - 0 1")
    # not valid-ish; build via moves instead
    board3 = chess.Board()
    # white knight to c3, black rook on d5 via ... too complex; use direct FEN with kings
    board4 = chess.Board("4k3/8/8/3r4/8/2N5/8/4K3 w - - 0 1")
    assert board4.is_valid()
    move = chess.Move.from_uci("c3d5")
    assert move in board4.legal_moves  # knight captures rook on d5
    m = encode_move(board4, move)
    # moving white knight -> plane 1
    assert m[128 + 1] == 1.0
    # captured black rook -> plane 9 -> captured_index 10 -> offset 140+10
    assert m[140 + 10] == 1.0


def test_capture_promotion():
    # white pawn captures and promotes: a7xb8q
    board = chess.Board("1k6/8/8/8/8/8/P7/1K6 w - - 0 1")
    # pawn on a7? a7 is rank7. Let's put pawn on a7 and black piece on b8
    board2 = chess.Board("1rbk4/8/8/8/8/8/P7/1K6 w - - 0 1")
    # a7 pawn, b8 has black rook; a7xb8 promotes
    move = chess.Move.from_uci("a7b8q")
    if move not in board2.legal_moves:
        # try without extra pieces blocking; rebuild
        board2 = chess.Board("1r1k4/P7/8/8/8/8/8/1K6 w - - 0 1")
        move = chess.Move.from_uci("a7b8q")
    assert board2.is_valid()
    assert move in board2.legal_moves, board2.fen()
    m = encode_move(board2, move)
    # captured piece present (not none)
    assert m[140] == 0.0
    assert m[140:153].sum() == 1.0
    # promotion queen
    assert m[153 + 1] == 1.0


def test_en_passant_encoding():
    # valid ep position with both kings and legal ep target e5d6
    # white pawn e5, black pawn d5 just moved d7->d5, white to move, ep d6
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    assert board.is_valid()
    move = chess.Move.from_uci("e5d6")
    assert move in board.legal_moves
    assert board.is_en_passant(move)
    m = encode_move(board, move)
    # captured black pawn -> plane 6 -> captured_index 7 -> offset 140+7
    assert m[140 + 7] == 1.0
    assert m[140] == 0.0


def test_promotion_encoding():
    # valid promotion position with both kings
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    assert board.is_valid()
    cases = {
        "a7a8q": 1,
        "a7a8r": 2,
        "a7a8b": 3,
        "a7a8n": 4,
    }
    for uci, idx in cases.items():
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves, uci
        m = encode_move(board, move)
        assert m[153 + idx] == 1.0, (uci, idx)
        assert m[153:158].sum() == 1.0


def test_from_fen_and_combined():
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    p = encode_position_from_fen(fen)
    assert p.shape == (POSITION_DIM,)
    board = chess.Board(fen)
    move = list(board.legal_moves)[0]
    pp, mm = encode_position_and_move(board, move)
    assert pp.shape == (POSITION_DIM,)
    assert mm.shape == (MOVE_DIM,)


def test_encode_all_legal_candidates():
    board = chess.Board()
    moves = list(board.legal_moves)
    assert len(moves) == 20
    encoded = [encode_move(board, mv) for mv in moves]
    assert len(encoded) == len(moves)
    for v in encoded:
        assert v.shape == (MOVE_DIM,)
        assert v.dtype == np.float32


def test_invalid_move_raises():
    board = chess.Board()
    with pytest.raises(ValueError):
        encode_move(board, chess.Move.from_uci("e2e5"))
    with pytest.raises(TypeError):
        encode_move(board, "e2e4")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        encode_move(None, chess.Move.from_uci("e2e4"))  # type: ignore[arg-type]


def test_encode_move_none_and_invalid_board():
    board = chess.Board()
    with pytest.raises(TypeError):
        encode_move(None, chess.Move.from_uci("e2e4"))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        encode_move("not a board", chess.Move.from_uci("e2e4"))  # type: ignore[arg-type]
    invalid = chess.Board()
    invalid.clear()
    with pytest.raises(ValueError):
        encode_move(invalid, chess.Move.from_uci("e2e4"))
    with pytest.raises(TypeError):
        encode_move(board, None)  # type: ignore[arg-type]


def test_invalid_board_raises():
    board = chess.Board()
    board.clear()
    with pytest.raises(ValueError):
        encode_position(board)
    with pytest.raises(TypeError):
        encode_position("not a board")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        encode_position(None)  # type: ignore[arg-type]


def test_non_string_fen_raises():
    with pytest.raises(TypeError):
        encode_position_from_fen(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        encode_position_from_fen(None)  # type: ignore[arg-type]


def test_finite():
    board = chess.Board()
    p = encode_position(board)
    m = encode_move(board, chess.Move.from_uci("e2e4"))
    assert np.isfinite(p).all()
    assert np.isfinite(m).all()


def test_package_imports():
    import chessevolve
    assert hasattr(chessevolve, "encode_position")
    assert hasattr(chessevolve, "POSITION_DIM")
    assert chessevolve.POSITION_DIM == 774
    assert chessevolve.MOVE_DIM == 158
