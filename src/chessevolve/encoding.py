"""Board and candidate-move encoding for ChessEvolve.

Converts a chess position and a legal candidate move into fixed-size,
deterministic NumPy feature vectors that a small neural network can consume.

The representation is intentionally small and explainable so that it can be
inspected by hand and debugged without tooling.  Every feature is a pure
function of the board and the move: the same position and move always produce
the same vectors, with no randomness and no dependence on object identity.

Feature layout (all vectors are ``dtype=float32``):

Position vector :attr:`POSITION_DIM == 774`::

    0-767   Twelve binary piece planes (one per piece type and color), each
            covering the 64 squares in rank-major order (index = rank*8 +
            file, a1 = 0 ... h8 = 63).  A cell is ``1.0`` where the piece
            sits and ``0.0`` otherwise.
            Plane 0  = white pawn
            Plane 1  = white knight
            Plane 2  = white bishop
            Plane 3  = white rook
            Plane 4  = white queen
            Plane 5  = white king
            Plane 6  = black pawn
            Plane 7  = black knight
            Plane 8  = black bishop
            Plane 9  = black rook
            Plane 10 = black queen
            Plane 11 = black king
            (Plane index * 64 + square index gives the flat offset.)
    768     Side to move: 0.0 = white, 1.0 = black.
    769     White kingside castling available (0.0 / 1.0).
    770     White queenside castling available (0.0 / 1.0).
    771     Black kingside castling available (0.0 / 1.0).
    772     Black queenside castling available (0.0 / 1.0).
    773     En-passant file (0.0-7.0) or -1.0 when there is no en-passant
            target square.

Move vector :attr:`MOVE_DIM == 158`::

    0-63    From-square one-hot (index = square).
    64-127  To-square one-hot (index = square).
    128-139 Moving piece one-hot (index = piece plane of the moving piece,
            i.e. 0-5 white P/N/B/R/Q/K, 6-11 black p/n/b/r/q/k).
    140-152 Captured piece one-hot (index 0 = no capture, 1-12 = piece
            planes).  The king slots (index 6 and 12) are reserved but
            inactive: a legal move can never capture a king.
    153-157 Promotion one-hot (0 = none, 1 = queen, 2 = rook, 3 = bishop,
            4 = knight).

The halfmove clock and fullmove number are deliberately omitted: they do not
affect the quality of a candidate move in a position, and excluding them keeps
the representation small while guaranteeing that identical boards encode
identically regardless of game-clock state.
"""

from __future__ import annotations

from typing import Tuple

import chess
import numpy as np

# --- Dimensions -----------------------------------------------------------

NUM_PIECE_PLANES = 12
SQUARES_PER_PLANE = 64
GLOBAL_FEATURES = 6
POSITION_DIM = NUM_PIECE_PLANES * SQUARES_PER_PLANE + GLOBAL_FEATURES  # 774
MOVE_DIM = 158

# --- Piece-plane mapping ---------------------------------------------------
# Planes 0-5 = white P/N/B/R/Q/K, planes 6-11 = black p/n/b/r/q/k.
# chess.PAWN..chess.KING are 1..6, so plane = (0 if white else 6) + (type-1).
_PIECE_TYPE_TO_PLANE_OFFSET = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

# Promotion piece type -> index in the move promotion one-hot.
_PROMOTION_TO_INDEX = {
    None: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
}


def piece_plane(piece: chess.Piece) -> int:
    """Return the piece plane index (0-11) for a :class:`chess.Piece`."""
    offset = _PIECE_TYPE_TO_PLANE_OFFSET[piece.piece_type]
    return offset if piece.color == chess.WHITE else offset + 6


# --- Position encoding -----------------------------------------------------

def encode_position(board: chess.Board) -> np.ndarray:
    """Encode a board position as a flat :attr:`POSITION_DIM` float32 vector.

    Parameters
    ----------
    board:
        A :class:`chess.Board`.  It must be valid (``board.is_valid()``).

    Returns
    -------
    np.ndarray
        Shape ``(POSITION_DIM,)``, dtype ``float32``.

    Raises
    ------
    TypeError
        If ``board`` is not a :class:`chess.Board`.
    ValueError
        If the board is not valid.
    """
    if not isinstance(board, chess.Board):
        raise TypeError(f"board must be a chess.Board, got {type(board).__name__}")
    if not board.is_valid():
        raise ValueError("board is not a valid position")

    features = np.zeros(POSITION_DIM, dtype=np.float32)

    # 0-767: twelve binary piece planes.
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            plane = piece_plane(piece)
            features[plane * SQUARES_PER_PLANE + square] = 1.0

    base = NUM_PIECE_PLANES * SQUARES_PER_PLANE  # 768
    # side to move
    features[base + 0] = 0.0 if board.turn == chess.WHITE else 1.0
    # castling rights
    features[base + 1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    features[base + 2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    features[base + 3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    features[base + 4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    # en-passant file
    if board.ep_square is not None:
        features[base + 5] = float(chess.square_file(board.ep_square))
    else:
        features[base + 5] = -1.0

    return features


def encode_position_from_fen(fen: str) -> np.ndarray:
    """Encode a FEN string as a position vector (thin wrapper over :func:`encode_position`)."""
    if not isinstance(fen, str):
        raise TypeError(f"fen must be a str, got {type(fen).__name__}")
    return encode_position(chess.Board(fen))


# --- Move encoding ---------------------------------------------------------

def encode_move(board: chess.Board, move: chess.Move) -> np.ndarray:
    """Encode a legal candidate move as a flat :attr:`MOVE_DIM` float32 vector.

    Parameters
    ----------
    board:
        The position the move is played from.  Must be a valid ``chess.Board``.
    move:
        A :class:`chess.Move` that must be legal in ``board``.

    Returns
    -------
    np.ndarray
        Shape ``(MOVE_DIM,)``, dtype ``float32``.

    Raises
    ------
    TypeError
        If ``board`` is not a :class:`chess.Board` or ``move`` is not a
        :class:`chess.Move`.
    ValueError
        If ``board`` is not valid or ``move`` is not legal in ``board``.
    """
    if not isinstance(board, chess.Board):
        raise TypeError(f"board must be a chess.Board, got {type(board).__name__}")
    if not board.is_valid():
        raise ValueError("board is not a valid position")
    if not isinstance(move, chess.Move):
        raise TypeError(f"move must be a chess.Move, got {type(move).__name__}")
    if move not in board.legal_moves:
        raise ValueError(f"Illegal move {move.uci()} for position {board.fen()}")

    features = np.zeros(MOVE_DIM, dtype=np.float32)

    # 0-63: from-square one-hot.
    features[move.from_square] = 1.0
    # 64-127: to-square one-hot.
    features[64 + move.to_square] = 1.0

    # 128-139: moving piece one-hot (plane index).
    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None:
        raise ValueError(f"No piece on from-square {chess.square_name(move.from_square)}")
    features[128 + piece_plane(moving_piece)] = 1.0

    # 140-152: captured piece one-hot (index 0 = no capture).
    captured_index = 0
    if board.is_capture(move):
        if board.is_en_passant(move):
            captured_piece = chess.Piece(chess.PAWN, not board.turn)
            captured_index = piece_plane(captured_piece) + 1
        else:
            captured_piece = board.piece_at(move.to_square)
            if captured_piece is not None:
                captured_index = piece_plane(captured_piece) + 1
    features[140 + captured_index] = 1.0

    # 153-157: promotion one-hot (0 = none).
    promotion_index = _PROMOTION_TO_INDEX.get(move.promotion)
    if promotion_index is None:
        raise ValueError(f"Unknown promotion piece type {move.promotion!r}")
    features[153 + promotion_index] = 1.0

    return features


def encode_position_and_move(
    board: chess.Board, move: chess.Move
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode a position and one of its legal moves together.

    Returns ``(position_vector, move_vector)``.
    """
    return encode_position(board), encode_move(board, move)
