"""Board and candidate-move encoding for ChessEvolve.

Converts a chess position and a legal candidate move into fixed-size,
deterministic NumPy feature vectors that a small neural network can consume.

The representation is intentionally small and explainable so that it can be
inspected by hand and debugged without tooling.  Every feature is a pure
function of the board and the move: the same position and move always produce
the same vectors, with no randomness and no dependence on object identity.

Feature layout (all vectors are ``dtype=float32``):

Position vector :attr:`POSITION_DIM == 1701`::

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
    768-773  Existing side-to-move, castling, and en-passant features.
    774-778  Normalized material balance for pawn through queen.
    779-780  White and black total material.
    781-784  Legal mobility: side-to-move, opponent, and both check flags.
    785-788  King safety: attacked king squares and attacked adjacent king
             squares for White and Black.
    789-796  Pawn structure: doubled, isolated, passed, and central pawn
             counts for White and Black.
    797-1564 Piece-square table values for the twelve piece/color planes.
    1565-1692 Attack maps for White and Black.
    1693-1700 Global check, material, and phase features.

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
    158-161 Candidate-specific check, capture, captured-piece value, and
            promotion-value features.

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
FEATURE_GROUPS = ("A", "B", "C", "D")
DEFAULT_FEATURE_GROUP = "D"
LEGACY_GLOBAL_FEATURES = 6
POSITION_FEATURES_C = 1573
MOVE_TACTICAL_FEATURES = 4
GLOBAL_FEATURES = 933
POSITION_DIM = NUM_PIECE_PLANES * SQUARES_PER_PLANE + GLOBAL_FEATURES  # D: 1701
MOVE_DIM = 158 + MOVE_TACTICAL_FEATURES  # B/C/D: 162
LEGACY_POSITION_DIM = 774
LEGACY_MOVE_DIM = 158

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

_PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

# Compact, symmetric piece-square priors. Values are deliberately bounded so
# they remain comparable with the binary piece planes.
_PST_CENTER = {
    chess.PAWN: 0.20,
    chess.KNIGHT: 0.35,
    chess.BISHOP: 0.25,
    chess.ROOK: 0.12,
    chess.QUEEN: 0.10,
    chess.KING: -0.15,
}


def _normalized_material(board: chess.Board, color: chess.Color) -> np.ndarray:
    return np.array(
        [
            sum(
                1.0
                for piece in board.piece_map().values()
                if piece.color == color and piece.piece_type == piece_type
            )
            / (8.0 if piece_type == chess.PAWN else 2.0 if piece_type in (chess.KNIGHT, chess.BISHOP) else 2.0)
            for piece_type in range(chess.PAWN, chess.QUEEN + 1)
        ],
        dtype=np.float32,
    )


def _pawn_structure(board: chess.Board, color: chess.Color) -> tuple[float, float, float, float]:
    pawns = board.pieces(chess.PAWN, color)
    files = [chess.square_file(square) for square in pawns]
    counts = np.bincount(files, minlength=8)
    doubled = float(np.maximum(counts - 1, 0).sum()) / 8.0
    isolated = sum(
        1 for file_index in files
        if not any(counts[neighbor] for neighbor in (file_index - 1, file_index + 1) if 0 <= neighbor < 8)
    ) / 8.0
    enemy_pawns = board.pieces(chess.PAWN, not color)
    enemy_by_file = {chess.square_file(square) for square in enemy_pawns}
    passed = sum(
        1 for square in pawns
        if not any(
            file_index in enemy_by_file
            for file_index in (
                chess.square_file(square) - 1,
                chess.square_file(square),
                chess.square_file(square) + 1,
            )
        )
    ) / 8.0
    central = sum(1 for square in pawns if chess.square_file(square) in (3, 4)) / 2.0
    return doubled, isolated, passed, central


def _attack_map(board: chess.Board, color: chess.Color) -> np.ndarray:
    return np.array(
        [1.0 if board.attackers(color, square) else 0.0 for square in chess.SQUARES],
        dtype=np.float32,
    )


def _piece_square_value(piece: chess.Piece, square: chess.Square) -> float:
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    if piece.color == chess.BLACK:
        rank_index = 7 - rank_index
    center_distance = abs(3.5 - file_index) + abs(3.5 - rank_index)
    advancement = rank_index / 7.0 if piece.piece_type == chess.PAWN else 0.0
    return _PST_CENTER[piece.piece_type] * (1.0 - center_distance / 7.0) + advancement * 0.15


def piece_plane(piece: chess.Piece) -> int:
    """Return the piece plane index (0-11) for a :class:`chess.Piece`."""
    offset = _PIECE_TYPE_TO_PLANE_OFFSET[piece.piece_type]
    return offset if piece.color == chess.WHITE else offset + 6


# --- Position encoding -----------------------------------------------------

def _encode_position_full(board: chess.Board) -> np.ndarray:
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

    cursor = base + 6
    white_material = _normalized_material(board, chess.WHITE)
    black_material = _normalized_material(board, chess.BLACK)
    features[cursor : cursor + 5] = white_material - black_material
    cursor += 5
    features[cursor] = float(
        sum(_PIECE_VALUES[piece.piece_type] for piece in board.piece_map().values() if piece.color == chess.WHITE)
        / 39.0
    )
    features[cursor + 1] = float(
        sum(_PIECE_VALUES[piece.piece_type] for piece in board.piece_map().values() if piece.color == chess.BLACK)
        / 39.0
    )
    cursor += 2

    white_mobility = board.legal_moves.count()
    mirror = board.copy(stack=False)
    mirror.turn = not board.turn
    black_mobility = mirror.legal_moves.count()
    features[cursor : cursor + 4] = (
        min(white_mobility, 40) / 40.0,
        min(black_mobility, 40) / 40.0,
        float(board.is_check() and board.turn == chess.WHITE),
        float(board.is_check() and board.turn == chess.BLACK),
    )
    cursor += 4

    for color in (chess.WHITE, chess.BLACK):
        king = board.king(color)
        king_zone = chess.SquareSet(chess.BB_KING_ATTACKS[king] | chess.BB_SQUARES[king]) if king is not None else chess.SquareSet()
        attacked_king = bool(king is not None and board.attackers(not color, king))
        attacked_zone = sum(bool(board.attackers(not color, square)) for square in king_zone)
        features[cursor : cursor + 2] = (
            float(attacked_king),
            attacked_zone / 9.0,
        )
        cursor += 2

    for color in (chess.WHITE, chess.BLACK):
        features[cursor : cursor + 4] = _pawn_structure(board, color)
        cursor += 4

    for plane in range(NUM_PIECE_PLANES):
        color = chess.WHITE if plane < 6 else chess.BLACK
        piece_type = (plane % 6) + 1
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None and piece.color == color and piece.piece_type == piece_type:
                features[cursor + plane * 64 + square] = _piece_square_value(piece, square)
    cursor += NUM_PIECE_PLANES * 64

    features[cursor : cursor + 64] = _attack_map(board, chess.WHITE)
    features[cursor + 64 : cursor + 128] = _attack_map(board, chess.BLACK)
    cursor += 128

    features[cursor : cursor + 8] = (
        float(board.is_check()),
        float(board.was_into_check()),
        float(board.fullmove_number / 100.0),
        float(board.halfmove_clock / 100.0),
        float(len(board.pieces(chess.PAWN, chess.WHITE)) / 8.0),
        float(len(board.pieces(chess.PAWN, chess.BLACK)) / 8.0),
        float(board.turn == chess.WHITE),
        float(board.is_checkmate()),
    )

    return features


def _validate_feature_group(feature_group: str) -> str:
    if feature_group not in FEATURE_GROUPS:
        raise ValueError(
            f"feature_group must be one of {FEATURE_GROUPS}, got {feature_group!r}"
        )
    return feature_group


def encode_position(
    board: chess.Board, feature_group: str = DEFAULT_FEATURE_GROUP
) -> np.ndarray:
    """Encode a board using feature group A, B, C, or D."""
    group = _validate_feature_group(feature_group)
    if group in {"A", "B"}:
        return encode_position_legacy(board)
    full = _encode_position_full(board)
    if group == "C":
        return full[:POSITION_FEATURES_C].copy()
    return full


def encode_position_from_fen(
    fen: str, feature_group: str = DEFAULT_FEATURE_GROUP
) -> np.ndarray:
    """Encode a FEN string as a position vector (thin wrapper over :func:`encode_position`)."""
    if not isinstance(fen, str):
        raise TypeError(f"fen must be a str, got {type(fen).__name__}")
    return encode_position(chess.Board(fen), feature_group)


def encode_position_legacy(board: chess.Board) -> np.ndarray:
    """Encode positions for checkpoints trained before feature expansion."""
    if not isinstance(board, chess.Board):
        raise TypeError(f"board must be a chess.Board, got {type(board).__name__}")
    if not board.is_valid():
        raise ValueError("board is not a valid position")
    features = np.zeros(LEGACY_POSITION_DIM, dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            features[piece_plane(piece) * 64 + square] = 1.0
    base = 768
    features[base] = float(board.turn == chess.BLACK)
    features[base + 1] = float(board.has_kingside_castling_rights(chess.WHITE))
    features[base + 2] = float(board.has_queenside_castling_rights(chess.WHITE))
    features[base + 3] = float(board.has_kingside_castling_rights(chess.BLACK))
    features[base + 4] = float(board.has_queenside_castling_rights(chess.BLACK))
    features[base + 5] = (
        float(chess.square_file(board.ep_square)) if board.ep_square is not None else -1.0
    )
    return features


# --- Move encoding ---------------------------------------------------------

def _encode_move_full(board: chess.Board, move: chess.Move) -> np.ndarray:
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

    # 158-161: candidate-specific tactical and value features.
    features[158] = float(board.gives_check(move))
    features[159] = float(board.is_capture(move))
    captured_value = 0.0
    if board.is_capture(move):
        captured = (
            chess.Piece(chess.PAWN, not board.turn)
            if board.is_en_passant(move)
            else board.piece_at(move.to_square)
        )
        if captured is not None:
            captured_value = _PIECE_VALUES[captured.piece_type] / 9.0
    features[160] = captured_value
    features[161] = (
        _PIECE_VALUES[move.promotion] / 9.0 if move.promotion is not None else 0.0
    )

    return features


def encode_move(
    board: chess.Board,
    move: chess.Move,
    feature_group: str = DEFAULT_FEATURE_GROUP,
) -> np.ndarray:
    """Encode a legal move using feature group A, B, C, or D."""
    group = _validate_feature_group(feature_group)
    if group == "A":
        return encode_move_legacy(board, move)
    return _encode_move_full(board, move)


def encode_move_legacy(board: chess.Board, move: chess.Move) -> np.ndarray:
    """Encode moves for checkpoints trained before feature expansion."""
    features = _encode_move_full(board, move)
    return features[:LEGACY_MOVE_DIM].copy()


def encode_position_and_move(
    board: chess.Board,
    move: chess.Move,
    feature_group: str = DEFAULT_FEATURE_GROUP,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode a position and one of its legal moves together.

    Returns ``(position_vector, move_vector)``.
    """
    return encode_position(board, feature_group), encode_move(board, move, feature_group)
