import { Chessboard } from "react-chessboard";

export type PieceKind = "q" | "r" | "b" | "n";

type CanDragPiece = (args: {
  isSparePiece: boolean;
  piece: { pieceType: string };
  square: string | null;
}) => boolean;

type PieceDropArgs = {
  piece: { isSparePiece: boolean; position: string; pieceType: string };
  sourceSquare: string;
  targetSquare: string | null;
};

interface Props {
  position: string; // FEN
  orientation: "white" | "black";
  allowDragging: boolean;
  canDragPiece: CanDragPiece | null;
  onPieceDrop: (source: string, target: string) => boolean;
  onPromotionSelect: (piece: PieceKind) => void;
  pendingPromotion: { from: string; to: string } | null;
}

export default function GameBoard({
  position,
  orientation,
  allowDragging,
  canDragPiece,
  onPieceDrop,
  onPromotionSelect,
  pendingPromotion,
}: Props) {
  const options = {
    position,
    boardOrientation: orientation,
    allowDragging,
    allowDragOffBoard: false,
    showNotation: true,
    canDragPiece: canDragPiece ?? undefined,
    onPieceDrop: (args: PieceDropArgs) => {
      if (!args.targetSquare) return false;
      return onPieceDrop(args.sourceSquare, args.targetSquare);
    },
  };

  return (
    <div className="board-wrap">
      <Chessboard options={options as Parameters<typeof Chessboard>[0]["options"]} />
      {pendingPromotion && (
        <div className="promotion-overlay">
          <span>Promote to:</span>
          {["q", "r", "b", "n"].map((piece: PieceKind) => (
            <button
              key={piece}
              type="button"
              onClick={() => onPromotionSelect(piece)}
              title={piece.toUpperCase()}
            >
              {piece.toUpperCase()}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
