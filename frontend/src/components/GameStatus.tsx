import { Chess } from "chess.js";

interface Props {
  fen: string;
  plyCapReached: boolean;
}

export default function GameStatus({ fen, plyCapReached }: Props) {
  const game = new Chess(fen);

  let kind: "ok" | "bad" | "warn" = "ok";
  let text: string;
  let result: string | null = null;

  if (plyCapReached) {
    kind = "warn";
    text = "Draw — 200-ply cap reached";
    result = "1/2-1/2";
  } else if (game.isCheckmate()) {
    kind = "bad";
    const winner = game.turn() === "w" ? "Black" : "White";
    text = `Checkmate — ${winner} wins`;
    result = game.turn() === "w" ? "0-1" : "1-0";
  } else if (game.isStalemate()) {
    kind = "warn";
    text = "Draw — stalemate";
    result = "1/2-1/2";
  } else if (game.isDraw()) {
    kind = "warn";
    text = "Draw — " + (game.isInsufficientMaterial() ? "insufficient material" : "50-move rule / threefold");
    result = "1/2-1/2";
  } else if (game.inCheck()) {
    kind = "warn";
    text = `${game.turn() === "w" ? "White" : "Black"} to move — in check`;
  } else {
    text = `${game.turn() === "w" ? "White" : "Black"} to move`;
  }

  return (
    <div className="status">
      <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.4px" }}>
        Game status
      </div>
      <span className={kind === "ok" ? "ok" : kind === "bad" ? "bad" : "warn"}>
        {text}
      </span>
      {result !== null && (
        <span style={{ marginLeft: 10, fontSize: 12, color: "#94a3b8" }}>Result: {result}</span>
      )}
    </div>
  );
}
