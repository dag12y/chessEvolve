interface Props {
  moves: string[]; // SAN moves in order played
}

export default function MoveList({ moves }: Props) {
  const pairs: { white: string; black?: string }[] = [];
  for (let i = 0; i < moves.length; i += 2) {
    pairs.push({ white: moves[i], black: moves[i + 1] });
  }

  return (
    <div className="moves">
      {pairs.map((pair, idx) => (
        <div key={idx} className="pair">
          <span className="num">{idx + 1}.</span>
          <span className="move white">{pair.white}</span>
          {pair.black !== undefined && <span className="move black">{pair.black}</span>}
        </div>
      ))}
      {pairs.length === 0 && <div style={{ color: "#64748b", fontSize: "12px" }}>No moves yet</div>}
    </div>
  );
}
