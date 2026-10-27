interface Props {
  onExport: () => void;
  disabled?: boolean;
}

export default function PGNExport({ onExport, disabled }: Props) {
  return (
    <button type="button" onClick={onExport} disabled={disabled} className="pgn-btn">
      Export PGN
    </button>
  );
}
