interface Props {
  mode: "human" | "model";
  onChange: (mode: "human" | "model") => void;
}

export default function ModeSelector({ mode, onChange }: Props) {
  return (
    <div style={{ display: "flex", gap: 6, border: "1px solid #334155", borderRadius: 6, overflow: "hidden" }}>
      <button
        type="button"
        onClick={() => onChange("human")}
        className={mode === "human" ? "primary" : undefined}
      >
        Human vs Model
      </button>
      <button
        type="button"
        onClick={() => onChange("model")}
        className={mode === "model" ? "primary" : undefined}
      >
        Model vs Model
      </button>
    </div>
  );
}
