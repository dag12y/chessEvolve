import type { Model } from "../lib/api";

interface Props {
  label: string;
  value: string | null;
  models: Model[];
  disabled?: boolean;
  onChange: (value: string | null) => void;
}

export default function ModelSelect({
  label,
  value,
  models,
  disabled,
  onChange,
}: Props) {
  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    onChange(event.target.value || null);
  };

  return (
    <div>
      <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 4 }}>{label}</div>
      <select
        value={value ?? ""}
        onChange={handleChange}
        disabled={disabled}
        style={{ width: "100%" }}
      >
        <option value="">— none —</option>
        {models.map((m) => (
          <option key={m.id} value={m.id} disabled={!m.available}>
            {m.name} {!m.available ? " (checkpoint missing)" : ""}
          </option>
        ))}
      </select>
    </div>
  );
}
