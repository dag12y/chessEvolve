interface Props {
  isPlaying: boolean;
  speed: number;
  onPlay: () => void;
  onPause: () => void;
  onStep: () => void;
  onSpeedChange: (value: number) => void;
}

export default function AutoplayControls({
  isPlaying,
  speed,
  onPlay,
  onPause,
  onStep,
  onSpeedChange,
}: Props) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", gap: 6 }}>
        <button type="button" onClick={onPause} disabled={!isPlaying}>
          Pause
        </button>
        <button type="button" onClick={onPlay} disabled={isPlaying}>
          Autoplay
        </button>
        <button type="button" onClick={onStep}>Step</button>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "#94a3b8" }}>
        <span>Speed (ms/move)</span>
        <input
          type="range"
          min="100"
          max="3000"
          step="100"
          value={speed}
          onChange={(e) => onSpeedChange(Number(e.target.value))}
        />
        <span>{speed}</span>
      </div>
    </div>
  );
}
