#!/usr/bin/env python3
"""Generate the current submission comparison chart from saved JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def load_models(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text())["models"]


def main() -> None:
    mixed = load_models(ROOT / "experiments/results/v2_mixed_B/test.json")
    values = {
        row["model"]: float(row["non_mate_regret_cp"])
        for row in mixed
    }
    labels = ["B teacher", "B ranked", "B mixed"]
    keys = ["B_teacher_seed1", "B_ranked_seed1", "B_mixed_seed1"]
    regrets = [values[key] for key in keys]

    output = ROOT / "reports/current_results.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(labels, regrets, color=["#64748b", "#f59e0b", "#2563eb"])
    axis.set_ylabel("Non-mate regret (centipawns; lower is better)")
    axis.set_title("Final-test B-model comparison")
    axis.bar_label(bars, fmt="%.1f")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
