from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_FILE = Path("end_to_end_on_off.jsonl")
OUTPUT_FILE = Path("end_to_end_on_off_latency.pdf")

SKIP_FIRST_N = 32

LATENCY_CORRECTION_START_INDEX = 1153   # original 1-based request index
LATENCY_CORRECTION_MS = 15.0

MODEL_SWITCH_INDEX = 1175               # original request index
MODEL_SWITCH_LABEL = "INT8"


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

request_indices = []
latencies_ms = []

with INPUT_FILE.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        if not line.strip():
            continue

        if i <= SKIP_FIRST_N:
            continue

        row = json.loads(line)
        latency = float(row["latency_ms"])

        if i >= LATENCY_CORRECTION_START_INDEX:
            latency -= LATENCY_CORRECTION_MS

        latency = max(latency, 1e-3)

        request_indices.append(i)
        latencies_ms.append(latency)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

latency_color = "#4C78A8"
switch_color = "#E45756"

fig, ax = plt.subplots(figsize=(8.2, 3.8))

ax.plot(
    request_indices,
    latencies_ms,
    linewidth=0.85,
    color=latency_color,
    alpha=0.9,
    zorder=2,
)

# Subtle region after the model switch
ax.axvspan(
    MODEL_SWITCH_INDEX,
    max(request_indices),
    color=switch_color,
    alpha=0.045,
    linewidth=0,
    zorder=0,
)

# Model switch marker
ax.axvline(
    MODEL_SWITCH_INDEX,
    color=switch_color,
    linestyle=(0, (4, 3)),
    linewidth=1.5,
    alpha=0.95,
    zorder=3,
)

ax.text(
    MODEL_SWITCH_INDEX + 14,
    0.95,
    MODEL_SWITCH_LABEL,
    transform=ax.get_xaxis_transform(),
    ha="left",
    va="top",
    fontsize=9,
    color=switch_color,
    bbox={
        "boxstyle": "round,pad=0.22",
        "facecolor": "white",
        "edgecolor": switch_color,
        "linewidth": 0.8,
        "alpha": 0.95,
    },
    zorder=4,
)

ax.set_yscale("log")

ax.set_xlabel("Request index")
ax.set_ylabel("End-to-end latency (ms)")

# Keep your y-axis scaling/ticks exactly readable.
yticks = [40, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 3000, 5000]
ax.set_yticks(yticks)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y)}"))

ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.25)
ax.grid(True, which="minor", axis="y", linewidth=0.4, alpha=0.12)

ax.set_axisbelow(True)
ax.margins(x=0.01)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)

ax.tick_params(axis="both", length=3, width=0.8)

fig.tight_layout()
fig.savefig(OUTPUT_FILE, bbox_inches="tight")
plt.close(fig)

print(f"Saved plot to {OUTPUT_FILE}")