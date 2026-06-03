from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

model_sizes = ["1B-7B"]

# Latencies in seconds
direct_evict = [0.000869]

wake_up_from_cpu = [1.159518]
l2_sleep = [0.099637]


# ---------------------------------------------------------------------------
# Safety check for log-scale plot
# ---------------------------------------------------------------------------

all_values = np.array(direct_evict + wake_up_from_cpu + l2_sleep)

if np.any(all_values <= 0):
    raise ValueError("All latency values must be greater than zero for a log-scale plot.")


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
# Plot
# ---------------------------------------------------------------------------

x = np.arange(len(model_sizes))
bar_width = 0.28

colors = {
    "evict": "#4C78A8",
    "wake": "#F58518",
    "l2": "#54A24B",
}

#fig, ax = plt.subplots(figsize=(6.6, 3.4))
fig, ax = plt.subplots(figsize=(4.2, 3.4))

bars_evict = ax.bar(
    x - bar_width / 2,
    direct_evict,
    width=bar_width,
    label="Direct evict",
    color=colors["evict"],
)

bars_wake = ax.bar(
    x + bar_width / 2,
    wake_up_from_cpu,
    width=bar_width,
    label="Wake-up (CPU)",
    color=colors["wake"],
)

bars_l2 = ax.bar(
    x + bar_width / 2,
    l2_sleep,
    width=bar_width,
    bottom=wake_up_from_cpu,
    label="L2 sleep",
    color=colors["l2"],
)


# Axes
ax.set_xlabel("Model size")
ax.set_ylabel("Eviction latency (s)")

ax.set_xticks(x)
ax.set_xticklabels(model_sizes)

#ax.set_xlim(-0.5, len(model_sizes) - 0.5)
ax.set_xlim(-0.45, 0.45)
ax.set_yscale("log")

indirect_total = np.array(wake_up_from_cpu) + np.array(l2_sleep)
ymin = min(direct_evict) * 0.6
ymax = max(indirect_total) * 1.8
ax.set_ylim(ymin, ymax)


# Grid and spines
ax.grid(axis="y", which="major", linewidth=0.6, alpha=0.25)
ax.grid(axis="y", which="minor", linewidth=0.4, alpha=0.12)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)

ax.tick_params(axis="both", length=3, width=0.8)


# Legend
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    handlelength=1.2,
    columnspacing=1.4,
)


# Value labels
ax.bar_label(
    bars_evict,
    labels=[f"{v:.4f}" for v in direct_evict],
    padding=2,
    fontsize=8,
)

for i, total in enumerate(indirect_total):
    ax.text(
        x[i] + bar_width / 2,
        total,
        f"{total:.2f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )


fig.tight_layout()

output_dir = Path(__file__).resolve().parent
fig.savefig(output_dir / "evict_latency.pdf", bbox_inches="tight")
#fig.savefig(output_dir / "evict_latency.png", dpi=300, bbox_inches="tight")

plt.show()