from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

model_sizes = ["1B-7B"]

l1_sleep = [1.114406]
l2_sleep = [0.099637]
l1_sleep_persistent = [0.112912]


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
bar_width = 0.22

colors = {
    "l1": "#4C78A8",
    "l2": "#F58518",
    "l1_cpu": "#54A24B",
}

#fig, ax = plt.subplots(figsize=(6.2, 3.1))
fig, ax = plt.subplots(figsize=(3.6, 3.1))

bars_l1 = ax.bar(
    x - bar_width,
    l1_sleep,
    width=bar_width,
    label="L1",
    color=colors["l1"],
)

bars_l2 = ax.bar(
    x,
    l2_sleep,
    width=bar_width,
    label="L2",
    color=colors["l2"],
)

bars_l1_cpu = ax.bar(
    x + bar_width,
    l1_sleep_persistent,
    width=bar_width,
    label="L1 (weights in CPU)",
    color=colors["l1_cpu"],
)


# Axes
ax.set_xlabel("Model size")
ax.set_ylabel("Sleep latency (s)")

ax.set_xticks(x)
ax.set_xticklabels(model_sizes)

ax.set_ylim(0, 1.55)
#ax.set_xlim(-0.5, len(model_sizes) - 0.5)
ax.set_xlim(-0.55, 0.55)


# Grid and spines
ax.grid(axis="y", linewidth=0.6, alpha=0.25)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)

ax.tick_params(axis="both", length=3, width=0.8)


# Legend above plot
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    handlelength=1.2,
    columnspacing=1.4,
)


# Optional: value labels
for bars in [bars_l1, bars_l2, bars_l1_cpu]:
    ax.bar_label(
        bars,
        labels=[f"{bar.get_height():.2f}" for bar in bars],
        padding=2,
        fontsize=8,
    )


fig.tight_layout()

output_dir = Path(__file__).resolve().parent
fig.savefig(output_dir / "sleep_latency.pdf", bbox_inches="tight")
#fig.savefig(output_dir / "sleep_latency.png", dpi=300, bbox_inches="tight")

plt.show()