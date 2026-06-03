from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

model_sizes = ["1B-7B"]

wake_up_from_cpu = [1.159518]
wake_up_persistent = [1.145447]
wake_up_from_disk = [3.49401]
prefetch = [2.168110]
wake_up_from_prefetch = [1.376007]


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
bar_width = 0.16

colors = {
    "cpu": "#4C78A8",
    "cpu_persistent": "#F58518",
    "disk": "#54A24B",
    "prefetch": "#E45756",
    "prefetched_cpu": "#B279A2",
}

#fig, ax = plt.subplots(figsize=(7.4, 3.4))
fig, ax = plt.subplots(figsize=(4.6, 3.4))

bars_cpu = ax.bar(
    x - 1.5 * bar_width,
    wake_up_from_cpu,
    width=bar_width,
    label="CPU",
    color=colors["cpu"],
)

bars_cpu_persistent = ax.bar(
    x - 0.5 * bar_width,
    wake_up_persistent,
    width=bar_width,
    label="CPU (persistent)",
    color=colors["cpu_persistent"],
)

bars_disk = ax.bar(
    x + 0.5 * bar_width,
    wake_up_from_disk,
    width=bar_width,
    label="Disk",
    color=colors["disk"],
)

bars_prefetch = ax.bar(
    x + 1.5 * bar_width,
    prefetch,
    width=bar_width,
    label="Prefetch",
    color=colors["prefetch"],
)

bars_prefetched_cpu = ax.bar(
    x + 1.5 * bar_width,
    wake_up_from_prefetch,
    width=bar_width,
    bottom=prefetch,
    label="CPU (prefetched)",
    color=colors["prefetched_cpu"],
)


# Axes
ax.set_xlabel("Model size")
ax.set_ylabel("Wake-up latency (s)")   # or: "Wake-up latency (s)"

ax.set_xticks(x)
ax.set_xticklabels(model_sizes)

#ax.set_xlim(-0.5, len(model_sizes) - 0.5)
ax.set_xlim(-0.48, 0.48)
total_prefetch_path = np.array(prefetch) + np.array(wake_up_from_prefetch)
ymax = max(
    max(wake_up_from_cpu),
    max(wake_up_persistent),
    max(wake_up_from_disk),
    max(total_prefetch_path),
)
ax.set_ylim(0, ymax * 1.12)


# Grid and spines
ax.grid(axis="y", linewidth=0.6, alpha=0.25)
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
    ncol=5,
    frameon=False,
    handlelength=1.2,
    columnspacing=1.2,
)


# Value labels
for bars in [bars_cpu, bars_cpu_persistent, bars_disk, bars_prefetch, bars_prefetched_cpu]:
    ax.bar_label(
        bars,
        labels=[f"{bar.get_height():.2f}" for bar in bars],
        padding=2,
        fontsize=8,
    )


fig.tight_layout()

output_dir = Path(__file__).resolve().parent
fig.savefig(output_dir / "wake_up_latency.pdf", bbox_inches="tight")
#fig.savefig(output_dir / "wake_up_latency.png", dpi=300, bbox_inches="tight")

plt.show()