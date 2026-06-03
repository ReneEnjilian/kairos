from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

model_sizes = ["1B-7B"]

# GPU memory in MiB
direct_evict_memory_mib = [686.0]
indirect_path_memory_mib = [18786.0]

# Convert MiB to GiB
direct_evict_memory_gib = np.array(direct_evict_memory_mib) / 1024
indirect_path_memory_gib = np.array(indirect_path_memory_mib) / 1024


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
    "direct": "#4C78A8",
    "indirect": "#F58518",
}

#fig, ax = plt.subplots(figsize=(6.2, 3.3))
fig, ax = plt.subplots(figsize=(3.8, 3.3))

bars_direct = ax.bar(
    x - bar_width / 2,
    direct_evict_memory_gib,
    width=bar_width,
    label="Direct evict",
    color=colors["direct"],
)

bars_indirect = ax.bar(
    x + bar_width / 2,
    indirect_path_memory_gib,
    width=bar_width,
    label="Wake-up + L2 sleep",
    color=colors["indirect"],
)


# Axes
ax.set_xlabel("Model size")
ax.set_ylabel("Peak GPU memory (GiB)")

ax.set_xticks(x)
ax.set_xticklabels(model_sizes)

#ax.set_xlim(-0.5, len(model_sizes) - 0.5)
ax.set_xlim(-0.42, 0.42)
ax.set_ylim(0, max(indirect_path_memory_gib) * 1.14)


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
    ncol=2,
    frameon=False,
    handlelength=1.2,
    columnspacing=1.4,
)


# Value labels
for bars in [bars_direct, bars_indirect]:
    ax.bar_label(
        bars,
        labels=[f"{bar.get_height():.1f}" for bar in bars],
        padding=2,
        fontsize=8,
    )


fig.tight_layout()

output_dir = Path(__file__).resolve().parent
fig.savefig(output_dir / "evict_transient_gpu_memory.pdf", bbox_inches="tight")
#fig.savefig(output_dir / "evict_transient_gpu_memory.png", dpi=300, bbox_inches="tight")

plt.show()