from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RATES = [10, 20, 50]

FILES = {
    "Concurrent dispatch": {
        10: "concurrent_dispatch_rate_10.jsonl",
        20: "concurrent_dispatch_rate_20.jsonl",
        50: "concurrent_dispatch_rate_50.jsonl",
    },
    "Sequential dispatch": {
        10: "sequential_dispatch_rate_10.jsonl",
        20: "sequential_dispatch_rate_20.jsonl",
        50: "sequential_dispatch_rate_50.jsonl",
    },
}

SKIP_FIRST_N = 70
ROLLING_WINDOW = 25

OUTPUT_STEM = "dispatch_concurrency_latency"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_latencies_ms(path: Path) -> list[float]:
    latencies = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            latencies.append(float(row["latency_ms"]))

    if len(latencies) <= SKIP_FIRST_N:
        raise ValueError(
            f"{path} contains only {len(latencies)} rows, "
            f"cannot skip the first {SKIP_FIRST_N}."
        )

    return latencies[SKIP_FIRST_N:]


def rolling_median(values: list[float], window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if window <= 1:
        return values

    result = np.empty_like(values)

    half = window // 2

    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        result[i] = np.median(values[lo:hi])

    return result


# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------

results = {}

for label, file_map in FILES.items():
    results[label] = {}

    for rate, filename in file_map.items():
        path = Path(filename)

        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        results[label][rate] = read_latencies_ms(path)


# ---------------------------------------------------------------------------
# Print summary statistics
# ---------------------------------------------------------------------------

print("End-to-end latency statistics after warm-up removal:")
print(f"Skipped first {SKIP_FIRST_N} requests per file.\n")

for rate in RATES:
    print(f"{rate} requests/s")

    for label in FILES.keys():
        latencies = np.array(results[label][rate])

        median = np.median(latencies)
        p95 = np.percentile(latencies, 95)
        mean = np.mean(latencies)

        print(
            f"  {label}: "
            f"mean={mean:.2f} ms, "
            f"median={median:.2f} ms, "
            f"p95={p95:.2f} ms"
        )

    print()


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

styles = {
    "Concurrent dispatch": {
        "color": "#4C78A8",
        "linestyle": "-",
        "marker": "o",
    },
    "Sequential dispatch": {
        "color": "#E45756",
        "linestyle": "--",
        "marker": "s",
    },
}

fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(7.4, 6.2),
    sharex=False,
    sharey=False,
)

for ax, rate in zip(axes, RATES):
    all_latencies_for_rate = []

    for label in FILES.keys():
        latencies = np.asarray(results[label][rate], dtype=float)
        x = np.arange(len(latencies))
        smooth = rolling_median(latencies, ROLLING_WINDOW)

        style = styles[label]
        all_latencies_for_rate.extend(latencies)

        # Raw trace, deliberately light
        ax.plot(
            x,
            latencies,
            linewidth=0.6,
            color=style["color"],
            alpha=0.22,
        )

        # Rolling median, visually dominant
        ax.plot(
            x,
            smooth,
            linewidth=1.9,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=3.5,
            markevery=max(1, len(latencies) // 18),
            label=label,
        )

    ax.set_title(f"{rate} requests/s", pad=6)
    ax.set_ylabel("Latency (ms)")

    ax.grid(axis="y", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(axis="both", length=3, width=0.8)

    # Small y-axis headroom per subplot
    ymin = min(all_latencies_for_rate)
    ymax = max(all_latencies_for_rate)
    padding = 0.08 * (ymax - ymin) if ymax > ymin else 1.0
    ax.set_ylim(max(0, ymin - padding), ymax + padding)

axes[-1].set_xlabel("Request index")


# Shared legend above the plots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
    frameon=False,
    handlelength=2.2,
    columnspacing=1.6,
)


fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

output_dir = Path(__file__).resolve().parent

fig.savefig(
    output_dir / f"{OUTPUT_STEM}.pdf",
    bbox_inches="tight",
)

fig.savefig(
    output_dir / f"{OUTPUT_STEM}.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()