from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RATES = [2, 5, 10]

DATASETS = ["boolq", "logiqa", "mmlu", "openbookqa"]
DATASET_LABELS = {
    "boolq": "BoolQ",
    "logiqa": "LogiQA",
    "mmlu": "MMLU",
    "openbookqa": "OpenBookQA",
}

MODELS = [
    {
        "folder": "llama-8B-base",
        "label": "BF16",
    },
    {
        "folder": "llama-8B-FP8",
        "label": "FP8",
    },
    {
        "folder": "llama-8B-INT8",
        "label": "INT8",
    },
    {
        "folder": "llama-8B-INT4",
        "label": "INT4",
    },
    {
        "folder": "llama-8B-NF4",
        "label": "NF4",
    },
]

OUTPUT_STEM = "p95_latency_quantized"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def result_file(folder: str, dataset: str, rate: int) -> Path:
    return Path(folder) / f"{folder}-{dataset}-rate-{rate}.jsonl"


def read_infer_latencies_ms(path: Path) -> list[float]:
    latencies = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if "response" in row and "infer_latency_ms" in row["response"]:
                latency = row["response"]["infer_latency_ms"]
            else:
                latency = row["infer_latency_ms"]

            latencies.append(float(latency))

    if not latencies:
        raise ValueError(f"No latency values found in {path}")

    return latencies


def p95_latency_ms(path: Path) -> float:
    latencies = read_infer_latencies_ms(path)
    return float(np.percentile(latencies, 95))


# ---------------------------------------------------------------------------
# Compute p95 latency
# ---------------------------------------------------------------------------

results = {}

for dataset in DATASETS:
    results[dataset] = {}

    for model in MODELS:
        model_label = model["label"]
        model_folder = model["folder"]

        values = []

        for rate in RATES:
            path = result_file(model_folder, dataset, rate)

            if not path.exists():
                raise FileNotFoundError(f"Missing file: {path}")

            values.append(p95_latency_ms(path))

        results[dataset][model_label] = values


# ---------------------------------------------------------------------------
# Print values
# ---------------------------------------------------------------------------

print("p95 inference latency in ms:")
for dataset in DATASETS:
    print(f"\n{DATASET_LABELS[dataset]}")
    for model in MODELS:
        label = model["label"]
        values = results[dataset][label]
        formatted = ", ".join(
            f"{rate} req/s = {value:.2f} ms"
            for rate, value in zip(RATES, values)
        )
        print(f"  {label}: {formatted}")


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

colors = {
    "BF16": "#4C78A8",
    "FP8": "#F58518",
    "INT8": "#54A24B",
    "INT4": "#B279A2",
    "NF4": "#E45756",
}

markers = {
    "BF16": "o",
    "FP8": "s",
    "INT8": "^",
    "INT4": "D",
    "NF4": "P",
}

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(7.4, 5.1),
    sharex=True,
    sharey=True,
)

axes = axes.flatten()

for ax, dataset in zip(axes, DATASETS):
    for model in MODELS:
        label = model["label"]
        values = results[dataset][label]

        ax.plot(
            RATES,
            values,
            marker=markers[label],
            linewidth=1.7,
            markersize=4.5,
            label=label,
            color=colors[label],
        )

    ax.set_title(DATASET_LABELS[dataset], pad=6)

    ax.set_xticks(RATES)
    ax.set_xlim(min(RATES) - 0.4, max(RATES) + 0.4)

    ax.grid(axis="y", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(axis="both", length=3, width=0.8)


# Shared axis labels
fig.supxlabel("Request rate (requests/s)", y=0.04)
fig.supylabel("p95 inference latency (ms)", x=0.02)


# Shared legend above the plots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=5,
    frameon=False,
    handlelength=1.6,
    columnspacing=1.3,
)


fig.tight_layout(rect=(0.04, 0.05, 1.0, 0.93))

output_dir = Path(__file__).resolve().parent
fig.savefig(output_dir / f"{OUTPUT_STEM}.pdf", bbox_inches="tight")
fig.savefig(output_dir / f"{OUTPUT_STEM}.png", dpi=300, bbox_inches="tight")

plt.show()