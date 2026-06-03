from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RATE = 2

DATASETS = ["boolq", "logiqa", "mmlu", "openbookqa"]
DATASET_LABELS = ["BoolQ", "LogiQA", "MMLU", "OpenBookQA"]

BASE_MODEL = {
    "folder": "llama-8B-base",
    "label": "Llama-8B",
}

MODELS = [
    {
        "folder": "olmoe-7B-moe",
        "label": "OLMoE-1B-7B",
    },
    {
        "folder": "qwen-7B-ind",
        "label": "Qwen-7B",
    },
    {
        "folder": "qwen-4B-ind",
        "label": "Qwen-4B",
    },
]

OUTPUT_STEM = "relative_accuracy_independent"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_task_accuracy(path: Path) -> float:
    correct = 0
    total = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if "response" in row and "correct" in row["response"]:
                is_correct = row["response"]["correct"]
            else:
                is_correct = row["correct"]

            correct += int(bool(is_correct))
            total += 1

    if total == 0:
        raise ValueError(f"No rows found in {path}")

    return correct / total


def result_file(folder: str, dataset: str, rate: int) -> Path:
    return Path(folder) / f"{folder}-{dataset}-rate-{rate}.jsonl"


# ---------------------------------------------------------------------------
# Compute relative accuracy
# ---------------------------------------------------------------------------

base_accuracies = {}

for dataset in DATASETS:
    path = result_file(BASE_MODEL["folder"], dataset, RATE)

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    base_accuracies[dataset] = read_task_accuracy(path)


plot_labels = [BASE_MODEL["label"]] + [model["label"] for model in MODELS]
relative_accuracies = []

# Base model is the reference.
relative_accuracies.append([1.0 for _ in DATASETS])

for model in MODELS:
    values = []

    for dataset in DATASETS:
        path = result_file(model["folder"], dataset, RATE)

        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        model_accuracy = read_task_accuracy(path)
        base_accuracy = base_accuracies[dataset]

        if base_accuracy == 0:
            raise ValueError(f"Base accuracy is zero for dataset {dataset}")

        relative_accuracy = model_accuracy / base_accuracy
        values.append(relative_accuracy)

    relative_accuracies.append(values)


# ---------------------------------------------------------------------------
# Print values
# ---------------------------------------------------------------------------

print("Base task accuracies:")
for dataset, accuracy in base_accuracies.items():
    print(f"  {dataset}: {accuracy:.4f}")

print("\nRelative accuracies:")
for label, values in zip(plot_labels, relative_accuracies):
    formatted = ", ".join(
        f"{dataset}={value:.4f}" for dataset, value in zip(DATASETS, values)
    )
    print(f"  {label}: {formatted}")


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

x = np.arange(len(DATASETS))
num_bars = len(plot_labels)
bar_width = 0.16

colors = {
    "Llama-8B": "#4C78A8",
    "OLMoE-1B-7B": "#F58518",
    "Qwen-7B": "#54A24B",
    "Qwen-4B": "#B279A2",
}

fig, ax = plt.subplots(figsize=(7.4, 3.4))

for i, (label, values) in enumerate(zip(plot_labels, relative_accuracies)):
    offset = (i - (num_bars - 1) / 2) * bar_width

    ax.bar(
        x + offset,
        values,
        width=bar_width,
        label=label,
        color=colors[label],
    )


# Reference line for the base model
ax.axhline(
    1.0,
    linewidth=0.9,
    linestyle="--",
    color="black",
    alpha=0.45,
)


# Axes
ax.set_xlabel("Task")
ax.set_ylabel("Relative accuracy")

ax.set_xticks(x)
ax.set_xticklabels(DATASET_LABELS)

ax.set_xlim(-0.5, len(DATASETS) - 0.5)

values_array = np.array(relative_accuracies)
ax.set_ylim(0, max(1.2, values_array.max() * 1.08))


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
    ncol=4,
    frameon=False,
    handlelength=1.2,
    columnspacing=1.3,
)


fig.tight_layout()

output_dir = Path(__file__).resolve().parent
fig.savefig(output_dir / f"{OUTPUT_STEM}.pdf", bbox_inches="tight")
fig.savefig(output_dir / f"{OUTPUT_STEM}.png", dpi=300, bbox_inches="tight")

plt.show()