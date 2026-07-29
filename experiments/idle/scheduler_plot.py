from __future__ import annotations

from pathlib import Path
import bisect
import json

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

METHODS = [
    {
        "folder": "moving-avg",
        "file_stem": "moving_average",
        "label": "Moving avg.",
    },
    {
        "folder": "ewma",
        "file_stem": "ewma",
        "label": "EWMA",
    },
    {
        "folder": "holt",
        "file_stem": "holt",
        "label": "Holt",
    },
    {
        "folder": "holt-winters",
        "file_stem": "holt_winters",
        "label": "Holt-Winters",
    },
]

PATTERN = "on_off"

# Method used for the detailed timeline plot
TIMELINE_METHOD = "ewma"

OUTPUT_DIR = Path("plots")


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


COLORS = {
    "precision": "#4C78A8",
    "recall": "#F58518",
    "f1": "#54A24B",
    "fpr": "#4C78A8",
    "actual": "#4C78A8",
    "predicted": "#E45756",
    "threshold": "#606060",
}


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(axis="both", length=3, width=0.8)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"No rows found in {path}")

    return rows


def load_arrival_times(path: Path) -> list[float]:
    rows = read_jsonl(path)
    times = [float(row["time"]) for row in rows]
    times.sort()
    return times


def load_predictions(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    rows.sort(key=lambda row: float(row["time"]))
    return rows


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def evaluate_predictions(
    predictions: list[dict],
    arrival_times: list[float],
) -> dict:
    """
    Compare each predicted idle window against the actual arrivals.

    For a prediction at time t with window W:
      actual_requests = number of arrivals in [t, t + W)
      actual_idle = actual_requests <= idle_request_threshold
    """

    first_arrival = arrival_times[0]
    last_arrival = arrival_times[-1]

    records = []

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    skipped = 0

    for prediction in predictions:
        t = float(prediction["time"])
        window = float(prediction["window"])
        threshold = float(prediction["idle_request_threshold"])

        # Skip predictions whose future window is not fully covered by the run.
        if t + window > last_arrival:
            skipped += 1
            continue

        left = bisect.bisect_left(arrival_times, t)
        right = bisect.bisect_left(arrival_times, t + window)

        actual_requests = right - left
        actual_idle = actual_requests <= threshold

        predicted_idle = bool(prediction["predicted_idle"])
        predicted_requests = float(prediction["predicted_requests"])

        if predicted_idle and actual_idle:
            tp += 1
        elif predicted_idle and not actual_idle:
            fp += 1
        elif not predicted_idle and actual_idle:
            fn += 1
        else:
            tn += 1

        records.append(
            {
                "time": t,
                "relative_time": t - first_arrival,
                "predicted_idle": predicted_idle,
                "actual_idle": actual_idle,
                "predicted_requests": predicted_requests,
                "actual_requests": actual_requests,
                "window": window,
                "threshold": threshold,
            }
        )

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    false_positive_rate = safe_div(fp, fp + tn)
    accuracy = safe_div(tp + tn, tp + fp + fn + tn)

    return {
        "records": records,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "skipped": skipped,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Load and evaluate all methods
# ---------------------------------------------------------------------------

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    results = {}

    for method in METHODS:
        folder = base_dir / method["folder"]
        file_stem = method["file_stem"]

        arrivals_path = folder / f"scheduler_arrivals_{file_stem}_{PATTERN}.jsonl"
        predictions_path = folder / f"scheduler_predictions_{file_stem}_{PATTERN}.jsonl"

        if not arrivals_path.exists():
            raise FileNotFoundError(f"Missing arrivals file: {arrivals_path}")

        if not predictions_path.exists():
            raise FileNotFoundError(f"Missing predictions file: {predictions_path}")

        arrival_times = load_arrival_times(arrivals_path)
        predictions = load_predictions(predictions_path)

        evaluation = evaluate_predictions(
            predictions=predictions,
            arrival_times=arrival_times,
        )

        results[method["file_stem"]] = {
            "label": method["label"],
            "arrival_times": arrival_times,
            "predictions": predictions,
            "evaluation": evaluation,
        }

    print_summary(results)

    plot_quality_metrics(results, output_dir)
    plot_false_positive_rate(results, output_dir)
    plot_timeline(results, output_dir, method_key=TIMELINE_METHOD)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_summary(results: dict) -> None:
    print("Idle-window forecasting results\n")

    for method_key, data in results.items():
        label = data["label"]
        eval_result = data["evaluation"]

        print(label)
        print(f"  TP: {eval_result['tp']}")
        print(f"  FP: {eval_result['fp']}")
        print(f"  FN: {eval_result['fn']}")
        print(f"  TN: {eval_result['tn']}")
        print(f"  Skipped predictions: {eval_result['skipped']}")
        print(f"  Precision: {eval_result['precision']:.4f}")
        print(f"  Recall: {eval_result['recall']:.4f}")
        print(f"  F1: {eval_result['f1']:.4f}")
        print(f"  False positive rate: {eval_result['false_positive_rate']:.4f}")
        print(f"  Accuracy: {eval_result['accuracy']:.4f}")
        print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_quality_metrics(results: dict, output_dir: Path) -> None:
    labels = [data["label"] for data in results.values()]

    precision = [data["evaluation"]["precision"] for data in results.values()]
    recall = [data["evaluation"]["recall"] for data in results.values()]
    f1 = [data["evaluation"]["f1"] for data in results.values()]

    x = np.arange(len(labels))
    bar_width = 0.24

    fig, ax = plt.subplots(figsize=(7.2, 3.4))

    bars_precision = ax.bar(
        x - bar_width,
        precision,
        width=bar_width,
        label="Precision",
        color=COLORS["precision"],
    )

    bars_recall = ax.bar(
        x,
        recall,
        width=bar_width,
        label="Recall",
        color=COLORS["recall"],
    )

    bars_f1 = ax.bar(
        x + bar_width,
        f1,
        width=bar_width,
        label="F1",
        color=COLORS["f1"],
    )

    ax.set_xlabel("Forecasting method")
    ax.set_ylabel("Score")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylim(0.0, 1.08)

    style_axis(ax)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.4,
    )

    for bars in [bars_precision, bars_recall, bars_f1]:
        ax.bar_label(
            bars,
            labels=[f"{bar.get_height():.2f}" for bar in bars],
            padding=2,
            fontsize=8,
        )

    fig.tight_layout()

    fig.savefig(output_dir / "idle_prediction_quality_on_off.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "idle_prediction_quality_on_off.png", dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_false_positive_rate(results: dict, output_dir: Path) -> None:
    labels = [data["label"] for data in results.values()]
    fpr = [data["evaluation"]["false_positive_rate"] for data in results.values()]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6.4, 3.3))

    bars = ax.bar(
        x,
        fpr,
        width=0.52,
        color=COLORS["fpr"],
    )

    ax.set_xlabel("Forecasting method")
    ax.set_ylabel("False positive rate")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylim(0.0, max(0.05, max(fpr) * 1.25))

    style_axis(ax)

    ax.bar_label(
        bars,
        labels=[f"{bar.get_height():.2f}" for bar in bars],
        padding=2,
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(output_dir / "idle_false_positive_rate_on_off.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "idle_false_positive_rate_on_off.png", dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_timeline(results: dict, output_dir: Path, method_key: str) -> None:
    if method_key not in results:
        raise ValueError(f"Unknown timeline method: {method_key}")

    data = results[method_key]
    label = data["label"]
    records = data["evaluation"]["records"]

    if not records:
        raise ValueError(f"No prediction records available for {label}")

    x = np.array([record["relative_time"] for record in records])
    predicted = np.array([record["predicted_requests"] for record in records])
    actual = np.array([record["actual_requests"] for record in records])

    threshold = records[0]["threshold"]

    fig, ax = plt.subplots(figsize=(7.6, 3.4))

    ax.plot(
        x,
        actual,
        linewidth=1.7,
        label="Actual requests",
        color=COLORS["actual"],
    )

    ax.plot(
        x,
        predicted,
        linewidth=1.7,
        linestyle="--",
        label="Predicted requests",
        color=COLORS["predicted"],
    )

    ax.axhline(
        threshold,
        linewidth=1.0,
        linestyle=":",
        label="Idle threshold",
        color=COLORS["threshold"],
    )

    ax.set_xlabel("Experiment time (s)")
    ax.set_ylabel("Requests in prediction window")

    ax.set_xlim(min(x), max(x))

    ymax = max(actual.max(), predicted.max(), threshold)
    ax.set_ylim(0, ymax * 1.12)

    style_axis(ax)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.4,
    )

    fig.tight_layout()

    fig.savefig(output_dir / f"idle_prediction_timeline_{method_key}_on_off.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"idle_prediction_timeline_{method_key}_on_off.png", dpi=300, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    main()