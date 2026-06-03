import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = "client_results.jsonl"
OUTPUT_FILE = "arrival_rate_poisson.png"


def extract_arrival_timestamp(row: dict) -> float | None:
    """
    Supports both:
    - row["arrival_timestamp"]
    - row["response"]["arrival_timestamp"]
    """
    ts = row.get("arrival_timestamp")

    if ts is None and isinstance(row.get("response"), dict):
        ts = row["response"].get("arrival_timestamp")

    if ts is None:
        return None

    return float(ts)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    input_path = script_dir / INPUT_FILE
    output_path = script_dir / OUTPUT_FILE

    timestamps = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            ts = extract_arrival_timestamp(row)

            if ts is not None:
                timestamps.append(ts)

    if not timestamps:
        raise ValueError("No valid arrival_timestamp values found.")

    df = pd.DataFrame({"arrival_timestamp": timestamps})
    df = df.sort_values("arrival_timestamp").reset_index(drop=True)

    start_time = df["arrival_timestamp"].iloc[0]
    df["elapsed_s"] = df["arrival_timestamp"] - start_time

    # One-second buckets: [0, 1), [1, 2), [2, 3), ...
    df["second"] = df["elapsed_s"].astype(int)

    requests_per_second = (
        df.groupby("second")
        .size()
        .rename("req_per_s")
        .reset_index()
    )

    # Fill missing seconds with 0, so idle periods are visible.
    all_seconds = pd.DataFrame({
        "second": range(
            int(requests_per_second["second"].min()),
            int(requests_per_second["second"].max()) + 1,
        )
    })

    requests_per_second = all_seconds.merge(
        requests_per_second,
        on="second",
        how="left",
    ).fillna({"req_per_s": 0})

    requests_per_second["req_per_s"] = requests_per_second["req_per_s"].astype(int)

    plt.figure(figsize=(8, 3.5))

    plt.bar(
        requests_per_second["second"],
        requests_per_second["req_per_s"],
        width=0.85,
    )

    plt.xlabel("Elapsed time (s)", fontsize=12)
    plt.ylabel("Request rate (req/s)", fontsize=12)
    plt.tick_params(axis="both", labelsize=10)

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig("arrival_rate_barplot.pdf", bbox_inches="tight")
    #plt.savefig("arrival_rate_barplot.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved plot to: {output_path}")
    print(f"Number of requests: {len(df)}")
    print(f"Duration: {df['elapsed_s'].max():.2f} seconds")
    print(f"Mean request rate: {len(df) / df['elapsed_s'].max():.2f} req/s")
    print(f"Max one-second bucket: {requests_per_second['req_per_s'].max()} req/s")


if __name__ == "__main__":
    main()