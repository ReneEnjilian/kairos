from __future__ import annotations


def forecast(
    counts: list[int],
    horizon: int,
    lookback: int | None = None,
) -> list[float]:
    """
    Forecast future request counts using a simple moving average.

    counts:
        Historical per-second request counts, ordered oldest -> newest.

    horizon:
        Number of future seconds to forecast.

    lookback:
        Number of recent seconds used for the average.
        If None, use the full counts list.
    """
    if horizon <= 0:
        return []

    if not counts:
        return [0.0 for _ in range(horizon)]

    if lookback is None:
        lookback = len(counts)

    recent_counts = counts[-lookback:]

    if not recent_counts:
        return [0.0 for _ in range(horizon)]

    avg = sum(recent_counts) / len(recent_counts)

    return [avg for _ in range(horizon)]



'''
Just count requests in the last N seconds. Even simpler than EWMA.
Good baseline. Weakness: all points weighted equally, slower to adapt.
'''