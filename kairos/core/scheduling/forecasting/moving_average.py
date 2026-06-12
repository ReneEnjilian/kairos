from __future__ import annotations


class MovingAverageForecaster:
    def __init__(self, lookback: int | None = None):
        if lookback is not None and lookback <= 0:
            raise ValueError("lookback must be positive or None.")

        self.lookback = lookback

    def forecast(self, counts: list[int], horizon: int) -> list[float]:
        """
        Forecast future request counts using a simple moving average.

        counts:
            Historical per-second request counts, ordered oldest -> newest.

        horizon:
            Number of future seconds to forecast.

        The moving average is computed over the last self.lookback seconds.
        If self.lookback is None, the full counts list is used.
        """
        if horizon <= 0:
            return []

        if not counts:
            return [0.0 for _ in range(horizon)]

        lookback = self.lookback or len(counts)
        recent_counts = counts[-lookback:]

        avg = sum(recent_counts) / len(recent_counts)

        return [avg for _ in range(horizon)]