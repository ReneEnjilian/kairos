from __future__ import annotations


class EWMAForecaster:
    def __init__(self, alpha: float = 0.5):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in the interval (0, 1].")

        self.alpha = alpha

    def forecast(self, counts: list[int], horizon: int) -> list[float]:
        """
        Forecast future request counts using exponential smoothing.

        counts:
            Historical per-second request counts, ordered oldest -> newest.

        horizon:
            Number of future seconds to forecast.

        alpha:
            Smoothing factor. Larger values react more strongly to recent load.
        """
        if horizon <= 0:
            return []

        if not counts:
            return [0.0 for _ in range(horizon)]

        level = float(counts[0])

        for count in counts[1:]:
            level = self.alpha * count + (1.0 - self.alpha) * level

        return [level for _ in range(horizon)]