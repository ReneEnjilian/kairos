from __future__ import annotations


class HoltForecaster:
    def __init__(self, alpha: float = 0.5, beta: float = 0.3):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in the interval (0, 1].")
        if not 0 < beta <= 1:
            raise ValueError("beta must be in the interval (0, 1].")

        self.alpha = alpha
        self.beta = beta

    def forecast(self, counts: list[int], horizon: int) -> list[float]:
        """
        Forecast future request counts using Holt's linear trend method.

        counts:
            Historical per-second request counts, ordered oldest -> newest.

        horizon:
            Number of future seconds to forecast.

        alpha:
            Smoothing factor for the level.

        beta:
            Smoothing factor for the trend.
        """
        if horizon <= 0:
            return []

        if not counts:
            return [0.0 for _ in range(horizon)]

        if len(counts) == 1:
            return [float(counts[0]) for _ in range(horizon)]

        level = float(counts[0])
        trend = float(counts[1] - counts[0])

        for count in counts[1:]:
            previous_level = level

            level = self.alpha * count + (1.0 - self.alpha) * (level + trend)
            trend = self.beta * (level - previous_level) + (1.0 - self.beta) * trend

        predictions = []

        for step in range(1, horizon + 1):
            prediction = level + step * trend
            predictions.append(max(0.0, prediction))

        return predictions