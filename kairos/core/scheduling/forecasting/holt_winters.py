from __future__ import annotations

from kairos.core.scheduling.forecasting.holt import HoltForecaster


class HoltWintersForecaster:
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.3,
        season_length: int = 10,
    ):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in the interval (0, 1].")
        if not 0 < beta <= 1:
            raise ValueError("beta must be in the interval (0, 1].")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in the interval (0, 1].")
        if season_length < 2:
            raise ValueError("season_length must be at least 2.")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.season_length = season_length

        self.fallback = HoltForecaster(alpha=alpha, beta=beta)

    def forecast(self, counts: list[int], horizon: int) -> list[float]:
        """
        Forecast future request counts using additive Holt-Winters smoothing.

        counts:
            Historical per-second request counts, ordered oldest -> newest.

        horizon:
            Number of future seconds to forecast.

        alpha:
            Smoothing factor for the level.

        beta:
            Smoothing factor for the trend.

        gamma:
            Smoothing factor for the seasonal component.

        season_length:
            Number of buckets after which the seasonal pattern repeats.
        """
        if horizon <= 0:
            return []

        if not counts:
            return [0.0 for _ in range(horizon)]

        season_length = self.season_length

        # Holt-Winters needs at least two seasons to initialize
        # the level, trend, and seasonal components safely.
        if len(counts) < 2 * season_length:
            return self.fallback.forecast(counts, horizon)

        first_season = counts[:season_length]
        second_season = counts[season_length : 2 * season_length]

        first_avg = sum(first_season) / season_length
        second_avg = sum(second_season) / season_length

        level = float(first_avg)
        trend = float((second_avg - first_avg) / season_length)

        seasonals = [
            float(first_season[i] - first_avg)
            for i in range(season_length)
        ]

        for t in range(season_length, len(counts)):
            count = float(counts[t])
            season_index = t % season_length

            previous_level = level
            previous_season = seasonals[season_index]

            level = (
                self.alpha * (count - previous_season)
                + (1.0 - self.alpha) * (level + trend)
            )

            trend = (
                self.beta * (level - previous_level)
                + (1.0 - self.beta) * trend
            )

            seasonals[season_index] = (
                self.gamma * (count - level)
                + (1.0 - self.gamma) * previous_season
            )

        predictions = []

        for step in range(1, horizon + 1):
            season_index = (len(counts) + step - 1) % season_length
            prediction = level + step * trend + seasonals[season_index]
            predictions.append(max(0.0, prediction))

        return predictions
