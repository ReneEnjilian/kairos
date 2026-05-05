from __future__ import annotations

import asyncio
import math
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

from client.config import (
    BurstyConfig,
    OnOffConfig,
    PeriodicConfig,
    PoissonConfig,
    RampConfig,
    StepConfig,
    WorkloadPatternConfig,
)


class RequestDistribution(ABC):
    @abstractmethod
    async def wait_next(self) -> None:
        ...


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _validate_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


class PoissonDistribution(RequestDistribution):
    def __init__(self, rate: float, rng: random.Random) -> None:
        _validate_positive("rate", rate)
        self.rate = rate
        self.rng = rng

    async def wait_next(self) -> None:
        delay = self.rng.expovariate(self.rate)
        await asyncio.sleep(delay)


class OnOffDistribution(RequestDistribution):
    def __init__(
        self,
        on_duration_seconds: float,
        off_duration_seconds: float,
        on_rate: float,
        rng: random.Random,
    ) -> None:
        _validate_positive("on_duration_seconds", on_duration_seconds)
        _validate_non_negative("off_duration_seconds", off_duration_seconds)
        _validate_positive("on_rate", on_rate)

        self.on_duration_seconds = on_duration_seconds
        self.off_duration_seconds = off_duration_seconds
        self.cycle_seconds = on_duration_seconds + off_duration_seconds
        self.on_rate = on_rate
        self.rng = rng
        self.start_time = time.monotonic()

    async def wait_next(self) -> None:
        while True:
            now = time.monotonic()
            elapsed = now - self.start_time
            cycle_pos = elapsed % self.cycle_seconds

            if cycle_pos >= self.on_duration_seconds:
                time_until_next_on = self.cycle_seconds - cycle_pos
                await asyncio.sleep(time_until_next_on)
                continue

            delay = self.rng.expovariate(self.on_rate)

            if cycle_pos + delay < self.on_duration_seconds:
                await asyncio.sleep(delay)
                return

            remaining_on = self.on_duration_seconds - cycle_pos
            delay_to_next_on = remaining_on + self.off_duration_seconds
            await asyncio.sleep(delay_to_next_on)
            return


class NonHomogeneousPoissonDistribution(RequestDistribution):
    """
    Generic time-varying Poisson arrival process using thinning.

    rate_fn(t) gives the expected request rate at elapsed time t.
    max_rate must be an upper bound on rate_fn(t).
    """

    def __init__(
        self,
        rate_fn: Callable[[float], float],
        max_rate: float,
        rng: random.Random,
    ) -> None:
        _validate_positive("max_rate", max_rate)

        self.rate_fn = rate_fn
        self.max_rate = max_rate
        self.rng = rng
        self.start_time = time.monotonic()

    async def wait_next(self) -> None:
        now_elapsed = time.monotonic() - self.start_time
        candidate_time = now_elapsed

        while True:
            candidate_time += self.rng.expovariate(self.max_rate)

            rate = self.rate_fn(candidate_time)
            _validate_non_negative("rate_fn(t)", rate)

            accept_probability = rate / self.max_rate

            if accept_probability > 1.0:
                raise ValueError(
                    f"rate_fn(t) returned {rate}, which exceeds max_rate={self.max_rate}"
                )

            if self.rng.random() <= accept_probability:
                delay = max(0.0, candidate_time - now_elapsed)
                await asyncio.sleep(delay)
                return


class PeriodicDistribution(NonHomogeneousPoissonDistribution):
    def __init__(
        self,
        base_rate: float,
        amplitude: float,
        period_seconds: float,
        rng: random.Random,
    ) -> None:
        _validate_positive("base_rate", base_rate)
        _validate_non_negative("amplitude", amplitude)
        _validate_positive("period_seconds", period_seconds)

        if base_rate - amplitude < 0:
            raise ValueError(
                "periodic workload requires base_rate - amplitude >= 0"
            )

        def rate_fn(t: float) -> float:
            return base_rate + amplitude * math.sin(
                2.0 * math.pi * t / period_seconds
            )

        super().__init__(
            rate_fn=rate_fn,
            max_rate=base_rate + amplitude,
            rng=rng,
        )


class RampDistribution(NonHomogeneousPoissonDistribution):
    def __init__(
        self,
        start_rate: float,
        end_rate: float,
        ramp_duration_seconds: float,
        rng: random.Random,
    ) -> None:
        _validate_non_negative("start_rate", start_rate)
        _validate_non_negative("end_rate", end_rate)
        _validate_positive("ramp_duration_seconds", ramp_duration_seconds)

        max_rate = max(start_rate, end_rate)
        _validate_positive("max(start_rate, end_rate)", max_rate)

        def rate_fn(t: float) -> float:
            if t >= ramp_duration_seconds:
                return end_rate

            progress = t / ramp_duration_seconds
            return start_rate + progress * (end_rate - start_rate)

        super().__init__(
            rate_fn=rate_fn,
            max_rate=max_rate,
            rng=rng,
        )


class StepDistribution(NonHomogeneousPoissonDistribution):
    def __init__(
        self,
        before_rate: float,
        after_rate: float,
        switch_time_seconds: float,
        rng: random.Random,
    ) -> None:
        _validate_non_negative("before_rate", before_rate)
        _validate_non_negative("after_rate", after_rate)
        _validate_positive("switch_time_seconds", switch_time_seconds)

        max_rate = max(before_rate, after_rate)
        _validate_positive("max(before_rate, after_rate)", max_rate)

        def rate_fn(t: float) -> float:
            if t < switch_time_seconds:
                return before_rate
            return after_rate

        super().__init__(
            rate_fn=rate_fn,
            max_rate=max_rate,
            rng=rng,
        )


class BurstyDistribution(RequestDistribution):
    def __init__(
        self,
        base_rate: float,
        burst_event_rate: float,
        burst_size: int,
        burst_spread_seconds: float,
        rng: random.Random,
    ) -> None:
        _validate_non_negative("base_rate", base_rate)
        _validate_non_negative("burst_event_rate", burst_event_rate)

        if base_rate == 0 and burst_event_rate == 0:
            raise ValueError("At least one of base_rate or burst_event_rate must be > 0")

        if burst_size <= 0:
            raise ValueError(f"burst_size must be > 0, got {burst_size}")

        _validate_non_negative("burst_spread_seconds", burst_spread_seconds)

        self.base_rate = base_rate
        self.burst_event_rate = burst_event_rate
        self.burst_size = burst_size
        self.burst_spread_seconds = burst_spread_seconds
        self.rng = rng
        self.start_time = time.monotonic()

        self.next_base_arrival = self._sample_next_base_arrival(0.0)
        self.next_burst_event = self._sample_next_burst_event(0.0)
        self.pending_burst_arrivals: list[float] = []

    def _sample_next_base_arrival(self, after_time: float) -> float:
        if self.base_rate == 0:
            return math.inf
        return after_time + self.rng.expovariate(self.base_rate)

    def _sample_next_burst_event(self, after_time: float) -> float:
        if self.burst_event_rate == 0:
            return math.inf
        return after_time + self.rng.expovariate(self.burst_event_rate)

    def _create_burst_arrivals(self, burst_start_time: float) -> None:
        offsets = [
            self.rng.uniform(0.0, self.burst_spread_seconds)
            for _ in range(self.burst_size)
        ]

        for offset in sorted(offsets):
            self.pending_burst_arrivals.append(burst_start_time + offset)

        self.pending_burst_arrivals.sort()

    async def wait_next(self) -> None:
        while True:
            now_elapsed = time.monotonic() - self.start_time

            next_burst_arrival = (
                self.pending_burst_arrivals[0]
                if self.pending_burst_arrivals
                else math.inf
            )

            next_time = min(
                self.next_base_arrival,
                self.next_burst_event,
                next_burst_arrival,
            )

            if next_time == self.next_burst_event:
                burst_start = self.next_burst_event
                self._create_burst_arrivals(burst_start)
                self.next_burst_event = self._sample_next_burst_event(burst_start)
                continue

            if next_time == self.next_base_arrival:
                self.next_base_arrival = self._sample_next_base_arrival(
                    self.next_base_arrival
                )

            elif next_time == next_burst_arrival:
                self.pending_burst_arrivals.pop(0)

            delay = max(0.0, next_time - now_elapsed)
            await asyncio.sleep(delay)
            return


def create_distribution(
    config: WorkloadPatternConfig,
    random_seed: int | None,
) -> RequestDistribution:
    rng = random.Random(random_seed)

    match config:
        case PoissonConfig():
            return PoissonDistribution(
                rate=config.rate,
                rng=rng,
            )

        case OnOffConfig():
            return OnOffDistribution(
                on_duration_seconds=config.on_duration_seconds,
                off_duration_seconds=config.off_duration_seconds,
                on_rate=config.on_rate,
                rng=rng,
            )

        case PeriodicConfig():
            return PeriodicDistribution(
                base_rate=config.base_rate,
                amplitude=config.amplitude,
                period_seconds=config.period_seconds,
                rng=rng,
            )

        case BurstyConfig():
            return BurstyDistribution(
                base_rate=config.base_rate,
                burst_event_rate=config.burst_event_rate,
                burst_size=config.burst_size,
                burst_spread_seconds=config.burst_spread_seconds,
                rng=rng,
            )

        case RampConfig():
            return RampDistribution(
                start_rate=config.start_rate,
                end_rate=config.end_rate,
                ramp_duration_seconds=config.ramp_duration_seconds,
                rng=rng,
            )

        case StepConfig():
            return StepDistribution(
                before_rate=config.before_rate,
                after_rate=config.after_rate,
                switch_time_seconds=config.switch_time_seconds,
                rng=rng,
            )

        case _:
            raise ValueError(f"Unsupported workload pattern config: {config}")