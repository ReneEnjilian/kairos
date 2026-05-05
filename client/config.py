from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from pathlib import Path
import yaml
import argparse


@dataclass(frozen=True)
class ClientConfig:
    kairos_port: int
    dataset: str
    keep_logs: bool
    endpoint: str
    max_in_flight: int
    max_requests: int | None
    random_seed: int | None
    workload_pattern: WorkloadPatternConfig


@dataclass(frozen=True)
class PoissonConfig:
    type: Literal["poisson"]
    rate: float


@dataclass(frozen=True)
class OnOffConfig:
    type: Literal["on_off"]
    on_duration_seconds: float
    off_duration_seconds: float
    on_rate: float


@dataclass(frozen=True)
class PeriodicConfig:
    type: Literal["periodic"]
    base_rate: float
    amplitude: float
    period_seconds: float


@dataclass(frozen=True)
class BurstyConfig:
    type: Literal["bursty"]
    base_rate: float
    burst_event_rate: float
    burst_size: int
    burst_spread_seconds: float


@dataclass(frozen=True)
class RampConfig:
    type: Literal["ramp"]
    start_rate: float
    end_rate: float
    ramp_duration_seconds: float


@dataclass(frozen=True)
class StepConfig:
    type: Literal["step"]
    before_rate: float
    after_rate: float
    switch_time_seconds: float


WorkloadPatternConfig = (
    PoissonConfig
    | OnOffConfig
    | PeriodicConfig
    | BurstyConfig
    | RampConfig
    | StepConfig
)


def parse_workload_pattern(raw: dict[str, Any]) -> WorkloadPatternConfig:
    pattern_type = raw["type"]

    match pattern_type:
        case "poisson":
            return PoissonConfig(
                type="poisson",
                rate=float(raw["rate"]),
            )

        case "on_off":
            return OnOffConfig(
                type="on_off",
                on_duration_seconds=float(raw["on_duration_seconds"]),
                off_duration_seconds=float(raw["off_duration_seconds"]),
                on_rate=float(raw["on_rate"]),
            )

        case "periodic":
            return PeriodicConfig(
                type="periodic",
                base_rate=float(raw["base_rate"]),
                amplitude=float(raw["amplitude"]),
                period_seconds=float(raw["period_seconds"]),
            )

        case "bursty":
            return BurstyConfig(
                type="bursty",
                base_rate=float(raw["base_rate"]),
                burst_event_rate=float(raw["burst_event_rate"]),
                burst_size=int(raw["burst_size"]),
                burst_spread_seconds=float(raw["burst_spread_seconds"]),
            )

        case "ramp":
            return RampConfig(
                type="ramp",
                start_rate=float(raw["start_rate"]),
                end_rate=float(raw["end_rate"]),
                ramp_duration_seconds=float(raw["ramp_duration_seconds"]),
            )

        case "step":
            return StepConfig(
                type="step",
                before_rate=float(raw["before_rate"]),
                after_rate=float(raw["after_rate"]),
                switch_time_seconds=float(raw["switch_time_seconds"]),
            )

        case _:
            raise ValueError(f"Unknown workload pattern type: {pattern_type}")


def load_config(config_file: str) -> ClientConfig:
    path = extract_config_file(config_file)
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    client_config = raw.get("client")
    workload_pattern_config = raw.get("workload_pattern")

    return ClientConfig(
        kairos_port=int(client_config["kairos_port"]),
        dataset=str(client_config["dataset"]),
        keep_logs=bool(client_config.get("keep_logs", False)),
        endpoint=str(client_config.get("endpoint", "/infer")),
        max_in_flight=int(client_config.get("max_in_flight", 128)),
        max_requests=client_config.get("max_requests"),
        random_seed=client_config.get("random_seed"),
        workload_pattern=parse_workload_pattern(workload_pattern_config),
    )


def extract_config_file(config_file: str) -> Path:
    path = Path(__file__).resolve().parent.parent / "configs/client" / config_file
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Config file does not exist: {config_file}")
    return path
