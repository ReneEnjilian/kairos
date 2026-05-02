from dataclasses import dataclass
from pathlib import Path
import yaml
import argparse


@dataclass(frozen=True)
class ClientConfig:
    kairos_port: int
    dataset: str
    keep_logs: bool
    request_distribution: str
    endpoint: str
    max_in_flight: int


def load_config(config_file: str) -> ClientConfig:
    path = extract_config_file(config_file)
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    client_config = raw.get("client")

    return ClientConfig(
        kairos_port=client_config["kairos_port"],
        dataset=client_config["dataset"],
        keep_logs=client_config.get("keep_logs", False),
        request_distribution=client_config["distribution"],
        endpoint=client_config["endpoint"],
        max_in_flight=client_config["max_in_flight"],
    )


def extract_config_file(config_file: str) -> Path:
    path = Path(__file__).resolve().parent.parent / "configs/client" / config_file
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Config file does not exist: {config_file}")
    return path
