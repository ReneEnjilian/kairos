from __future__ import annotations
import argparse
from pathlib import Path
from typing import Sequence
import subprocess
import sys
import os
from kairos.api.server import Server
from kairos.logger import init_logger

logger = init_logger(__name__)


def existing_file(path_str: str) -> Path:
    path = Path(__file__).resolve().parent.parent / "configs" / path_str
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Config file does not exist: {path_str}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kairos")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument(
        "config",
        type=existing_file,
        help="Path to the Kairos YAML config file.",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to.",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to.",)

    # In the future if another command is added just do
    # train_parser = subparses.add_parser("train")
    # train_parser.add_argument(....)

    #serve_parser.set_defaults(handler=test)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        core_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "kairos.core.core",
                str(args.config),
                str(args.port),
            ]
        )
        try:
            server = Server(host=args.host, port=args.port)
            server.run()
        finally:
            core_process.terminate()
            core_process.wait()









