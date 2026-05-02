import argparse
from pathlib import Path
from client.config import load_config
from datasets import load_dataset

from client.workload import Workload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config_file)
    #print(cfg)
    workload = Workload(cfg.dataset)

    for request in workload:
        print(request)
        break


if __name__ == '__main__':
    main()
