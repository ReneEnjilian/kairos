import argparse
from pathlib import Path
from client.config import load_config
from datasets import load_dataset

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config_file)
    print(cfg)
    #ds = load_dataset("google/boolq")
    #ds = load_dataset("lucasmccabe/logiqa", revision="refs/convert/parquet")
    #ds = load_dataset("cais/mmlu", "all")
    ds = load_dataset("allenai/openbookqa", "main")
    #print(ds.keys())
    first_row = ds["train"][0]
    #print(len(ds['validation']))
    print(first_row)


if __name__ == '__main__':
    main()
