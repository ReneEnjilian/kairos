from datasets import load_dataset
from pathlib import Path
import json
import argparse


RAW_DATA_DIR = Path(__file__).resolve().parent
BOOLQ_DIR = RAW_DATA_DIR / "boolq"
LOGIQA_DIR = RAW_DATA_DIR / "logiqa"
MMLU_DIR = RAW_DATA_DIR / "mmlu"
OPENBOOKQA_DIR = RAW_DATA_DIR / "openbookqa"

COLUMNS = ["Instruction", "Prompt", "Answer", "Kairos", "Correct"]

BOOLQ_INSTRUCTION = (
    "Answer the question using only the given passage. "
    "Reply with only 'yes' or 'no'."
)

LOGIQA_INSTRUCTION = (
    "Use the given context to answer the multiple-choice logic question. "
    "Choose the single best option. "
    "Reply with only the option index: "
)

MMLU_INSTRUCTION = (
    "Answer the multiple-choice question. "
    "Choose the single best option. "
    "Reply with only the option index: "
)

OPENBOOKQA_INSTRUCTION = (
    "Answer the multiple-choice question. "
    "Choose the single best option. "
    "Reply with only the answer label: "
)


def format_option_indices(num_options: int) -> str:
    indices = [str(i) for i in range(num_options)]

    if num_options == 1:
        return indices[0]

    return ", ".join(indices[:-1]) + f", or {indices[-1]}"


def format_option_labelset(labels: list) -> str:
    if len(labels) == 1:
        label_options = labels[0]
    elif len(labels) == 2:
        label_options = f"{labels[0]} or {labels[1]}"
    else:
        label_options = ", ".join(labels[:-1]) + f", or {labels[-1]}"

    return label_options


def preprocess_openbookqa():
    OPENBOOKQA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OPENBOOKQA_DIR / "openbookqa.jsonl"
    ds = load_dataset("allenai/openbookqa", name="main", split="test")

    with output_path.open("w", encoding="utf-8") as f:
        for example in ds:
            labels = example["choices"]["label"]
            texts = example["choices"]["text"]
            label_options = format_option_labelset(labels)

            instruction = OPENBOOKQA_INSTRUCTION + label_options + "."

            options_text = "\n".join(
                f"{label}. {text}"
                for label, text in zip(labels, texts)
            )

            row = {
                "instruction": instruction,
                "prompt": (
                    f"Question:\n{example['question_stem']}\n\n"
                    f"Options:\n{options_text}\n\n"
                    "Answer:"
                ),
                "answer": example["answerKey"],
                "kairos": None,
                "correct": None,
                "arrival_timestamp": None,
                "infer_latency_ms": None,
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def preprocess_mmlu():
    MMLU_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MMLU_DIR / "mmlu.jsonl"
    ds = load_dataset("cais/mmlu", name="all", split="test")

    with output_path.open("w", encoding="utf-8") as f:
        for example in ds:
            num_options = len(example["choices"])
            option_indices = format_option_indices(num_options)

            instruction = MMLU_INSTRUCTION + option_indices + "."

            options_text = "\n".join(
                f"{idx}. {option}"
                for idx, option in enumerate(example["choices"])
            )

            row = {
                "instruction": instruction,
                "prompt": (
                    f"Subject:\n{example['subject']}\n\n"
                    f"Question:\n{example['question']}\n\n"
                    f"Options:\n{options_text}\n\n"
                    "Answer:"
                ),
                "answer": str(example["answer"]),
                "kairos": None,
                "correct": None,
                "arrival_timestamp": None,
                "infer_latency_ms": None,
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def preprocess_logiqa():
    LOGIQA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LOGIQA_DIR / "logiqa.jsonl"

    ds = load_dataset("lucasmccabe/logiqa", revision="refs/convert/parquet")
    ds = ds["test"]

    with output_path.open("w", encoding="utf-8") as f:
        for example in ds:
            num_options = len(example["options"])
            option_indices = format_option_indices(num_options)
            instruction = LOGIQA_INSTRUCTION + option_indices + "."
            options_text = "\n".join(
                f"{idx}. {option}"
                for idx, option in enumerate(example["options"])
            )

            row = {
                "instruction": instruction,
                "prompt": (
                    f"Context:\n{example['context']}\n\n"
                    f"Question:\n{example['query']}\n\n"
                    f"Options:\n{options_text}\n\n"
                    "Answer:"
                ),
                "answer": str(example["correct_option"]),
                "kairos": None,
                "correct": None,
                "arrival_timestamp": None,
                "infer_latency_ms": None,
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def preprocess_boolq():
    BOOLQ_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BOOLQ_DIR / "boolq.jsonl"

    ds = load_dataset("google/boolq", split="validation")

    with output_path.open("w", encoding="utf-8") as f:
        for example in ds:
            row = {
                "instruction": BOOLQ_INSTRUCTION,
                "prompt": (
                    f"Passage:\n{example['passage']}\n\n"
                    f"Question: {example['question']}\n\n"
                    "Answer:"
                ),
                "answer": "yes" if example["answer"] else "no",
                "kairos": None,
                "correct": None,
                "arrival_timestamp": None,
                "infer_latency_ms": None,
            }
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["boolq", "logiqa", "mmlu", "openbookqa"],
        default=None,
    )

    args = parser.parse_args()

    preprocessors = {
        "boolq": preprocess_boolq,
        "logiqa": preprocess_logiqa,
        "mmlu": preprocess_mmlu,
        "openbookqa": preprocess_openbookqa,
    }

    if args.dataset is None:
        for preprocess in preprocessors.values():
            preprocess()
    else:
        preprocessors[args.dataset]()


if __name__ == '__main__':
    main()
