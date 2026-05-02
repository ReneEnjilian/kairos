from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

'''reads dataset and produces requests'''

JsonObject = dict[str, Any]


class Workload:
    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.path = self._resolve_dataset_path(dataset)

    def _resolve_dataset_path(self, dataset: str) -> Path:
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / "data" / self.dataset / f"{dataset}.jsonl"

        if not path.is_file():
            raise FileNotFoundError(f"Dataset file {self.dataset}.jsonl does not exist.")

        return path

    def __iter__(self) -> Iterator[JsonObject]:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)