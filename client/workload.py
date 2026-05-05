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

    def _iter_once(self) -> Iterator[JsonObject]:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    def iter_requests(self, max_requests: int | None) -> Iterator[JsonObject]:
        if max_requests is None:
            yield from self._iter_once()
            return

        if max_requests < 0:
            raise ValueError(f"max_requests must be >= 0, got {max_requests}")

        emitted = 0

        while emitted < max_requests:
            yielded_any = False

            for request in self._iter_once():
                yielded_any = True
                yield request

                emitted += 1
                if emitted >= max_requests:
                    break

            if not yielded_any:
                raise ValueError(f"Dataset file is empty: {self.path}")

    def __iter__(self) -> Iterator[JsonObject]:
        return self._iter_once()