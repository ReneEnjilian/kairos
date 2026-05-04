from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any


class ResultLogger:
    def __init__(self, filename: str) -> None:
        project_root = Path(__file__).resolve().parent.parent
        self.log_dir = project_root / "logs"
        self.log_path = self.log_dir / filename
        self._lock = asyncio.Lock()
        self._file = None

    def open(self):
        self.log_dir.mkdir(exist_ok=True)
        self._file = self.log_path.open("w",  encoding="utf-8")

    async def write_result(
        self,
        request_id: int,
        latency_ms: float,
        response: dict[str, Any],
    ) -> None:
        if self._file is None:
            raise RuntimeError("Logger is not open.")

        row = {
            "request_id": request_id,
            "latency_ms": latency_ms,
            "response": response,
        }

        line = json.dumps(row, ensure_ascii=False)

        async with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None