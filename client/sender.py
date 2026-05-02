from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class SenderResult:
    request_index: int
    request: JsonObject
    response: JsonObject
    latency_ms: float


class Sender:
    def __init__(
        self,
        kairos_port: int,
        endpoint: str,
        timeout: float | None = None,
    ) -> None:
        self.url = f"http://localhost:{kairos_port}{endpoint}"
        self.timeout = timeout
        self.client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Sender:
        self.client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        if self.client is not None:
            await self.client.aclose()

    async def send(self, request_index: int, request: JsonObject) -> SenderResult:
        if self.client is None:
            raise RuntimeError("Sender must be used as an async context manager.")

        start = time.perf_counter()
        response = await self.client.post(self.url, json=request)
        end = time.perf_counter()

        response.raise_for_status()

        return SenderResult(
            request_index=request_index,
            request=request,
            response=response.json(),
            latency_ms=(end - start) * 1000,
        )