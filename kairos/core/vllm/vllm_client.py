from __future__ import annotations

from typing import Any
from dataclasses import dataclass
import httpx
import time
from kairos.logger import init_logger

logger = init_logger(__name__)


@dataclass(slots=True)
class VLLMCompletionResult:
    text: str
    latency_ms: float


class VLLMClient:
    def __init__(self):
        self._client = httpx.AsyncClient()

    async def close(self) -> None:
        await self._client.aclose()

    async def _post(
        self,
        url: str,
        timeout: float = 7.0,
        json_body: dict[str, Any] | None = None,
    ) -> bool:
        try:
            response = await self._client.post(
                url,
                json=json_body,
                timeout=timeout,
            )
            return response.status_code == 200

        except httpx.HTTPError as e:
            logger.warning(f"HTTP request failed for {url}: {e}")
            return False

    '''Sleep-related methods for vLLM'''
    async def sleep_level_1(self, port: int) -> bool:
        return await self._post(
            f"http://localhost:{port}/sleep?level=1",
            timeout=7.0,
        )

    async def sleep_level_2(self, port: int) -> bool:
        return await self._post(
            f"http://localhost:{port}/sleep?level=2",
            timeout=7.0,
        )

    async def wake_up(self, port: int) -> bool:
        return await self._post(
            f"http://localhost:{port}/wake_up",
            timeout=7.0,
        )

    async def wake_up_persistent(self, port: int) -> bool:
        return await self._post(
            f"http://localhost:{port}/wake_up?persistent=true",
            timeout=7.0,
        )

    async def reset_prefix_cache(self, port: int) -> bool:
        return await self._post(
            f"http://localhost:{port}/reset_prefix_cache",
            timeout=7.0,
        )

    async def reload_weights(self, port: int) -> bool:
        return await self._post(
            f"http://localhost:{port}/collective_rpc",
            json_body={"method": "reload_weights"},
            timeout=7.0,
        )

    async def reload_weights_from_prefetch(self, port: int) -> bool:
        return await self._post(
            f"http://localhost:{port}/collective_rpc",
            json_body={"method": "reload_weights_from_prefetch"},
            timeout=7.0,
        )

    async def prefetch(self, port: int) -> bool:
        return await self._post(
            f"http://localhost:{port}/prefetch",
            timeout=10.0,
        )

    async def evict(self, port: int) -> bool:
        return await self._post(
            f"http://localhost:{port}/evict",
            timeout=7.0,
        )

    '''Inference via chat-completion'''

    async def chat_completion(
        self,
        port: int,
        model_id: str,
        instruction: str,
        prompt: str,
        max_tokens: int = 1,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> VLLMCompletionResult:
        url = f"http://localhost:{port}/v1/chat/completions"

        request_body: dict[str, Any] = {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": instruction,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start = time.perf_counter()

        response = await self._client.post(
            url,
            json=request_body,
            timeout=timeout,
        )

        end = time.perf_counter()
        latency_ms = (end - start) * 1000

        response.raise_for_status()

        response_body = response.json()
        content = response_body["choices"][0]["message"]["content"]

        return VLLMCompletionResult(
            text=content.strip().lower(),
            latency_ms=latency_ms,
        )


