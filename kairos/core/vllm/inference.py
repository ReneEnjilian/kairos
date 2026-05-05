from __future__ import annotations

from typing import Any

import httpx


async def infer_chat_completion(
    client: httpx.AsyncClient,
    port: int,
    model_id: str,
    payload: dict[str, Any],
    timeout: float = 30.0,
) -> str:
    url = f"http://localhost:{port}/v1/chat/completions"

    request_body = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": payload["instruction"],
            },
            {
                "role": "user",
                "content": payload["prompt"],
            },
        ],
        "temperature": 0.0,
        "max_tokens": 16,
    }

    response = await client.post(
        url,
        json=request_body,
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()
