import httpx
from kairos.logger import init_logger

logger = init_logger(__name__)


async def _post(url: str, timeout: float = 7.0, json: dict | None = None) -> bool:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=json, timeout=timeout)
        return response.status_code == 200
    except httpx.HTTPError as e:
        logger.warning(f"HTTP request failed for {url}: {e}")
        return False


async def sleep_level_1(port: int) -> bool:
    return await _post(f"http://localhost:{port}/sleep?level=1", timeout=7.0)


async def sleep_level_2(port: int) -> bool:
    return await _post(f"http://localhost:{port}/sleep?level=2", timeout=7.0)


async def wake_up(port: int) -> bool:
    return await _post(f"http://localhost:{port}/wake_up", timeout=7.0)


async def wake_up_persistent(port: int) -> bool:
    return await _post(
        f"http://localhost:{port}/wake_up?persistent=true",
        timeout=7.0,
    )


async def reset_prefix_cache(port: int) -> bool:
    return await _post(
        f"http://localhost:{port}/reset_prefix_cache",
        timeout=7.0,
    )


async def reload_weights(port: int) -> bool:
    return await _post(
        f"http://localhost:{port}/collective_rpc",
        json={"method": "reload_weights"},
        timeout=7.0,
    )


async def reload_weights_from_prefetch(port: int) -> bool:
    return await _post(
        f"http://localhost:{port}/collective_rpc",
        json={"method": "reload_weights_from_prefetch"},
        timeout=7.0,
    )


async def prefetch(port: int) -> bool:
    return await _post(
        f"http://localhost:{port}/prefetch",
        timeout=10.0,
    )


async def evict(port: int) -> bool:
    return await _post(
        f"http://localhost:{port}/evict",
        timeout=7.0,
    )