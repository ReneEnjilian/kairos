import httpx
from kairos.logger import init_logger

logger = init_logger(__name__)


async def sleep_level_1(port: int) -> bool:
    try:
        response = httpx.post(f"http://localhost:{port}/sleep?level=1", timeout=7.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


async def sleep_level_2(port: int) -> bool:
    try:
        response = httpx.post(f"http://localhost:{port}/sleep?level=2", timeout=7.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


async def wake_up(port: int) -> bool:
    try:
        response = httpx.post(f"http://localhost:{port}/wake_up", timeout=7.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


async def wake_up_persistent(port: int) -> bool:
    try:
        response = httpx.post(f"http://localhost:{port}/wake_up?persistent=true", timeout=7.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


async def reset_prefix_cache(port: int) -> bool:
    try:
        response = httpx.post(f"http://localhost:{port}/reset_prefix_cache", timeout=7.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


async def reload_weights(port: int) -> bool:
    pass


async def reload_weights_from_prefetch(port: int) -> bool:
    pass


async def prefetch(port: int) -> bool:
    try:
        response = httpx.post(f"http://localhost:{port}/prefetch", timeout=10.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


async def evict(port: int) -> bool:
    try:
        response = httpx.post(f"http://localhost:{port}/evict", timeout=7.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False



