from __future__ import annotations

import argparse
import asyncio

from client.config import load_config
from client.sender import Sender, SenderResult
from client.workload import Workload
from client.logs import ResultLogger
from client.distribution import create_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    return parser.parse_args()


async def send_and_record(
    sender: Sender,
    request_index: int,
    request: dict,
    semaphore: asyncio.Semaphore,
    logger: ResultLogger | None,
) -> None:
    try:
        result: SenderResult = await sender.send(request_index, request)

        if logger is not None:
            await logger.write_result(
                request_id=result.request_index,
                latency_ms=result.latency_ms,
                response=result.response,
            )

    except Exception as e:
        print(f"request={request_index} failed: {e}")

    finally:
        semaphore.release()


async def run() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    workload = Workload(cfg.dataset)
    semaphore = asyncio.Semaphore(cfg.max_in_flight)

    distribution = create_distribution(
        config=cfg.workload_pattern,
        random_seed=cfg.random_seed,
    )

    logger = ResultLogger() if cfg.keep_logs else None

    if logger is not None:
        logger.open()

    try:
        async with Sender(
            kairos_port=cfg.kairos_port,
            endpoint=cfg.endpoint,
        ) as sender:
            async with asyncio.TaskGroup() as task_group:
                first_request = True

                for request_index, request in enumerate(workload):
                    if cfg.max_requests is not None and request_index >= cfg.max_requests:
                        break

                    if not first_request:
                        await distribution.wait_next()

                    first_request = False

                    await semaphore.acquire()

                    task_group.create_task(
                        send_and_record(
                            sender=sender,
                            request_index=request_index,
                            request=request,
                            semaphore=semaphore,
                            logger=logger,
                        )
                    )

                    # Later:
                    # await distribution.wait_next()
    finally:
        if logger is not None:
            logger.close()


def main() -> None:
    asyncio.run(run())


if __name__ == '__main__':
    main()
