from __future__ import annotations

import argparse
import asyncio

from client.config import load_config
from client.sender import Sender, SenderResult
from client.workload import Workload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    return parser.parse_args()


async def send_and_record(
    sender: Sender,
    request_index: int,
    request: dict,
    semaphore: asyncio.Semaphore,
    keep_logs: bool,
) -> None:
    try:
        result: SenderResult = await sender.send(request_index, request)

        if keep_logs:
            print(
                f"request={result.request_index} "
                f"latency_ms={result.latency_ms:.2f} "
                f"response={result.response}"
            )

    except Exception as e:
        if keep_logs:
            print(f"request={request_index} failed: {e}")

    finally:
        semaphore.release()


async def run() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    workload = Workload(cfg.dataset)
    semaphore = asyncio.Semaphore(cfg.max_in_flight)

    async with Sender(
        kairos_port=cfg.kairos_port,
        endpoint=cfg.endpoint,
    ) as sender:
        async with asyncio.TaskGroup() as task_group:
            for request_index, request in enumerate(workload):
                await semaphore.acquire()

                task_group.create_task(
                    send_and_record(
                        sender=sender,
                        request_index=request_index,
                        request=request,
                        semaphore=semaphore,
                        keep_logs=cfg.keep_logs,
                    )
                )
                #break

                # Later:
                # await distribution.wait_next()


def main() -> None:
    asyncio.run(run())


if __name__ == '__main__':
    main()
