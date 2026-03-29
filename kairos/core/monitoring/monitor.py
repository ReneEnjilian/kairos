from __future__ import annotations

import asyncio
from dataclasses import dataclass

from kairos.core.control.commands import ControlCommand
from kairos.logger import init_logger

logger = init_logger(__name__)


@dataclass(slots=True)
class CompletionItem:
    payload: str
    result: str


class CoreMonitor:
    def __init__(
        self,
        control_queue: asyncio.Queue[ControlCommand],
        pause_after: int | None = None,
    ) -> None:
        self.control_queue = control_queue
        self.pause_after = pause_after

        self.completion_queue: asyncio.Queue[CompletionItem] = asyncio.Queue()
        self.completion_count = 0

    async def notify_completion(self, payload: str, result: str) -> None:
        await self.completion_queue.put(
            CompletionItem(payload=payload, result=result)
        )

    async def pause_dispatch(self, reason: str = "monitor request") -> None:
        await self.control_queue.put(
            ControlCommand(kind="PAUSE_DISPATCH", reason=reason)
        )

    async def resume_dispatch(self, reason: str = "monitor request") -> None:
        await self.control_queue.put(
            ControlCommand(kind="RESUME_DISPATCH", reason=reason)
        )

    async def run(self) -> None:
        while True:
            item = await self.completion_queue.get()

            try:
                self.completion_count += 1

                # Minimal prototype behavior:
                # if pause_after is set, pause dispatch once after N completions.
                if (
                    self.pause_after is not None
                    and self.completion_count >= self.pause_after
                ):
                    logger.info(
                        f"Monitor requests dispatch pause after "
                        f"{self.completion_count} completions."
                    )
                    await self.pause_dispatch(
                        reason=f"pause_after={self.pause_after}"
                    )
                    self.pause_after = None
            finally:
                self.completion_queue.task_done()