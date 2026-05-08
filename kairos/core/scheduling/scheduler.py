from __future__ import annotations

import asyncio
from collections import deque
from kairos.core.memory.memory_manager import MemoryManager
from kairos.core.control.commands import ControlCommand
from kairos.logger import init_logger

logger = init_logger(__name__)


class CoreScheduler:
    def __init__(
        self,
        memory_manager: MemoryManager,
        control_queue: asyncio.Queue[ControlCommand],
        method: str | None = "ewma",
        window: int | None = None,
    ):
        self.method = method
        self.window = window
        self.mem = memory_manager
        self.control_queue = control_queue

        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.arrival_timestamps: deque[float] = deque(maxlen=1000)

    async def scheduling_loop(self) -> None:
        print(self.method)
        while True:
            job = await self.job_queue.get()

            try:
                await self._handle_job(job)
            finally:
                self.job_queue.task_done()

    async def _handle_job(self, job) -> None:
        # 1. inspect current model state
        # 2. check memory constraints
        # 3. forecast idle window
        # 4. emit controller commands in safe order
        pass

    def record_arrival(self, timestamp: float) -> None:
        self.arrival_timestamps.append(timestamp)
