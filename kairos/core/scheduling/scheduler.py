from __future__ import annotations

import time
import asyncio
from dataclasses import dataclass
from collections import deque
from kairos.core.memory.memory_manager import MemoryManager
from kairos.core.control.commands import ControlCommand, ControlKind
from kairos.core.catalog.model_variants_catalog import ModelVariantsCatalog
from kairos.core.catalog.model import Model
from kairos.logger import init_logger

logger = init_logger(__name__)


@dataclass(slots=True)
class ScheduleCommand:
    kind: ControlKind
    model: list[Model]
    interference: bool
    samples: list[dict] | None = None


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

        self.schedule_queue: asyncio.Queue[ScheduleCommand] = asyncio.Queue()
        self.arrival_timestamps: deque[float] = deque(maxlen=1000)
        self.catalog = ModelVariantsCatalog()
        self.poll_interval = 1.0
        self.history_seconds = 30
        self.idle_request_threshold = 1.0

    async def scheduling_loop(self) -> None:
        #await self.initiate_model_servers()
        #await self.set_active_model(self.catalog.get_base())
        while True:
            job = await self.schedule_queue.get()

            try:
                if job.interference:
                    await self._handle_job(job)
                else:
                    # case with forecasting
                    pass
            finally:
                self.schedule_queue.task_done()

    async def _handle_job(self, job) -> None:
        # 1. inspect current model state
        # 2. check memory constraints
        # 3. forecast idle window
        # 4. emit controller commands in safe order
        await self.control_queue.put(
            ControlCommand(
                model=job.model,
                samples=job.samples,
                kind=job.kind,
            )
        )

    async def initiate_model_servers(self) -> None:
        models = self.catalog.get_catalog()
        base_model = None
        for model in models.values():
            if model.relation == "base":
                base_model = model
            if model.relation != "base":
                await self.control_queue.put(
                   ControlCommand(
                       kind=ControlKind.START_MODEL_SERVER,
                       model=[model],
                   )
                )
        # ensure that base model is last in queue
        if base_model is not None:
            await self.control_queue.put(
                ControlCommand(
                    kind=ControlKind.START_MODEL_SERVER,
                    model=[base_model],
                )
            )

    async def set_active_model(self, model: Model) -> None:
        await self.control_queue.put(
            ControlCommand(
                kind=ControlKind.SET_ACTIVE_MODEL,
                model=[model],
            )
        )

    async def add_to_schedule_queue(self, command: ScheduleCommand) -> None:
        await self.schedule_queue.put(command)

    '''Timestamps-related methods'''

    def record_arrival(self, timestamp: float) -> None:
        self.arrival_timestamps.append(timestamp)

    def get_arrival_counts(self, now: float | None = None) -> list[int]:
        """
        Convert raw arrival timestamps into per-second request counts.

        Returns a list ordered from oldest second to newest second.
        Example:
            [0, 2, 5, 1]
        means:
            4 seconds ago: 0 requests
            3 seconds ago: 2 requests
            2 seconds ago: 5 requests
            latest second: 1 request
        """
        if now is None:
            now = time.time()

        end_second = int(now)
        start_second = end_second - self.history_seconds + 1

        counts = [0 for _ in range(self.history_seconds)]

        # Copy the deque so we work on a stable snapshot.
        timestamps = list(self.arrival_timestamps)

        for timestamp in timestamps:
            second = int(timestamp)

            if start_second <= second <= end_second:
                index = second - start_second
                counts[index] += 1

        return counts


