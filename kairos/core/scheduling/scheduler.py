from __future__ import annotations

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

        self.job_queue: asyncio.Queue[ScheduleCommand] = asyncio.Queue()
        self.arrival_timestamps: deque[float] = deque(maxlen=1000)
        self.catalog = ModelVariantsCatalog()

    async def scheduling_loop(self) -> None:
        #await self.initiate_model_servers()
        #await self.set_active_model(self.catalog.get_base())
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

    async def warmup(self) -> None:
        '''
        - use baseline model for this
        start -> GPU
        L2 -> disk
        prefetch -> RAM
        wake_from_prefetch -> GPU
        L1 -> RAM
        wake_persistent -> GPU
        L1 -> RAM
        evict -> disk
        '''
        pass

    async def initiate_model_servers(self) -> None:
        models = self.catalog.get_catalog()
        base_model = None
        for model in models.values():
            if model.relation == "base":
                base_model = model
            if model.relation != "base":
                await self.control_queue.put(
                   ControlCommand(
                       kind="START_MODEL_SERVER",
                       model=model,
                   )
                )
        # ensure that base model is last in queue
        if base_model is not None:
            await self.control_queue.put(
                ControlCommand(
                    kind="START_MODEL_SERVER",
                    model=base_model,
                )
            )

    async def set_active_model(self, model: Model) -> None:
        await self.control_queue.put(
            ControlCommand(
                kind="SET_ACTIVE_MODEL",
                model=model,
            )
        )

    '''Timestamps-related methods'''

    def record_arrival(self, timestamp: float) -> None:
        self.arrival_timestamps.append(timestamp)
