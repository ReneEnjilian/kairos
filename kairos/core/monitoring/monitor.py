from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List
from collections import deque

from kairos.core.catalog.model import Model
from kairos.core.memory.memory_manager import MemoryManager
from kairos.logger import init_logger
from kairos.core.catalog.model_variants_catalog import ModelVariantsCatalog

logger = init_logger(__name__)


@dataclass(slots=True)
class SampleItem:
    payload: dict
    result: dict


class CoreMonitor:
    def __init__(
        self,
        memory_manager: MemoryManager,
        accuracy: float | None = None,
        latency: float | None = None,
        monotonicity: bool = False,
        discard: bool = False,
        recycle: bool = False,
        sample_size: int = 100,
        sample_rate: int = 50,
    ) -> None:

        self.sample_queue: asyncio.Queue[SampleItem] = asyncio.Queue()
        self.catalog = ModelVariantsCatalog()

        self.accuracy = accuracy
        self.latency = latency
        self.monotonicity = monotonicity
        self.discard = discard
        self.recycle = recycle
        self.sample_size = sample_size
        self.mem = memory_manager
        self.sample_rate = sample_rate

        self.request_counter = 0
        self.samples: deque[dict] = deque(maxlen=self.sample_size)

    async def monitor_loop(self) -> None:

        while True:
            item = await self.sample_queue.get()

            try:
                self.samples.append(item.payload)

            finally:
                self.sample_queue.task_done()

    '''Monitoring methods'''

    async def sample_request(self, payload: dict, result: dict) -> None:
        await self.sample_queue.put(
            SampleItem(payload=payload, result=result)
        )

    def should_sample_request(self) -> bool:
        self.request_counter += 1
        return self.request_counter % self.sample_rate == 0

    '''
    missing_samples = [
    sample
    for sample in evaluation_snapshot
    if not model.has_result(sample.sample_id)
    ]
    '''


