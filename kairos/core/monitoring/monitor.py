from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List

from kairos.core.catalog.model import Model
from kairos.core.memory.memory_manager import MemoryManager
from kairos.logger import init_logger
from kairos.core.catalog.model_variants_catalog import ModelVariantsCatalog

logger = init_logger(__name__)


@dataclass(slots=True)
class CompletionItem:
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
    ) -> None:

        self.completion_queue: asyncio.Queue[CompletionItem] = asyncio.Queue()
        self.completion_count = 0
        self.samples: List[str] = []    # only payload, results are stored in individual models
        self.catalog = ModelVariantsCatalog()
        self.current_model = self.catalog.get_baseline()
        self.accuracy = accuracy
        self.latency = latency
        self.monotonicity = monotonicity
        self.discard = discard
        self.recycle = recycle
        self.sample_size = sample_size
        self.memory_manager = memory_manager

    async def monitor_loop(self) -> None:

        while True:
            item = await self.completion_queue.get()
            #self.samples.append(item.payload)
            #self.current_model.add_sample(item.result, 0.0) # ignore latency for now

            try:
                print("test")

            finally:
                self.completion_queue.task_done()

    '''Methods used '''
    async def notify_completion(self, payload: dict, result: dict) -> None:
        await self.completion_queue.put(
            CompletionItem(payload=payload, result=result)
        )




