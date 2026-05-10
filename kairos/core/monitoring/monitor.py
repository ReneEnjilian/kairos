from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TypeAlias, Any
from collections import OrderedDict
from uuid import uuid4

from kairos.core.catalog.model import Model
from kairos.core.memory.memory_manager import MemoryManager
from kairos.logger import init_logger
from kairos.core.catalog.model_variants_catalog import ModelVariantsCatalog

logger = init_logger(__name__)


@dataclass(slots=True)
class SampleItem:
    payload: dict
    result: dict


@dataclass(slots=True)
class EvaluationItem:
    evaluation: dict[str, list[dict[str, Any]]]


MonitorEvent: TypeAlias = SampleItem | EvaluationItem


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

        self.monitoring_queue: asyncio.Queue[MonitorEvent] = asyncio.Queue()
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
        self.samples: OrderedDict[str, dict] = OrderedDict()

    async def monitor_loop(self) -> None:

        while True:
            item = await self.monitoring_queue.get()

            try:
                if isinstance(item, SampleItem):
                    sample_id = f"sample-{uuid4()}"
                    payload = dict(item.payload)

                    self.samples[sample_id] = payload
                    self.samples.move_to_end(sample_id)

                    if len(self.samples) > self.sample_size:
                        self.samples.popitem(last=False)

                    active_model = item.result["active_model"]
                    model = self.catalog.get_model(active_model)
                    model.add_result(sample_id, item.result)
                elif isinstance(item, EvaluationItem):
                    pass
                else:
                    raise TypeError(f"Unknown monitoring item: {type(item)}")
            finally:
                self.monitoring_queue.task_done()

    '''Monitoring methods'''

    async def sample_request(self, payload: dict, result: dict) -> None:
        await self.monitoring_queue.put(
            SampleItem(payload=payload, result=result)
        )

    def should_sample_request(self) -> bool:
        self.request_counter += 1
        return self.request_counter % self.sample_rate == 0

    async def evaluate_samples(self, evaluation: dict) -> None:
        await self.monitoring_queue.put(
            EvaluationItem(evaluation=evaluation)
        )

    '''
    missing_samples = [
    sample
    for sample in evaluation_snapshot
    if not model.has_result(sample.sample_id)
    ]
    '''


