from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List
from kairos.core.control.commands import ControlCommand
from kairos.core.memory.memory_manager import MemoryManager
from kairos.logger import init_logger
from kairos.core.catalog.model_variants_catalog import ModelVariantsCatalog
logger = init_logger(__name__)


@dataclass(slots=True)
class CompletionItem:
    payload: str
    result: str


class CoreMonitor:
    def __init__(
        self,
        control_queue: asyncio.Queue[ControlCommand],
        memory_manager: MemoryManager,
        pause_after: int | None = None,
        accuracy: float | None = None,
        latency: float | None = None,
        monotonicity: bool = False,
        discard: bool = False,
        recycle: bool = False,
        sample_size: int = 100,
    ) -> None:
        self.control_queue = control_queue
        self.pause_after = pause_after

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
        #await self.initiate_model_servers()
        while True:
            item = await self.completion_queue.get()
            self.samples.append(item.payload)
            self.current_model.add_sample(item.result, 0.0) # ignore latency for now

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
                    self.internal_work()
                    self.pause_after = None
                    self.completion_count = 0

                    await self.resume_dispatch(
                        reason="because"
                    )

            finally:
                self.completion_queue.task_done()

    '''Methods used '''
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

    def internal_work(self) -> None:
        for i in range(5):
            print(f"internal op: {i}")

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

