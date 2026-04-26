from __future__ import annotations

from kairos.core.memory.memory_manager import MemoryManager
from kairos.logger import init_logger

logger = init_logger(__name__)


class CoreScheduler:
    def __init__(
        self,
        memory_manager: MemoryManager,
        scheduler: str | None = "ewma",
    ):
        self.scheduler = scheduler
        self.memory_manager = memory_manager

    async def scheduling_loop(self) -> None:
        print(self.scheduler)
        while True:
            break
