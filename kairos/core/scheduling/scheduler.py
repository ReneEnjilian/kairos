from __future__ import annotations


from kairos.logger import init_logger

logger = init_logger(__name__)


class CoreScheduler:
    def __init__(
        self,
        scheduler: str | None = "ewma",
    ):
        self.scheduler = scheduler

    async def scheduling_loop(self) -> None:
        print(self.scheduler)
        while True:
            break
