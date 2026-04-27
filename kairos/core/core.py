from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from kairos.core.memory.memory_manager import MemoryManager
from kairos.core.config.loader import (
    parse_models_from_config, parse_monitoring_from_config,
    parse_objectives_from_config, parse_scheduling_from_config)
from kairos.core.control.commands import ControlCommand
from kairos.core.control.controller import CoreController
from kairos.core.monitoring.monitor import CoreMonitor
from kairos.core.scheduling.scheduler import CoreScheduler
from kairos.ipc.server import CoreIpcServer
from kairos.logger import init_logger

logger = init_logger(__name__)


async def async_main() -> None:
    config_path = Path(sys.argv[1])
    api_port = int(sys.argv[2])

    parse_models_from_config(config_path, api_port)

    ipc_server = CoreIpcServer()

    control_queue: asyncio.Queue[ControlCommand] = asyncio.Queue()
    accuracy, latency = parse_objectives_from_config(config_path)
    knobs = parse_monitoring_from_config(config_path)
    # For normal operation, use pause_after=None.
    # For testing, you can set pause_after=5 and watch dispatch stop.
    memory_manager = MemoryManager()
    monitor = CoreMonitor(memory_manager=memory_manager, control_queue=control_queue, pause_after=5,
                          accuracy=accuracy, latency=latency, **knobs["monitoring"])

    controller = CoreController(
        ipc_server=ipc_server,
        control_queue=control_queue,
        monitor=monitor,
        **knobs["control"]
    )

    scheduling = parse_scheduling_from_config(config_path)
    scheduler = CoreScheduler(
        memory_manager=memory_manager,
        **scheduling,
    )

    ipc_task = asyncio.create_task(
        ipc_server.recv_loop(controller.handle_ipc_message)
    )
    dispatch_task = asyncio.create_task(controller.dispatch_loop())
    control_task = asyncio.create_task(controller.control_loop())
    monitor_task = asyncio.create_task(monitor.monitor_loop())
    schedule_task = asyncio.create_task(scheduler.scheduling_loop())

    try:
        await asyncio.gather(
            ipc_task,
            dispatch_task,
            control_task,
            monitor_task,
            schedule_task,
        )
    finally:
        for task in (ipc_task, dispatch_task, control_task, monitor_task, schedule_task):
            task.cancel()

        await asyncio.gather(
            ipc_task,
            dispatch_task,
            control_task,
            monitor_task,
            schedule_task,
            return_exceptions=True,
        )
        controller.shutdown_containers()
        ipc_server.close()


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()