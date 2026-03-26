from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from kairos.core.config.loader import parse_config_file
from kairos.ipc.server import CoreIpcServer
from kairos.logger import init_logger

logger = init_logger(__name__)


async def async_main() -> None:
    config_path = Path(sys.argv[1])
    api_port = int(sys.argv[2])
    parse_config_file(config_path, api_port)
    # TODO: Add here parsing of config-yaml and starting of vllm servers subsequently
    ipc_server = CoreIpcServer()

    try:
        await ipc_server.start()
    finally:
        ipc_server.close()


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()