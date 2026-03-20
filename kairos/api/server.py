from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter
import uvicorn
import os
from kairos.api.routes.api_router import router as inference_router
from kairos.ipc.client import CoreIpcClient
from kairos.logger import init_logger
from kairos.logger import build_uvicorn_log_config

logger = init_logger(__name__)


class Server:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.host = host
        self.port = port

        core_client = CoreIpcClient()

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await core_client.start()
            app.state.core_client = core_client
            try:
                yield
            finally:
                await core_client.close()

        self.app = FastAPI(lifespan=lifespan)
        self.app.include_router(inference_router)

    def run(self) -> None:
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_config=build_uvicorn_log_config()
        )


