from __future__ import annotations

import asyncio
import contextlib
import uuid
import zmq
import zmq.asyncio

from kairos.logger import init_logger

logger = init_logger(__name__)


class CoreIpcClient:
    def __init__(self, address: str = "tcp://127.0.0.1:5555") -> None:
        self.address = address

        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.DEALER)

        self._pending: dict[str, asyncio.Future[dict]] = {}
        self._recv_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        logger.info(f"Starting IPC client on {self.address}")
        self.socket.connect(self.address)
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def infer(self, payload: dict) -> dict:
        request_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict] = loop.create_future()
        self._pending[request_id] = future

        message = {
            "kind": "infer_request",
            "request_id": request_id,
            "payload": payload,
        }

        try:
            await self.socket.send_json(message)
            return await future
        finally:
            _ = self._pending.pop(request_id, None)

    # awaits response from CoreIPCServer
    async def _recv_loop(self) -> None:
        while True:
            reply = await self.socket.recv_json()
            request_id = reply["request_id"]
            future = self._pending.get(request_id)

            if future is None or future.done():
                continue

            if reply["ok"]:
                future.set_result(reply["result"])
            else:
                future.set_exception(RuntimeError(reply["error"]))

    async def close(self) -> None:
        logger.info("Shutting down IPC client.")
        if self._recv_task is not None:
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task

        self.socket.close(0)
        self.context.term()
