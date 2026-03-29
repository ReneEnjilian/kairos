from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

import zmq
import zmq.asyncio

from kairos.logger import init_logger

logger = init_logger(__name__)


class CoreIpcServer:
    def __init__(self, address: str = "tcp://127.0.0.1:5555") -> None:
        self.address = address

        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.ROUTER)

        self._is_bound = False
        self._send_lock = asyncio.Lock()

    def bind(self) -> None:
        if self._is_bound:
            return
        self.socket.bind(self.address)
        self._is_bound = True
        logger.info(f"Starting IPC server on {self.address}")

    async def recv_loop(
        self,
        on_request: Callable[[bytes, dict], Awaitable[None]],
    ) -> None:
        self.bind()

        while True:
            identity, payload = await self.socket.recv_multipart()
            message = self._decode_message(payload)
            await on_request(identity, message)

    async def send_reply(self, identity: bytes, reply: dict) -> None:
        async with self._send_lock:
            await self.socket.send_multipart(
                [identity, self._encode_message(reply)]
            )

    def _encode_message(self, message: dict) -> bytes:
        return json.dumps(message).encode("utf-8")

    def _decode_message(self, payload: bytes) -> dict:
        return json.loads(payload.decode("utf-8"))

    def close(self) -> None:
        logger.info("Shutting down IPC server.")
        self.socket.close(0)
        self.context.term()