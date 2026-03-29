'''from __future__ import annotations
from kairos.logger import init_logger
import asyncio
import os
import zmq
import zmq.asyncio

logger = init_logger(__name__)


class CoreIpcServer:
    def __init__(self, address: str = "tcp://127.0.0.1:5555") -> None:
        self.address = address

        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.ROUTER)

    async def start(self) -> None:
        self.socket.bind(self.address)
        logger.info(f"Starting IPC server on {self.address}")

        while True:
            identity, payload = await self.socket.recv_multipart()
            message = self._decode_message(payload)

            kind = message["kind"]
            request_id = message["request_id"]

            if kind == "infer_request":
                user_payload = message["payload"]

                print(f"[CORE][IPC] got request: {user_payload}")

                # placeholder for now
                result = await self.handle_infer(user_payload)

                reply = {
                    "kind": "infer_result",
                    "request_id": request_id,
                    "ok": True,
                    "result": result,
                }
            else:
                reply = {
                    "kind": "infer_result",
                    "request_id": request_id,
                    "ok": False,
                    "error": f"Unknown kind: {kind}",
                }

            await self.socket.send_multipart([identity, self._encode_message(reply)])

    async def handle_infer(self, payload: str) -> str:
        #await asyncio.sleep(0.01)
        return f"processed by core: {payload}"

    def _encode_message(self, message: dict) -> bytes:
        import json
        return json.dumps(message).encode("utf-8")

    def _decode_message(self, payload: bytes) -> dict:
        import json
        return json.loads(payload.decode("utf-8"))

    def close(self) -> None:
        logger.info("Shutting down IPC server.")
        self.socket.close(0)
        self.context.term()'''