from __future__ import annotations

import asyncio
from dataclasses import dataclass

from kairos.core.control.commands import ControlCommand
from kairos.core.monitoring.monitor import CoreMonitor
from kairos.ipc.server import CoreIpcServer
from kairos.logger import init_logger

logger = init_logger(__name__)


@dataclass(slots=True)
class RequestItem:
    identity: bytes
    request_id: str
    payload: str


class CoreController:
    def __init__(
        self,
        ipc_server: CoreIpcServer,
        control_queue: asyncio.Queue[ControlCommand],
        monitor: CoreMonitor | None = None,
    ) -> None:
        self.ipc_server = ipc_server
        self.control_queue = control_queue
        self.monitor = monitor

        self.request_queue: asyncio.Queue[RequestItem] = asyncio.Queue()

        self.dispatch_enabled = asyncio.Event()
        self.dispatch_enabled.set()

        self.active_model = "baseline"

    async def handle_ipc_message(self, identity: bytes, message: dict) -> None:
        kind = message.get("kind")
        request_id = message.get("request_id")

        if kind != "infer_request":
            reply = {
                "kind": "infer_result",
                "request_id": request_id,
                "ok": False,
                "error": f"Unknown kind: {kind}",
            }
            await self.ipc_server.send_reply(identity, reply)
            return

        payload = message.get("payload")
        if not isinstance(payload, str):
            reply = {
                "kind": "infer_result",
                "request_id": request_id,
                "ok": False,
                "error": "Payload must be a string.",
            }
            await self.ipc_server.send_reply(identity, reply)
            return

        await self.request_queue.put(
            RequestItem(
                identity=identity,
                request_id=request_id,
                payload=payload,
            )
        )

    async def dispatch_loop(self) -> None:
        while True:
            request = await self.request_queue.get()

            try:
                await self.dispatch_enabled.wait()

                result = await self.handle_infer(request.payload)

                reply = {
                    "kind": "infer_result",
                    "request_id": request.request_id,
                    "ok": True,
                    "result": result,
                }
                await self.ipc_server.send_reply(request.identity, reply)

                if self.monitor is not None:
                    await self.monitor.notify_completion(
                        payload=request.payload,
                        result=result,
                    )

            except Exception as e:
                reply = {
                    "kind": "infer_result",
                    "request_id": request.request_id,
                    "ok": False,
                    "error": str(e),
                }
                await self.ipc_server.send_reply(request.identity, reply)

            finally:
                self.request_queue.task_done()

    async def control_loop(self) -> None:
        while True:
            command = await self.control_queue.get()

            try:
                if command.kind == "PAUSE_DISPATCH":
                    if self.dispatch_enabled.is_set():
                        self.dispatch_enabled.clear()
                        logger.info(
                            "Dispatch paused."
                            if command.reason is None
                            else f"Dispatch paused: {command.reason}"
                        )

                elif command.kind == "RESUME_DISPATCH":
                    if not self.dispatch_enabled.is_set():
                        self.dispatch_enabled.set()
                        logger.info(
                            "Dispatch resumed."
                            if command.reason is None
                            else f"Dispatch resumed: {command.reason}"
                        )

            finally:
                self.control_queue.task_done()

    async def handle_infer(self, payload: str) -> str:
        # Placeholder.
        # Later this becomes the real call into your vLLM-side logic.
        return f"processed by core: {payload}"