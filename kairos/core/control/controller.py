from __future__ import annotations

import asyncio
from dataclasses import dataclass

from kairos.core.control.commands import ControlCommand
from kairos.core.monitoring.monitor import CoreMonitor
from kairos.ipc.server import CoreIpcServer
from kairos.logger import init_logger
from kairos.core.vllm.docker import DockerContainer
from kairos.core.catalog.model import Model
# from kairos.core.vllm.sleep_mode import *
from kairos.core.vllm.vllm_client import VllmClient

logger = init_logger(__name__)


@dataclass(slots=True)
class RequestItem:
    identity: bytes
    request_id: str
    payload: dict


class CoreController:
    def __init__(
        self,
        ipc_server: CoreIpcServer,
        control_queue: asyncio.Queue[ControlCommand],
        monitor: CoreMonitor | None = None,
        sample_rate: int = 30
    ) -> None:
        self.ipc_server = ipc_server
        self.control_queue = control_queue
        self.monitor = monitor

        self.request_queue: asyncio.Queue[RequestItem] = asyncio.Queue()

        self.dispatch_enabled = asyncio.Event()
        self.dispatch_enabled.set()

        self.active_model: Model | None = None

        self.docker = DockerContainer()
        self.vllm_client = VllmClient()

        self.sample_rate = sample_rate

        # For continous batching
        self.max_in_flight = 32
        self.inflight_semaphore = asyncio.Semaphore(self.max_in_flight)
        self._inflight_tasks: set[asyncio.Task[None]] = set()

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
        if not isinstance(payload, dict):
            reply = {
                "kind": "infer_result",
                "request_id": request_id,
                "ok": False,
                "error": "Payload must be an object.",
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
            print("dispatch")
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

    '''
    async def dispatch_loop(self) -> None:
        while True:
            request = await self.request_queue.get()
            await self.dispatch_enabled.wait()

            await self.inflight_semaphore.acquire()

            task = asyncio.create_task(
                self._process_request(request)
            )
            self._inflight_tasks.add(task)
            task.add_done_callback(self._inflight_tasks.discard)
    '''
    async def _process_request(self, request: RequestItem) -> None:
        try:
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
            self.inflight_semaphore.release()

    async def control_loop(self) -> None:
        # await self.initiate_model_servers()
        while True:
            command = await self.control_queue.get()

            try:
                if command.kind == "START_MODEL_SERVER":
                    if self.dispatch_enabled.is_set():
                        self.dispatch_enabled.clear()
                    await self.start_model_server(command.model)
                    self.dispatch_enabled.set()

                if command.kind == "STOP_MODEL_SERVER":
                    await self.stop_model_server(command.model)

                if command.kind == "L1_SLEEP":
                    if command.model.is_baseline() and self.dispatch_enabled.is_set():
                        self.dispatch_enabled.clear()
                    await self.sleep_level_1(command.model)
                    self.dispatch_enabled.set()

                if command.kind == "L2_SLEEP":
                    if command.model.is_baseline() and self.dispatch_enabled.is_set():
                        self.dispatch_enabled.clear()
                    await self.sleep_level_2(command.model)
                    self.dispatch_enabled.set()

                if command.kind == "WAKE_UP_FROM_CPU":
                    await self.wake_up_from_cpu(command.model)

                if command.kind == "WAKE_UP_PERSISTENT":
                    await self.wake_up_from_cpu(command.model)

                if command.kind == "WAKE_UP_FROM_DISK":
                    await self.wake_up_from_disk(command.model)

                if command.kind == "PREFETCH":
                    await self.prefetch_weights(command.model)

                if command.kind == "WAKE_UP_FROM_PREFETCH":
                    await self.wake_up_from_prefetch(command.model)

                if command.kind == "PAUSE_DISPATCH":
                    if self.dispatch_enabled.is_set():
                        self.dispatch_enabled.clear()
                        logger.info(
                            "Dispatch paused."
                        )

                if command.kind == "RESUME_DISPATCH":
                    if not self.dispatch_enabled.is_set():
                        self.dispatch_enabled.set()
                        logger.info(
                            "Dispatch resumed."
                        )

                if command.kind == "SET_ACTIVE_MODEL":
                    # TODO: Think about requests not returned yet
                    self.active_model = command.model

            finally:
                self.control_queue.task_done()

    '''
    Methods for dispatcher:
    '''

    '''
    async def handle_infer(self, payload: dict) -> dict:
        # Later this becomes the real call into your vLLM-side logic.
        # avoid mutating the original payload object in-place.
        # Why: payload is also passed to the monitor later
        # Keeping the original request unchanged is cleaner
        result = dict(payload)

        result["kairos"] = "no"
        result["correct"] = result["answer"] == result["kairos"]

        return result
    '''

    def _normalize_answer(self, answer: str) -> str:
        return answer.strip().lower()

    async def handle_infer(self, payload: dict) -> dict:
        active_model = self.active_model

        kairos_answer = await self.vllm_client.chat_completion(
            port=active_model.port,
            model_id=active_model.model_id,
            instruction=payload["instruction"],
            prompt=payload["prompt"]
        )

        result = dict(payload)

        result["kairos"] = kairos_answer
        result["correct"] = self._normalize_answer(payload["answer"]) == self._normalize_answer(kairos_answer)

        return result

    '''
    Methods for controller:
    '''

    async def start_model_server(self, model: Model) -> None:
        logger.info(f"Starting model {model.name}.")
        await asyncio.to_thread(
            self.docker.start_container,
            model.name,
            model.model_id,
            model.port,
            model.gpu_memory_allocation
        )
        # ensure that base model starts in GPU, quantized in CPU, rest in RAM
        # TODO: add new field to command -> location
        if model.relation == "quantized":
            await self.vllm_client.sleep_level_1(model.port)
            model.set_storage_location_to_cpu()
        elif model.relation == "base":
            model.set_storage_location_to_gpu()
        else:
            await self.vllm_client.sleep_level_2(model.port)
            model.set_storage_location_to_disk()

    async def stop_model_server(self, model: Model) -> None:
        logger.info(f"Stopping model {model.name}.")
        await asyncio.to_thread(
            self.docker.stop_container,
            model.name,
        )
        model.set_storage_location_to_disk()

    def shutdown_containers(self):
        self.docker.stop_all_containers()
        logger.info("Shutting down containers.")

    async def sleep_level_1(self, model: Model) -> None:
        await self.vllm_client.sleep_level_1(model.port)
        model.set_storage_location_to_cpu()
        logger.info(f"Sleeping {model.name} on CPU")

    async def sleep_level_2(self, model: Model) -> None:
        await self.vllm_client.sleep_level_2(model.port)
        model.set_storage_location_to_disk()
        logger.info(f"Sleeping {model.name} on disk")

    async def wake_up_from_cpu(self, model: Model) -> None:
        await self.vllm_client.wake_up(model.port)
        model.set_storage_location_to_gpu()
        logger.info(f"waking {model.name} from CPU RAM.")

    async def wake_up_from_cpu_persistent(self, model: Model) -> None:
        await self.vllm_client.wake_up_persistent(model.port)
        model.set_storage_location_to_gpu()
        logger.info(f"waking {model.name} from CPU RAM while keeping weight in RAM.")

    async def wake_up_from_disk(self, model: Model) -> None:
        port = model.port
        await self.vllm_client.wake_up(port)
        await self.vllm_client.reload_weights(port)
        await self.vllm_client.reset_prefix_cache(port)
        model.set_storage_location_to_gpu()
        logger.info(f"Waking {model.name} from disk.")

    async def prefetch_weights(self, model: Model) -> None:
        await self.vllm_client.prefetch(model.port)
        model.set_storage_location_to_cpu()
        logger.info(f"Prefetching {model.name} weights into CPU RAM.")

    async def wake_up_from_prefetch(self, model: Model) -> None:
        port = model.port
        await self.vllm_client.wake_up(port)
        await self.vllm_client.reload_weights_from_prefetch(port)
        await self.vllm_client.reset_prefix_cache(port)
        model.set_storage_location_to_gpu()
        logger.info(f"Waking {model.name} from prefetch.")
