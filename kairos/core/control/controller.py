from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from kairos.core.control.commands import ControlCommand, ControlKind
from kairos.core.memory.memory_manager import MemoryManager
from kairos.core.monitoring.monitor import CoreMonitor
from kairos.core.scheduling.scheduler import CoreScheduler
from kairos.ipc.server import CoreIpcServer
from kairos.logger import init_logger
from kairos.core.vllm.docker import DockerContainer
from kairos.core.catalog.model import Model
# from kairos.core.vllm.sleep_mode import *
from kairos.core.vllm.vllm_client import VLLMClient

logger = init_logger(__name__)


@dataclass(slots=True)
class RequestItem:
    identity: bytes
    request_id: str
    payload: dict
    sampled: bool = False


class CoreController:
    def __init__(
        self,
        ipc_server: CoreIpcServer,
        control_queue: asyncio.Queue[ControlCommand],
        monitor: CoreMonitor,
        scheduler: CoreScheduler,
        memory_manager: MemoryManager,
    ) -> None:
        self.ipc_server = ipc_server
        self.control_queue = control_queue
        self.monitor = monitor
        self.scheduler = scheduler

        self.request_queue: asyncio.Queue[RequestItem] = asyncio.Queue()

        self.dispatch_enabled = asyncio.Event()
        self.dispatch_enabled.set()

        self.active_model: Model | None = None
        self.mem = memory_manager
        self.docker = DockerContainer(self.mem)
        self.vllm_client = VLLMClient()

        # For continuous batching
        self.max_in_flight = 256
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

        arrival_timestamp = payload.get("arrival_timestamp")
        if arrival_timestamp is not None:
            self.scheduler.record_arrival(arrival_timestamp)

        sampled = False
        sampled = self.monitor.should_sample_request()

        await self.request_queue.put(
            RequestItem(
                identity=identity,
                request_id=request_id,
                payload=payload,
                sampled=sampled,
            )
        )

    '''
    async def dispatch_loop(self) -> None:
        print("DISPATCH LOOP STARTED", id(asyncio.current_task()))

        while True:
            request = await self.request_queue.get()

            try:
                await self.dispatch_enabled.wait()

                #print("START", request.request_id)

                result = await self.handle_infer(request.payload)

                #print("END", request.request_id)

                reply = {
                    "kind": "infer_result",
                    "request_id": request.request_id,
                    "ok": True,
                    "result": result,
                }
                await self.ipc_server.send_reply(request.identity, reply)


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

            if request.sampled:
                await self.monitor.sample_request(
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
                if command.kind == ControlKind.START_MODEL_SERVER:
                    if self.dispatch_enabled.is_set():
                        self.dispatch_enabled.clear()
                    await self.start_model_server(command.model[0])
                    self.dispatch_enabled.set()

                if command.kind == ControlKind.STOP_MODEL_SERVER:
                    await self.stop_model_server(command.model[0])

                if command.kind == ControlKind.L1_SLEEP:
                    await self.sleep_level_1(command.model[0])

                if command.kind == ControlKind.L2_SLEEP:
                    await self.sleep_level_2(command.model[0])

                if command.kind == ControlKind.WAKE_UP_FROM_CPU:
                    await self.wake_up_from_cpu(command.model[0])

                if command.kind == ControlKind.WAKE_UP_PERSISTENT:
                    await self.wake_up_from_cpu(command.model[0])

                if command.kind == ControlKind.WAKE_UP_FROM_DISK:
                    await self.wake_up_from_disk(command.model[0])

                if command.kind == ControlKind.PREFETCH:
                    await self.prefetch_weights(command.model[0])

                if command.kind == ControlKind.WAKE_UP_FROM_PREFETCH:
                    await self.wake_up_from_prefetch(command.model[0])

                if command.kind == ControlKind.PAUSE_DISPATCH:
                    if self.dispatch_enabled.is_set():
                        self.dispatch_enabled.clear()
                        logger.info(
                            "Dispatch paused."
                        )

                if command.kind == ControlKind.RESUME_DISPATCH:
                    if not self.dispatch_enabled.is_set():
                        self.dispatch_enabled.set()
                        logger.info(
                            "Dispatch resumed."
                        )

                if command.kind == ControlKind.SET_ACTIVE_MODEL:
                    # TODO: Think about requests not returned yet
                    self.active_model = command.model

                if command.kind == ControlKind.EVALUATE_MODEL:
                    pass

            finally:
                self.control_queue.task_done()

    '''
    Methods for inference:
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
        text = answer.strip().lower()

        match = re.match(r"(yes|no|[0-3]|[a-d])\b", text)

        if match is None:
            return text

        return match.group(0)

    async def handle_infer(self, payload: dict) -> dict:
        active_model = self.active_model

        completion = await self.vllm_client.chat_completion(
            port=active_model.port,
            model_id=active_model.model_id,
            instruction=payload["instruction"],
            prompt=payload["prompt"]
        )

        result = dict(payload)

        result["kairos"] = completion.text
        result["correct"] = self._normalize_answer(result["answer"]) == self._normalize_answer(result["kairos"])
        result["infer_latency_ms"] = completion.latency_ms
        result["active_model"] = active_model.model_id

        return result

    async def evaluate_sample_on_model(
            self,
            model: Model,
            sample: dict[str, Any],
    ) -> dict[str, Any]:
        completion = await self.vllm_client.chat_completion(
            port=model.port,
            model_id=model.model_id,
            instruction=sample["instruction"],
            prompt=sample["prompt"],
        )

        result = dict(sample)

        result["kairos"] = completion.text
        result["correct"] = (
                self._normalize_answer(result["answer"])
                == self._normalize_answer(completion.text)
        )
        result["infer_latency_ms"] = completion.latency_ms
        result["active_model"] = model.model_id

        return result

    async def evaluate_model_on_samples(
            self,
            model: Model,
            samples: list[dict[str, Any]],
            max_in_flight: int,
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(max_in_flight)

        async def run_one(sample: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await self.evaluate_sample_on_model(
                    model=model,
                    sample=sample,
                )

        results = await asyncio.gather(
            *(run_one(sample) for sample in samples)
        )

        return list(results)

    async def evaluate_models_on_same_samples(
            self,
            models: list[Model],
            samples: list[dict[str, Any]],
            max_in_flight_per_model: int,
    ) -> dict[str, list[dict[str, Any]]]:
        results = await asyncio.gather(
            *(
                self.evaluate_model_on_samples(
                    model=model,
                    samples=samples,
                    max_in_flight=max_in_flight_per_model,
                )
                for model in models
            )
        )

        return {
            model.model_id: model_results
            for model, model_results in zip(models, results)
        }

    '''
    later usage: next to active model, and multiple evaluating at once
    results = await self.evaluate_models_on_same_samples(
    models=[candidate_model],
    samples=samples,
    max_in_flight_per_model=4,
    )
    
    results = await self.evaluate_models_on_same_samples(
    models=[model_a, model_b],
    samples=samples,
    max_in_flight_per_model=8,
    )
    '''

    '''
    Methods for model placement:
    '''

    async def start_model_server(self, model: Model) -> None:
        logger.info(f"Starting model {model.model_id}.")

        vllm_engine_pid = await asyncio.to_thread(
            self.docker.start_container,
            model.model_id,
            model.port,
            model.gpu_memory_allocation_estimate,
        )
        model.set_engine_pid(vllm_engine_pid)

        # measure standby GPU memory
        await self.sleep_level_1(model)
        gpu_standby_mem = self.get_current_gpu_memory_usage(model.vllm_engine_pid)
        model.set_gpu_standby_memory_allocation(gpu_standby_mem)

        # warm-up and measure full GPU memory
        await self.wake_up_from_cpu_persistent(model)
        gpu_mem_model = self.get_current_gpu_memory_usage(model.vllm_engine_pid)
        model.set_gpu_memory_allocation(gpu_mem_model)

        # default state after startup
        await self.sleep_level_1(model)

        # ensure base model starts in GPU
        if model.is_base():
            await self.wake_up_from_cpu_persistent(model)

    async def stop_model_server(self, model: Model) -> None:
        logger.info(f"Stopping model {model.model_id}.")
        await asyncio.to_thread(
            self.docker.stop_container,
            model.model_id,
        )
        model.set_storage_location_to_disk()

    def shutdown_containers(self):
        self.docker.stop_all_containers()
        logger.info("Shutting down containers.")

    async def sleep_level_1(self, model: Model) -> None:
        await self.vllm_client.sleep_level_1(model.port)
        model.set_storage_location_to_cpu()
        logger.info(f"Sleeping {model.model_id} on CPU")

    async def sleep_level_2(self, model: Model) -> None:
        await self.vllm_client.sleep_level_2(model.port)
        model.set_storage_location_to_disk()
        logger.info(f"Sleeping {model.model_id} on disk")

    async def wake_up_from_cpu(self, model: Model) -> None:
        await self.vllm_client.wake_up(model.port)
        model.set_storage_location_to_gpu()
        logger.info(f"waking {model.model_id} from CPU RAM.")

    async def wake_up_from_cpu_persistent(self, model: Model) -> None:
        await self.vllm_client.wake_up_persistent(model.port)
        model.set_storage_location_to_gpu()
        logger.info(f"waking {model.model_id} from CPU RAM while keeping weight in RAM.")

    async def wake_up_from_disk(self, model: Model) -> None:
        port = model.port
        await self.vllm_client.wake_up(port)
        await self.vllm_client.reload_weights(port)
        await self.vllm_client.reset_prefix_cache(port)
        model.set_storage_location_to_gpu()
        logger.info(f"Waking {model.model_id} from disk.")

    async def prefetch_weights(self, model: Model) -> None:
        await self.vllm_client.prefetch(model.port)
        model.set_storage_location_to_cpu()
        logger.info(f"Prefetching {model.model_id} weights into CPU RAM.")

    async def wake_up_from_prefetch(self, model: Model) -> None:
        port = model.port
        await self.vllm_client.wake_up(port)
        await self.vllm_client.reload_weights_from_prefetch(port)
        await self.vllm_client.reset_prefix_cache(port)
        model.set_storage_location_to_gpu()
        logger.info(f"Waking {model.model_id} from prefetch.")

    def get_current_gpu_memory_usage(self, vllm_engine_pid: int) -> int:
        gpu_mem = self.mem.get_gpu_memory_used_by_pid(
            0,
            vllm_engine_pid
        )
        return gpu_mem
