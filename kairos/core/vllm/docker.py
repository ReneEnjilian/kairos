from __future__ import annotations

from typing import Dict
from pathlib import Path
import httpx
import docker
import time
from docker.types import DeviceRequest
from docker.errors import NotFound, APIError

from kairos.core.memory.memory_manager import MemoryManager
from kairos.logger import init_logger

logger = init_logger(__name__)


class DockerContainer:
    def __init__(
            self,
            memory_manager: MemoryManager,
    ):
        self.client = docker.from_env()
        self.running_containers: Dict[str, str] = {}
        self.docker_image = "vllm-openai:thesis-v0.13.0"
        self.timeout = 700  # time to wait in seconds
        self.memory_manager = memory_manager

    def start_container(self, model_id: str, port: int, gpu_memory_allocation_estimate: int) -> int:
        total_gpu_bytes = self.memory_manager.get_total_gpu_bytes(0)
        gpu_memory_utilization = gpu_memory_allocation_estimate / total_gpu_bytes
        hf_cache = str(Path.home() / ".cache" / "huggingface")
        hf_token = __import__("os").environ.get("HF_TOKEN")
        container_name = f"kairos-{model_id.lower().replace('/', '-')}"

        self.client.containers.run(
            image=self.docker_image,
            command=[
                model_id,
                "--enable-sleep-mode",
                "--max-model-len", "1024",
                "--gpu-memory-utilization", str(gpu_memory_utilization),
            ],
            detach=True,
            auto_remove=True,
            ports={"8000/tcp": port},
            volumes={
                hf_cache: {
                    "bind": "/root/.cache/huggingface",
                    "mode": "rw",
                }
            },
            environment={
                "HF_TOKEN": hf_token,
            },
            ipc_mode="host",
            device_requests=[
                DeviceRequest(count=-1, capabilities=[["gpu"]])
            ],
            name=container_name,
        )

        deadline = time.monotonic() + self.timeout  # 5 minutes

        while True:
            if self.is_running(port):
                break

            if time.monotonic() > deadline:
                raise RuntimeError(f"Model server {model_id} did not become ready.")

            time.sleep(3)  # every 3 seconds

        self.running_containers[model_id] = container_name
        logger.info(f"Docker container for model {model_id} started.")
        engine_pid = self.get_vllm_engine_pid(container_name)
        return engine_pid

    def is_running(self, port: int) -> bool:
        try:
            response = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def stop_container(self, model_id: str) -> None:
        try:
            container_name = self.running_containers[model_id]
            container = self.client.containers.get(container_name)
            container.stop()
            self.running_containers.pop(model_id, None)
            logger.info(f"Docker container for model {model_id} stopped.")
        except NotFound:
            return

    def stop_all_containers(self) -> None:
        for model_id in list(self.running_containers):
            try:
                self.stop_container(model_id)
            except APIError:
                pass

    def get_vllm_engine_pid(self, container_name: str) -> int:
        container = self.client.containers.get(container_name)
        top = container.top()

        pid_index = top["Titles"].index("PID")
        cmd_index = top["Titles"].index("CMD")

        matching_pids: list[int] = []

        for process in top["Processes"]:
            pid = int(process[pid_index])
            cmd = process[cmd_index]

            if "VLLM::EngineCore" in cmd:
                matching_pids.append(pid)

        if not matching_pids:
            raise RuntimeError(
                f"No VLLM::EngineCore process found in container {container_name}."
            )

        if len(matching_pids) > 1:
            raise RuntimeError(
                f"Multiple VLLM::EngineCore processes found in container {container_name}: "
                f"{matching_pids}"
            )

        return matching_pids[0]
