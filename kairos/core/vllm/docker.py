from __future__ import annotations

from typing import Dict
from pathlib import Path
import httpx
import docker
import time
from docker.types import DeviceRequest
from docker.errors import NotFound, APIError
from kairos.logger import init_logger

logger = init_logger(__name__)


class DockerContainer:
    def __init__(self):
        self.client = docker.from_env()
        self.running_containers: Dict[str, str] = {}
        self.docker_image = "vllm-openai:thesis-v0.13.0"
        self.timeout = 300  # time to wait in seconds

    def start_container(self, model_name: str, model_id: str, port: int) -> None:

        hf_cache = str(Path.home() / ".cache" / "huggingface")
        hf_token = __import__("os").environ.get("HF_TOKEN")
        container_name = f"kairos-{model_name.lower().replace('/', '-')}"

        self.client.containers.run(
            image=self.docker_image,
            command=[
                model_id,
                "--enable-sleep-mode",
                "--max-model-len", "32768",
            ],
            detach=True,
            #auto_remove=True,
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
                raise RuntimeError(f"Model server {model_name} did not become ready.")

            time.sleep(3)  # every 3 seconds

        self.running_containers[model_name] = container_name
        logger.info(f"Docker container for model {model_name} started.")

    def is_running(self, port: int) -> bool:
        try:
            response = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def stop_container(self, model_name: str) -> None:
        try:
            container_name = self.running_containers[model_name]
            container = self.client.containers.get(container_name)
            container.stop()
            self.running_containers.pop(model_name, None)
            logger.info(f"Docker container for model {model_name} stopped.")
        except NotFound:
            return

    def stop_all_containers(self) -> None:
        for model_name in list(self.running_containers):
            try:
                self.stop_container(model_name)
            except APIError:
                pass
