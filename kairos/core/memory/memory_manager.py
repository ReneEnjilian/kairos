import pynvml
import psutil
import shutil
from pathlib import Path


class MemoryManager:
    def __init__(
        self,
        disk_path: str | None = None,
        gpu_device_ids: list[int] | None = None,
        cpu_safety_margin_percent: float = 0.10,
        gpu_safety_margin_percent: float = 0.10,
        disk_safety_margin_percent: float = 0.05,
    ):

        pynvml.nvmlInit()

        if gpu_device_ids is None:
            gpu_count = pynvml.nvmlDeviceGetCount()
            self.gpu_device_ids = list(range(gpu_count))
        else:
            self.gpu_device_ids = gpu_device_ids

        if disk_path is None:
            self.disk_path = str(Path.home())
        else:
            self.disk_path = disk_path
        self.cpu_safety_margin_percent = cpu_safety_margin_percent
        self.gpu_safety_margin_percent = gpu_safety_margin_percent
        self.disk_safety_margin_percent = disk_safety_margin_percent

    '''GPU-related methods'''

    def _get_gpu_handle(self, device_id: int):
        if device_id not in self.gpu_device_ids:
            raise ValueError(f"GPU device_id {device_id} is not managed by this MemoryManager.")
        return pynvml.nvmlDeviceGetHandleByIndex(device_id)

    def get_total_gpu_bytes(self, device_id: int) -> int:
        handle = self._get_gpu_handle(device_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(mem_info.total)

    def get_free_gpu_bytes(self, device_id: int) -> int:
        handle = self._get_gpu_handle(device_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(mem_info.free)

    def can_move_to_gpu(self, device_id: int, required_gpu_bytes: int) -> bool:
        total_bytes = self.get_total_gpu_bytes(device_id)
        free_bytes = self.get_free_gpu_bytes(device_id)

        safety_margin_bytes = int(total_bytes * self.gpu_safety_margin_percent)
        effective_free_bytes = free_bytes - safety_margin_bytes

        return required_gpu_bytes <= effective_free_bytes

    '''CPU-related methods'''

    def get_total_cpu_bytes(self) -> int:
        return psutil.virtual_memory().total

    def get_free_cpu_bytes(self) -> int:
        return psutil.virtual_memory().free

    def get_available_cpu_bytes(self) -> int:
        return psutil.virtual_memory().available

    def can_move_to_cpu(self, model_size_bytes: int) -> bool:
        total_bytes = self.get_total_cpu_bytes()
        available_bytes = self.get_available_cpu_bytes()

        safety_margin_bytes = int(total_bytes * self.cpu_safety_margin_percent)
        effective_free_bytes = available_bytes - safety_margin_bytes

        return model_size_bytes <= effective_free_bytes

    '''Disk-related methods'''

    def get_total_disk_bytes(self) -> int:
        usage = shutil.disk_usage(self.disk_path)
        return usage.total

    def get_free_disk_bytes(self) -> int:
        usage = shutil.disk_usage(self.disk_path)
        return usage.free

    def can_move_to_disk(self, model_size_bytes: int) -> bool:
        total_bytes = self.get_total_disk_bytes()
        free_bytes = self.get_free_disk_bytes()

        safety_margin_bytes = int(total_bytes * self.disk_safety_margin_percent)
        effective_free_bytes = free_bytes - safety_margin_bytes

        return model_size_bytes <= effective_free_bytes






