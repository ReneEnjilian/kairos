import pynvml


class MemoryManager:
    def __init__(
        self,
        gpu_device_ids: list[int],
        disk_path: str,
        cpu_safety_margin_percent: float = 0.10,
        gpu_safety_margin_percent: float = 0.10,
        disk_safety_margin_percent: float = 0.05,
    ):

        self.gpu_device_ids = gpu_device_ids
        self.disk_path = disk_path
        self.cpu_safety_margin_percent = cpu_safety_margin_percent
        self.gpu_safety_margin_percent = gpu_safety_margin_percent
        self.disk_safety_margin_percent = disk_safety_margin_percent

        # NVML must be initialized
        pynvml.nvmlInit()

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

    def can_move_to_gpu(self, device_id: int, model_size_bytes: int) -> bool:
        total_bytes = self.get_total_gpu_bytes(device_id)
        free_bytes = self.get_free_gpu_bytes(device_id)

        safety_margin_bytes = int(total_bytes * self.gpu_safety_margin_percent)
        effective_free_bytes = free_bytes - safety_margin_bytes

        return model_size_bytes <= effective_free_bytes





