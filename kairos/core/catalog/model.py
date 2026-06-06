import math


class Model:
    def __init__(
            self,
            port: int,
            relation: str,
            model_id: str,
            size: float,
            compute_dtype: str,
            lineage: str = None,
            quant_method: str = None,
            weight_bits: int = None,
            weight_type: str = None,
            activation_bits: int = None,
            activation_type: str = None,
            storage_location: str = None,
            gpu_factor: float = 2.0,
            max_cached_results: int = 1000,
    ):

        # Specified in configuration file
        self.relation = relation
        self.port = port
        self.model_id = model_id

        # Retrieved from huggingface hub
        self.compute_dtype = compute_dtype  # or fp16, bf16
        self.lineage = lineage  # None in case of baseline model
        self.size = size

        # Optional parameters in case of quantization
        self.quant_method = quant_method
        self.weight_bits = weight_bits  # 4, 8 ...
        self.weight_type = weight_type  # "fp4", nf4, int ...
        self.activation_bits = activation_bits
        self.activation_type = activation_type

        self.storage_location = storage_location  # options: disk, cpu, gpu

        # TODO: includes latency, energy, accuracy
        self.latency: float = 0.0
        self.accuracy: float = 1.0 if self.is_base() else 0.0
        # self.avg_energy_consumption: float = 0.0
        self.max_cached_results = max_cached_results
        self.evaluation_results: list[dict] = []
        self.failed_rounds: int = 0
        self.discarded: bool = False

        # calculate memory requirement (estimate)
        self.gpu_factor = gpu_factor
        self.gpu_memory_allocation_estimate: int = math.ceil(self.size * self.gpu_factor)
        self.gpu_memory_allocation: int | None = None
        self.gpu_sleep_memory_allocation: int | None = None
        self.gpu_wake_memory_allocation: int | None = None
        self.vllm_engine_pid: int | None = None

    def print_all_fields(self) -> None:
        print(f"relation: {self.relation}")
        print(f"port: {self.port}")
        print(f"model_id: {self.model_id}")
        print(f"compute_dtype: {self.compute_dtype}")
        print(f"lineage: {self.lineage}")
        print(f"size: {self.size}")
        print(f"quant_method: {self.quant_method}")
        print(f"weights_bits: {self.weight_bits}")
        print(f"weight_type: {self.weight_type}")
        print(f"activation_bits: {self.activation_bits}")
        print(f"activation_type: {self.activation_type}")
        print(f"storage_location: {self.storage_location}")
        print(f"gpu_factor: {self.gpu_factor}")
        print(f"gpu_memory_allocation_estimate: {self.gpu_memory_allocation_estimate}")
        print(f"gpu_memory_allocation: {self.gpu_memory_allocation}")
        print(f"gpu_standby_memory_allocation: {self.gpu_standby_memory_allocation}")
        print(f"vllm_engine_pid: {self.vllm_engine_pid}")
        print(f"latency: {self.latency}")
        print(f"accuracy: {self.accuracy}")

    def set_evaluation_results(self, results: list[dict]) -> None:
        self.evaluation_results = list(results)

    def update_failed_rounds(self, failed: bool, threshold: int) -> None:
        if failed:
            self.failed_rounds += 1
        else:
            self.failed_rounds = 0

        if not self.is_base() and self.failed_rounds >= threshold:
            self.discarded = True

    def get_evaluation_results(self):
        return self.evaluation_results

    def set_accuracy(self, accuracy: float) -> None:
        self.accuracy = accuracy

    def get_accuracy(self) -> float:
        return self.accuracy

    def set_latency(self, latency: float) -> None:
        self.latency = latency

    def get_latency(self) -> float:
        return self.latency

    def set_storage_location_to_disk(self) -> None:
        self.storage_location = "disk"

    def set_storage_location_to_cpu(self) -> None:
        self.storage_location = "cpu"

    def set_storage_location_to_gpu(self) -> None:
        self.storage_location = "gpu"

    def get_storage_location(self) -> str:
        return self.storage_location

    def is_base(self) -> bool:
        return self.relation == "base"

    def is_quantized(self) -> bool:
        return self.relation == "quantized"

    def is_independent(self) -> bool:
        return self.relation == "independent"

    def set_gpu_memory_allocation(self, gpu_memory_allocation: int) -> None:
        self.gpu_memory_allocation = gpu_memory_allocation

    def set_gpu_sleep_memory_allocation(self, gpu_sleep_memory_allocation: int) -> None:
        self.gpu_sleep_memory_allocation = gpu_sleep_memory_allocation
        self.gpu_wake_memory_allocation = self.gpu_memory_allocation - gpu_sleep_memory_allocation

    def set_engine_pid(self, pid: int) -> None:
        self.vllm_engine_pid = pid



