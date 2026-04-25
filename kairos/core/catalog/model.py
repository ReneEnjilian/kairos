from typing import List
from dataclasses import dataclass


@dataclass(slots=True)
class SampleItem:
    result: str
    latency: float


class Model:
    def __init__(
            self,
            name: str,
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
    ):

        # Specified in configuration file
        self.name = name
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

        # TODO: SLOs and how to describe them (avg vs full data)
        # TODO: includes latency, energy, accuracy
        self.avg_latency: float = 0.0
        self.avg_accuracy: float = 0.0
        self.avg_energy_consumption: float = 0.0
        self.sample_results: List[SampleItem] = []

        # calculate memory requirement (estimate)
        self.gpu_memory_allocation: float = self.size


    def print_all_fields(self) -> None:
        print(f"name: {self.name}")
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
        print(f"is_quantized: {self.is_quantized}")

    def add_sample(self, result: str, latency: float) -> None:
        self.sample_results.append(SampleItem(result, latency))
        if len(self.sample_results) >= self.sample_limit:
            # TODO: compute avg_latency, avg_accuracy
            pass

    def set_storage_location_to_disk(self) -> None:
        self.storage_location = "disk"

    def set_storage_location_to_cpu(self) -> None:
        self.storage_location = "cpu"

    def set_storage_location_to_gpu(self) -> None:
        self.storage_location = "gpu"

    def is_baseline(self) -> bool:
        return self.relation == "baseline"

    def is_quantized(self) -> bool:
        return self.relation == "quantized"

    def is_independent(self) -> bool:
        return self.relation == "independent"

    def update_gpu_allocation(self, gpu_factor: float) -> None:
        self.gpu_memory_allocation *= gpu_factor



