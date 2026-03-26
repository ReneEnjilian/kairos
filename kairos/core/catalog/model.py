

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
        self.is_quantized = True if self.relation == "quantized" else False

        # TODO: SLOs and how to describe them (avg vs full data)
        # TODO: includes latency, energy, accuracy

    def update_location(self, storage_location: str) -> None:
        self.storage_location = storage_location
