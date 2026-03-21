


class Model:
    def __init__(
            self,
            name: str,
            role: str,
            port: int,
            model_id: str,
            quantization: str,
            quant_lib: str,
            lineage: str,
            size: float
    ):

        self.name = name
        self.role = role
        self.port = port
        self.model_id = model_id
        self.location = None # current location of weights in hierarchy
        self.quantization = quantization # quant bit
        self.quant_lib = quant_lib
        self.lineage = lineage
        self.size = size

        # TODO: SLOs and how to describe them (avg vs full data)
        # TODO: includes latency, energy, accuracy

    def update_location(self, location: str) -> None:
        self.location = location
