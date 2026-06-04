from kairos.core.memory.memory_manager import MemoryManager
from kairos.core.catalog.model_variants_catalog import ModelVariantsCatalog

from kairos.logger import init_logger


logger = init_logger(__name__)


class CoreReconfiguration:
    def __init__(
            self,
            memory_manager: MemoryManager,
            weight_accuracy: float = 0.5,
            weight_latency: float = 0.5,

    ):
        self.snapshot: list[dict] = []
        self.monitoring_round = 0
        self.weight_accuracy = weight_accuracy
        self.weight_latency = weight_latency
        self.catalog = ModelVariantsCatalog()
        self.command_sequence: list = []

    # maybe async ?
    def trigger_reconfiguration(self, snapshot: list[dict], monitoring_round: int):
        self.snapshot = snapshot
        self.monitoring_round = monitoring_round

