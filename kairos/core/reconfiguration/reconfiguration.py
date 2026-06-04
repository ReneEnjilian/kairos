from kairos.core.memory.memory_manager import MemoryManager
from kairos.core.catalog.model_variants_catalog import ModelVariantsCatalog
from kairos.core.catalog.model import Model

from kairos.logger import init_logger


logger = init_logger(__name__)


class CoreReconfiguration:
    def __init__(
            self,
            accuracy_slo: float,
            latency_slo: float,
            memory_manager: MemoryManager,
            weight_accuracy: float = 0.5,
            weight_latency: float = 0.5,
    ):
        self.accuracy_slo = accuracy_slo
        self.latency_slo = latency_slo
        self.memory_manager = memory_manager
        self.weight_accuracy = weight_accuracy
        self.weight_latency = weight_latency
        self.constant = 1000

        self.catalog = ModelVariantsCatalog()

        self.snapshot: list[dict] = []
        self.monitoring_round = 0
        self.command_sequence: list = []
        self.models: dict[str, Model] = {}
        self.model_scores: dict[str, float] = {}
        # self.model_ranking: list[Model] = []

    # maybe async ?
    def trigger_reconfiguration(self, snapshot: list[dict], monitoring_round: int):
        self.command_sequence.clear()
        self.model_scores.clear()
        self.snapshot = snapshot
        self.monitoring_round = monitoring_round
        self.reconfiguration_algorithm()

    def get_reconfigurable_models(self) -> dict[str, Model]:
        return {
            model_id: model
            for model_id, model in self.catalog.get_catalog().items()
            if not model.discarded
        }

    def compute_model_scores(self) -> None:
        feasible: list[Model] = []
        infeasible: list[Model] = []

        for model in self.models.values():
            if model.failed_rounds == 0:
                feasible.append(model)
            else:
                infeasible.append(model)

        for model in feasible:
            score_acc = (model.accuracy - self.accuracy_slo) / self.accuracy_slo
            score_lat = (self.latency_slo - model.latency) / self.latency_slo
            feasible_score = (self.weight_accuracy * score_acc + self.weight_latency * score_lat) + self.constant
            self.model_scores[model.model_id] = feasible_score

        for model in infeasible:
            acc_violation = max(0.0, self.accuracy_slo - model.accuracy) / self.accuracy_slo
            lat_violation = max(0.0, model.latency - self.latency_slo) / self.latency_slo
            violation = self.weight_accuracy * acc_violation + self.weight_latency * lat_violation
            infeasible_score = 1 / (1+violation)
            self.model_scores[model.model_id] = infeasible_score

        '''
        # concat feasible and infeasible and sort in descending order
        self.model_ranking = sorted(
            feasible + infeasible,
            key=lambda model: self.model_scores[model.model_id],
            reverse=True,
        )
        '''

    def reconfiguration_algorithm(self):
        # get models we want to reconfigure
        self.models = self.get_reconfigurable_models()

        # TODO: if model is discarded and not on disk -> add to command sequence to get sent to disk (this step should come here)

        # compute model scores for feasible and infeasible models
        self.compute_model_scores()



