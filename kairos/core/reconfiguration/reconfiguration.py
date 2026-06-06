from kairos.core.memory.memory_manager import MemoryManager
from kairos.core.catalog.model_variants_catalog import ModelVariantsCatalog
from kairos.core.catalog.model import Model
from kairos.core.scheduling.scheduler import ScheduleCommand
from kairos.core.control.commands import ControlKind
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
        self.mem = memory_manager
        self.weight_accuracy = weight_accuracy
        self.weight_latency = weight_latency
        self.constant = 1000

        self.catalog = ModelVariantsCatalog()

        self.snapshot: list[dict] = []
        self.monitoring_round = 0
        self.command_sequence: list[ScheduleCommand] = []
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

    def reconfiguration_algorithm(self):
        # get models we want to reconfigure
        self.models = self.get_reconfigurable_models()

        # discarded models are sent to disk
        # self.discard_models_to_disk()

        # compute model scores for feasible and infeasible models
        self.compute_model_scores()

        # compute target placement
        target_placement = self.compute_serving_placement()
        logger.info("Model scores: %s", {
            model.model_id: self.model_scores[model.model_id]
            for model in self.models.values()
        })

        logger.info("Current placement: %s", {
            location: [model.model_id for model in models]
            for location, models in self.get_current_placement().items()
        })

        logger.info("Target placement: %s", {
            location: [model.model_id for model in models]
            for location, models in target_placement.items()
        })

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
    def get_current_placement(self) -> dict:
        current_placement = dict()
        current_placement["disk"] = []
        current_placement["cpu"] = []
        current_placement["gpu"] = []

        for model in self.models.values():
            placement = model.get_storage_location()
            current_placement[placement].append(model)
        return current_placement

    def discard_models_to_disk(self) -> None:
        all_models = self.catalog.get_catalog().values()
        for model in all_models:
            if model.discarded:
                location = model.get_storage_location()
                if location == "cpu":
                    self.command_sequence.append(
                        ScheduleCommand(
                            kind=ControlKind.EVICT,
                            model=[model],
                            interference=False,
                        )
                    )
                elif location == "gpu":
                    self.command_sequence.append(
                        ScheduleCommand(
                            kind=ControlKind.L1_SLEEP,
                            model=[model],
                            interference=False,
                        )
                    )
                    self.command_sequence.append(
                        ScheduleCommand(
                            kind=ControlKind.EVICT,
                            model=[model],
                            interference=False,
                        )
                    )

    def compute_serving_placement(self) -> dict[str, list[Model]]:
        target_placement: dict[str, list[Model]] = {
            "cpu": [],
            "gpu": [],
        }

        best_model = max(
            self.models.values(),
            key=lambda candidate: self.model_scores[candidate.model_id],
        )

        target_placement["gpu"].append(best_model)

        current_placement = self.get_current_placement()
        capacity = self.mem.get_free_gpu_bytes(0)

        # original, restore later
        #for model in current_placement["gpu"]:
        #    capacity += model.gpu_wake_memory_allocation

        capacity -= best_model.gpu_wake_memory_allocation

        # just for offline testing, delete afterwards
        for model in self.models.values():
            capacity -= model.gpu_sleep_memory_allocation

        candidates = [
            model
            for model in self.models.values()
            if model.model_id != best_model.model_id
        ]

        selected_models = self._knapsack_by_wake_memory(candidates, capacity)

        target_placement["gpu"].extend(selected_models)

        selected_ids = {model.model_id for model in target_placement["gpu"]}

        for model in self.models.values():
            if model.model_id not in selected_ids:
                target_placement["cpu"].append(model)

        return target_placement

    def _knapsack_by_wake_memory(
            self,
            candidates: list[Model],
            capacity: int,
    ) -> list[Model]:
        if capacity <= 0:
            return []

        # used_memory -> (total_score, selected_models)
        states: dict[int, tuple[float, tuple[Model, ...]]] = {
            0: (0.0, ())
        }

        for model in candidates:
            weight = model.gpu_wake_memory_allocation
            if weight is None:
                raise ValueError(
                    f"Model {model.model_id} has no gpu_wake_memory_allocation"
                )

            if weight > capacity:
                continue

            value = self.model_scores[model.model_id]

            new_states: dict[int, tuple[float, tuple[Model, ...]]] = {}

            for used_memory, (total_score, selected_models) in states.items():
                new_used_memory = used_memory + weight

                if new_used_memory > capacity:
                    continue

                new_score = total_score + value
                new_selected_models = selected_models + (model,)

                existing_state = states.get(new_used_memory)
                new_existing_state = new_states.get(new_used_memory)

                best_existing = existing_state
                if new_existing_state is not None:
                    if best_existing is None or new_existing_state[0] > best_existing[0]:
                        best_existing = new_existing_state

                if best_existing is None or new_score > best_existing[0]:
                    new_states[new_used_memory] = (new_score, new_selected_models)

            states.update(new_states)

        best_used_memory, best_state = max(
            states.items(),
            key=lambda item: (item[1][0], -item[0]),
        )

        return list(best_state[1])
