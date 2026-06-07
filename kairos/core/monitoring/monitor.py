from __future__ import annotations

import asyncio
import re
import math
from dataclasses import dataclass
from typing import TypeAlias, Any
from collections import deque
#from kairos.core.catalog.model import Model
from kairos.core.memory.memory_manager import MemoryManager
from kairos.logger import init_logger
from kairos.core.catalog.model_variants_catalog import ModelVariantsCatalog
from kairos.core.reconfiguration.reconfiguration import CoreReconfiguration

logger = init_logger(__name__)


@dataclass(slots=True)
class SampleItem:
    payload: dict
    result: dict


@dataclass(slots=True)
class EvaluationItem:
    evaluation: dict[str, list[dict[str, Any]]]


MonitorEvent: TypeAlias = SampleItem | EvaluationItem


class CoreMonitor:
    def __init__(
        self,
        memory_manager: MemoryManager,
        reconfiguration: CoreReconfiguration,
        accuracy_slo: float | None = None,
        latency_slo: float | None = None,
        monotonicity: bool = False,
        discard: int = 0,
        recycle: int = 0,
        sample_size: int = 100,
        sample_rate: int = 50,
        #weight_accuracy: float = 0.5,
        #weight_latency: float = 0.5,
    ) -> None:

        self.monitoring_queue: asyncio.Queue[MonitorEvent] = asyncio.Queue()
        self.catalog = ModelVariantsCatalog()

        self.accuracy_slo = accuracy_slo
        self.latency_slo = latency_slo
        #self.weight_accuracy = weight_accuracy
        #self.weight_latency = weight_latency
        self.monotonicity = monotonicity
        self.discard = discard
        self.recycle = recycle
        self.sample_size = sample_size
        self.mem = memory_manager
        self.sample_rate = sample_rate

        self.request_counter = 0
        self.samples: deque[dict] = deque(maxlen=self.sample_size)
        self.monitoring_round = 0
        self.first_round_results: list[dict] = []
        self.expected_evaluation_models: set[str] = set()
        self.completed_evaluation_models: set[str] = set()
        self.evaluation_snapshot: list[dict] = []
        self.reconfiguration = reconfiguration

    async def monitor_loop(self) -> None:
        models = self.catalog.get_catalog()
        self.expected_evaluation_models = set(models.keys())
        await self.test_reconfiguration()
        while True:
            item = await self.monitoring_queue.get()

            try:
                if isinstance(item, SampleItem):
                    payload = dict(item.payload)
                    self.samples.append(payload)

                    if self.monitoring_round == 0 and len(self.first_round_results) < self.sample_size:
                        result = dict(item.result)
                        self.first_round_results.append(result)

                        if len(self.first_round_results) == self.sample_size:
                            self.evaluation_snapshot = list(self.samples)

                            active_model = result["active_model"]
                            model = self.catalog.get_model(active_model)
                            model.set_evaluation_results(self.first_round_results)
                            self.completed_evaluation_models.add(active_model)

                            # call reconfiguration
                            await self.reconfiguration.trigger_reconfiguration(
                                list(self.evaluation_snapshot),
                                self.monitoring_round
                            )

                elif isinstance(item, EvaluationItem):
                    for model_id, results in item.evaluation.items():
                        model = self.catalog.get_model(model_id)
                        model.set_evaluation_results(results)
                        self.completed_evaluation_models.add(model_id)
                    if self.expected_evaluation_models <= self.completed_evaluation_models:
                        self.compute_slis()

                        for model_id in self.expected_evaluation_models:
                            model = self.catalog.get_model(model_id)
                            model_acc = model.get_accuracy()
                            model_lat = model.get_latency()

                            failed = model_acc < self.accuracy_slo or model_lat > self.latency_slo
                            if self.discard > 0:
                                model.update_failed_rounds(failed, self.discard)

                        self.completed_evaluation_models = set()

                        self.expected_evaluation_models = {
                            model_id
                            for model_id, model in self.catalog.get_catalog().items()
                            if not model.discarded
                        }

                        self.monitoring_round += 1
                        self.evaluation_snapshot = list(self.samples)
                        # call reconfiguration
                        await self.reconfiguration.trigger_reconfiguration(
                            list(self.evaluation_snapshot),
                            self.monitoring_round
                        )
                else:
                    raise TypeError(f"Unknown monitoring item: {type(item)}")
            finally:
                self.monitoring_queue.task_done()

    '''Monitoring methods'''

    async def sample_request(self, payload: dict, result: dict) -> None:
        await self.monitoring_queue.put(
            SampleItem(payload=payload, result=result)
        )

    def should_sample_request(self) -> bool:
        self.request_counter += 1
        return self.request_counter % self.sample_rate == 0

    async def evaluate_samples(self, evaluation: dict) -> None:
        await self.monitoring_queue.put(
            EvaluationItem(evaluation=evaluation)
        )

    def compute_slis(self) -> None:
        base = self.catalog.get_base()
        ground_truth = base.get_evaluation_results()

        if not ground_truth:
            raise ValueError("Cannot compute SLIs: base model has no evaluation results.")

        for model_id in self.expected_evaluation_models:
            model = self.catalog.get_model(model_id)
            evaluation_results = model.get_evaluation_results()

            if not evaluation_results:
                raise ValueError(
                    f"Cannot compute SLIs: model {model_id} has no evaluation results."
                )

            if len(evaluation_results) != len(ground_truth):
                raise ValueError(
                    f"Cannot compute SLIs for {model_id}: "
                    f"expected {len(ground_truth)} results, got {len(evaluation_results)}."
                )

            # Accuracy relative to base model output
            correct = 0

            for base_result, model_result in zip(ground_truth, evaluation_results):
                base_answer = self._normalize_answer(base_result["kairos"])
                model_answer = self._normalize_answer(model_result["kairos"])

                if base_answer == model_answer:
                    correct += 1

            accuracy = correct / len(ground_truth)
            model.set_accuracy(accuracy)

            # Latency: p95 after ignoring first 5 results
            latencies = [
                result["infer_latency_ms"]
                for result in evaluation_results[5:]
            ]

            if not latencies:
                raise ValueError(
                    f"Cannot compute latency for {model_id}: not enough results after warmup removal."
                )

            latency = self._p95(latencies)
            model.set_latency(latency)

    def _p95(self, values: list[float]) -> float:
        sorted_values = sorted(values)
        index = math.ceil(0.95 * len(sorted_values)) - 1
        return sorted_values[index]

    def _normalize_answer(self, answer: str) -> str:
        text = answer.strip().lower()

        match = re.match(r"(yes|no|[0-3]|[a-d])\b", text)

        if match is None:
            return text

        return match.group(0)

    async def test_reconfiguration(self) -> None:
        models = self.catalog.get_catalog()
        for model in models.values():
            if model.is_base():
                model.latency = 90
                model.accuracy = 1.0
                model.set_storage_location_to_gpu()
                model.gpu_memory_allocation = 9774825472
                model.gpu_sleep_memory_allocation = 966787072
                model.gpu_wake_memory_allocation = 8808038400
            if model.is_quantized() and model.weight_bits == 8:
                model.accuracy = 0.95
                model.latency = 80
                model.set_storage_location_to_cpu()
                model.gpu_memory_allocation = 6239027200
                model.gpu_sleep_memory_allocation = 1071644672
                model.gpu_wake_memory_allocation = 5167382528
            if model.is_quantized() and model.weight_bits == 4:
                model.accuracy = 0.93
                model.latency = 60
                model.set_storage_location_to_cpu()
                model.gpu_memory_allocation = 5454692352
                model.gpu_sleep_memory_allocation = 1019215872
                model.gpu_wake_memory_allocation = 4435476480
            if model.is_independent():
                model.accuracy = 0.8
                model.latency = 80
                model.set_storage_location_to_cpu()
                model.gpu_memory_allocation = 8594128896
                model.gpu_sleep_memory_allocation = 866123776
                model.gpu_wake_memory_allocation = 7728005120
                model.failed_rounds = 1
        snapshot = []
        x = 1
        actual_free_bytes = self.mem.get_free_gpu_bytes(0)
        model_bytes = 0
        for model in models.values():
            if model.is_base():
                model_bytes += model.gpu_memory_allocation
            else:
                model_bytes += model.gpu_sleep_memory_allocation

        fake_free_bytes = actual_free_bytes - model_bytes
        self.reconfiguration.free_gpu_bytes_override = fake_free_bytes
        await self.reconfiguration.trigger_reconfiguration(snapshot, x)



