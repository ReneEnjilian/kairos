from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from kairos.core.catalog.model import Model
from enum import StrEnum


class ControlKind(StrEnum):
    PAUSE_DISPATCH = "PAUSE_DISPATCH"
    RESUME_DISPATCH = "RESUME_DISPATCH"
    SET_ACTIVE_MODEL = "SET_ACTIVE_MODEL"
    START_MODEL_SERVER = "START_MODEL_SERVER"
    STOP_MODEL_SERVER = "STOP_MODEL_SERVER"
    L1_SLEEP = "L1_SLEEP"
    L2_SLEEP = "L2_SLEEP"
    WAKE_UP_FROM_CPU = "WAKE_UP_FROM_CPU"
    WAKE_UP_FROM_DISK = "WAKE_UP_FROM_DISK"
    WAKE_UP_PERSISTENT = "WAKE_UP_PERSISTENT"
    WAKE_UP_FROM_PREFETCH = "WAKE_UP_FROM_PREFETCH"
    PREFETCH = "PREFETCH"
    EVALUATE_MODEL = "EVALUATE_MODEL"
    EVICT = "EVICT"


@dataclass(slots=True)
class ControlCommand:
    kind: ControlKind
    model: list[Model]
    samples: list[dict] | None = None
