from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from kairos.core.catalog.model import Model

ControlKind = Literal[
    "PAUSE_DISPATCH",
    "RESUME_DISPATCH",
    "SET_ACTIVE_MODEL",
    "START_MODEL_SERVER",
    "STOP_MODEL_SERVER",
    "L1_SLEEP",
    "L2_SLEEP",
    "WAKE_UP_FROM_CPU",
    "WAKE_UP_FROM_DISK",
    "WAKE_UP_PERSISTENT",
    "WAKE_UP_FROM_PREFETCH",
    "PREFETCH",
]


@dataclass(slots=True)
class ControlCommand:
    kind: ControlKind
    model: Model
    samples: list[str] | None = None
