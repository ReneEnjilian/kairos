from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from kairos.core.catalog.model import Model

ControlKind = Literal[
    "PAUSE_DISPATCH",
    "RESUME_DISPATCH",
    "START_MODEL_SERVER",
    "STOP_MODEL_SERVER",
]


@dataclass(slots=True)
class ControlCommand:
    kind: ControlKind
    model: Model
    samples: list[str] | None = None
