from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ControlKind = Literal["PAUSE_DISPATCH", "RESUME_DISPATCH"]


@dataclass(slots=True)
class ControlCommand:
    kind: ControlKind
    reason: str | None = None