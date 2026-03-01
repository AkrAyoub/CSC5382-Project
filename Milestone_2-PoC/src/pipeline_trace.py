from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TraceStep:
    name: str
    status: str = "PENDING"   # PENDING | OK | FAIL
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def end_ok(self):
        self.status = "OK"
        self.ended_at = time.time()

    def end_fail(self, err: Exception):
        self.status = "FAIL"
        self.error = f"{type(err).__name__}: {err}"
        self.ended_at = time.time()

    @property
    def duration_s(self) -> Optional[float]:
        if self.ended_at is None:
            return None
        return float(self.ended_at - self.started_at)


@dataclass
class PipelineTrace:
    steps: List[TraceStep] = field(default_factory=list)

    def start(self, name: str) -> TraceStep:
        step = TraceStep(name=name)
        self.steps.append(step)
        return step