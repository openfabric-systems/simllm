"""Framework-agnostic core: virtual clock, step records, compute-cost model.

Nothing in this package may import vLLM or SGLang.
"""

from simllm.core.clock import VirtualClock
from simllm.core.step import (
    RESULT_SCHEMA,
    STEP_SCHEMA,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    StepResult,
)

__all__ = [
    "RESULT_SCHEMA",
    "STEP_SCHEMA",
    "RequestPhase",
    "ScheduledRequest",
    "StepRecord",
    "StepResult",
    "VirtualClock",
]
