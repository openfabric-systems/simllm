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
    step_record_to_json,
    step_records_to_json,
    write_step_records,
)

__all__ = [
    "RESULT_SCHEMA",
    "STEP_SCHEMA",
    "RequestPhase",
    "ScheduledRequest",
    "StepRecord",
    "StepResult",
    "VirtualClock",
    "step_record_to_json",
    "step_records_to_json",
    "write_step_records",
]
