"""Framework-agnostic core: virtual clock, step records, compute-cost model.

Nothing in this package may import vLLM or SGLang.
"""

from simllm.core.step import RequestPhase, ScheduledRequest, StepRecord, StepResult

__all__ = ["RequestPhase", "ScheduledRequest", "StepRecord", "StepResult"]
