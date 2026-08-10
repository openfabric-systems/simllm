"""CPU pre-play inference and the strict trace artifact boundary."""

from simllm.preplay.runner import (
    PreplayRequest,
    TransformersCpuRunner,
    run_transformers_preplay,
)
from simllm.preplay.schema import (
    PREPLAY_TRACE_SCHEMA,
    ForwardPhase,
    ForwardTokenTrace,
    LayerRouting,
    PreplayTrace,
    PromptFormat,
    RequestTrace,
    SamplingConfig,
    SamplingMode,
    StopReason,
    TraceProvenance,
    validate_preplay_trace,
    validate_request_trace,
    validate_sampling_config,
    validate_trace_provenance,
)
from simllm.preplay.trace import (
    PreplayTraceReader,
    PreplayTraceWriter,
    read_preplay_trace,
    trace_provenance_from_json,
    trace_provenance_to_json,
    write_preplay_trace,
)

__all__ = [
    "PREPLAY_TRACE_SCHEMA",
    "ForwardPhase",
    "ForwardTokenTrace",
    "LayerRouting",
    "PreplayRequest",
    "PreplayTrace",
    "PreplayTraceReader",
    "PreplayTraceWriter",
    "PromptFormat",
    "RequestTrace",
    "SamplingConfig",
    "SamplingMode",
    "StopReason",
    "TraceProvenance",
    "TransformersCpuRunner",
    "read_preplay_trace",
    "run_transformers_preplay",
    "trace_provenance_from_json",
    "trace_provenance_to_json",
    "validate_preplay_trace",
    "validate_request_trace",
    "validate_sampling_config",
    "validate_trace_provenance",
    "write_preplay_trace",
]
