"""Produce the canonical mixed-locality bypass record for TRAF-76."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import fields, is_dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(
    os.environ.get("SIMLLM_BYPASS_REPOSITORY_ROOT", Path(__file__).resolve().parents[2])
).resolve()
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
from simllm.backends.htsim_rnic import FlowCompletion
from simllm.compute import GPU_ENVELOPES, ModelDims, RooflineProvider
from simllm.core import (
    RequestPhase,
    ScheduledRequest,
    StepRecord,
)
from simllm.placement import PlacementManifest, RankPlacement

SCHEMA = "simllm-collective-floor-pre-wave-bypass-golden-v1"
PRE_WAVE_COMMIT = "06fc199783e364c2eaa6a7c917a1f9f2c84d79ac"
SCENARIO_ID = "traf76-mixed-locality-bypass-v1"
_SEND_RE = re.compile(rb"\bsend ([0-9]+)b\b")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, separators=(",", ": "))
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _normalize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, Path):
        return value.name
    if isinstance(value, dict):
        return {
            str(key): _normalize(member)
            for key, member in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normalize(member) for member in value]
    return value


def _manifest() -> PlacementManifest:
    hosts = ("node-a", "node-a", "node-b", "node-b")
    local_counts: dict[str, int] = {}
    ranks = []
    for global_rank, hostname in enumerate(hosts):
        local_rank = local_counts.get(hostname, 0)
        local_counts[hostname] = local_rank + 1
        ranks.append(
            RankPlacement(
                global_rank=global_rank,
                hostname=hostname,
                local_rank=local_rank,
            )
        )
    return PlacementManifest(ranks=ranks)


def _dims() -> ModelDims:
    return ModelDims(
        num_layers=1,
        hidden_size=1_024,
        intermediate_size=2_048,
        num_heads=8,
        num_kv_heads=8,
        head_size=128,
        vocab_size=256,
        dtype_bytes=2,
    )


def _records() -> tuple[StepRecord, ...]:
    return (
        StepRecord(
            step_index=0,
            virtual_time_ps=0,
            scheduled=[
                ScheduledRequest(
                    "r0",
                    RequestPhase.PREFILL,
                    64,
                    context_length=64,
                )
            ],
            num_sampled=1,
        ),
        StepRecord(
            step_index=1,
            virtual_time_ps=0,
            scheduled=[
                ScheduledRequest(
                    "r0",
                    RequestPhase.DECODE,
                    1,
                    context_length=65,
                )
            ],
            num_sampled=1,
        ),
    )


def _goal_projection(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = path.read_bytes()
    return {
        "name": path.name,
        "goal_text_bytes": len(payload),
        "goal_sha256": _sha256_bytes(payload),
        "wire_send_bytes": sum(int(value) for value in _SEND_RE.findall(payload)),
    }


def _plan_projection(plan: object) -> dict[str, Any]:
    locality = plan.locality
    phase_application_bytes = []
    for phase in locality.phases:
        phase_application_bytes.append(
            {
                "phase_id": phase.phase.phase_id,
                "total_directed_bytes": sum(
                    segment.payload_bytes for segment in phase.phase.segments
                ),
                "local_directed_bytes": sum(
                    segment.payload_bytes for segment in phase.nvlink_segments
                ),
                "fabric_directed_bytes": sum(
                    segment.payload_bytes for segment in phase.fabric_segments
                ),
            }
        )
    artifacts = []
    for artifact in plan.artifacts:
        artifacts.append(
            {
                "artifact_id": artifact.artifact_id,
                "operation_ids": list(artifact.operation_ids),
                "local_service_ps": artifact.local_service_ps,
                "collective_operation_id": artifact.collective_operation_id,
                "goal": _goal_projection(artifact.goal_path),
            }
        )
    return {
        "step_index": plan.step_index,
        "virtual_time_ps": plan.virtual_time_ps,
        "locality": _normalize(locality),
        "artifacts": artifacts,
        "application_bytes": {
            "total_directed_bytes": locality.total_directed_bytes,
            "local_directed_bytes": locality.nvlink_bytes,
            "fabric_directed_bytes": locality.fabric_bytes,
            "phases": phase_application_bytes,
        },
    }


class _InspectingSink(HtsimStepSink):
    def __init__(
        self,
        config: HtsimStepSinkConfig,
        *,
        backend_replay: list[dict[str, Any]] | None,
    ) -> None:
        super().__init__(config)
        self.backend_replay = backend_replay
        self.backend_invocations: list[dict[str, Any]] = []
        self.plan_projections: list[dict[str, Any]] = []

    def _execute_plan(self, plan):
        projection = _plan_projection(plan)
        simulation = super()._execute_plan(plan)
        locality = simulation.locality_outcome
        if locality is None:
            raise RuntimeError("the bypass fixture produced no locality outcome")
        phase_timestamps = []
        cursor_ps = plan.virtual_time_ps
        for artifact, service_ps in zip(
            plan.artifacts,
            locality.composed_phase_service_ps,
            strict=True,
        ):
            phase_timestamps.append(
                {
                    "artifact_id": artifact.artifact_id,
                    "started_at_ps": cursor_ps,
                    "finished_at_ps": cursor_ps + service_ps,
                }
            )
            cursor_ps += service_ps
        projection["phase_timestamps"] = phase_timestamps
        self.plan_projections.append(projection)
        return simulation

    def _run_goal(self, plan, goal_path, completion_csv):
        goal = _goal_projection(goal_path)
        if goal is None:
            raise AssertionError("backend invocation has no GOAL")
        if self.backend_replay is None:
            run = super()._run_goal(plan, goal_path, completion_csv)
        else:
            index = len(self.backend_invocations)
            expected = self.backend_replay[index]
            for field in ("name", "goal_text_bytes", "goal_sha256", "wire_send_bytes"):
                if goal[field] != expected[field]:
                    raise RuntimeError(
                        f"replayed backend GOAL diverged at invocation {index} field {field}"
                    )
            flows = tuple(FlowCompletion(**row) for row in expected["flows"])
            service_ps = expected["service_ps"]
            run = _ReplayedRun(
                flows=flows,
                quiescent=expected["quiescent"],
                service_ps=service_ps,
            )
        invocation = {
            **goal,
            "service_ps": run.job_completion_time_ps(),
            "quiescent": run.quiescent,
            "flows": [_normalize(flow) for flow in run.flows],
        }
        self.backend_invocations.append(invocation)
        return run


class _ReplayedRun:
    def __init__(self, *, flows, quiescent: bool, service_ps: int) -> None:
        self.flows = flows
        self.quiescent = quiescent
        self.service_ps = service_ps

    def job_completion_time_ps(self) -> int:
        return self.service_ps


def produce_bypass_record(
    workdir: Path,
    *,
    backend_replay: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute the pinned scenario and return its canonical frozen fields."""

    sink = _InspectingSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1, 2, 3),
            dims=_dims(),
            workdir=workdir,
            placement_manifest=_manifest(),
            provider=RooflineProvider(efficiency=0.7),
            gpu=GPU_ENVELOPES["b100"],
        ),
        backend_replay=backend_replay,
    )
    random.seed(76)
    random_before = _sha256_bytes(repr(random.getstate()).encode("utf-8"))
    results = []
    virtual_time_ps = 0
    for record in _records():
        record.virtual_time_ps = virtual_time_ps
        result = sink(record)
        if result is None:
            raise RuntimeError("the bypass fixture produced no StepResult")
        results.append(
            {
                "step_result": _normalize(result),
            }
        )
        virtual_time_ps = result.completed_at_ps
    random_after = _sha256_bytes(repr(random.getstate()).encode("utf-8"))
    wire_bytes = [row["wire_send_bytes"] for row in sink.backend_invocations]
    record = {
        "plans": sink.plan_projections,
        "results": results,
        "completion_order": [row["step_result"]["step_index"] for row in results],
        "backend_invocation_order": [
            row["name"] for row in sink.backend_invocations
        ],
        "backend_invocations": sink.backend_invocations,
        "wire_bytes": {
            "fabric_goal_send_bytes": sum(wire_bytes),
            "per_invocation": wire_bytes,
        },
        "random_generator_state": {
            "before": random_before,
            "after": random_after,
        },
    }
    if not any(
        phase["local_directed_bytes"] > 0
        for plan in record["plans"]
        for phase in plan["application_bytes"]["phases"]
    ):
        raise RuntimeError("the bypass fixture did not exercise a local phase")
    if not any(
        phase["fabric_directed_bytes"] > 0
        for plan in record["plans"]
        for phase in plan["application_bytes"]["phases"]
    ):
        raise RuntimeError("the bypass fixture did not exercise a fabric phase")
    if random_before != random_after:
        raise RuntimeError("the bypass fixture changed the random-generator state")
    if backend_replay is not None and len(sink.backend_invocations) != len(
        backend_replay
    ):
        raise RuntimeError("the bypass fixture did not consume every replay row")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generating-commit", required=True)
    args = parser.parse_args()
    payload = {
        "schema": SCHEMA,
        "generating_commit": args.generating_commit,
        "scenario": {
            "scenario_id": SCENARIO_ID,
            "profile": "rnic-nn-fluid",
            "tp_ranks": [0, 1, 2, 3],
            "hosts": ["node-a", "node-a", "node-b", "node-b"],
            "prompt_tokens": 64,
            "steps": 2,
        },
        "record": produce_bypass_record(args.workdir),
    }
    output = _json_bytes(payload)
    if b"\r" in output:
        raise RuntimeError("the bypass golden must use LF line endings")
    args.output.write_bytes(output)
    print(f"golden_sha256={_sha256_bytes(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
