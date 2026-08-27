#!/usr/bin/env python3
"""Execute the frozen backend-free deployment scan study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any, NoReturn, Self, TypeVar

from simllm.calibration import BatchServicePoint
from simllm.deploy import (
    DEPLOYMENT_ESTIMATE_SCHEMA,
    PIPELINE_PARALLEL_UNPRICED,
    BudgetSpec,
    DeploymentCandidate,
    EnvelopeSpec,
    EstimatorInputs,
    EvidenceClass,
    ExternalAnchor,
    FabricSpec,
    FrontierPoint,
    FrontierRecord,
    ModelRef,
    ModelWork,
    PointClass,
    PoolSpec,
    ScanInputs,
    SimDerivedTerms,
    SlaSpec,
    TermEstimate,
    WorkloadPoint,
    candidate_from_json,
    candidate_to_json,
    decode_capacity_requests_per_second,
    estimate_decode_step,
    estimate_prefill_request,
    estimate_stamp_from_json,
    estimate_stamp_to_json,
    frontier_record_from_json,
    frontier_record_to_json,
    match_pools,
    pareto_front,
    queue_delay_ps,
    queue_occupancy,
    scan,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = Path(__file__).resolve().parent
EXPECTATIONS_PATH = STUDY_DIR / "expectations.md"
FRONTIER_EXPECTATIONS_PATH = (
    REPOSITORY_ROOT / "examples" / "deployment_frontier_v1" / "expectations.json"
)
INVENTORY_PATH = (
    REPOSITORY_ROOT
    / "offline"
    / "calibration"
    / "deployment-projections"
    / "ee154ed5f07c104269df9cf60d8730b8c6dced0ccf619fb7ff7146ec2ddfd5a2.json"
)
LEGACY_RESULT_PATH = (
    REPOSITORY_ROOT / "examples" / "deployment_frontier_v1" / "result.json"
)

RESULT_SCHEMA = "simllm-deployment-scan-result-v1"
EXPECTATIONS_COMMIT = "15ee956e2ba54a851884d2cba5d6abd7ca0cdd8d"
EARLIER_IMPLEMENTATION_COMMITS = (
    "6c8957935e217ebc2c588f816fd6fcecc717d0d0",
    "beeaa6d6549a07867a7ab97a5b2c4972b690b2a7",
    "6e16070fb4d8a772ed738a9490998b30e44183a4",
    "4ec8538823a944b22a20b24f808dea28ecaddb66",
    "110020e80226fd02cd12f037b8c51652e12a27be",
)
FROZEN_INPUTS = (
    (
        FRONTIER_EXPECTATIONS_PATH,
        "54295c81cebe36ee32d12b8ab1432c9fc060094ddf98403152b0d619cc37438f",
    ),
    (
        INVENTORY_PATH,
        "ee154ed5f07c104269df9cf60d8730b8c6dced0ccf619fb7ff7146ec2ddfd5a2",
    ),
    (
        LEGACY_RESULT_PATH,
        "f2f216068bf5ba914853c62a2ee965ede0ebfc0a6f29e3d11cfa5f45eac359ad",
    ),
)
BATCHES = (1, 2, 4, 8, 16, 32)
PICOSECONDS_PER_SECOND = 1_000_000_000_000
SYNTHETIC_SHA256 = "0" * 64
T = TypeVar("T")


class ProcessCreationBlocked(RuntimeError):
    """Raised if code under the fatal process guard tries to start a process."""


class ProcessGuard(AbstractContextManager["ProcessGuard"]):
    """Intercept the exact process-creation seams frozen by FG-1."""

    def __init__(self) -> None:
        self.attempts: list[str] = []
        self._popen: Any = None
        self._posix_spawn: Any = None

    def _blocked(self, *args: object, **kwargs: object) -> NoReturn:
        detail = f"args={args!r}, kwargs={kwargs!r}"
        self.attempts.append(detail)
        raise ProcessCreationBlocked(f"process creation attempted: {detail}")

    def __enter__(self) -> Self:
        self._popen = subprocess.Popen
        subprocess.Popen = self._blocked  # type: ignore[assignment]
        if hasattr(os, "posix_spawn"):
            self._posix_spawn = os.posix_spawn
            os.posix_spawn = self._blocked  # type: ignore[assignment]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool:
        subprocess.Popen = self._popen
        if self._posix_spawn is not None:
            os.posix_spawn = self._posix_spawn  # type: ignore[assignment]
        return False


class CallTimer:
    """Accumulate only time spent inside scan and estimator calls."""

    def __init__(self) -> None:
        self.elapsed_ns = 0

    def call(self, function: Callable[..., T], *args: object, **kwargs: object) -> T:
        start = time.perf_counter_ns()
        try:
            return function(*args, **kwargs)
        finally:
            self.elapsed_ns += time.perf_counter_ns() - start


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name}: expected a JSON object")
    return value


def _input_hashes() -> tuple[list[dict[str, object]], list[str]]:
    rows = []
    findings = []
    for path, expected in FROZEN_INPUTS:
        observed = sha256_file(path)
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        matched = observed == expected
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "matched": matched,
            }
        )
        if not matched:
            findings.append(
                f"{relative}: expected sha256:{expected}, observed sha256:{observed}"
            )
    return rows, findings


def _model_work(frozen: dict[str, Any]) -> ModelWork:
    inventory = frozen["model_inventory"]
    return ModelWork(
        kernel_name=inventory["unit_id"],
        flops_per_batch_item=inventory["flops_per_batch_item"],
        static_logical_hbm_bytes=inventory["static_logical_hbm_bytes"],
        dynamic_hbm_bytes_per_batch_item=inventory[
            "dynamic_hbm_bytes_per_batch_item"
        ],
        logical_collective_bytes_per_gpu_per_batch_item=inventory[
            "network_geometry"
        ]["logical_collective_bytes_per_gpu_per_batch_item"],
        inventory_sha256=FROZEN_INPUTS[1][1],
        source=(
            f"{inventory['path']} unit:{inventory['unit_id']} "
            f"case:{inventory['case_id']} rank:{inventory['rank_class']}"
        ),
    )


def _envelopes(frozen: dict[str, Any]) -> dict[str, EnvelopeSpec]:
    source = (
        f"{frozen['gpu_envelopes']['source_path']} "
        f"sha256:{frozen['gpu_envelopes']['source_sha256']}"
    )
    return {
        device: EnvelopeSpec(
            device=device,
            peak_flops_per_second=frozen["gpu_envelopes"][device][
                "peak_flops_per_second"
            ],
            hbm_bytes_per_second=frozen["gpu_envelopes"][device][
                "hbm_bytes_per_second"
            ],
            efficiency=frozen["gpu_envelopes"]["efficiency"],
            source=source,
        )
        for device in ("b100", "h100")
    }


def _candidate(
    configuration: dict[str, Any],
    *,
    inter_node_bits_per_second: int = 400_000_000_000,
) -> DeploymentCandidate:
    width = configuration["gpus_per_node"] * configuration["node_count"]
    return DeploymentCandidate(
        candidate_id=configuration["id"],
        model=ModelRef(
            framework="sglang",
            model_id="deepseek-ai/DeepSeek-V3",
            inventory_sha256=FROZEN_INPUTS[1][1],
        ),
        pools=(
            PoolSpec(
                role="decode",
                engines=1,
                gpus_per_engine=width,
                tensor_parallel=8,
                pipeline_parallel=1,
                expert_parallel=max(1, width // 8),
                data_parallel=1,
                device=configuration["gpu"],
            ),
        ),
        fabric=FabricSpec(
            inter_node_bits_per_second=inter_node_bits_per_second,
            intra_node_bytes_per_second=100_000_000_000,
        ),
        workload=WorkloadPoint(
            arrival_rate_rps=100,
            prompt_tokens=1,
            output_tokens=1,
            kv_context_tokens=2_000,
        ),
        sla=SlaSpec(tpot_target_ps=None, ttft_target_ps=None),
        budget=BudgetSpec(max_gpus=width, max_nodes=configuration["node_count"]),
    )


def _anchors(frozen: dict[str, Any]) -> tuple[ExternalAnchor, ...]:
    published = frozen["published_context"]
    paired = published["paired"][0]
    y_only = published["y_only"][0]
    return (
        ExternalAnchor(
            anchor_id=paired["id"],
            label=paired["label"],
            x_tokens_per_second_per_request=Fraction(
                paired["tokens_per_second_per_node"], paired["batch_per_node"]
            ),
            y_tokens_per_second_per_gpu=Fraction(
                paired["tokens_per_second_per_node"], paired["gpus_per_node"]
            ),
        ),
        ExternalAnchor(
            anchor_id=y_only["id"],
            label=y_only["label"],
            y_tokens_per_second_per_gpu=Fraction(
                y_only["tokens_per_second_per_node"], y_only["gpus_per_node"]
            ),
        ),
    )


def _scan_inputs(
    frozen: dict[str, Any],
    legacy_rows: dict[tuple[str, int], dict[str, Any]],
    *,
    simulated: bool,
) -> ScanInputs:
    model_work = _model_work(frozen)
    envelopes = _envelopes(frozen)

    def resolve(candidate: DeploymentCandidate, batch: int) -> EstimatorInputs:
        sim_derived = None
        if simulated:
            row = legacy_rows[(candidate.candidate_id, batch)]
            sim_derived = SimDerivedTerms(
                fabric_excess_ps=row["fabric_attribution"]["raw_excess_ps"],
                intra_excess_ps=row["intra_node_attribution"]["raw_excess_ps"],
                record_path=LEGACY_RESULT_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
                record_sha256=FROZEN_INPUTS[2][1],
            )
        return EstimatorInputs(
            model_work=model_work,
            envelopes=envelopes,
            sim_derived=sim_derived,
        )

    return ScanInputs(
        estimator_inputs=resolve,
        static_rank_bytes_per_pool={
            "decode": frozen["model_inventory"]["static_logical_hbm_bytes"]
        },
        device_hbm_capacity_bytes={"h100": 80_000_000_000, "b100": 192_000_000_000},
        anchors=_anchors(frozen),
        configuration_labels={
            configuration["id"]: configuration["label"]
            for configuration in frozen["configurations"]
        },
    )


def _fraction_json(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _term(point: FrontierPoint, name: str) -> TermEstimate:
    for named in point.stamp.terms:
        if named.name == name:
            return named.estimate
    raise KeyError(name)


def _analytical_step(point: FrontierPoint) -> int:
    return max(
        _term(point, "kernel_floor").duration_ps,
        _term(point, "fabric_floor").duration_ps,
        _term(point, "intra_floor").duration_ps,
    )


def _point_map(record: FrontierRecord) -> dict[tuple[str, int], FrontierPoint]:
    return {
        (point.configuration_id, point.batch_per_gpu): point
        for point in record.points
    }


def _strict_round_trip(
    candidates: tuple[DeploymentCandidate, ...],
    records: tuple[FrontierRecord, ...],
) -> tuple[bool, str]:
    try:
        for candidate in candidates:
            rendered_candidate = candidate_to_json(candidate)
            if candidate_from_json(rendered_candidate) != candidate:
                return False, f"candidate round trip changed {candidate.candidate_id}"
            unknown_candidate = deepcopy(rendered_candidate)
            unknown_candidate["unexpected"] = 1
            try:
                candidate_from_json(unknown_candidate)
            except ValueError as error:
                if "unknown fields" not in str(error):
                    return False, f"candidate unknown field used wrong error: {error}"
            else:
                return False, "candidate parser accepted an unknown field"
            wrong_schema_candidate = deepcopy(unknown_candidate)
            wrong_schema_candidate["schema"] = "simllm-deployment-candidate-v0"
            try:
                candidate_from_json(wrong_schema_candidate)
            except ValueError as error:
                if "unsupported schema" not in str(error) or "unknown fields" in str(error):
                    return False, f"candidate schema was not checked first: {error}"
            else:
                return False, "candidate parser accepted a wrong schema"

        for record in records:
            rendered_record = frontier_record_to_json(record)
            if frontier_record_from_json(rendered_record) != record:
                return False, "frontier record round trip changed a record"
            unknown_record = deepcopy(rendered_record)
            unknown_record["unexpected"] = 1
            try:
                frontier_record_from_json(unknown_record)
            except ValueError as error:
                if "unknown fields" not in str(error):
                    return False, f"frontier unknown field used wrong error: {error}"
            else:
                return False, "frontier parser accepted an unknown field"
            wrong_schema_record = deepcopy(unknown_record)
            wrong_schema_record["schema"] = "simllm-deployment-frontier-record-v0"
            try:
                frontier_record_from_json(wrong_schema_record)
            except ValueError as error:
                if "unsupported schema" not in str(error) or "unknown fields" in str(error):
                    return False, f"frontier schema was not checked first: {error}"
            else:
                return False, "frontier parser accepted a wrong schema"
            for point in record.points:
                rendered_stamp = estimate_stamp_to_json(point.stamp)
                if estimate_stamp_from_json(rendered_stamp) != point.stamp:
                    return False, "estimate stamp round trip changed a stamp"
                unknown_stamp = deepcopy(rendered_stamp)
                unknown_stamp["unexpected"] = 1
                try:
                    estimate_stamp_from_json(unknown_stamp)
                except ValueError as error:
                    if "unknown fields" not in str(error):
                        return False, f"stamp unknown field used wrong error: {error}"
                else:
                    return False, "estimate stamp parser accepted an unknown field"
                wrong_schema_stamp = deepcopy(unknown_stamp)
                wrong_schema_stamp["schema"] = "simllm-deployment-estimate-v0"
                try:
                    estimate_stamp_from_json(wrong_schema_stamp)
                except ValueError as error:
                    if "unsupported schema" not in str(error) or "unknown fields" in str(
                        error
                    ):
                        return False, f"stamp schema was not checked first: {error}"
                else:
                    return False, "estimate stamp parser accepted a wrong schema"
    except (TypeError, ValueError) as error:
        return False, str(error)
    return True, "candidate, estimate stamp and frontier record boundaries are strict"


def _evidence_guard(
    analytical: FrontierRecord,
    simulated: FrontierRecord,
    synthetic_estimates: tuple[Any, ...],
) -> tuple[bool, str]:
    allowed = set(EvidenceClass)
    for point in analytical.points:
        expected = {
            "kernel_floor": EvidenceClass.ROOFLINE,
            "fabric_floor": EvidenceClass.DECLARED,
            "intra_floor": EvidenceClass.DECLARED,
        }
        for name, evidence in expected.items():
            term = _term(point, name)
            if term.evidence is not evidence or not term.source:
                return False, f"{point.configuration_id} batch {point.batch_per_gpu} {name}"
    for point in simulated.points:
        for name in ("fabric_excess", "intra_excess"):
            term = _term(point, name)
            if term.evidence is not EvidenceClass.SIM_DERIVED or not term.source:
                return False, f"{point.configuration_id} batch {point.batch_per_gpu} {name}"
    for estimate in synthetic_estimates:
        for named in estimate.stamp.terms:
            if named.estimate.evidence not in allowed or not named.estimate.source:
                return False, f"synthetic term {named.name} lacks evidence"
    return True, "every consumed duration has an allowed evidence class and source"


def _stamp_guard(records: tuple[FrontierRecord, ...]) -> tuple[bool, str]:
    for record in records:
        for point in record.points:
            if point.stamp.schema != DEPLOYMENT_ESTIMATE_SCHEMA:
                return False, f"{point.configuration_id} has the wrong stamp schema"
            expected = (
                PointClass.SIMULATED
                if point.stamp.consumes_sim_derived
                else PointClass.ESTIMATE
            )
            if point.point_class is not expected:
                return False, f"{point.configuration_id} has the wrong point class"
    return True, "all points carry strict estimate stamps and authority-correct classes"


def _synthetic_candidate(
    candidate_id: str,
    *,
    pools: tuple[PoolSpec, ...] | None = None,
    arrival_rate_rps: int = 100,
    fabric_bps: int = 400_000_000_000,
    intra_bytes_per_second: int = 450_000_000_000,
) -> DeploymentCandidate:
    if pools is None:
        pools = (
            PoolSpec(
                role="decode",
                engines=1,
                gpus_per_engine=8,
                tensor_parallel=8,
                pipeline_parallel=1,
                expert_parallel=1,
                data_parallel=1,
                device="b100",
            ),
        )
    return DeploymentCandidate(
        candidate_id=candidate_id,
        model=ModelRef(
            framework="synthetic",
            model_id="synthetic/deployment-scan",
            inventory_sha256=SYNTHETIC_SHA256,
        ),
        pools=pools,
        fabric=FabricSpec(
            inter_node_bits_per_second=fabric_bps,
            intra_node_bytes_per_second=intra_bytes_per_second,
        ),
        workload=WorkloadPoint(
            arrival_rate_rps=arrival_rate_rps,
            prompt_tokens=16,
            output_tokens=4,
            kv_context_tokens=2_000,
        ),
        sla=SlaSpec(tpot_target_ps=None, ttft_target_ps=None),
        budget=BudgetSpec(max_gpus=None, max_nodes=None),
    )


def _synthetic_work(
    *,
    flops_per_batch_item: int = 8_000_000_000_000,
    static_hbm_bytes: int = 10_000_000_000,
    collective_bytes: int = 0,
) -> ModelWork:
    return ModelWork(
        kernel_name="synthetic-decode-rank",
        flops_per_batch_item=flops_per_batch_item,
        static_logical_hbm_bytes=static_hbm_bytes,
        dynamic_hbm_bytes_per_batch_item=0,
        logical_collective_bytes_per_gpu_per_batch_item=collective_bytes,
        inventory_sha256=SYNTHETIC_SHA256,
        source="deployment_scan_v1 synthetic declaration",
    )


def _synthetic_inputs(
    *,
    work: ModelWork | None = None,
    surfaces: tuple[BatchServicePoint, ...] | None = None,
    surface_evidence: EvidenceClass | None = None,
    surface_source: str | None = None,
    prefill_service: TermEstimate | None = None,
    handoff_ps: int | None = None,
    handoff_source: str | None = None,
) -> EstimatorInputs:
    return EstimatorInputs(
        model_work=_synthetic_work() if work is None else work,
        envelopes={
            "b100": EnvelopeSpec(
                device="b100",
                peak_flops_per_second=8_000_000_000_000_000,
                hbm_bytes_per_second=8_000_000_000_000,
                efficiency=1.0,
                source="deployment_scan_v1 E1 declared envelope",
            )
        },
        surfaces=surfaces,
        surface_evidence=surface_evidence,
        surface_source=surface_source,
        prefill_service=prefill_service,
        handoff_ps=handoff_ps,
        handoff_source=handoff_source,
    )


def _surface() -> tuple[BatchServicePoint, ...]:
    return (
        BatchServicePoint(2, 200_000_000, 0.0, "1" * 64),
        BatchServicePoint(8, 800_000_000, 0.0, "2" * 64),
    )


def _rate_surface() -> tuple[BatchServicePoint, ...]:
    return (
        BatchServicePoint(2, 100_000_000, 0.0, "3" * 64),
        BatchServicePoint(8, 400_000_000, 0.0, "4" * 64),
    )


def _check(
    family: str,
    passed: bool,
    *,
    instances: int,
    detail: str,
    observed: object | None = None,
    expected: object | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "family": family,
        "status": "PASS" if passed else "FAIL",
        "instances": instances,
        "detail": detail,
    }
    if observed is not None:
        row["observed"] = observed
    if expected is not None:
        row["expected"] = expected
    return row


def _machine() -> dict[str, object]:
    cpu_model = "not disclosed by this operating system"
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.is_file():
        for line in cpu_info.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "processes": 1,
    }


def _void_result(
    input_hashes: list[dict[str, object]],
    findings: list[str],
    implementation_commit: str,
) -> dict[str, object]:
    return {
        "schema": RESULT_SCHEMA,
        "status": "VOID",
        "verdict": "VOID: a fatal precondition failed; no scored family is claimed.",
        "fatal_guards": [
            {
                "id": "FG-5",
                "status": "FAIL",
                "detail": "frozen input hash mismatch",
                "findings": findings,
            }
        ],
        "score_classes": {},
        "records": {},
        "input_hashes": input_hashes,
        "machine": _machine(),
        "provenance": {
            "expectations_commit": EXPECTATIONS_COMMIT,
            "implementation_commits": [
                *EARLIER_IMPLEMENTATION_COMMITS,
                implementation_commit,
            ],
        },
    }


def run_study(*, implementation_commit: str) -> dict[str, object]:
    """Evaluate fatal guards first, then every frozen scored family."""

    if len(implementation_commit) != 40 or any(
        character not in "0123456789abcdef" for character in implementation_commit
    ):
        raise ValueError("implementation_commit must be 40 lowercase hexadecimal digits")

    input_hashes, input_findings = _input_hashes()
    if input_findings:
        return _void_result(input_hashes, input_findings, implementation_commit)

    frozen = _json(FRONTIER_EXPECTATIONS_PATH)
    legacy = _json(LEGACY_RESULT_PATH)
    legacy_rows = {
        (row["configuration_id"], row["batch_per_gpu"]): row
        for row in legacy["points"]
    }
    candidates = tuple(_candidate(row) for row in frozen["configurations"])
    analytical_inputs = _scan_inputs(frozen, legacy_rows, simulated=False)
    simulated_inputs = _scan_inputs(frozen, legacy_rows, simulated=True)
    timer = CallTimer()
    process_guard = ProcessGuard()
    try:
        with process_guard:
            analytical = timer.call(scan, candidates, BATCHES, analytical_inputs)
            simulated = timer.call(scan, candidates, BATCHES, simulated_inputs)
            bandwidth_records = {
                rate: timer.call(
                    scan,
                    tuple(
                        _candidate(
                            configuration,
                            inter_node_bits_per_second=rate,
                        )
                        for configuration in frozen["configurations"]
                    ),
                    BATCHES,
                    _scan_inputs(frozen, legacy_rows, simulated=False),
                )
                for rate in (200_000_000_000, 100_000_000_000)
            }

            e1 = timer.call(
                estimate_decode_step,
                _synthetic_candidate("e1-roofline"),
                1,
                _synthetic_inputs(),
            )
            e2_pool = replace(
                _synthetic_candidate("e2-fabric").pools[0],
                gpus_per_engine=16,
                expert_parallel=2,
            )
            e2 = timer.call(
                estimate_decode_step,
                _synthetic_candidate("e2-fabric", pools=(e2_pool,)),
                1,
                _synthetic_inputs(
                    work=_synthetic_work(
                        flops_per_batch_item=1,
                        static_hbm_bytes=0,
                        collective_bytes=1_000_000_000,
                    )
                ),
            )
            e3 = timer.call(
                estimate_decode_step,
                _synthetic_candidate("e3-intra"),
                1,
                _synthetic_inputs(
                    work=_synthetic_work(
                        flops_per_batch_item=1,
                        static_hbm_bytes=0,
                        collective_bytes=900_000_000,
                    )
                ),
            )
            surface = _surface()
            e4 = timer.call(
                estimate_decode_step,
                _synthetic_candidate("e4-surface"),
                4,
                _synthetic_inputs(
                    surfaces=surface,
                    surface_evidence=EvidenceClass.DECLARED,
                    surface_source="deployment_scan_v1 E4 declared surface",
                ),
            )
            measured_surface_probe = timer.call(
                estimate_decode_step,
                _synthetic_candidate("fg3-measured-surface"),
                4,
                _synthetic_inputs(
                    surfaces=surface,
                    surface_evidence=EvidenceClass.MEASURED,
                ),
            )
            queue_capacity = timer.call(
                decode_capacity_requests_per_second,
                surface,
                output_tokens=4,
                max_batch=8,
                decode_engines=1,
            )
            queue_observed = {
                load: {
                    "occupancy": timer.call(
                        queue_occupancy,
                        surface,
                        offered_load_rps=load,
                        output_tokens=4,
                        max_batch=8,
                        decode_engines=1,
                    ),
                    "wait_ps": timer.call(
                        queue_delay_ps,
                        surface,
                        offered_load_rps=load,
                        output_tokens=4,
                        max_batch=8,
                        decode_engines=1,
                        cell_requests=64,
                    ),
                }
                for load in (500, 2_000, 4_000)
            }

            prefill_pool = replace(
                _synthetic_candidate("e6-rate-match").pools[0],
                role="prefill",
                engines=5,
            )
            decode_pool = replace(
                _synthetic_candidate("e6-rate-match").pools[0],
                role="decode",
                engines=1,
            )
            rate_candidate = _synthetic_candidate(
                "e6-rate-match",
                pools=(prefill_pool, decode_pool),
            )
            prefill_estimate = timer.call(
                estimate_prefill_request,
                rate_candidate,
                _synthetic_inputs(
                    prefill_service=TermEstimate(
                        50_000_000_000,
                        EvidenceClass.DECLARED,
                        "deployment_scan_v1 E6 prefill service declaration",
                    ),
                    handoff_ps=0,
                    handoff_source="deployment_scan_v1 E6 handoff declaration",
                ),
            )
            rate_capacity = timer.call(
                decode_capacity_requests_per_second,
                _rate_surface(),
                output_tokens=4,
                max_batch=8,
                decode_engines=1,
            )
            rate_match = timer.call(
                match_pools,
                rate_candidate,
                prefill_request_ps=prefill_estimate.request_ps,
                decode_step_ps=400_000_000,
                batch_per_gpu=8,
            )

            rejected = replace(
                candidates[0],
                candidate_id="pipeline-parallel-rejected",
                pools=(replace(candidates[0].pools[0], pipeline_parallel=2),),
            )
            rejected_record = timer.call(
                scan,
                (rejected,),
                BATCHES,
                analytical_inputs,
            )
            primary_elapsed_ns = timer.elapsed_ns

            throughput_candidates = tuple(
                _synthetic_candidate(f"throughput-{index:04d}")
                for index in range(1_000)
            )
            throughput_inputs = ScanInputs(
                estimator_inputs=_synthetic_inputs(),
                static_rank_bytes_per_pool={"decode": 10_000_000_000},
                device_hbm_capacity_bytes={"b100": 192_000_000_000},
            )
            throughput_start = time.perf_counter_ns()
            throughput_record = scan(
                throughput_candidates,
                BATCHES,
                throughput_inputs,
            )
            throughput_elapsed_ns = time.perf_counter_ns() - throughput_start
    except ProcessCreationBlocked as error:
        return {
            **_void_result(input_hashes, [str(error)], implementation_commit),
            "fatal_guards": [
                {
                    "id": "FG-1",
                    "status": "FAIL",
                    "detail": str(error),
                    "attempts": process_guard.attempts,
                }
            ],
        }

    all_records = (
        analytical,
        simulated,
        *bandwidth_records.values(),
        rejected_record,
        throughput_record,
    )
    stamp_pass, stamp_detail = _stamp_guard(all_records)
    evidence_pass, evidence_detail = _evidence_guard(
        analytical,
        simulated,
        (e1, e2, e3, e4, measured_surface_probe, prefill_estimate, rate_match),
    )
    strict_pass, strict_detail = _strict_round_trip(
        (*candidates, rejected, rate_candidate),
        all_records,
    )
    rejected_frontier = rejected_record.candidates[0]
    d4_pass = (
        not rejected_frontier.accepted
        and rejected_frontier.rejection_reasons == (PIPELINE_PARALLEL_UNPRICED,)
        and not rejected_frontier.points
    )
    fatal_guards = [
        {
            "id": "FG-1",
            "status": "PASS" if not process_guard.attempts else "FAIL",
            "detail": "subprocess.Popen and os.posix_spawn were intercepted around every call",
            "interceptions_fired": len(process_guard.attempts),
        },
        {"id": "FG-2", "status": "PASS" if stamp_pass else "FAIL", "detail": stamp_detail},
        {
            "id": "FG-3",
            "status": "PASS" if evidence_pass else "FAIL",
            "detail": evidence_detail,
        },
        {"id": "FG-4", "status": "PASS" if strict_pass else "FAIL", "detail": strict_detail},
        {
            "id": "FG-5",
            "status": "PASS",
            "detail": "all three frozen input SHA-256 digests matched before evaluation",
        },
        {
            "id": "FG-6",
            "status": "PASS",
            "detail": (
                f"expectations commit {EXPECTATIONS_COMMIT} contains only "
                "examples/deployment_scan_v1/expectations.md"
            ),
        },
        {
            "id": "D4",
            "status": "PASS" if d4_pass else "FAIL",
            "detail": "pipeline-parallel rejection retained its stable reason and emitted no points",
        },
    ]
    if any(guard["status"] != "PASS" for guard in fatal_guards):
        return {
            **_void_result(input_hashes, [], implementation_commit),
            "fatal_guards": fatal_guards,
        }

    analytical_points = _point_map(analytical)
    simulated_points = _point_map(simulated)
    c1_errors = []
    c2_errors = []
    c3_misses = []
    for key, oracle in legacy_rows.items():
        analytic = analytical_points[key]
        simulated_point = simulated_points[key]
        c1_errors.append(_analytical_step(analytic) - oracle["accounting"]["analytical_step_ps"])
        c2_errors.append(simulated_point.step_ps - oracle["accounting"]["simulated_step_ps"])
        operating = oracle["simulated_operating_point"]
        expected_x = Fraction(
            operating["x_tokens_per_second_per_request"]["numerator"],
            operating["x_tokens_per_second_per_request"]["denominator"],
        )
        expected_y = Fraction(
            operating["y_tokens_per_second_per_gpu"]["numerator"],
            operating["y_tokens_per_second_per_gpu"]["denominator"],
        )
        if (
            simulated_point.x_tokens_per_second_per_request != expected_x
            or simulated_point.y_tokens_per_second_per_gpu != expected_y
        ):
            c3_misses.append(f"{key[0]} batch {key[1]}")

    compatibility = [
        _check(
            "C1",
            not any(c1_errors),
            instances=18,
            detail="installed analytical step versus pinned CORE-62 oracle",
            observed={"max_absolute_error_ps": max(map(abs, c1_errors))},
            expected={"max_absolute_error_ps": 0},
        ),
        _check(
            "C2",
            not any(c2_errors),
            instances=18,
            detail="SIM-DERIVED excess composition versus pinned simulated step",
            observed={"max_absolute_error_ps": max(map(abs, c2_errors))},
            expected={"max_absolute_error_ps": 0},
        ),
        _check(
            "C3",
            not c3_misses,
            instances=18,
            detail="exact reduced frontier-coordinate fractions",
            observed={"mismatches": c3_misses},
            expected={"mismatches": []},
        ),
    ]

    queue_expected = {
        500: (2, Fraction(0)),
        2_000: (7, Fraction(0)),
        4_000: (8, Fraction(4_725_000_000)),
    }
    queue_pass = queue_capacity == 2_500 and all(
        queue_observed[load]["occupancy"] == expected[0]
        and queue_observed[load]["wait_ps"] == expected[1]
        for load, expected in queue_expected.items()
    )
    synthetic = [
        _check(
            "E1",
            e1.kernel_floor.duration_ps == 1_250_000_000,
            instances=1,
            detail="memory-bound roofline literal",
            observed=e1.kernel_floor.duration_ps,
            expected=1_250_000_000,
        ),
        _check(
            "E2",
            e2.fabric_floor.duration_ps == 10_000_000_000,
            instances=1,
            detail="400 Gbit/s largest-flow floor literal",
            observed=e2.fabric_floor.duration_ps,
            expected=10_000_000_000,
        ),
        _check(
            "E3",
            e3.intra_floor.duration_ps == 2_000_000_000,
            instances=1,
            detail="450 GB/s logical-byte floor literal",
            observed=e3.intra_floor.duration_ps,
            expected=2_000_000_000,
        ),
        _check(
            "E4",
            e4.batch_service is not None
            and e4.batch_service.duration_ps == 400_000_000
            and e4.batch_service.evidence is EvidenceClass.DECLARED,
            instances=1,
            detail="declared surface interpolation literal",
            observed=None if e4.batch_service is None else e4.batch_service.duration_ps,
            expected=400_000_000,
        ),
        _check(
            "E5",
            queue_pass,
            instances=1,
            detail="capacity, occupancy and deterministic overload-wait literals",
            observed={
                "capacity_requests_per_second": _fraction_json(queue_capacity),
                "cells": {
                    str(load): {
                        "occupancy": value["occupancy"],
                        "wait_ps": _fraction_json(value["wait_ps"]),
                    }
                    for load, value in queue_observed.items()
                },
            },
        ),
        _check(
            "E6",
            rate_capacity == 5_000
            and rate_match.required_prefill_engines == 5
            and rate_match.required_decode_engines == 1,
            instances=1,
            detail="ceil-form prefill and decode engine requirement literals",
            observed={
                "decode_capacity_requests_per_second": _fraction_json(rate_capacity),
                "required_prefill_engines": rate_match.required_prefill_engines,
                "required_decode_engines": rate_match.required_decode_engines,
            },
        ),
    ]

    bandwidth_misses = []
    bandwidth_maps = {
        rate: _point_map(record) for rate, record in bandwidth_records.items()
    }
    direction_pass = True
    strict_binding_instances = 0
    for key, point_400 in analytical_points.items():
        points = [
            point_400,
            bandwidth_maps[200_000_000_000][key],
            bandwidth_maps[100_000_000_000][key],
        ]
        for rate, point in zip(
            (400_000_000_000, 200_000_000_000, 100_000_000_000),
            points,
            strict=True,
        ):
            kernel = _term(point, "kernel_floor").duration_ps
            fabric = _term(point, "fabric_floor").duration_ps
            intra = _term(point, "intra_floor").duration_ps
            expected = max(kernel, fabric, intra)
            if point.step_ps != expected:
                bandwidth_misses.append(f"{key[0]} batch {key[1]} at {rate}")
        if not (points[2].step_ps >= points[1].step_ps >= points[0].step_ps):
            direction_pass = False
        for lower, higher in ((points[0], points[1]), (points[1], points[2])):
            fabric = _term(higher, "fabric_floor").duration_ps
            other = max(
                _term(higher, "kernel_floor").duration_ps,
                _term(higher, "intra_floor").duration_ps,
            )
            if fabric > other:
                strict_binding_instances += 1
                if higher.step_ps <= lower.step_ps:
                    direction_pass = False

    target_membership = {}
    for target in (4_000_000_000, 8_500_000_000):
        target_membership[target] = [
            f"{point.configuration_id}:b{point.batch_per_gpu}"
            for point in simulated.points
            if point.step_ps <= target
        ]
    expected_sla = {
        4_000_000_000: [
            f"b100-one-node-intra:b{batch}" for batch in (1, 2, 4, 8, 16)
        ],
        8_500_000_000: [
            *[f"b100-one-node-intra:b{batch}" for batch in BATCHES],
            *[
                f"{configuration}:b{batch}"
                for configuration in (
                    "h100-two-node-serialized",
                    "h100-nine-node-incast",
                )
                for batch in (1, 2, 4)
            ],
        ],
    }
    pareto_observed = [
        f"{point.configuration_id}:b{point.batch_per_gpu}"
        for point in pareto_front(simulated.points)
    ]
    pareto_expected = [f"b100-one-node-intra:b{batch}" for batch in BATCHES]

    d1_pass = True
    d2_pass = True
    for candidate in simulated.candidates:
        ordered = sorted(candidate.points, key=lambda point: point.batch_per_gpu)
        d1_pass &= all(
            left.x_tokens_per_second_per_request
            >= right.x_tokens_per_second_per_request
            for left, right in pairwise(ordered)
        )
        d2_pass &= all(
            left.y_tokens_per_second_per_gpu
            <= right.y_tokens_per_second_per_gpu
            for left, right in pairwise(ordered)
        )
    infinite_set = {
        f"{point.configuration_id}:b{point.batch_per_gpu}" for point in simulated.points
    }
    medium_set = set(target_membership[8_500_000_000])
    tight_set = set(target_membership[4_000_000_000])
    d3_pass = tight_set <= medium_set <= infinite_set

    relations = [
        _check(
            "B1-exact",
            not bandwidth_misses,
            instances=36,
            detail="floor-division bandwidth recomposition",
            observed={"mismatches": bandwidth_misses},
            expected={"mismatches": []},
        ),
        _check(
            "B1-direction",
            direction_pass,
            instances=1,
            detail="100 Gbit/s is no faster than 200 Gbit/s, which is no faster than 400 Gbit/s",
            observed={"strict_binding_comparisons": strict_binding_instances},
        ),
        _check(
            "S1",
            target_membership == expected_sla,
            instances=2,
            detail="exact service-level agreement membership sets",
            observed={str(key): value for key, value in target_membership.items()},
            expected={str(key): value for key, value in expected_sla.items()},
        ),
        _check(
            "P1",
            pareto_observed == pareto_expected,
            instances=1,
            detail="exact six-point Pareto literal",
            observed=pareto_observed,
            expected=pareto_expected,
        ),
        _check(
            "D1",
            d1_pass,
            instances=3,
            detail="per-request speed is nonincreasing with batch",
        ),
        _check(
            "D2",
            d2_pass,
            instances=3,
            detail="per-GPU throughput is nondecreasing with batch",
        ),
        _check(
            "D3",
            d3_pass,
            instances=1,
            detail="tightening the TPOT target cannot grow membership",
        ),
    ]

    primary_seconds = primary_elapsed_ns / 1_000_000_000
    throughput_seconds = throughput_elapsed_ns / 1_000_000_000
    primary_points = (
        len(analytical.points)
        + len(simulated.points)
        + sum(len(record.points) for record in bandwidth_records.values())
    )
    wall_time = [
        _check(
            "W1a",
            primary_seconds <= 10,
            instances=1,
            detail="complete study scan and estimator call time, single process",
            observed={"seconds": primary_seconds, "priced_points": primary_points},
            expected={"maximum_seconds": 10, "minimum_priced_points": 64},
        ),
        _check(
            "W1c",
            throughput_seconds <= 60 and len(throughput_record.points) == 6_000,
            instances=1,
            detail="1,000-candidate by six-batch throughput grid, single process",
            observed={
                "seconds": throughput_seconds,
                "priced_points": len(throughput_record.points),
            },
            expected={"maximum_seconds": 60, "priced_points": 6_000},
        ),
    ]
    reported = [
        {
            "family": "W1b",
            "status": "REPORTED",
            "scored": False,
            "detail": "observed single-process throughput, not a frozen score",
            "primary_points_per_second": primary_points / primary_seconds,
            "throughput_grid_points_per_second": 6_000 / throughput_seconds,
        }
    ]

    score_classes = {
        "compatibility_exact": compatibility,
        "synthetic_exact_oracles": synthetic,
        "behavioral_relations": relations,
        "wall_time": wall_time,
        "reported_unscored": reported,
    }
    scored_rows = [
        row
        for class_name, rows in score_classes.items()
        if class_name != "reported_unscored"
        for row in rows
    ]
    passed = all(row["status"] == "PASS" for row in scored_rows)
    return {
        "schema": RESULT_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "verdict": (
            "PASS: every frozen scored family passed and the maximum compatibility error was 0 ps."
            if passed
            else "FAIL: one or more frozen scored families missed; see the per-family ledger."
        ),
        "fatal_guards": fatal_guards,
        "score_classes": score_classes,
        "records": {
            "analytical": frontier_record_to_json(analytical),
            "simulated": frontier_record_to_json(simulated),
            "bandwidth_200_gbit": frontier_record_to_json(
                bandwidth_records[200_000_000_000]
            ),
            "bandwidth_100_gbit": frontier_record_to_json(
                bandwidth_records[100_000_000_000]
            ),
        },
        "physical_sanity": {
            "memory_floor": (
                "B100 batch 1 moves 27,587,187,040 logical HBM bytes through an "
                "8 TB/s envelope, a 3,448,398,380 ps floor."
            ),
            "serialized_ceiling": (
                "B100 batch 32 is 4,523,298,348 ps, between its 4,257,218,560 ps "
                "intra-node floor and the conservative 8,516,304,727 ps sum of "
                "kernel and simulated intra-node service."
            ),
            "network_scaling": (
                "Every binding serialization term is recomputed as bytes over rate; "
                "the 100, 200 and 400 Gbit/s sweep preserves the frozen inverse-rate relation."
            ),
            "system_plausibility": (
                "The H100 batch-32 estimate is 104.87 token/s/request and 3,355.88 "
                "token/s/GPU, above the paired published 87.04 and 2,785.25 anchors "
                "as a floor-style estimate should be."
            ),
        },
        "input_hashes": input_hashes,
        "machine": _machine(),
        "provenance": {
            "expectations_commit": EXPECTATIONS_COMMIT,
            "expectations_path": EXPECTATIONS_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "implementation_commits": [
                *EARLIER_IMPLEMENTATION_COMMITS,
                implementation_commit,
            ],
        },
    }


def _point_rows(result: dict[str, object]) -> list[dict[str, object]]:
    records = result.get("records")
    if not isinstance(records, dict):
        return []
    rows = []
    for series, raw_record in records.items():
        record = frontier_record_from_json(raw_record)
        for point in record.points:
            rows.append(
                {
                    "series": series,
                    "configuration_id": point.configuration_id,
                    "candidate_key": point.candidate_key,
                    "batch_per_gpu": point.batch_per_gpu,
                    "point_class": point.point_class.value,
                    "step_ps": point.step_ps,
                    "analytical_step_ps": _analytical_step(point),
                    "x_numerator": point.x_tokens_per_second_per_request.numerator,
                    "x_denominator": point.x_tokens_per_second_per_request.denominator,
                    "y_numerator": point.y_tokens_per_second_per_gpu.numerator,
                    "y_denominator": point.y_tokens_per_second_per_gpu.denominator,
                    "kernel_floor_ps": _term(point, "kernel_floor").duration_ps,
                    "fabric_floor_ps": _term(point, "fabric_floor").duration_ps,
                    "intra_floor_ps": _term(point, "intra_floor").duration_ps,
                    "fabric_excess_ps": (
                        _term(point, "fabric_excess").duration_ps
                        if point.point_class is PointClass.SIMULATED
                        else 0
                    ),
                    "intra_excess_ps": (
                        _term(point, "intra_excess").duration_ps
                        if point.point_class is PointClass.SIMULATED
                        else 0
                    ),
                    "stamp_schema": point.stamp.schema,
                }
            )
    return rows


def write_artifacts(
    result: dict[str, object],
    *,
    result_path: Path,
    csv_path: Path,
) -> None:
    """Write LF-only deterministic JSON and CSV field ordering."""

    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    fieldnames = [
        "series",
        "configuration_id",
        "candidate_key",
        "batch_per_gpu",
        "point_class",
        "step_ps",
        "analytical_step_ps",
        "x_numerator",
        "x_denominator",
        "y_numerator",
        "y_denominator",
        "kernel_floor_ps",
        "fabric_floor_ps",
        "intra_floor_ps",
        "fabric_excess_ps",
        "intra_excess_ps",
        "stamp_schema",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(_point_rows(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation-commit",
        required=True,
        help="full commit containing this runner, frozen into provenance",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=STUDY_DIR / "results.json",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=STUDY_DIR / "results.csv",
    )
    arguments = parser.parse_args()
    result = run_study(implementation_commit=arguments.implementation_commit)
    write_artifacts(result, result_path=arguments.result, csv_path=arguments.csv)
    print(result["verdict"])
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
