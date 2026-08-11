from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rnic_authority_v1 import (
    check_results,
    produce_observations,
    run_study,
)
from simllm.backends import (
    BypassArtifacts,
    ComposedRnicCell,
    ComposedWqeObservation,
)
from simllm.core import (
    CoarseDeviceRuntime,
    CompletionReducer,
    ControlMode,
    ControlWork,
    ExecutionGraph,
    ExecutionResult,
    RequestPhase,
    RuntimeReport,
    StepRecord,
    execution_graph_to_json,
    step_record_to_json,
)

STUDY_DIR = REPO_ROOT / "examples" / "rnic_authority_v1"
EXPECTATIONS = STUDY_DIR / "expectations.json"


def _expectations() -> dict[str, Any]:
    return produce_observations.load_expectations(EXPECTATIONS)


def _cell(rate_gbps: int, doorbell_ps: int) -> ComposedRnicCell:
    wire_ps = 4096 * 8 * 1000 // rate_gbps
    wqes = tuple(
        ComposedWqeObservation(
            ordinal=ordinal,
            native_wqe_id=ordinal + 1,
            eligible_at_ps=doorbell_ps,
            network_started_at_ps=doorbell_ps + ordinal * wire_ps,
            network_finished_at_ps=doorbell_ps + (ordinal + 1) * wire_ps,
            completed_at_ps=doorbell_ps + (ordinal + 1) * wire_ps,
        )
        for ordinal in range(2)
    )
    return ComposedRnicCell(
        payload_bytes=4096,
        link_rate_gbps=rate_gbps,
        doorbell_service_ps=doorbell_ps,
        wqes=wqes,
        jct_ps=doorbell_ps + 2 * wire_ps,
    )


def _metric(mode: dict[str, Any], request_id: str) -> dict[str, Any]:
    return next(
        metric
        for metric in mode["step_result"]["request_metrics"]
        if metric["request_id"] == request_id
    )


def _fraction(value: int) -> dict[str, int]:
    return {"numerator": value, "denominator": 1}


def _summary_ps(summary: dict[str, Any], metric: str) -> int:
    value = summary[metric]
    if isinstance(value, dict):
        assert value["denominator"] == 1
        return value["numerator"]
    assert isinstance(value, int)
    return value


def test_fixed_graph_and_step_record_match_the_frozen_semantic_input() -> None:
    expectations = _expectations()

    graph = produce_observations.build_fixed_graph(expectations)
    record = produce_observations.build_fixed_step_record(expectations)

    assert graph == ExecutionGraph(
        execution_id="core21-fixed-contended-graph",
        step_index=1,
        released_at_ps=17_000,
        operations=graph.operations,
        completion_operation_ids=("core21-contended-send",),
    )
    assert len(graph.operations) == 1
    operation = graph.operations[0]
    assert operation.operation_id == "core21-contended-send"
    assert operation.rank == 0
    assert operation.logical_queue == "core21-control"
    assert operation.depends_on == ()
    assert operation.participant_local_depends_on == ()
    assert operation.not_before_ps == 0
    assert operation.priority == 0
    assert operation.placement_epoch == 0
    assert operation.correlation.request_ids == (
        "core21-prefill",
        "core21-decode",
    )
    assert operation.correlation.batch_id is None
    assert operation.correlation.layer is None
    assert operation.correlation.microbatch is None
    assert operation.correlation.iteration is None
    assert isinstance(operation.work, ControlWork)
    assert operation.work.destination_ranks == (8, 16)
    assert operation.work.payload_bytes == 4096
    assert operation.work.mode is ControlMode.SYNCHRONOUS

    assert isinstance(record, StepRecord)
    assert record.step_index == graph.step_index
    assert record.virtual_time_ps == graph.released_at_ps
    assert [request.request_id for request in record.scheduled] == [
        "core21-prefill",
        "core21-decode",
    ]
    assert [request.phase for request in record.scheduled] == [
        RequestPhase.PREFILL,
        RequestPhase.DECODE,
    ]
    assert [request.num_new_tokens for request in record.scheduled] == [1, 1]
    assert record.num_sampled == 2
    assert record.sampled_request_ids == ["core21-prefill", "core21-decode"]
    assert record.preempted_request_ids == []
    assert record.finished_request_ids == []
    assert record.num_tokens_after_padding is None


def test_bypass_and_structural_modes_share_objects_and_use_the_real_reducer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expectations = _expectations()
    calls: list[tuple[StepRecord, ExecutionGraph, ExecutionResult, RuntimeReport]] = []
    original_reduce = CompletionReducer.reduce

    def observe_reduce(
        self: CompletionReducer,
        record: StepRecord,
        graph: ExecutionGraph,
        result: ExecutionResult,
        report: RuntimeReport,
    ):
        calls.append((record, graph, result, report))
        return original_reduce(self, record, graph, result, report)

    monkeypatch.setattr(CompletionReducer, "reduce", observe_reduce)

    raw = produce_observations.run_authority_cell(_cell(400, 1_000), expectations)

    assert set(raw) == set(expectations["raw_cell_keys"])
    assert set(raw["bypass"]) == set(expectations["raw_mode_keys"])
    assert set(raw["structural"]) == set(expectations["raw_mode_keys"])
    history_calls = [call for call in calls if call[1].execution_id == "core21-history-seed"]
    assert len(history_calls) == 2
    assert history_calls[0][0] is history_calls[1][0]
    assert history_calls[0][1] is history_calls[1][1]
    assert history_calls[0][2] is history_calls[1][2]
    assert history_calls[0][3] is history_calls[1][3]
    decision_calls = [
        call for call in calls if call[1].execution_id == "core21-fixed-contended-graph"
    ]
    assert len(decision_calls) == 2
    assert decision_calls[0][0] is decision_calls[1][0]
    assert decision_calls[0][1] is decision_calls[1][1]
    assert all(isinstance(call[2], ExecutionResult) for call in decision_calls)
    assert all(isinstance(call[3], RuntimeReport) for call in decision_calls)
    assert raw["execution_graph"] == execution_graph_to_json(decision_calls[0][1])
    assert raw["step_record"] == step_record_to_json(decision_calls[0][0])
    assert raw["history_seed"]["step_result"]["step_latency_ps"] == 10_000
    history_metrics = raw["history_seed"]["step_result"]["request_metrics"]
    assert len(history_metrics) == 1
    assert history_metrics[0]["request_id"] == "core21-decode"
    assert history_metrics[0]["token_index"] == 1
    assert history_metrics[0]["ttft_ps"] == 10_000
    assert history_metrics[0]["tpot_ps"] is None

    assert raw["bypass"]["hardware_mode"] == "bypass"
    assert raw["bypass"]["authority"] == "AtlahsWqeLedger"
    assert raw["bypass"]["runtime_report"]["authority"] == "AtlahsWqeLedger"
    assert raw["structural"]["hardware_mode"] == "structural"
    assert raw["structural"]["authority"] == "SimllmNativeRnicSession"
    assert raw["structural"]["runtime_report"]["authority"] == "SimllmNativeRnicSession"
    assert raw["bypass"]["wqe_count"] == 2
    assert raw["structural"]["wqe_count"] == 2
    assert {wqe["authority"] for wqe in raw["bypass"]["runtime_report"]["wqes"]} == {
        "AtlahsWqeLedger"
    }
    assert {wqe["authority"] for wqe in raw["structural"]["runtime_report"]["wqes"]} == {
        "SimllmNativeRnicSession"
    }
    assert raw["structural"]["bypass_artifacts"] is None
    assert {visit["stage"] for visit in raw["structural"]["runtime_report"]["visits"]} >= {
        "native_doorbell",
        "native_network",
    }


def test_mixed_request_metrics_expose_the_exact_signed_authority_delta() -> None:
    expectations = _expectations()
    raw = produce_observations.run_authority_cell(_cell(400, 1_000), expectations)
    bypass = raw["bypass"]
    structural = raw["structural"]

    assert bypass["execution_result"]["completed_at_ps"] == 180_840
    assert structural["execution_result"]["completed_at_ps"] == 181_840
    assert bypass["step_result"]["step_latency_ps"] == 163_840
    assert structural["step_result"]["step_latency_ps"] == 164_840

    bypass_prefill = _metric(bypass, "core21-prefill")
    structural_prefill = _metric(structural, "core21-prefill")
    bypass_decode = _metric(bypass, "core21-decode")
    structural_decode = _metric(structural, "core21-decode")
    assert bypass_prefill["ttft_ps"] == 163_840
    assert structural_prefill["ttft_ps"] == 164_840
    assert bypass_decode["token_index"] == 2
    assert structural_decode["token_index"] == 2
    assert bypass_decode["tpot_ps"] == _fraction(163_840)
    assert structural_decode["tpot_ps"] == _fraction(164_840)
    assert bypass["request_summary"] == {
        "jct_ps": 163_840,
        "prefill_ttft_ps": 163_840,
        "decode_tpot_ps": _fraction(163_840),
    }
    assert structural["request_summary"] == {
        "jct_ps": 164_840,
        "prefill_ttft_ps": 164_840,
        "decode_tpot_ps": _fraction(164_840),
    }
    assert {
        metric: _summary_ps(structural["request_summary"], metric)
        - _summary_ps(bypass["request_summary"], metric)
        for metric in (
            "jct_ps",
            "prefill_ttft_ps",
            "decode_tpot_ps",
        )
    } == {
        "jct_ps": 1_000,
        "prefill_ttft_ps": 1_000,
        "decode_tpot_ps": 1_000,
    }


def test_registered_cell_runs_transaction_failure_before_one_successful_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[tuple[str, str, str, bool]] = []
    original_execute = CoarseDeviceRuntime.execute

    def observe_execute(
        self: CoarseDeviceRuntime,
        graph: ExecutionGraph,
        **kwargs: Any,
    ) -> ExecutionResult:
        try:
            result = original_execute(self, graph, **kwargs)
        except BaseException as error:
            attempts.append(
                (
                    self.authority_name,
                    graph.execution_id,
                    type(error).__name__,
                    self.last_report is None,
                )
            )
            raise
        attempts.append(
            (
                self.authority_name,
                graph.execution_id,
                "success",
                self.last_report is None,
            )
        )
        return result

    monkeypatch.setattr(CoarseDeviceRuntime, "execute", observe_execute)
    raw = produce_observations.run_authority_cell(_cell(400, 1_000), _expectations())
    failure = raw["transaction_failure"]

    structural_attempts = [
        attempt for attempt in attempts if attempt[0] == "SimllmNativeRnicSession"
    ]
    assert structural_attempts == [
        (
            "SimllmNativeRnicSession",
            "core21-incomplete-transaction",
            "ValueError",
            True,
        ),
        (
            "SimllmNativeRnicSession",
            "core21-fixed-contended-graph",
            "success",
            False,
        ),
    ]

    assert failure["exception_type"] == "ValueError"
    assert "consume every WQE" in failure["exception_message"]
    assert failure["before"] == {
        "committed_transactions": 0,
        "committed_wqes": 0,
    }
    assert failure["after"] == failure["before"]
    assert failure["retry"] == {
        "committed_transactions": 1,
        "committed_wqes": 2,
    }
    assert failure["runtime_last_report_is_none"] is True
    assert failure["bypass_ledger_is_none"] is True
    assert failure["authority"] == "SimllmNativeRnicSession"
    assert raw["structural"]["runtime_report"]["random_draw_count"] == 0


def test_bypass_bundle_uses_the_repository_artifact_comparator() -> None:
    expectations = _expectations()
    mode_row, reference = produce_observations.run_bypass_replay(expectations, rate_gbps=400)
    candidate = produce_observations.bypass_artifacts_from_mode(mode_row)

    assert isinstance(reference, BypassArtifacts)
    assert isinstance(candidate, BypassArtifacts)
    assert run_study.compare_bypass_replay(reference, mode_row).equivalent
    changed_mode = copy.deepcopy(mode_row)
    changed_mode["bypass_artifacts"]["step_results_hex"] += "0a"
    comparison = run_study.compare_bypass_replay(reference, changed_mode)
    assert not comparison.equivalent
    assert comparison.changed_artifacts == ("step_results",)


@pytest.mark.parametrize(
    "script,arguments",
    [
        (
            "produce_observations.py",
            [
                "--expectations",
                str(EXPECTATIONS),
                "--observations",
                "{out}/raw_observations.json",
                "--tier-a-producer",
                "{tmp}/missing-tier-a",
                "--htsim-rnic",
                "{tmp}/missing-htsim-rnic",
                "--txt2bin",
                "{tmp}/missing-txt2bin",
            ],
        ),
        (
            "check_results.py",
            [
                "--expectations",
                str(EXPECTATIONS),
                "--observations",
                "{out}/raw_observations.json",
                "--negative-evidence",
                "{out}/negative.json",
                "--results",
                "{out}/results.json",
            ],
        ),
    ],
)
def test_registered_check_only_commands_create_no_output(
    script: str,
    arguments: list[str],
    tmp_path: Path,
) -> None:
    out = tmp_path / "must-remain-absent"
    formatted = [argument.format(out=out, tmp=tmp_path) for argument in arguments]
    completed = subprocess.run(
        [
            sys.executable,
            str(STUDY_DIR / script),
            *formatted,
            "--check-only",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert not out.exists()


def test_registered_runner_check_only_creates_no_output_when_source_is_available(
    tmp_path: Path,
) -> None:
    source_value = os.environ.get("SIMLLM_HTSIM_PIN_ROOT")
    if source_value is None or not Path(source_value).is_dir():
        pytest.skip(
            "the registered runner validates the external pinned htsim source, "
            "which CI does not initialize"
        )
    out = tmp_path / "must-remain-absent"
    completed = subprocess.run(
        [
            sys.executable,
            str(STUDY_DIR / "run_study.py"),
            "--out",
            str(out),
            "--htsim-source",
            source_value,
            "--check-only",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert not out.exists()


def _observations(expectations: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": expectations["observation_schema"],
        "simllm_source_commit": expectations["simllm_base_commit"],
        "htsim_source_commit": expectations["htsim_source_commit"],
        "native_observations_sha256": "1" * 64,
        "preflight": {
            "profile": "rnic-nn",
            "manifest": [
                ("hardware_mode=structural wqe_authority=simllm-native-rnic-session"),
                "physical_quiescence=verified",
            ],
            "flow_count": 1,
            "quiescent": True,
        },
        "cells": [
            produce_observations.run_authority_cell(_cell(rate, doorbell), expectations)
            for rate in (200, 400)
            for doorbell in (0, 1_000)
        ],
    }


def _negative_evidence(expectations: dict[str, Any]) -> dict[str, Any]:
    preserved = {
        name: str(index % 10) * 64
        for index, name in enumerate(sorted(check_results.POSITIVE_ASSET_NAMES), start=1)
    }
    preserved["native_observations_sha256"] = "1" * 64
    return {
        "schema": "simllm-rnic-authority-negative-evidence-v1",
        "htsim_source_commit": expectations["htsim_source_commit"],
        "cache_value": "OFF",
        "negative_main_sha256": "6" * 64,
        "positive_assets_before": preserved,
        "positive_assets_after": dict(preserved),
        "positive_assets_preserved": True,
        "bypass_comparison": {
            "input_matches": [
                [name, True]
                for name in (
                    "goal_text",
                    "goal_binary",
                    "topology",
                    "profile",
                    "seed",
                    "baseline_parameters",
                )
            ],
            "behavioral_matches": [
                [name, True]
                for name in (
                    "completion_csv",
                    "canonical_completion",
                    "step_results",
                    "replay_summary",
                )
            ],
            "equivalent": True,
        },
        "negative_tier_a_target_absent": True,
        "producer_returncode": 2,
        "producer_stderr": ("composed preflight lacks structural native authority\n"),
        "checker_returncode": 2,
        "checker_stderr": "authority observations do not exist\n",
        "forbidden_outputs_absent": {
            name: True for name in expectations["negative_control"]["forbidden_outputs"]
        },
    }


def test_checker_evaluates_metric_families_before_entailing_exact_oracles() -> None:
    expectations = _expectations()
    report = check_results.check_observations(
        _observations(expectations),
        expectations,
        _negative_evidence(expectations),
    )

    order = report["evaluation_order"]
    signed = order.index("signed_authority_delta")
    inverse = order.index("inverse_rate_delta")
    first_exact = min(
        index for index, name in enumerate(order) if name.startswith(("exact_", "fatal_"))
    )
    assert signed < first_exact
    assert inverse < first_exact
    assert report["behavioral_families"]["signed_authority_delta"]["genuine_risk_fraction"] == "6/6"
    assert report["behavioral_families"]["inverse_rate_delta"]["genuine_risk_fraction"] == "12/12"


def test_signed_metric_mutant_reaches_the_scored_family_before_exact_oracles() -> None:
    expectations = _expectations()
    observations = _observations(expectations)
    target = next(
        cell
        for cell in observations["cells"]
        if cell["link_rate_gbps"] == 400 and cell["doorbell_service_ps"] == 1_000
    )
    _metric(target["structural"], "core21-prefill")["ttft_ps"] += 1

    with pytest.raises(
        check_results.AuthorityCheckError,
        match="signed_authority_delta",
    ):
        check_results.check_observations(
            observations,
            expectations,
            _negative_evidence(expectations),
        )


def test_inverse_rate_metric_mutant_reaches_scoring_before_d_zero_identity() -> None:
    expectations = _expectations()
    observations = _observations(expectations)
    target = next(
        cell
        for cell in observations["cells"]
        if cell["link_rate_gbps"] == 200 and cell["doorbell_service_ps"] == 0
    )
    _metric(target["bypass"], "core21-decode")["tpot_ps"]["numerator"] += 1

    with pytest.raises(
        check_results.AuthorityCheckError,
        match="inverse_rate_delta",
    ):
        check_results.check_observations(
            observations,
            expectations,
            _negative_evidence(expectations),
        )


def test_checker_missing_observations_uses_the_registered_negative_diagnostic(
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-remain-absent" / "results.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(STUDY_DIR / "check_results.py"),
            "--expectations",
            str(EXPECTATIONS),
            "--observations",
            str(tmp_path / "missing-observations.json"),
            "--negative-evidence",
            str(tmp_path / "missing-negative-evidence.json"),
            "--results",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stderr.strip() == "authority observations do not exist"
    assert not output.exists()
    assert not Path(f"{output}.tmp").exists()
