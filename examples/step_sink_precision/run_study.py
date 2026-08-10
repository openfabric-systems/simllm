"""Run the checks frozen in expectations.md for BACK-5, BACK-6 and BACK-7.

Usage:
    SIMLLM_HTSIM_RNIC=... \
    SIMLLM_TXT2BIN=... \
    SIMLLM_DATA_ROOT=... python examples/step_sink_precision/run_study.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

from simllm._local_config import path_from_env
from simllm.backends import HtsimStepSink, HtsimStepSinkConfig, parse_completion_csv
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.core import (
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    step_record_from_json,
    step_record_to_json,
)

EXPECTATIONS_COMMIT = "9a8c05e"
DEFAULT_GOAL_SHA256 = "f8aade109ba8e3a581b7d965b3a0c76c1247016a1e37491fa84efbbf377677a5"
TOPOLOGY = Path(__file__).resolve().parents[1] / "m1" / "topologies" / "clos_64_400g.topo"

G = 1_000_000_000
LINKSPEED_BPS = 400 * G
PS_PER_BYTE_400G = 20
PROPAGATION_PS = 2_000_000

FROZEN_A = {
    (2, 2): ((2_600, 4_600), (2, 5), 7_200, 16_017_240, 16),
    (2, 4): ((4_600, 8_600), (4, 9), 13_200, 48_028_360, 96),
    (4, 2): ((2_600, 4_600, 6_600, 8_600), (2, 5, 6, 9), 22_400, 32_042_480, 32),
    (4, 4): (
        (4_600, 8_600, 12_600, 16_600),
        (4, 9, 12, 17),
        42_400,
        96_072_720,
        192,
    ),
}


class RegisteredLayerProvider(ComputeProvider):
    """The exact unequal pattern frozen for check A."""

    def __init__(self, layers: int, world: int):
        self.layer_ps = tuple(1_000 * world * (layer + 1) + 600 for layer in range(layers))

    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=sum(self.layer_ps), bound="measured")

    def estimate_layers(self, kernel, gpu, num_layers):
        return tuple(
            DurationEstimate(duration_ps=duration_ps, bound="measured")
            for duration_ps in self.layer_ps
        )


class FlopProvider(ComputeProvider):
    """One modeled FLOP costs one picosecond, as frozen for checks B and C."""

    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=int(kernel.flops), bound="compute")


class LiteralLayerProvider(ComputeProvider):
    """Structural guard provider used only for invalid-breakdown checks."""

    def __init__(self, layer_ps, fused_adjustment=0):
        self.layer_ps = tuple(layer_ps)
        self.fused_adjustment = fused_adjustment

    def estimate(self, kernel, gpu):
        return DurationEstimate(
            duration_ps=sum(self.layer_ps) + self.fused_adjustment,
            bound="measured",
        )

    def estimate_layers(self, kernel, gpu, num_layers):
        return tuple(
            DurationEstimate(duration_ps=duration_ps, bound="measured")
            for duration_ps in self.layer_ps
        )


def dims(layers: int) -> ModelDims:
    return ModelDims(
        num_layers=layers,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        num_kv_heads=4,
        head_size=16,
        vocab_size=256,
        dtype_bytes=2,
    )


def default_dims() -> ModelDims:
    return ModelDims(
        num_layers=2,
        hidden_size=1024,
        intermediate_size=4096,
        num_heads=8,
        num_kv_heads=8,
        head_size=128,
        vocab_size=32000,
        dtype_bytes=2,
    )


def one_decode(*, exact: bool = True, context_length: int = 16) -> StepRecord:
    return StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                "d",
                RequestPhase.DECODE,
                num_new_tokens=1,
                context_length=context_length,
            )
        ],
        num_sampled=1 if exact else None,
    )


def layer_calcs(path: Path, rank: int) -> tuple[int, ...]:
    text = path.read_text()
    match = re.search(rf"rank {rank} \{{\n(.*?)\n\}}", text, flags=re.DOTALL)
    if match is None:
        raise AssertionError(f"rank {rank} missing from {path}")
    return tuple(int(value) for value in re.findall(r": calc (\d+)(?:\s|$)", match.group(1)))


def normalized_flows(path: Path, rank_offset: int) -> list[tuple[int, ...]]:
    return sorted(
        (
            flow.source - rank_offset,
            flow.destination - rank_offset,
            flow.tag,
            flow.payload_bytes,
            flow.start_time_ps,
            flow.completion_time_ps,
            flow.fct_ps,
        )
        for flow in parse_completion_csv(path)
    )


def frozen_a_jct(layers: int, world: int, layer_calc_ns: tuple[int, ...]) -> int:
    chunk = 128 // world
    round_ps = chunk * PS_PER_BYTE_400G + PROPAGATION_PS
    return sum(layer_calc_ns) * 1_000 + 2 * layers * 2 * (world - 1) * round_ps


def run_check_a(out: Path, emit) -> None:
    measured_jct = {}
    for (layers, world), frozen in FROZEN_A.items():
        expected_layer_ps, expected_calc_ns, expected_estimate, expected_jct, expected_flows = (
            frozen
        )
        assert frozen_a_jct(layers, world, expected_calc_ns) == expected_jct
        workdir = out / f"a-l{layers}-w{world}"
        sink = HtsimStepSink(
            HtsimStepSinkConfig(
                profile="rnic-nn-fluid",
                tp_ranks=tuple(range(world)),
                dims=dims(layers),
                workdir=workdir,
                linkspeed_bps=LINKSPEED_BPS,
                provider=RegisteredLayerProvider(layers, world),
            )
        )
        result = sink(one_decode())
        assert result is not None
        outcome = sink.outcomes[0]
        goal_path = workdir / "step-000000.goal"

        assert outcome.compute_estimate_ps == expected_estimate
        assert outcome.layer_calc_ns == expected_calc_ns
        assert outcome.per_layer_calc_ns is None
        assert sum(outcome.layer_calc_ns) == expected_estimate // 1000
        assert outcome.num_flows == expected_flows
        assert result.step_latency_ps == expected_jct
        assert tuple(sink.config.provider.layer_ps) == expected_layer_ps
        for rank in range(world):
            assert layer_calcs(goal_path, rank) == expected_calc_ns

        measured_jct[(layers, world)] = result.step_latency_ps
        emit(
            section="A",
            layers=layers,
            world=world,
            provider_layer_ps=";".join(map(str, expected_layer_ps)),
            calc_ns=";".join(map(str, outcome.layer_calc_ns)),
            calc_sum_ns=sum(outcome.layer_calc_ns),
            estimate_ps=outcome.compute_estimate_ps,
            jct_ps=result.step_latency_ps,
            expected_jct_ps=expected_jct,
            residual_ps=result.step_latency_ps - expected_jct,
            flows=outcome.num_flows,
            expected_flows=expected_flows,
            quiescence="verified",
        )

    assert measured_jct[(2, 2)] < measured_jct[(4, 2)]
    assert measured_jct[(2, 4)] < measured_jct[(4, 4)]
    assert measured_jct[(2, 2)] < measured_jct[(2, 4)]
    assert measured_jct[(4, 2)] < measured_jct[(4, 4)]


def run_sample_sink(out: Path, label: str, record: StepRecord) -> HtsimStepSink:
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=dims(2),
            workdir=out / label,
            linkspeed_bps=LINKSPEED_BPS,
            provider=FlopProvider(),
        )
    )
    assert sink(record) is not None
    return sink


def run_check_b(out: Path, emit) -> None:
    mixed = [
        ScheduledRequest("p", RequestPhase.PREFILL, 4, context_length=8),
        ScheduledRequest("d", RequestPhase.DECODE, 1, context_length=32),
    ]
    approximate = run_sample_sink(out, "b-chunk-approximate", StepRecord(0, 0, mixed))
    exact = run_sample_sink(
        out,
        "b-chunk-exact",
        StepRecord(0, 0, mixed, num_sampled=1),
    )
    approx_outcome = approximate.outcomes[0]
    exact_outcome = exact.outcomes[0]
    assert approx_outcome.compute_estimate_ps == 912_896
    assert exact_outcome.compute_estimate_ps == 880_128
    assert approx_outcome.compute_estimate_ps - exact_outcome.compute_estimate_ps == 32_768
    assert approx_outcome.layer_calc_ns == (456, 456)
    assert exact_outcome.layer_calc_ns == (440, 440)
    assert approx_outcome.makespan_ps == 16_963_200
    assert exact_outcome.makespan_ps == 16_931_200
    assert approx_outcome.makespan_ps - exact_outcome.makespan_ps == 32_000
    emit(
        section="B",
        case="chunked-prefill",
        approximate_samples=approx_outcome.num_sampled,
        exact_samples=exact_outcome.num_sampled,
        estimate_delta_ps=(
            approx_outcome.compute_estimate_ps - exact_outcome.compute_estimate_ps
        ),
        expected_estimate_delta_ps=32_768,
        jct_delta_ps=approx_outcome.makespan_ps - exact_outcome.makespan_ps,
        expected_jct_delta_ps=32_000,
        approximate_jct_ps=approx_outcome.makespan_ps,
        exact_jct_ps=exact_outcome.makespan_ps,
        quiescence="verified",
    )

    decode_rows = [
        ScheduledRequest("d0", RequestPhase.DECODE, 1, context_length=32),
        ScheduledRequest("d1", RequestPhase.DECODE, 1, context_length=32),
    ]
    absent = run_sample_sink(out, "b-all-sample-absent", StepRecord(0, 0, decode_rows))
    present = run_sample_sink(
        out,
        "b-all-sample-exact",
        StepRecord(0, 0, decode_rows, num_sampled=2),
    )
    absent_goal = (out / "b-all-sample-absent" / "step-000000.goal").read_bytes()
    present_goal = (out / "b-all-sample-exact" / "step-000000.goal").read_bytes()
    assert absent.outcomes[0].compute_estimate_ps == 424_960
    assert present.outcomes[0].compute_estimate_ps == 424_960
    assert absent.outcomes[0].layer_calc_ns == (212, 212)
    assert present.outcomes[0].layer_calc_ns == (212, 212)
    assert absent.outcomes[0].makespan_ps == 16_444_480
    assert present.outcomes[0].makespan_ps == 16_444_480
    assert absent_goal == present_goal
    emit(
        section="B",
        case="all-sample-identity",
        absent_estimate_ps=absent.outcomes[0].compute_estimate_ps,
        exact_estimate_ps=present.outcomes[0].compute_estimate_ps,
        goal_bytes_equal=True,
        absent_jct_ps=absent.outcomes[0].makespan_ps,
        exact_jct_ps=present.outcomes[0].makespan_ps,
        jct_residual_ps=present.outcomes[0].makespan_ps - absent.outcomes[0].makespan_ps,
        quiescence="verified",
    )


def run_check_c(out: Path, emit) -> None:
    assert TOPOLOGY.is_file()
    registered_probe = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=dims(2),
            workdir=out / "c-registered-configuration-probe",
            linkspeed_bps=LINKSPEED_BPS,
            topology=TOPOLOGY,
            num_goal_ranks=64,
            provider=FlopProvider(),
        )
    )
    try:
        registered_probe(one_decode(exact=False))
    except RuntimeError as exc:
        message = str(exc)
        assert "physical Clos options are valid only" in message
        emit(
            section="C-config",
            registered_profile="rnic-nn-fluid",
            registered_topology_flag_supported=False,
            backend_exit=2,
            classification="preregistration-defect",
        )
    else:
        raise AssertionError("rnic-nn-fluid unexpectedly accepted a physical topology")

    profiles = (
        ("rnic-nn-fluid", None, "post-specified-fluid-correction"),
        ("rnic-cn", TOPOLOGY, "post-specified-topology-supplement"),
    )
    for profile, topology, classification in profiles:
        for world in (2, 4):
            explicit_dir = out / f"c-{profile}-w{world}-explicit"
            workaround_dir = out / f"c-{profile}-w{world}-workaround"
            explicit = HtsimStepSink(
                HtsimStepSinkConfig(
                    profile=profile,
                    tp_ranks=tuple(range(world)),
                    dims=dims(2),
                    workdir=explicit_dir,
                    linkspeed_bps=LINKSPEED_BPS,
                    topology=topology,
                    num_goal_ranks=64,
                    provider=FlopProvider(),
                )
            )
            rank_offset = 64 - world
            workaround = HtsimStepSink(
                HtsimStepSinkConfig(
                    profile=profile,
                    tp_ranks=tuple(range(rank_offset, 64)),
                    dims=dims(2),
                    workdir=workaround_dir,
                    linkspeed_bps=LINKSPEED_BPS,
                    topology=topology,
                    provider=FlopProvider(),
                )
            )
            explicit_result = explicit(one_decode(exact=False))
            workaround_result = workaround(one_decode(exact=False))
            assert explicit_result is not None
            assert workaround_result is not None

            explicit_goal = explicit_dir / "step-000000.goal"
            workaround_goal = workaround_dir / "step-000000.goal"
            assert explicit_goal.read_text().startswith("num_ranks 64\n")
            assert workaround_goal.read_text().startswith("num_ranks 64\n")
            explicit_csv = explicit_dir / f"step-000000.{profile}.csv"
            workaround_csv = workaround_dir / f"step-000000.{profile}.csv"
            explicit_flows = normalized_flows(explicit_csv, rank_offset=0)
            workaround_flows = normalized_flows(workaround_csv, rank_offset=rank_offset)
            assert explicit_flows == workaround_flows
            assert explicit_result.step_latency_ps == workaround_result.step_latency_ps
            emit(
                section="C",
                classification=classification,
                profile=profile,
                topology_enabled=topology is not None,
                world=world,
                explicit_jct_ps=explicit_result.step_latency_ps,
                workaround_jct_ps=workaround_result.step_latency_ps,
                residual_ps=(
                    explicit_result.step_latency_ps - workaround_result.step_latency_ps
                ),
                flow_ledgers_equal=True,
                flows=len(explicit_flows),
                goal_ranks=64,
                quiescence="verified",
            )


def run_check_d(out: Path, emit) -> None:
    record = StepRecord(
        0,
        0,
        [ScheduledRequest("prefill", RequestPhase.PREFILL, 256, context_length=256)],
    )
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=default_dims(),
            workdir=out / "d-default-baseline",
            linkspeed_bps=LINKSPEED_BPS,
        )
    )
    result = sink(record)
    assert result is not None
    goal_path = out / "d-default-baseline" / "step-000000.goal"
    digest = hashlib.sha256(goal_path.read_bytes()).hexdigest()
    assert digest == DEFAULT_GOAL_SHA256
    assert sink.outcomes[0].per_layer_calc_ns == 12_030
    assert sink.outcomes[0].layer_calc_ns == (12_030, 12_030)

    legacy_payload = step_record_to_json(record)
    assert "num_sampled" not in legacy_payload
    assert step_record_from_json(legacy_payload) == record
    exact_record = StepRecord(
        record.step_index,
        record.virtual_time_ps,
        record.scheduled,
        num_sampled=0,
    )
    exact_payload = step_record_to_json(exact_record)
    assert exact_payload["num_sampled"] == 0
    assert step_record_from_json(exact_payload) == exact_record

    invalid = (
        LiteralLayerProvider((1_000,)),
        LiteralLayerProvider((-1, 1_001)),
        LiteralLayerProvider((1_000, 2_000), fused_adjustment=1),
    )
    rejected = 0
    for index, provider in enumerate(invalid):
        guard_sink = HtsimStepSink(
            HtsimStepSinkConfig(
                profile="rnic-nn-fluid",
                tp_ranks=(0, 1),
                dims=default_dims(),
                workdir=out / f"d-invalid-{index}",
                provider=provider,
            )
        )
        try:
            guard_sink.compute_estimate_ps(record)
        except ValueError:
            rejected += 1
    assert rejected == len(invalid)
    emit(
        section="D",
        default_goal_sha256=digest,
        expected_goal_sha256=DEFAULT_GOAL_SHA256,
        goal_bytes_equal=True,
        per_layer_calc_ns=sink.outcomes[0].per_layer_calc_ns,
        absent_sample_field_omitted=True,
        present_sample_field_round_trips=True,
        invalid_breakdowns_rejected=rejected,
        quiescence="verified",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out is None:
        data_root = path_from_env("SIMLLM_DATA_ROOT")
        if data_root is None:
            parser.error("--out is required when SIMLLM_DATA_ROOT is not set")
        args.out = data_root / "step_sink_precision"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    def emit(**row) -> None:
        rows.append(row)
        print(" ".join(f"{key}={value}" for key, value in row.items()))

    run_check_a(out, emit)
    run_check_b(out, emit)
    run_check_c(out, emit)
    run_check_d(out, emit)

    summary = out / "summary.csv"
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with open(summary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"completed {len(rows)} result rows against expectations commit "
        f"{EXPECTATIONS_COMMIT}; summary={summary}"
    )


if __name__ == "__main__":
    main()
