"""M4 closed-loop slice: run the pre-registered checks of expectations.md.

Sections mirror the frozen registration: (a) standalone ring allreduces on
the fluid profile, (b) the full-step makespan grid on fluid through
HtsimStepSink, (c) the same grid on rnic-nn against the banded point form,
(d) TTFT/TPOT of a replayed synthetic step sequence in virtual mode, and
(e) replays of the recorded vLLM/SGLang smoke JSONLs behind a declared
tp=8 manifest. One CSV row per (run, metric) goes to <out>/summary.csv;
raw GOALs and completion CSVs land under <out> as well (keep <out> on a
data volume, e.g. /data3/yifeng/simllm-dev/m4-runs).

Usage:
    SIMLLM_HTSIM_RNIC=... SIMLLM_TXT2BIN=... \\
    python examples/m4/run_m4.py --out /data3/yifeng/simllm-dev/m4-runs
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from simllm.backends import (
    HtsimRnicConfig,
    HtsimStepSink,
    HtsimStepSinkConfig,
    run_htsim_rnic,
)
from simllm.compute import ModelDims
from simllm.core import (
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    step_records_from_jsonl,
)
from simllm.goal import GoalTrace, to_binary
from simllm.placement import declared_manifest
from simllm.traffic import ring_allreduce

G = 1_000_000_000

# M1-calibrated constants (examples/m1/RESULTS.md C1-C3), reused not rederived
PS_PER_BYTE_400G = 20
PROPAGATION_PS = 2_000_000
SLOT_PS = 83_200
PKT_PAYLOAD = 4096

# Declared llama-8b-shaped geometry, per TP rank (frozen in expectations.md)
LAYERS = 32
HIDDEN = 4096
DTYPE_BYTES = 2

# Frozen check-B inputs (expectations.md B1-B6): whole-step roofline
# estimate E in ps and per-layer calc units c in ns, registered before any
# run. The grid compares against these constants, never against the
# runtime cost model: a drifted roofline must surface as a failed check,
# not as a silently matching residual. The runtime values are still
# emitted and checked against the frozen ones per cell.
FROZEN_B_INPUTS: dict[tuple[str, int], tuple[int, int]] = {
    ("decode8x2048", 2): (1_625_667_291, 50_802),
    ("decode8x2048", 4): (906_643_748, 28_332),
    ("decode8x2048", 8): (547_131_977, 17_097),
    ("prefill2048", 2): (11_781_315_593, 368_166),
    ("prefill2048", 4): (5_891_074_730, 184_096),
    ("prefill2048", 8): (2_945_954_299, 92_061),
}

# Frozen check-D expectations: TTFT equals the B6 makespan and every
# decode latency equals the B3 makespan.
FROZEN_TTFT_PS = 42_318_915_840
FROZEN_DECODE_PS = 2_485_904_640

_HERE = Path(__file__).resolve().parent
DEFAULT_VLLM_JSONL = _HERE / "fixtures" / "vllm-m2-steps.jsonl"
DEFAULT_SGLANG_JSONL = _HERE / "fixtures" / "sglang-m3-steps.jsonl"


def dims_tp(tp: int) -> ModelDims:
    return ModelDims(
        num_layers=LAYERS,
        hidden_size=HIDDEN,
        intermediate_size=14336 // tp,
        num_heads=32 // tp,
        num_kv_heads=max(8 // tp, 1),
        head_size=128,
        vocab_size=128256,
        dtype_bytes=DTYPE_BYTES,
    )


def decode_record(step_index: int = 0, virtual_time_ps: int = 0) -> StepRecord:
    """decode8x2048: 8 requests, 1 new token each, context 2048."""
    return StepRecord(step_index=step_index, virtual_time_ps=virtual_time_ps, scheduled=[
        ScheduledRequest(f"d{i}", RequestPhase.DECODE, num_new_tokens=1,
                         context_length=2048)
        for i in range(8)
    ])


def prefill_record(step_index: int = 0, virtual_time_ps: int = 0) -> StepRecord:
    """prefill2048: one 2048-token prefill chunk at context 2048."""
    return StepRecord(step_index=step_index, virtual_time_ps=virtual_time_ps, scheduled=[
        ScheduledRequest("p0", RequestPhase.PREFILL, num_new_tokens=2048,
                         context_length=2048)
    ])


STEP_SHAPES = {"decode8x2048": decode_record, "prefill2048": prefill_record}


def allreduce_payload(record: StepRecord) -> int:
    return record.total_new_tokens * HIDDEN * DTYPE_BYTES


def fluid_allreduce_ps(payload: int, world: int) -> int:
    """Frozen check-A form: 2(W-1) chained rounds of chunk*20 + P each."""
    chunk = payload // world
    return 2 * (world - 1) * (chunk * PS_PER_BYTE_400G + PROPAGATION_PS)


def fluid_makespan_ps(per_layer_calc_ns: int, payload: int, world: int) -> int:
    """Frozen check-B form: L * calc + 2L allreduces."""
    calc_eff_ps = max(per_layer_calc_ns, 1) * 1000
    return LAYERS * calc_eff_ps + 2 * LAYERS * fluid_allreduce_ps(payload, world)


def nn_point_and_band_ps(per_layer_calc_ns: int, payload: int, world: int) -> tuple[int, int]:
    """Frozen check-C point form and its +-1 slot-per-round band."""
    chunk = payload // world
    npkt = chunk // PKT_PAYLOAD
    if chunk % PKT_PAYLOAD:
        raise ValueError("registered grid uses whole-packet chunks only")
    rounds = 2 * LAYERS * 2 * (world - 1)
    round_nn = (npkt + 1) * SLOT_PS + PROPAGATION_PS
    point = LAYERS * max(per_layer_calc_ns, 1) * 1000 + rounds * round_nn
    return point, rounds * SLOT_PS


def make_sink(profile: str, tp: int, workdir: Path) -> HtsimStepSink:
    manifest = declared_manifest(tp=tp, pp=1, dp=1)
    return HtsimStepSink(HtsimStepSinkConfig(
        profile=profile,
        tp_ranks=manifest.group_ranks(0, "tp"),
        dims=dims_tp(tp),
        workdir=workdir,
        linkspeed_bps=400 * G,
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/m4")
    parser.add_argument("--sections", default="a,b,c,d,e")
    parser.add_argument("--vllm-jsonl", default=DEFAULT_VLLM_JSONL)
    parser.add_argument("--sglang-jsonl", default=DEFAULT_SGLANG_JSONL)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sections = set(args.sections.split(","))
    rows: list[dict] = []

    def emit(**kw) -> None:
        rows.append(kw)
        print(" ".join(f"{k}={v}" for k, v in kw.items()))

    if "a" in sections:
        for payload in (65536, 16777216):
            for world in (2, 4, 8):
                trace = GoalTrace(world)
                ring_allreduce(trace, ranks=list(range(world)),
                               size_bytes=payload, base_tag=1)
                name = f"a-s{payload}-w{world}"
                goal_bin = to_binary(trace.write(out / f"{name}.goal"))
                result = run_htsim_rnic(HtsimRnicConfig(
                    goal_bin=goal_bin, profile="rnic-nn-fluid",
                    linkspeed_bps=400 * G,
                    completion_csv=out / f"{name}.csv",
                ))
                measured = result.job_completion_time_ps()
                expected = fluid_allreduce_ps(payload, world)
                emit(section="A", payload=payload, world=world,
                     jct_ps=measured, expected_ps=expected,
                     residual_ps=measured - expected)

    grid_outcomes: dict[tuple[str, int, str], tuple[int, int]] = {}
    if "b" in sections or "c" in sections:
        profiles = [p for p, s in (("rnic-nn-fluid", "b"), ("rnic-nn", "c")) if s in sections]
        for profile in profiles:
            for shape, build in STEP_SHAPES.items():
                for tp in (2, 4, 8):
                    workdir = out / f"grid-{shape}-tp{tp}-{profile}"
                    sink = make_sink(profile, tp, workdir)
                    record = build()
                    result = sink(record)
                    assert result is not None
                    outcome = sink.outcomes[0]
                    measured = result.step_latency_ps
                    payload = allreduce_payload(record)
                    grid_outcomes[(shape, tp, profile)] = (
                        measured, outcome.per_layer_calc_ns)
                    frozen_e, frozen_c = FROZEN_B_INPUTS[(shape, tp)]
                    if profile == "rnic-nn-fluid":
                        expected = fluid_makespan_ps(frozen_c, payload, tp)
                        emit(section="B", shape=shape, tp=tp,
                             estimate_ps=outcome.compute_estimate_ps,
                             estimate_matches_frozen=(
                                 outcome.compute_estimate_ps == frozen_e),
                             calc_ns=outcome.per_layer_calc_ns,
                             calc_matches_frozen=(
                                 outcome.per_layer_calc_ns == frozen_c),
                             makespan_ps=measured, expected_ps=expected,
                             residual_ps=measured - expected)
                    else:
                        point, band = nn_point_and_band_ps(
                            frozen_c, payload, tp)
                        fluid = grid_outcomes.get((shape, tp, "rnic-nn-fluid"))
                        emit(section="C", shape=shape, tp=tp,
                             calc_matches_frozen=(
                                 outcome.per_layer_calc_ns == frozen_c),
                             makespan_ps=measured, point_ps=point,
                             deviation_ps=measured - point, band_ps=band,
                             in_band=abs(measured - point) <= band,
                             ge_fluid=(fluid is None or measured >= fluid[0]))

    if "d" in sections:
        sink = make_sink("rnic-nn-fluid", 8, out / "d-replay-tp8")
        virtual_time = 0
        latencies = []
        for index in range(9):
            build = prefill_record if index == 0 else decode_record
            result = sink(build(step_index=index, virtual_time_ps=virtual_time))
            assert result is not None
            assert result.completed_at_ps == virtual_time + result.step_latency_ps
            latencies.append(result.step_latency_ps)
            virtual_time = result.completed_at_ps
        ttft = latencies[0]
        decode_latencies = latencies[1:]
        tpot = sum(decode_latencies) / len(decode_latencies)
        emit(section="D", ttft_ps=ttft,
             ttft_residual_ps=ttft - FROZEN_TTFT_PS,
             tpot_ps=round(tpot),
             tpot_residual_ps=round(tpot) - FROZEN_DECODE_PS,
             decode_latencies_distinct=len(set(decode_latencies)),
             final_virtual_time_ps=virtual_time)

    if "e" in sections:
        smoke_jsonls = {
            "vllm-m2": Path(args.vllm_jsonl),
            "sglang-m3": Path(args.sglang_jsonl),
        }
        for label, path in smoke_jsonls.items():
            records = step_records_from_jsonl(path)
            times = [r.virtual_time_ps for r in records]
            assert times == sorted(times) and len(set(times)) == len(times), (
                f"{label}: recorded virtual time not strictly monotonic")
            sink = make_sink("rnic-nn-fluid", 8, out / f"e-{label}")
            for record in records:
                estimate_ps = sink.compute_estimate_ps(record)
                result = sink(record)
                assert result is not None, f"{label} step {record.step_index}"
                assert result.completed_at_ps == record.virtual_time_ps + result.step_latency_ps
                outcome = sink.outcomes[-1]
                emit(section="E", frontend=label, step=record.step_index,
                     new_tokens=record.total_new_tokens,
                     compute_estimate_ps=estimate_ps,
                     step_latency_ps=result.step_latency_ps,
                     exceeds_compute=result.step_latency_ps > estimate_ps,
                     network_share=round(outcome.network_share_for(LAYERS), 4))

    with open(out / "summary.csv", "w", newline="") as handle:
        fields: list[str] = []
        for row in rows:
            fields += [k for k in row if k not in fields]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} rows -> {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
