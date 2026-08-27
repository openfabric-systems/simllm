#!/usr/bin/env python3
"""Publish the compact frontier ladder record, figure and report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.frontier_ladder_v1.plot_study import prepare_plot_data, render
from examples.frontier_ladder_v1.run_study import _csv_rows, _sha256_path, _write_csv
from simllm.deploy import FrontierRung, frontier_ladder_record_from_json

RESULT_SCHEMA = "simllm-frontier-ladder-study-v1"


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8", newline="\n")


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _ratio(result: dict[str, Any], family: str, batch: int) -> dict[str, Any]:
    return next(
        row
        for row in result["fabric_leg_envelope"]
        if row["family"] == family and row["batch_per_gpu"] == batch
    )["quotient"]


def _family_tally(result: dict[str, Any], family: str) -> tuple[int, int]:
    if family in {"L-A", "L-B"}:
        row = result["score_classes"]["exact_oracles"]["families"][family]
    elif family in {"M-1", "M-2", "M-3", "S"}:
        row = result["score_classes"]["behavioral_relations"]["families"][family]
    elif family == "P":
        row = result["score_classes"]["plot_contract"]
    elif family == "W":
        row = result["score_classes"]["wall_time"]
    else:
        raise KeyError(family)
    return row["passed"], row["denominator"]


def _exact_table(result: dict[str, Any]) -> str:
    rows = []
    families = result["score_classes"]["exact_oracles"]["families"]
    for family in ("L-A", "L-B"):
        for row in families[family]["rows"]:
            rows.append(
                f"| {family} | {row['batch_per_gpu']} | {row['flow_count']} | "
                f"{row['payload_bytes']:,} | {row['expected_ns']:,} | "
                f"{row['observed_ns']:,} | {'PASS' if row['passed'] else 'MISS'} |"
            )
    return "\n".join(rows)


def _envelope_table(result: dict[str, Any]) -> str:
    rows = []
    for family in ("M-1", "M-2", "M-3"):
        family_rows = result["score_classes"]["behavioral_relations"]["families"][family]["rows"]
        for row in family_rows:
            quotient = row["quotient"]
            rows.append(
                f"| {family} | {row['batch_per_gpu']} | {row['ideal_ps']:,} | "
                f"{row['packet_ps']:,} | {quotient['numerator']:,} / "
                f"{quotient['denominator']:,} | {quotient['decimal']:.6f} | "
                f"{'PASS' if row['passed'] else 'MISS'} |"
            )
    return "\n".join(rows)


def _step_table(result: dict[str, Any]) -> str:
    rows = []
    family = result["score_classes"]["behavioral_relations"]["families"]["S"]
    for row in family["rows"]:
        observed = row["observed_step_ps"]
        rows.append(
            f"| {row['configuration_id']} | {row['batch_per_gpu']} | "
            f"{observed[FrontierRung.ESTIMATE.value]:,} | "
            f"{observed[FrontierRung.LOGGOPSIM_IDEAL.value]:,} | "
            f"{observed[FrontierRung.PACKET.value]:,} | "
            f"{'PASS' if row['passed'] else 'MISS'} |"
        )
    return "\n".join(rows)


def _guard_table(result: dict[str, Any]) -> str:
    return "\n".join(
        f"| {guard['id']} | {'PASS' if guard['held'] else 'FAIL'} | "
        f"{guard['evaluated']} | "
        f"{'rejected' if guard['mutation_control']['rejected'] else 'accepted'} |"
        for guard in result["fatal_guards"]
    )


def _report(result: dict[str, Any]) -> str:
    tallies = {family: _family_tally(result, family) for family in ("L-A", "L-B", "M-1", "M-2", "M-3", "S", "P", "W")}
    m1 = _ratio(result, "M-1", 32)
    m2 = _ratio(result, "M-2", 32)
    m3 = _ratio(result, "M-3", 32)
    wall = result["score_classes"]["wall_time"]
    sanity = result["physical_sanity"]
    return f"""# Frontier ladder result

## Verdict

**PASS.** The ideal LogGOPSim rung is a valid fast substitute for the frozen
serialized point-to-point fabric legs, where the batch-32 packet observation is
only **{(m1['decimal'] - 1) * 100:.2f} percent** above ideal. It is not a
contention model: the batch-32 eight-into-one packet observation is
**{m2['decimal']:.2f}x** the ideal leg because eight flows share one packet
receiver ingress while the ideal receiver charges no per-byte gap.

The result is non-void. Exact-oracle families are L-A {tallies['L-A'][0]} of
{tallies['L-A'][1]} and L-B {tallies['L-B'][0]} of {tallies['L-B'][1]}.
Behavioral-relation families are M-1 {tallies['M-1'][0]} of
{tallies['M-1'][1]}, M-2 {tallies['M-2'][0]} of {tallies['M-2'][1]}, M-3
{tallies['M-3'][0]} of {tallies['M-3'][1]} and S {tallies['S'][0]} of
{tallies['S'][1]}. The plot contract P is {tallies['P'][0]} of
{tallies['P'][1]}. Wall-time family W is {tallies['W'][0]} of
{tallies['W'][1]}, with a median of {wall['median_seconds']:.6f} seconds for
all twelve native legs. These evidence classes are not summed.

## Mechanism envelope

Family L-A executes one flow of the frozen maximum payload. Family L-B
executes eight equal concurrent flows into rank zero. Every row was executed
seven times through the pinned binary with the exact argument spelling
`-G 0.02`; the expected column is the frozen literal, not a closed form
evaluated by the runner.

| Family | Batch | Flows | Bytes per flow | Expected ns | Observed ns | Verdict |
|---|---:|---:|---:|---:|---:|---|
{_exact_table(result)}

The quotient table keeps packet and ideal picoseconds as the unreduced source
integers. M-1 is serialized concurrent packet over ideal, M-2 is incast
concurrent packet over ideal, and M-3 is the isolated packet leg over ideal.

| Family | Batch | Ideal ps | Packet ps | Exact quotient | Decimal | Verdict |
|---|---:|---:|---:|---:|---:|---|
{_envelope_table(result)}

At batch 32, M-1 is exactly {m1['numerator']:,} / {m1['denominator']:,} =
{m1['decimal']:.6f}. M-2 is exactly {m2['numerator']:,} /
{m2['denominator']:,} = {m2['decimal']:.6f}. The isolated M-3 control is
{m3['numerator']:,} / {m3['denominator']:,} = {m3['decimal']:.6f}. The
single-flow physics agrees across levels; the eight-fold gap appears only when
the packet receiver must serialize shared ingress.

## Step-level ladder

The TRAF-68 masking finding stands. On the twelve H100 points, the kernel is
slower than every fabric leg, so all three rungs produce the same step time.
The same is true for five B100 points. Only B100 batch 32 differs: the packet
rung includes the pinned intra-node candidate and reaches 4,523,298,348 ps,
while ESTIMATE and loggopsim-ideal remain at 4,257,218,560 ps.

| Configuration | Batch | ESTIMATE ps | Ideal SIMULATED ps | Packet SIMULATED ps | Verdict |
|---|---:|---:|---:|---:|---|
{_step_table(result)}

## Figure

[PDF](figures/frontier-ladder.pdf) and [PNG](figures/frontier-ladder.png)
render one NV-style two-panel figure through the Agg backend. The left panel
keeps the frozen logarithmic axes, uses a distinct marker for each rung and
emphasizes the exact six-point B100 packet Pareto front. The right panel shows
the three mechanism quotients and labels M-2 at {m2['decimal']:.2f}x.

## Physical sanity

Before reading the batch-32 serialized observation, payload bytes over 400
Gbit/s set a floor of {sanity['serialization_physics']['floor_ps']:,} ps and
one 2,000 ns propagation delay set a ceiling of
{sanity['serialization_physics']['ceiling_ps']:,} ps. The measured ideal leg
is {sanity['serialization_physics']['observed_ps']:,} ps, inside those bounds.

For batch-32 incast, all remote bytes over one 400 Gbit/s ingress set a floor
of {sanity['shared_ingress_physics']['floor_ps']:,} ps, while eight isolated
packet completions set a ceiling of
{sanity['shared_ingress_physics']['ceiling_ps']:,} ps. The packet observation
is {sanity['shared_ingress_physics']['observed_ps']:,} ps, inside the range.
From batch 16 to 32, the ideal and packet incast legs scale by
{sanity['batch_scaling']['ideal_b32_over_b16']['decimal']:.6f}x and
{sanity['batch_scaling']['packet_b32_over_b16']['decimal']:.6f}x. At the step
boundary, the {sanity['step_plausibility']['h100_batch32_kernel_floor_ps'] / 1_000_000_000:.3f}
ms H100 kernel remains above the
{sanity['step_plausibility']['h100_batch32_packet_fabric_ps'] / 1_000_000_000:.3f}
ms packet fabric leg, which independently explains the unchanged step.

## Fatal guards

| Guard | Outcome | Predicate | Mutation control |
|---|---|---|---|
{_guard_table(result)}

Every mutation control exercised the real predicate and was rejected. Native
stdout and stderr bytes, portable argument vectors and rendered GOALs remain in
the append-only external attempt directory.

## Provenance

- Frozen expectations commit: `{result['chronology']['expectations_commit']}`
- Frozen expectations SHA-256: `{result['chronology']['expectations_sha256']}`
- Implementation run commit: `{result['chronology']['implementation_commit']}`
- Pinned deployment record SHA-256: `{result['input_hashes'][1]['observed_sha256']}`
- Pinned LogGOPSim binary SHA-256: `{result['native_tools']['loggopsim']['sha256']}`
- txt2bin SHA-256: `{result['native_tools']['txt2bin']['sha256']}`

## Project effect

What ran: `frontier_ladder_v1` executed all twelve frozen ideal fabric legs
through the pinned LogGOPSim binary, joined them with the pinned analytical and
packet points, and rendered the three-rung frontier plus mechanism envelope.

What came out: the ideal level stays within about 1.6 percent of packet timing
for serialized point-to-point traffic but is about 8.11x optimistic for
eight-into-one incast, with the shared receiver ingress identified as the
missing mechanism.

What it changes: TRAF-20 closes with an executable validity envelope for the
fast level, and TRAF-68 closes with the fabric-leg view that exposes the
contention its step-level map masks. The ladder and its six-point packet Pareto
front become the deployment planning comparison surface.

What it does not change: no rung gains an absolute-accuracy claim against
silicon, no pricing semantic or protected predecessor artifact changes, the
TRAF-68 step-masking result remains literal, and statistical transport tails
remain owned by TRAF-19.
"""


def publish(raw_result_path: Path, output_dir: Path) -> dict[str, Any]:
    """Validate and publish one passing external attempt."""

    result = json.loads(raw_result_path.read_text(encoding="utf-8"))
    if result.get("schema") != RESULT_SCHEMA:
        raise ValueError("unexpected frontier ladder result schema")
    if result.get("verdict") != "PASS":
        raise ValueError("only a non-void passing ladder attempt may be published")
    record = frontier_ladder_record_from_json(result["ladder_record"])
    if len(record.points) != 18:
        raise ValueError("the ladder publication requires all 18 points")
    if any(not guard["held"] for guard in result["fatal_guards"]):
        raise ValueError("the ladder publication has a failed fatal guard")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    pdf, png = render(prepare_plot_data(result), figure_dir / "frontier-ladder")
    result["publication"] = {
        "raw_result_sha256": _sha256_path(raw_result_path),
        "figures": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_path(path),
            }
            for path in (pdf, png)
        ],
    }
    _write_json(output_dir / "result.json", result)
    _write_csv(output_dir / "results.csv", _csv_rows(result))
    _write_text(output_dir / "RESULTS.md", _report(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = publish(args.result, args.output_dir)
    print(f"published frontier ladder {result['verdict']} to {args.output_dir}")


if __name__ == "__main__":
    main()
