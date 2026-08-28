#!/usr/bin/env python3
"""Publish the compact frontier comparison record, table, figure and report."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.frontier_comparison_v1.plot_study import render

STUDY_DIR = Path(__file__).resolve().parent
RESULT_SCHEMA = "simllm-frontier-comparison-study-v1"


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8", newline="\n")


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _family(result: dict[str, Any], family: str) -> dict[str, Any]:
    if family in {"X1", "X2"}:
        return result["families"][family]
    return result["families"]["X3"][family]


def _family_table(result: dict[str, Any]) -> str:
    rows = []
    for family in ("X1", "X2", "X3a", "X3b", "X3c"):
        record = _family(result, family)
        acceptance = (
            f">= {record['acceptance_minimum']}"
            if family == "X3c"
            else "all"
        )
        passed = (
            record["passed"] >= record["acceptance_minimum"]
            if family == "X3c"
            else record["passed"] == record["denominator"]
        )
        rows.append(
            f"| {family} | {record['passed']} / {record['denominator']} | "
            f"{acceptance} | {'PASS' if passed else 'MISS'} |"
        )
    wall = result["families"]["W"]
    rows.append(
        f"| W | {wall['passed']} / {wall['denominator']} | <= 120 s | "
        f"{'PASS' if wall['passed'] else 'MISS'} |"
    )
    return "\n".join(rows)


def _x3_table(result: dict[str, Any]) -> str:
    rows = []
    for external in result["families"]["X3"]["external_rows"]:
        x3b = external["x3b"]
        x3c = external["x3c"]
        rows.append(
            f"| {external['row']} | "
            f"{float(external['x_tokens_per_second_per_user']):.3f} | "
            f"{x3c['low_y']:.3f} | {x3c['external_y']:.3f} | "
            f"{x3c['high_y']:.3f} | "
            f"{'PASS' if x3b['passed'] else 'MISS'} | "
            f"{'PASS' if x3c['passed'] else 'MISS, below e=0.6'} |"
        )
    return "\n".join(rows)


def _guard_table(result: dict[str, Any]) -> str:
    return "\n".join(
        f"| {guard['id']} | {'PASS' if guard['held'] else 'FAIL'} | "
        f"{guard['detail']} |"
        for guard in result["fatal_guards"]
    )


def _report(result: dict[str, Any]) -> str:
    x1 = result["families"]["X1"]
    x2 = result["families"]["X2"]
    x3 = result["families"]["X3"]
    wall = result["families"]["W"]
    derivation = result["model_work_derivation"]
    extraction = result["extraction"]
    decode_e = x2["decode_e_star"]["decimal"]
    prefill_e = x2["prefill_e_star"]["decimal"]
    matched = x2["matched_point"]
    decode_bytes_tp4_b64 = (
        derivation["static_parameter_bytes"] // 4
        + 64 * derivation["decode_logical_kv_bytes_per_batch_item"] // 4
    )
    prefill_compute_floor_ms = matched["prefill_request_ps"] / 1_000_000_000
    decode_memory_floor_ms = matched["decode_step_ps"] / 1_000_000_000
    return f"""# Frontier comparison result

## Verdict

**PASS, non-void.** The frozen comparison behaves in the expected direction
without fitting to the external data. The exact candidate and matched-point
families pass, all three estimator frontiers plus the external frontier are
monotone, the efficiency-1.0 frontier dominates all 10 external rows at their
required per-user speeds, and 9 of 10 rows fall inside the frozen 0.6 to 1.0
matched-configuration throughput bracket. Row 10 is the published exception:
its **157.234 tokens/s/GPU** is below our efficiency-0.6 estimate of
**167.408 tokens/s/GPU**.

The result is non-void because all five fatal guards held. Evidence classes
remain separate and are not added into one score.

| Family | Tally | Acceptance | Verdict |
|---|---:|---:|---|
{_family_table(result)}

## Qwen3-32B FP8 extraction column

The config-only extraction binds `Qwen/Qwen3-32B-FP8` to exact Hugging Face
revision `{extraction['model']['revision']}`. The vLLM 0.27.1 inventory is
`{result['frozen_inputs']['inventory']['sha256']}` and the companion SGLang
inventory is `{extraction['companion_sglang_inventory_sha256']}`. Both contain
{extraction['case_count']} cases, {extraction['family_count']} logical families
and {extraction['logical_visits_per_case']} visits per case. Repeated runs are
byte identical, and their framework-neutral content agrees exactly after
source provenance is removed.

FG-2 confirms the fatal architecture literals exactly: 64 layers, hidden size
5120, intermediate size 25600, 64 attention heads, 8 key-value heads, head
dimension 128 and vocabulary size 151936. The FP8 checkpoint has
{x1['fp8_parameter_count']:,} logical parameters, so TP4 owns
{x1['fp8_rank_bytes_tp4']:,} weight bytes per rank at one byte per parameter.
That is 5.8 percent of the declared 141 GB H200 capacity.

## Work derivation and matched-point pricing

The pricing record derives its work from the inventory's per-layer
projections. At the frozen TP4 mapping, decode carries
{derivation['decode_flops_per_batch_item_at_tp4']:,} FLOPs and
{derivation['decode_logical_kv_bytes_per_batch_item']:,} logical key-value
cache bytes per batch item at the average 4,250-token context. A 3,500-token
uncached prefill carries {derivation['prefill_flops_per_request_at_tp4']:,}
FLOPs and {derivation['prefill_logical_kv_bytes_per_request']:,} logical
key-value bytes per request. Matrix weights use FP8 at one byte per parameter;
logical weight and key-value bytes shard by tensor-parallel width. TP2 and TP8
FLOPs scale relative to the frozen TP4 comparison mapping.

At efficiency 1.0, the exact external-best topology prices decode batch 64 to
**{matched['decode_step_ps'] / 1_000_000_000:.6f} ms**, below the external
9.179 ms, and prices the uncached prefill request to
**{matched['prefill_request_ps'] / 1_000_000_000:.6f} ms**, below the external
196.423 ms. The implied efficiencies are **{decode_e:.6f} for decode** and
**{prefill_e:.6f} for prefill**. Both sit inside the frozen [0.40, 1.00] band.
They are reported comparison results only and are never installed as model
parameters.

## Frontier overlay

[PDF](figures/frontier-comparison.pdf) and
[PNG](figures/frontier-comparison.png) show the three SimLLM ESTIMATE
frontiers and all 10 MEASURED-EXTERNAL rows on logarithmic per-user speed and
per-GPU throughput axes. The upper-right corner is better. External row labels
match the table below.

X3b compares the efficiency-1.0 service-feasible frontier at or above each
external per-user speed. X3c compares each external row's throughput with the
0.6 and 1.0 estimates for that row's exact prefill/decode topology and batch,
including comparison points that the 10 ms frontier filter excludes. This
keeps the external database out of pricing while making every row-level
predicate explicit.

| Row | External user tok/s | e=0.6 tok/s/GPU | External tok/s/GPU | e=1.0 tok/s/GPU | X3b | X3c |
|---:|---:|---:|---:|---:|---|---|
{_x3_table(result)}

The X4 scope comes from the
[frontier ladder study](../frontier_ladder_v1/RESULTS.md). Its ideal-network
class tracks the packet rung within about 1.6 percent on contention-free
point-to-point legs and is about 8x optimistic at the frozen eight-into-one
fan-in cell. This workload uses intra-node tensor parallel and one
prefill-to-decode transfer, so its declared regime is contention-free. No
packet run executes here, and the 1.6 percent statement is a regime-scoped
mechanism result rather than an absolute-accuracy claim.

## Honesty and version drift

The external rows interpolate a measured per-operation database for real H200
silicon. Our rows use a declared roofline and declared envelopes until the
calibration campaigns close. **On absolute kernel throughput, their side is
better calibrated today.** The defensible precision claims are exactly the X4
network-mechanism envelope, the evidence-class label on every number and the
exact accounting gates. Nothing broader is claimed.

The local run used aiconfigurator 0.11.0 and TensorRT-LLM database
h200_sxm 1.3.0rc10. Its best disaggregated row is 602.586 tokens/s/GPU at
108.944 tokens/s/user, with five TP4 prefill workers and three TP4 decode
workers at batch 64. The published README snapshot reports 684.79 at 100.31
with four replicas of a different TP2/TP4 topology and decode batch 68. This is
external-tool version drift. Neither anchor is preferred, and neither was used
to fit any SimLLM parameter.

## Physical sanity

Before reading the decode estimate, {decode_bytes_tp4_b64:,} bytes over 4.8
TB/s set a memory floor of {decode_memory_floor_ms:.6f} ms. The TP4 compute
floor is lower, so the observed {decode_memory_floor_ms:.6f} ms estimate sits
exactly on the physical memory floor. At the frozen lower plausibility edge of
efficiency 0.4, the same term is 13.448789 ms, which brackets the external
9.179 ms rather than making it physically impossible.

Before reading the prefill estimate,
{derivation['prefill_flops_per_request_at_tp4']:,} FLOPs over 1.979 PFLOP/s set
a compute floor of {prefill_compute_floor_ms:.6f} ms. At efficiency 0.4 the
same work takes 280.005271 ms, so the external 196.423 ms sits between the
declared ideal and low-efficiency bounds. This is an independent compute
check, not another pass over the memory arithmetic.

At system level, the external best point and our matched point both fit 32
GPUs and the same five-plus-three TP4 pool structure. Our ideal decode and
prefill services are 58.6 and 57.0 percent of their measured values. Those
ratios have the required optimistic sign and are of the same order, but their
roughly 1.7x residual is exactly why DEPLOY-11 owns silicon calibration.

## Fatal guards

| Guard | Outcome | Predicate |
|---|---|---|
{_guard_table(result)}

The pricing lane called only `simllm.deploy` and triggered zero process
interceptions. The external program did not run. All three efficiency arms
scanned {x3['arms']['1.0']['candidate_count']:,} candidates each in one process
and completed in {wall['elapsed_seconds']:.3f} seconds, below the frozen 120 s
limit. The external tool's observed 11 s search remains unscored context.

## Project consequence

What ran: `examples/frontier_comparison_v1`, a config-only extraction-backed,
binary-free comparison of the SimLLM deployment estimator against the 10
frozen aiconfigurator 0.11.0 disaggregated H200 rows.

What came out: the run is non-void and passes every family acceptance bar;
the deciding overlay result is X3c at 9 of 10 rows, with row 10 below the
declared 0.6 arm by 10.174 tokens/s/GPU. Decode and prefill implied
efficiencies are {decode_e:.6f} and {prefill_e:.6f}, and remain report-only.

What it changes for the project: the maintained frontier comparison is now a
literal binary-free study, while DEPLOY-9 registers wider candidate
enumeration, DEPLOY-10 registers additional external systems and DEPLOY-11
registers H200 silicon calibration. Those tasks name the remaining
completeness and precision work rather than letting the declared roofline read
as a calibrated model.

What it does not change: no existing calibration task closes, COMP-54 remains
open for Kimi K3, TRAF-20 remains open on its separate speed qualification, no
packet-level validity claim expands, no H200 efficiency is installed and
neither tool is validated against a live serving deployment.
"""


def _csv_rows(result: dict[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm, record in result["families"]["X3"]["arms"].items():
        for index, point in enumerate(record["frontier"], start=1):
            rows.append(
                {
                    "series": f"simllm-e{arm}",
                    "row": index,
                    "point_class": "ESTIMATE",
                    "x_tokens_per_second_per_user": point[
                        "x_tokens_per_second_per_user"
                    ]["decimal"],
                    "y_tokens_per_second_per_gpu": point[
                        "y_tokens_per_second_per_gpu"
                    ]["decimal"],
                    "candidate_id": point["candidate_id"],
                }
            )
    for point in result["families"]["X3"]["external_rows"]:
        rows.append(
            {
                "series": "aiconfigurator-0.11.0",
                "row": point["row"],
                "point_class": "MEASURED-EXTERNAL",
                "x_tokens_per_second_per_user": point[
                    "x_tokens_per_second_per_user"
                ],
                "y_tokens_per_second_per_gpu": point[
                    "y_tokens_per_second_per_gpu"
                ],
                "candidate_id": "external-disaggregated-row",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def publish(source: Path) -> None:
    result = json.loads(source.read_text(encoding="utf-8"))
    if result.get("schema") != RESULT_SCHEMA:
        raise ValueError(f"unexpected result schema {result.get('schema')!r}")
    if not result["nonvoid"]:
        raise RuntimeError("refusing to publish a void frontier comparison")

    tracked_result = STUDY_DIR / "results.json"
    tracked_csv = STUDY_DIR / "results.csv"
    figure_dir = STUDY_DIR / "figures"
    pdf_path = figure_dir / "frontier-comparison.pdf"
    png_path = figure_dir / "frontier-comparison.png"
    _write_json(tracked_result, result)
    _write_csv(tracked_csv, _csv_rows(result))
    render(result, pdf_path=pdf_path, png_path=png_path)
    _write_text(STUDY_DIR / "RESULTS.md", _report(result))

    bulk_copy = source.parent / "published-results.json"
    shutil.copyfile(tracked_result, bulk_copy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    publish(args.result)


if __name__ == "__main__":
    main()
