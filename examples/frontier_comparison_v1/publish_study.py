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


def _family_passes(family: str, record: dict[str, Any]) -> bool:
    if family == "X3c":
        return record["passed"] >= record["acceptance_minimum"]
    return record["passed"] == record["denominator"]


def _family_table(result: dict[str, Any]) -> str:
    rows = []
    for family in ("X1", "X2", "X3a", "X3b", "X3c"):
        record = _family(result, family)
        acceptance = (
            f">= {record['acceptance_minimum']}"
            if family == "X3c"
            else "all"
        )
        passed = _family_passes(family, record)
        rows.append(
            f"| {family} | {record['passed']} / {record['denominator']} | "
            f"{acceptance} | {'PASS' if passed else 'FAIL'} |"
        )
    wall = result["families"]["W"]
    rows.append(
        f"| W | {wall['passed']} / {wall['denominator']} | <= 120 s | "
        f"{'PASS' if wall['passed'] else 'FAIL'} |"
    )
    return "\n".join(rows)


def _x3c_verdict(row: dict[str, Any]) -> str:
    if row["passed"]:
        return "PASS"
    direction, arm = row["miss_direction"].split("-", maxsplit=1)
    return f"FAIL, {direction} e={arm}"


def _x3_table(result: dict[str, Any]) -> str:
    rows = []
    for external in result["families"]["X3"]["external_rows"]:
        x3b = external["x3b"]
        x3c = external["x3c"]
        rows.append(
            f"| {external['row']} | "
            f"{float(external['x_tokens_per_second_per_user']):.3f} | "
            f"{x3b['frontier_y_at_or_above_external_x']:.3f} | "
            f"{'PASS' if x3b['passed'] else 'FAIL'} | "
            f"{x3c['low_y']:.3f} | {x3c['external_y']:.3f} | "
            f"{x3c['high_y']:.3f} | {_x3c_verdict(x3c)} |"
        )
    return "\n".join(rows)


def _x3b_answer_table(result: dict[str, Any]) -> str:
    rows = []
    for external in result["families"]["X3"]["external_rows"]:
        answer = external["x3b"]["frontier_answer"]
        if answer is None:
            point = "none"
            mechanism = "no point at or above external x"
            candidate = "none"
        else:
            point = (
                f"F{answer['frontier_index']} "
                f"({answer['x_tokens_per_second_per_user']:.3f}, "
                f"{answer['y_tokens_per_second_per_gpu']:.3f})"
            )
            mechanism = answer["selection_mechanism"].replace("-", " ")
            candidate = f"`{answer['candidate_id']}`"
        rows.append(
            f"| {external['row']} | {mechanism} | {point} | {candidate} |"
        )
    return "\n".join(rows)


def _x3b_degeneracy(result: dict[str, Any]) -> str:
    groups: dict[tuple[object, ...], list[int]] = {}
    mechanisms: dict[tuple[object, ...], str] = {}
    for external in result["families"]["X3"]["external_rows"]:
        answer = external["x3b"]["frontier_answer"]
        if answer is None:
            key = (None,)
            mechanism = "no-point sentinel"
        else:
            key = (answer["frontier_index"], answer["candidate_id"])
            mechanism = answer["selection_mechanism"].replace("-", " ")
        groups.setdefault(key, []).append(external["row"])
        mechanisms[key] = mechanism
    disclosures = []
    for key, rows in groups.items():
        if len(rows) <= 1 and "endpoint" not in mechanisms[key] and "clamp" not in mechanisms[key]:
            continue
        point = "the no-point sentinel" if key == (None,) else f"frontier point F{key[0]}"
        row_text = ", ".join(str(row) for row in rows)
        disclosures.append(
            f"{point} answers external rows {row_text} through the "
            f"{mechanisms[key]} rule"
        )
    if not disclosures:
        return "No corrected X3b point answers more than one external row."
    return "; ".join(disclosures) + "."


def _x2_table(result: dict[str, Any]) -> str:
    rows = []
    for row in result["families"]["X2"]["rows"]:
        if "e_star" in row:
            detail = f"e-star {row['e_star']['decimal']:.6f}"
        else:
            detail = (
                f"{row['predicted_ps'] / 1_000_000_000:.6f} ms <= "
                f"{row['external_ps'] / 1_000_000_000:.6f} ms"
            )
        rows.append(
            f"| {row['id']} | {detail} | {'PASS' if row['passed'] else 'FAIL'} |"
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
    band = x2["e_star_band"]
    x2_rows_by_id = {row["id"]: row for row in x2["rows"]}
    decode_external_ms = x2_rows_by_id["X2a"]["external_ps"] / 1_000_000_000
    prefill_external_ms = x2_rows_by_id["X2b"]["external_ps"] / 1_000_000_000
    passed_families = []
    failed_families = []
    for family in ("X1", "X2", "X3a", "X3b", "X3c"):
        target = _family(result, family)
        outcome = passed_families if _family_passes(family, target) else failed_families
        outcome.append(family)
    (passed_families if wall["passed"] else failed_families).append("W")
    pass_text = ", ".join(passed_families)
    fail_text = ", ".join(failed_families)
    x3c_misses = [row for row in x3["X3c"]["rows"] if not row["passed"]]
    miss_rows = ", ".join(str(row["row"]) for row in x3c_misses)
    decode_band_status = (
        "inside" if band["minimum"] <= decode_e <= band["maximum"] else "outside"
    )
    prefill_band_status = (
        "inside" if band["minimum"] <= prefill_e <= band["maximum"] else "outside"
    )
    decode_bytes_tp4_b64 = (
        derivation["static_parameter_bytes"] // 4
        + 64 * derivation["decode_logical_kv_bytes_per_batch_item"] // 4
    )
    decode_flops_tp4_b64 = (
        derivation["decode_flops_per_batch_item_per_rank_tp4"] * 64
    )
    prefill_bytes_tp4 = (
        derivation["static_parameter_bytes"] // 4
        + derivation["prefill_logical_kv_bytes_per_request"] // 4
    )
    prefill_flops_tp4 = derivation["prefill_flops_per_request_per_rank_tp4"]
    peak_flops = result["declared_h200_envelope"][
        "peak_dense_fp8_flops_per_second"
    ]
    hbm_rate = result["declared_h200_envelope"]["hbm_bytes_per_second"]
    decode_compute_floor_ms = decode_flops_tp4_b64 / peak_flops * 1000
    decode_memory_floor_ms = matched["decode_step_ps"] / 1_000_000_000
    prefill_compute_floor_ms = matched["prefill_request_ps"] / 1_000_000_000
    prefill_memory_floor_ms = prefill_bytes_tp4 / hbm_rate * 1000
    decode_low_efficiency_ceiling_ms = decode_memory_floor_ms / band["minimum"]
    prefill_low_efficiency_ceiling_ms = prefill_compute_floor_ms / band["minimum"]
    prefill_external_over_ceiling = (
        prefill_external_ms / prefill_low_efficiency_ceiling_ms
    )
    semantics = result["honesty"]["external_ttft_semantics"]
    attention = derivation["attention_score_projection_inconsistency"]
    attempt = result["attempt_evidence"]
    conduct_commits = ", ".join(
        f"`{row['commit']}`"
        for row in result["honesty"]["conduct_deviation"]["commits"]
    )
    return f"""# Frontier comparison result

## Verdict

**{result['verdict']}, non-void.** The corrected scoring record passes
{pass_text} and fails {fail_text}. X2c-prefill is outside the frozen
[{band['minimum']:.2f}, {band['maximum']:.2f}] band at e-star
**{prefill_e:.6f}**, so X2 fails. X3c passes **{x3['X3c']['passed']} of
{x3['X3c']['denominator']}** rows against the frozen minimum of
{x3['X3c']['acceptance_minimum']}, so X3c fails; the misses are rows
{miss_rows}. The bands are unchanged and every miss remains in the scoring
record.

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
and {extraction['logical_visits_per_case']} visits per case. The tracked tests
verify that each committed inventory is canonical at its content-addressed
filename and that their framework-neutral content agrees exactly after source
provenance is removed. No tracked second extraction run exists, so this report
does not claim repeat-run evidence.

FG-2 confirms the fatal architecture literals exactly: 64 layers, hidden size
5120, intermediate size 25600, 64 attention heads, 8 key-value heads, head
dimension 128 and vocabulary size 151936. The FP8 checkpoint has
{x1['fp8_parameter_count']:,} logical parameters, so TP4 owns
{x1['fp8_rank_bytes_tp4']:,} weight bytes per rank at one byte per parameter.
That is 5.8 percent of the declared 141 GB H200 capacity.

## Work derivation and matched-point pricing

The pricing record derives its work from the inventory's per-layer
projections. Decode carries {derivation['decode_total_flops_per_batch_item']:,}
whole-model FLOPs per batch item and a TP4 rank owns
{derivation['decode_flops_per_batch_item_per_rank_tp4']:,}; a 3,500-token
uncached prefill carries {derivation['prefill_total_flops_per_request']:,}
whole-model FLOPs and a TP4 rank owns
{derivation['prefill_flops_per_request_per_rank_tp4']:,}. Logical weight,
key-value bytes and FLOPs now all divide by tensor-parallel width exactly once.
This is the physical ownership correction that replaced the prior whole-model
FLOP charge on every GPU.

At efficiency 1.0, the exact external-best topology prices decode batch 64 to
**{matched['decode_step_ps'] / 1_000_000_000:.6f} ms**, below the external
{decode_external_ms:.3f} ms, and prices the uncached prefill request to
**{matched['prefill_request_ps'] / 1_000_000_000:.6f} ms**, below the external
{prefill_external_ms:.3f} ms. The implied efficiencies are **{decode_e:.6f} for decode** and
**{prefill_e:.6f} for prefill**. Decode is {decode_band_status} the frozen band;
prefill is {prefill_band_status}. Both remain report-only and neither is
installed as a model parameter.

| X2 row | Corrected predicate value | Verdict |
|---|---:|---|
{_x2_table(result)}

The frozen inventories also expose an unresolved convention mismatch:
decode `attn_score` projects {attention['decode_flops_per_token_pair']:,} FLOPs
per token pair while prefill projects
{attention['prefill_flops_per_token_pair']:,}, exactly
{attention['decode_over_prefill']}x lower. COMP-81 owns the successor
reconciliation; neither frozen inventory changed in this repair.

## Frontier overlay

[PDF](figures/frontier-comparison.pdf) and
[PNG](figures/frontier-comparison.png) show the three SimLLM ESTIMATE
frontiers and all 10 MEASURED-EXTERNAL rows on logarithmic per-user speed and
per-GPU throughput axes. The upper-right corner is better. External row labels
match the table below.

X3b compares the efficiency-1.0 service-feasible frontier at or above each
external per-user speed. X3c separately compares each external row's
throughput with the 0.6 and 1.0 estimates for that row's exact prefill/decode
topology and batch, including comparison points that the 10 ms frontier filter
excludes. The X3b frontier value and X3c matched-topology value have separate
columns so one cannot be read as the other.

| Row | External user tok/s | X3b frontier e=1.0 tok/s/GPU | X3b | X3c matched e=0.6 | External tok/s/GPU | X3c matched e=1.0 | X3c |
|---:|---:|---:|---|---:|---:|---:|---|
{_x3_table(result)}

X3b uses the frozen step-frontier rule rather than interpolation.
{_x3b_degeneracy(result)} The complete answer identity is:

| External row | X3b selection | Answering frontier point (user tok/s, tok/s/GPU) | Candidate |
|---:|---|---|---|
{_x3b_answer_table(result)}

The X4 scope comes from the
[frontier ladder study](../frontier_ladder_v1/RESULTS.md). Its ideal-network
class tracks the packet rung within about 1.6 percent on contention-free
point-to-point legs and is about 8x optimistic at the frozen eight-into-one
fan-in cell. This workload uses intra-node tensor parallel and one
prefill-to-decode transfer, so its declared regime is contention-free. No
packet run executes here, and the 1.6 percent statement is a regime-scoped
mechanism result rather than an absolute-accuracy claim.

Every candidate declares zero logical collective bytes per GPU per batch item;
the X4 1 to 2 percent bound applies only to ideal-versus-packet pricing of the
represented contention-free legs, not to the omitted tensor-parallel
collective service.

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

The corrected prefill miss has a candidate semantic explanation, but it is not
used to rescore this frozen study. All 10 external rows carry operating-point
fields: concurrency ranges from {semantics['concurrency_min']} to
{semantics['concurrency_max']}, request rate ranges from
{semantics['request_rate_min']:.3f} to {semantics['request_rate_max']:.3f}, and
their TTFT column is the same {semantics['ttft_ms_values'][0]} ms value. The
external TTFT is therefore an operating-point quantity at concurrency, while
the SimLLM value is isolated prefill service. The frozen matched-point premise
conflates queueing with service. DEPLOY-12 owns a v2 comparison freeze that
must clarify the external semantics before any new score is defined.

Commits {conduct_commits} used the nonconforming `feat:` prefix. This conduct
deviation is recorded without rewriting history.

## Physical sanity

Per TP4 rank, decode batch 64 executes {decode_flops_tp4_b64:,} FLOPs and moves
{decode_bytes_tp4_b64:,} bytes. The compute floor is
{decode_compute_floor_ms:.6f} ms and the memory floor is
{decode_memory_floor_ms:.6f} ms, so the corrected ideal service sits exactly on
the larger memory floor. The frozen e=0.4 edge gives a
{decode_low_efficiency_ceiling_ms:.6f} ms envelope ceiling; the external
{decode_external_ms:.3f} ms TPOT lies inside that {decode_memory_floor_ms:.6f} to
{decode_low_efficiency_ceiling_ms:.6f} ms bracket.

Per TP4 rank, the uncached prefill executes {prefill_flops_tp4:,} FLOPs and
moves {prefill_bytes_tp4:,} bytes. Its compute floor is
{prefill_compute_floor_ms:.6f} ms while its memory floor is only
{prefill_memory_floor_ms:.6f} ms, so compute is decisive. The frozen e=0.4 edge
gives a {prefill_low_efficiency_ceiling_ms:.6f} ms service-envelope ceiling;
the external {prefill_external_ms:.3f} ms TTFT is
{prefill_external_over_ceiling:.3f}x above it.
That outside result is exactly what X2c was frozen to detect.

At system level, the external best point and our matched point both fit 32
GPUs and the same five-plus-three TP4 pool structure. Our ideal decode and
prefill values are {decode_e * 100:.1f} and {prefill_e * 100:.1f} percent of
their external columns. Decode remains a clean service comparison inside the
frozen bracket. Prefill does not: its operating-point-versus-service semantic
confound is the DEPLOY-12 residual, not a reason to widen or reinterpret the
v1 band.

## Fatal guards

| Guard | Outcome | Predicate |
|---|---|---|
{_guard_table(result)}

The pricing lane called only `simllm.deploy` and triggered zero process
interceptions. The external program did not run. All three efficiency arms
scanned {x3['arms']['1.0']['candidate_count']:,} candidates each in one process
and completed in {wall['elapsed_seconds']:.3f} seconds, below the frozen 120 s
limit. The external tool's observed 11 s search remains unscored context.

The scoring record is {attempt['attempt_id']}. Its full predecessor is
{attempt['previous_attempt_id']}, and the two deterministic projections have
the same SHA-256 `{attempt['deterministic_projection_sha256']}`. The comparison
excludes only wall-clock values, their W outcome, the overall verdict that
includes W, and attempt metadata.

## Project consequence

What ran: `examples/frontier_comparison_v1`, a config-only extraction-backed,
binary-free comparison of the SimLLM deployment estimator against the 10
frozen aiconfigurator 0.11.0 disaggregated H200 rows.

What came out: the run is non-void and the corrected result is
{result['verdict']}. X2 is {x2['passed']} of {x2['denominator']} because
prefill e-star is {prefill_e:.6f} outside the frozen band, and X3c is
{x3['X3c']['passed']} of {x3['X3c']['denominator']} against a minimum of
{x3['X3c']['acceptance_minimum']}. Decode e-star remains {decode_e:.6f} and
inside the band. Every X3c row and miss direction is published above.

What it changes for the project: the comparison study now delivers a validated
decode bracket and a refuted prefill matched-point premise. DEPLOY-12 registers
the v2 external-semantics freeze and COMP-81 registers the 8x attention-score
projection inconsistency. No implementation or calibration task closes; this
change closes nothing beyond publication of the corrected scoring record.

What it does not change: COMP-54 remains open for Kimi K3, DEPLOY-9 through
DEPLOY-11 remain open, TRAF-20 remains open on its separate speed
qualification, no packet-level validity claim expands, no H200 efficiency is
installed, neither tool is validated against a live serving deployment, and
the frozen expectations, anchors, earlier studies and inventories remain
unchanged.
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
    attempt = result.get("attempt_evidence", {})
    if attempt.get("deterministic_reproduction_matched") is not True:
        raise RuntimeError(
            "refusing to publish without a matching full deterministic rerun"
        )

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
