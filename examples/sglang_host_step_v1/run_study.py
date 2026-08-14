"""Run the frozen SGLang per-step host cost replay study (SGL-23).

The ``--check-only`` gate re-derives every frozen literal in
``expectations.json`` from the physical constants and record shapes the same
file freezes. It imports no simllm code, invokes no backend and writes
nothing, so it can run before the seam exists.

The measuring half replays the tracked SGLang smoke capture through the
selector and the packet-level sink, once per cell::

    tracked step records -> select_sglang_host_model(profile, N)
      -> HtsimStepSink (rnic-nn-fluid, ep_ranks 0..7)
      -> SerialStepLowerer -> GOAL -> htsim_rnic -> StepResult

Every cell replays the same nine records, so the only thing that moves between
cells is the per-step host cost. Run it with an explicit ``--run-dir``; nothing
is written anywhere else. The backend binaries come from ``SIMLLM_HTSIM_RNIC``
and ``SIMLLM_TXT2BIN`` as usual.

The ``results.json`` this writes is exactly the file the study tracks: the
per-file artifact digests are reduced to a per-cell rollup here, inside the
run, so the tracked artifact needs no post-processing step and reproduces
byte for byte from a rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
EXPECTATIONS = STUDY_DIR / "expectations.json"
SCHEMA = "simllm-sglang-host-step-expectations-v1"
RESULTS_SCHEMA = "simllm-sglang-host-step-results-v1"

#: expert-parallel width of the declared topology; tp_ranks stays a single rank
EP_WORLD = 8
LINKSPEED_BPS = 400_000_000_000
PROFILE = "rnic-nn-fluid"


def load_expectations() -> dict[str, Any]:
    document = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        raise SystemExit(f"expectations schema must be {SCHEMA}")
    return document


# The check-only derivation: standard library only, no simllm import


def derived_geometry(dims: dict[str, int]) -> dict[str, int]:
    """Rebuild the frozen parameter and byte counts from the model dimensions."""

    layers = dims["num_layers"]
    hidden = dims["hidden_size"]
    query_dim = dims["num_heads"] * dims["head_size"]
    kv_dim = dims["num_kv_heads"] * dims["head_size"]
    attention_params = layers * (
        hidden * (query_dim + 2 * kv_dim) + query_dim * hidden
    )
    expert_block = 3 * hidden * dims["moe_intermediate_size"] * layers
    mlp_resident_params = expert_block * dims["local_num_experts"]
    mlp_active_params = expert_block * dims["top_k"]
    weight_bytes = (attention_params + mlp_resident_params) * dims["dtype_bytes"]
    lm_head_bytes = hidden * dims["vocab_size"] * dims["dtype_bytes"]
    return {
        "attention_params": attention_params,
        "mlp_resident_params": mlp_resident_params,
        "mlp_active_params": mlp_active_params,
        "weight_bytes": weight_bytes,
        "lm_head_bytes": lm_head_bytes,
        "resident_bytes": weight_bytes + lm_head_bytes,
        "kv_bytes_per_token": 2
        * layers
        * dims["num_kv_heads"]
        * dims["head_size"]
        * dims["dtype_bytes"],
    }


def derived_provider_service_ps(document: dict[str, Any]) -> list[int]:
    """Rebuild the per-record roofline service from the frozen constants."""

    dims = document["dims"]
    envelope = document["envelope"]
    geometry = derived_geometry(dims)
    values = []
    for shape in document["input"]["shapes"]:
        flops = (
            2
            * shape["new_tokens"]
            * (geometry["attention_params"] + geometry["mlp_active_params"])
            + 4
            * shape["attention_pairs"]
            * dims["num_layers"]
            * dims["num_heads"]
            * dims["head_size"]
            + 2 * shape["sampled"] * dims["hidden_size"] * dims["vocab_size"]
        )
        moved = (
            geometry["resident_bytes"]
            + shape["kv_tokens"] * geometry["kv_bytes_per_token"]
        )
        seconds = max(
            flops / (envelope["peak_flops"] * envelope["efficiency"]),
            moved / (envelope["mem_bandwidth"] * envelope["efficiency"]),
        )
        values.append(int(seconds * 10**12))
    return values


def derived_enclosures(document: dict[str, Any]) -> dict[str, list[int]]:
    """Rebuild each cell's whole-nanosecond layer-compute total."""

    layers = document["dims"]["num_layers"]
    service = derived_provider_service_ps(document)
    enclosures = {"ideal": [layers * (value // (layers * 1000)) for value in service]}
    for cell in document["cells"]:
        if cell["profile"] == "ideal":
            continue
        floor = cell["launch_floor_ps"]
        enclosures[cell["name"]] = [
            math.ceil(max(value, floor) / 1000) for value in service
        ]
    return enclosures


def check_only(document: dict[str, Any]) -> None:
    """Re-derive every frozen literal, or exit non-zero naming the first gap."""

    failures: list[str] = []

    def expect(name: str, derived: Any, frozen: Any) -> None:
        if derived != frozen:
            failures.append(f"{name}: derived {derived!r}, frozen {frozen!r}")

    geometry = derived_geometry(document["dims"])
    for name, value in geometry.items():
        expect(f"geometry.{name}", value, document["geometry"][name])

    service = derived_provider_service_ps(document)
    expect("provider_service_ps", service, document["provider_service_ps"])

    points = document["points_ps_per_launch"]
    for cell in document["cells"]:
        floor = 0 if cell["profile"] == "ideal" else cell["launch_count"] * points[
            cell["profile"]
        ]
        expect(f"cells.{cell['name']}.launch_floor_ps", floor, cell["launch_floor_ps"])

    enclosures = derived_enclosures(document)
    expect("enclosed_calc_ns", enclosures, document["enclosed_calc_ns"])

    deltas = {
        name: [
            1000 * (value - base)
            for value, base in zip(values, enclosures["ideal"], strict=True)
        ]
        for name, values in enclosures.items()
        if name != "ideal"
    }
    expect("step_latency_delta_ps", deltas, document["step_latency_delta_ps"])

    masked = sorted(
        cell["name"]
        for cell in document["cells"]
        if cell["profile"] != "ideal"
        and all(cell["launch_floor_ps"] <= value for value in service)
    )
    expect("masked_cells", masked, sorted(document["masked_cells"]))
    host_bound = sorted(
        cell["name"]
        for cell in document["cells"]
        if cell["profile"] != "ideal"
        and all(cell["launch_floor_ps"] > value for value in service)
    )
    expect("host_bound_cells", host_bound, sorted(document["host_bound_cells"]))

    covariates = document["covariates"]
    expect(
        "covariates.graph440_minus_graph123_ps",
        1000 * (enclosures["graph440"][0] - enclosures["graph123"][0]),
        covariates["graph440_minus_graph123_ps"],
    )
    expect(
        "covariates.eager567_minus_eager42_ps",
        1000 * (enclosures["eager567"][0] - enclosures["eager42"][0]),
        covariates["eager567_minus_eager42_ps"],
    )

    bounds = document["napkin_bounds_ps"]
    envelope = document["envelope"]
    expect(
        "napkin.compute_floor_no_derate",
        int(geometry["resident_bytes"] / envelope["mem_bandwidth"] * 10**12),
        bounds["compute_floor_no_derate"],
    )
    expect(
        "napkin.compute_floor_derated",
        int(
            geometry["resident_bytes"]
            / (envelope["mem_bandwidth"] * envelope["efficiency"])
            * 10**12
        ),
        bounds["compute_floor_derated"],
    )

    if failures:
        for failure in failures:
            print(f"MISMATCH {failure}")
        raise SystemExit(1)
    print("check-only: every frozen literal re-derives from the frozen constants")


# The measuring half


def records_path(document: dict[str, Any]) -> Path:
    return REPOSITORY_ROOT / document["input"]["records_path"]


def load_records(document: dict[str, Any]) -> list[Any]:
    """Load the tracked stream after checking the frozen digest (guard G1)."""

    from simllm.core import step_records_from_jsonl

    path = records_path(document)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != document["input"]["sha256"]:
        raise SystemExit(
            f"G1 violated: {document['input']['records_path']} hashes {digest}, "
            f"frozen {document['input']['sha256']}"
        )
    records = list(step_records_from_jsonl(path))
    if len(records) != document["input"]["record_count"]:
        raise SystemExit(
            f"G1 violated: {len(records)} records, frozen "
            f"{document['input']['record_count']}"
        )
    return records


def study_dims(document: dict[str, Any]) -> Any:
    from simllm.compute import ModelDims

    return ModelDims(**document["dims"])


def build_sink(document: dict[str, Any], selection: Any, workdir: Path) -> Any:
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig

    return HtsimStepSink(
        HtsimStepSinkConfig(
            profile=PROFILE,
            tp_ranks=(0,),
            ep_ranks=tuple(range(EP_WORLD)),
            dims=study_dims(document),
            workdir=workdir,
            linkspeed_bps=LINKSPEED_BPS,
            provider=selection.provider,
            gpu=selection.gpu,
            host_model=selection.host_model,
        )
    )


def build_reference_sink(document: dict[str, Any], workdir: Path) -> Any:
    """The pre-seam construction, built by hand exactly as SGL-8's demo did."""

    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
    from simllm.compute import GPU_ENVELOPES, HostInitiationModel, RooflineProvider

    return HtsimStepSink(
        HtsimStepSinkConfig(
            profile=PROFILE,
            tp_ranks=(0,),
            ep_ranks=tuple(range(EP_WORLD)),
            dims=study_dims(document),
            workdir=workdir,
            linkspeed_bps=LINKSPEED_BPS,
            provider=RooflineProvider(document["envelope"]["efficiency"]),
            gpu=GPU_ENVELOPES[document["envelope"]["name"]],
            host_model=HostInitiationModel.ideal(),
        )
    )


def artifact_digests(workdir: Path) -> list[tuple[str, str]]:
    """Every rendered artifact under one cell workdir, by relative name."""

    rows = []
    for path in sorted(workdir.rglob("*")):
        if path.is_file():
            rows.append(
                (
                    str(path.relative_to(workdir)),
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    return rows


def artifact_digest_summary(rows: list[tuple[str, str]]) -> dict[str, Any]:
    """Reduce one cell's per-file digests to the form the study records.

    The per-file list is thousands of rows per cell and is bulk run output, so
    only this reduction is written. ``rollup_sha256`` hashes the ordered
    ``name digest`` lines, which makes it exactly as discriminating as the list
    it replaces: two cells agree here if and only if every file agrees.
    """

    payload = "\n".join(f"{name} {digest}" for name, digest in rows)
    return {
        "artifact_count": len(rows),
        "goal_files": sum(1 for name, _ in rows if name.endswith(".goal")),
        "rollup_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def replay_cell(document: dict[str, Any], sink: Any, records: list[Any]) -> list[dict]:
    rows = []
    for record in records:
        result = sink(record)
        if result is None:
            raise SystemExit(
                f"G4 violated: step {record.step_index} produced no StepResult"
            )
        outcome = sink.outcomes[-1]
        rows.append(
            {
                "step_index": record.step_index,
                "step_latency_ps": result.step_latency_ps,
                "completed_at_ps": result.completed_at_ps,
                "virtual_time_ps": record.virtual_time_ps,
                "provider_compute_ps": outcome.provider_compute_ps,
                "compute_estimate_ps": outcome.compute_estimate_ps,
                "layer_calc_ns": list(outcome.layer_calc_ns),
                "enclosed_calc_ns": sum(outcome.layer_calc_ns),
                "host_profile": outcome.host_profile,
                "host_launch_count": outcome.host_launch_count,
                "host_launch_floor_ps": outcome.host_launch_floor_ps,
                "exposed_host_ps": outcome.exposed_host_ps,
                "represented_bound": outcome.represented_bound,
                "num_sampled": outcome.num_sampled,
                "sample_count_exact": outcome.sample_count_exact,
            }
        )
    return rows


def run(document: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    from simllm.adapters.sglang.host import select_sglang_host_model

    records = load_records(document)
    efficiency = document["envelope"]["efficiency"]
    cells: dict[str, Any] = {}
    digests: dict[str, list[tuple[str, str]]] = {}

    for cell in document["cells"]:
        name = cell["name"]
        selection = (
            select_sglang_host_model("ideal", efficiency=efficiency)
            if cell["profile"] == "ideal"
            else select_sglang_host_model(
                cell["profile"],
                launch_count=cell["launch_count"],
                efficiency=efficiency,
            )
        )
        if selection.launch_floor_ps != cell["launch_floor_ps"]:
            raise SystemExit(
                f"{name}: selector reports {selection.launch_floor_ps} ps, frozen "
                f"{cell['launch_floor_ps']} ps"
            )
        workdir = run_dir / name
        sink = build_sink(document, selection, workdir)
        cells[name] = {
            "profile": selection.profile,
            "launch_count": selection.launch_count,
            "launch_floor_ps": selection.launch_floor_ps,
            "device_key": selection.gpu.name,
            "provider_envelope": selection.provider_envelope,
            "is_default": selection.is_default,
            "transfer_disclosure": selection.transfer_disclosure,
            "steps": replay_cell(document, sink, records),
        }
        digests[name] = artifact_digests(workdir)

    reference_dir = run_dir / "reference-preseam"
    reference = build_reference_sink(document, reference_dir)
    reference_steps = replay_cell(document, reference, records)
    digests["reference-preseam"] = artifact_digests(reference_dir)

    return {
        "schema": RESULTS_SCHEMA,
        "study": document["study"],
        "task": document["task"],
        "expectations_schema": SCHEMA,
        "records": document["input"]["records_path"],
        "records_sha256": document["input"]["sha256"],
        "profile": PROFILE,
        "ep_world": EP_WORLD,
        "linkspeed_bps": LINKSPEED_BPS,
        "cells": cells,
        "reference_preseam_steps": reference_steps,
        "artifact_digest_summary": {
            name: artifact_digest_summary(rows) for name, rows in digests.items()
        },
    }


# Scoring


def score(document: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    """Score R0, R1 and R2 and evaluate every fatal guard."""

    cells = results["cells"]
    ideal = cells["ideal"]["steps"]
    service = document["provider_service_ps"]
    deltas = document["step_latency_delta_ps"]

    r0 = [
        {
            "step_index": row["step_index"],
            "measured_ps": row["provider_compute_ps"],
            "frozen_ps": service[index],
            "pass": row["provider_compute_ps"] == service[index],
        }
        for index, row in enumerate(ideal)
    ]

    r1 = []
    for name, frozen in deltas.items():
        for index, row in enumerate(cells[name]["steps"]):
            measured = row["step_latency_ps"] - ideal[index]["step_latency_ps"]
            r1.append(
                {
                    "cell": name,
                    "step_index": row["step_index"],
                    "measured_delta_ps": measured,
                    "frozen_delta_ps": frozen[index],
                    "pass": measured == frozen[index],
                }
            )

    # R2 is the frozen conjunction in full: zero exposure AND the provider's
    # own bound on the masked side, positive exposure AND host-initiation on
    # the bound side. The provider's own bound is read from the ideal arm of
    # the same record rather than hardcoded, so the comparison cannot drift
    # from what the provider actually reported.
    r2 = []
    for masked, bound in (("graph122", "graph123"), ("eager41", "eager42")):
        for index in range(len(ideal)):
            masked_row = cells[masked]["steps"][index]
            bound_row = cells[bound]["steps"][index]
            provider_bound = ideal[index]["represented_bound"]
            exposure_half = (
                masked_row["exposed_host_ps"] == 0
                and bound_row["exposed_host_ps"] > 0
            )
            bound_half = (
                masked_row["represented_bound"] == provider_bound
                and bound_row["represented_bound"] == "host-initiation"
            )
            r2.append(
                {
                    "masked_cell": masked,
                    "bound_cell": bound,
                    "step_index": masked_row["step_index"],
                    "masked_exposed_ps": masked_row["exposed_host_ps"],
                    "bound_exposed_ps": bound_row["exposed_host_ps"],
                    "provider_bound": provider_bound,
                    "masked_bound": masked_row["represented_bound"],
                    "bound_cell_bound": bound_row["represented_bound"],
                    "exposure_half": exposure_half,
                    "bound_half": bound_half,
                    "pass": exposure_half and bound_half,
                }
            )

    guards = {}
    guards["G1"] = results["records_sha256"] == document["input"]["sha256"]
    # The rollup hashes the ordered per-file digest list, so comparing rollups
    # is exactly the per-file comparison. Note that every cell renders the same
    # artifacts here, so the informative half of G2 is the per-step value
    # equality against the hand-built pre-seam sink.
    guards["G2"] = (
        results["artifact_digest_summary"]["ideal"]
        == results["artifact_digest_summary"]["reference-preseam"]
        and list(cells["ideal"]["steps"]) == results["reference_preseam_steps"]
    )
    guards["G3"] = all(
        cell["device_key"] == document["calibrated_device_key"]
        and cell["provider_envelope"] == document["envelope"]["name"]
        for name, cell in cells.items()
        if name != "ideal"
    )
    guards["G4"] = all(
        row["step_latency_ps"] > 0
        and row["completed_at_ps"] == row["virtual_time_ps"] + row["step_latency_ps"]
        for cell in cells.values()
        for row in cell["steps"]
    )
    guards["G5"] = all(
        cells[name]["steps"][index]["provider_compute_ps"]
        == ideal[index]["provider_compute_ps"]
        for name in cells
        for index in range(len(ideal))
    )
    guards["G6"] = "sglang" not in sys.modules
    guards["G7"] = all(
        row["sample_count_exact"] is False
        for cell in cells.values()
        for row in cell["steps"]
    )

    covariates = {
        "graph440_minus_graph123_ps": sorted(
            {
                cells["graph440"]["steps"][index]["step_latency_ps"]
                - cells["graph123"]["steps"][index]["step_latency_ps"]
                for index in range(len(ideal))
            }
        ),
        "eager567_minus_eager42_ps": sorted(
            {
                cells["eager567"]["steps"][index]["step_latency_ps"]
                - cells["eager42"]["steps"][index]["step_latency_ps"]
                for index in range(len(ideal))
            }
        ),
        "ideal_step_latency_ps": [row["step_latency_ps"] for row in ideal],
        "eager567_step_latency_ps": [
            row["step_latency_ps"] for row in cells["eager567"]["steps"]
        ],
        "transferred_share": {
            name: [
                1000 * row["enclosed_calc_ns"] / row["step_latency_ps"]
                for row in cells[name]["steps"]
            ]
            for name in ("graph440", "eager567")
        },
    }

    return {
        "R0": {"class": "exact-oracle", "rows": r0},
        "R1": {"class": "exact-oracle", "rows": r1},
        "R2": {"class": "behavioral", "rows": r2},
        "fatal_guards": guards,
        "covariates": covariates,
        "void": not all(guards.values()),
    }


def summarize(scored: dict[str, Any]) -> str:
    lines = []
    for name in ("R0", "R1", "R2"):
        rows = scored[name]["rows"]
        passed = sum(1 for row in rows if row["pass"])
        lines.append(
            f"{name} ({scored[name]['class']}): {passed} of {len(rows)} instances pass"
        )
    held = [name for name, ok in scored["fatal_guards"].items() if ok]
    broken = [name for name, ok in scored["fatal_guards"].items() if not ok]
    lines.append(f"fatal guards held: {', '.join(held) if held else 'none'}")
    if broken:
        lines.append(f"FATAL GUARDS VIOLATED: {', '.join(broken)}; the run is void")
    else:
        lines.append("no fatal guard violated; the run is not void")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="re-derive the frozen literals and exit; imports no simllm code",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="directory for cell workdirs and results.json; required to measure",
    )
    args = parser.parse_args()
    document = load_expectations()

    if args.check_only:
        check_only(document)
        return
    if args.run_dir is None:
        parser.error("--run-dir is required when measuring")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    results = run(document, args.run_dir)
    scored = score(document, results)
    results["scoring"] = scored
    (args.run_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(summarize(scored))


if __name__ == "__main__":
    main()
