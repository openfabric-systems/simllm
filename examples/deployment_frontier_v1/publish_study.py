#!/usr/bin/env python3
"""Publish compact CORE-62 and TRAF-68 records from one external run."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.deployment_frontier_v1.frontier import sha256_file
from examples.deployment_frontier_v1.plot_study import (
    prepare_plot,
    render_bottleneck_figure,
    render_deployment_figure,
)

RESULT_SCHEMA = "simllm-deployment-frontier-result-v1"


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8", newline="\n")


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _counts(rows: list[str]) -> str:
    counts = Counter(rows)
    return ", ".join(f"{name} {counts[name]}" for name in sorted(counts))


def _report(result: dict[str, Any]) -> str:
    points = result["points"]
    residuals = [point["accounting"]["residual_ps"] for point in points]
    bottlenecks = _counts(
        [point["bottleneck"]["classification"] for point in points]
    )
    mechanisms = _counts(
        [point["fabric_attribution"]["dominant_mechanism"] for point in points]
    )
    modules = _counts(
        [point["intra_node_attribution"]["dominant_module"] for point in points]
    )
    direction_rows = "\n".join(
        f"- {'PASS' if row['passed'] else 'MISS'}: {row['name']}"
        for row in result["expected_direction_checks"]
    )
    table_rows = []
    for point in points:
        accounting = point["accounting"]
        table_rows.append(
            "| {configuration} | {batch} | {analytical:.6f} | {simulated:.6f} | "
            "{inter:.6f} | {intra:.6f} | {residual} | {fabric} | {module} | {binding} |".format(
                configuration=point["configuration_id"],
                batch=point["batch_per_gpu"],
                analytical=accounting["analytical_step_ps"] / 1_000_000_000,
                simulated=accounting["simulated_step_ps"] / 1_000_000_000,
                inter=accounting["inter_node_attributed_ps"] / 1_000_000_000,
                intra=accounting["intra_node_attributed_ps"] / 1_000_000_000,
                residual=accounting["residual_ps"],
                fabric=point["fabric_attribution"]["dominant_mechanism"],
                module=point["intra_node_attribution"]["dominant_module"],
                binding=point["bottleneck"]["classification"],
            )
        )
    return f"""# CORE-62 analytical frontier and TRAF-68 two-network result

## Gate verdict

**{result['status']}**: {result['verdict']} The maximum absolute unexplained
residual is **{max(abs(value) for value in residuals)} ps** across all
{len(points)} swept points. Both attributed terms use the frozen inter-then-intra
telescoping order; no residual was absorbed.

This is a roofline replay of the declared disaggregated-session decode step,
not a live SGLang frontend run. Kernel simulation is off and
`RooflineProvider(efficiency=1.0)` is the only kernel price.

## Bottleneck map

The binding classifications are {bottlenecks}. Dominant fabric mechanism rows
are {mechanisms}. Dominant candidate-module rows are {modules}. A raw network
excess remains in [results.csv](results.csv) even when the roofline or the other
network masks it from elapsed step time.

The intra-node timing is cross-architecture candidate evidence. The existing
A100 NVLink3 three-module profile prices two independent four-endpoint domains
for each eight-GPU node. It is not H100 or B100 measurement evidence.

## Figures

- [Deployment frontier](figures/deployment-frontier.pdf) and
  [PNG](figures/deployment-frontier.png)
- [Two-network bottleneck attribution](figures/two-network-bottleneck.pdf) and
  [PNG](figures/two-network-bottleneck.png)

The analytical reference is a line and the roofline simulation is a dot at each
batch. Both frontier axes are logarithmic: X is per-request decode speed and Y
is aggregate output throughput normalized per GPU. Analytical lines are
floor-style step-time bounds, so comparable real and simulated points sit on or
below them. The published standard-decode measurement retains its white diamond
marker. The y-only production anchor is a dashed horizontal line because its
batch and context were not disclosed.

## Per-point accounting

Times are milliseconds except for the exact residual column.

| Configuration | B/GPU | Analytical | Simulated | Inter attributed | Intra attributed | Residual ps | Fabric mechanism | Candidate module | Binds |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
{chr(10).join(table_rows)}

## Frozen direction checks

{direction_rows}

## Provenance and preservation

- Expectations commit: `{result['provenance']['expectations_commit']}`
- Expectations SHA-256: `{result['provenance']['expectations_sha256']}`
- Implementation run commit: `{result['provenance']['head_commit']}`
- htsim rnic-nn binary SHA-256: `{result['provenance']['htsim_rnic']['sha256']}`
txt2bin SHA-256: `{result['provenance']['txt2bin']['sha256']}`

All {result['preservation_lock']['artifacts_checked']} artifacts in the expanded
preservation class remained byte-identical. No prior flagship runner was
invoked and no prior record or figure was rewritten. TRAF-69 and COMP-77 remain
reserved for a fabric residual or a compute and composition residual.
"""


def publish(raw_result_path: Path, output_dir: Path) -> dict[str, Any]:
    """Validate, render, and write one compact repository publication."""

    result = json.loads(raw_result_path.read_text(encoding="utf-8"))
    if result.get("schema") != RESULT_SCHEMA:
        raise ValueError("unexpected deployment-frontier result schema")
    if len(result.get("points", [])) != 18:
        raise ValueError("the compact result must contain all 18 frozen points")
    if result["provenance"]["preservation_artifacts_checked"] != 43:
        raise ValueError("the compact result did not check all 43 preservation entries")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    plot = prepare_plot(result)
    deployment = render_deployment_figure(plot, figure_dir / "deployment-frontier")
    bottleneck = render_bottleneck_figure(
        plot,
        figure_dir / "two-network-bottleneck",
    )
    result["publication"] = {
        "figures": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in (*deployment, *bottleneck)
        ],
        "result_source_sha256": sha256_file(raw_result_path),
    }
    _write_json(output_dir / "result.json", result)
    source_csv = raw_result_path.with_name("points.csv")
    _write_text(
        output_dir / "results.csv",
        source_csv.read_text(encoding="utf-8"),
    )
    _write_text(output_dir / "RESULTS.md", _report(result))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = publish(args.result, args.output_dir)
    print(f"published CORE-62/TRAF-68 {result['status']} to {args.output_dir.as_posix()}")


if __name__ == "__main__":
    main()
