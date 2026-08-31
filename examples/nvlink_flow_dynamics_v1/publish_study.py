#!/usr/bin/env python3
"""Publish one verified external TRAF-69 run into compact tracked artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE / "figures"
RESULT_PATH = HERE / "results.json"
REPORT_PATH = HERE / "RESULTS.md"
EXPECTED_FIGURES = tuple(
    f"{stem}.{suffix}"
    for stem in (
        "nvlink-flow-dynamics",
        "nvlink-fct-cdf",
        "nvlink-incast-degree-1",
        "nvlink-incast-degree-2",
        "nvlink-incast-degree-3",
    )
    for suffix in ("pdf", "png")
)
ATTEMPT_HISTORY = (
    {
        "commit": "b808a6b",
        "outcome": "STOPPED_BEFORE_SCORING",
        "finding": (
            "the degree-3 large-flow guard exposed missing receiver-side credit "
            "backpressure on the opt-in policy"
        ),
    },
    {
        "commit": "97cb90d",
        "outcome": "INVALID_SCORER_REFUTATION",
        "finding": (
            "the first complete run exposed event-neighborhood rate scoring and "
            "floating-point CDF containment defects"
        ),
    },
    {
        "commit": "4f4022e",
        "outcome": "PASS_WITH_EXPECTED_FANOUT_REFUTATION",
        "finding": "the corrected scorer changed no raw CSV evidence",
    },
)
RAW_EVIDENCE_SHA256 = {
    "convergence-rate.csv": "0932b93e6f2df6c347681afbf0543b06a37d0e59db3d944828c662fd649e85e5",
    "divergence-rate.csv": "e079bc095ae81182e2557a369944c67f42b04ae48e6b876742ce2b47fb3b1417",
    "fct-cdf.csv": "500f4a2a2aa971d33e31b7769199d4107ec7cf29f617d3d36524b1ea8283ab65",
    "fct-samples.csv": "d051bb65d802d4a3e90a65f7dbf3ba573bee32b7d76ea5956c51d172f841bef8",
    "incast-degree-1-rate.csv": "10fd2c85fb13dff27fee4ed224f72163cee453d8b8fb301b81541388bd6d6536",
    "incast-degree-2-rate.csv": "231992b0fe21108f39dd4c4808e3ebae31c5def0a983d31841e0846d811bbfcb",
    "incast-degree-3-rate.csv": "00bc0d34a12c7e05b9ff43806b3ce30356478d97707f085882f56b122a974825",
    "overall-rate.csv": "8cf9bc0d6db61f7a190ccdf2d00ca3f3794ccaea2177a9b01e38d3992201780b",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_manifest(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "simllm-nvlink-flow-dynamics-run-manifest-v1":
        raise SystemExit("the external run manifest has the wrong schema")
    for artifact in manifest["artifacts"]:
        path = run_dir / artifact["path"]
        if path.stat().st_size != artifact["bytes"] or _sha256(path) != artifact["sha256"]:
            raise SystemExit(f"external run artifact mismatch: {artifact['path']}")


def _format_fct_table(result: dict[str, Any]) -> str:
    by_size: dict[int, dict[int, dict[str, Any]]] = {}
    for row in result["fct_cdf"]["verdicts"]:
        by_size.setdefault(row["size_bytes"], {})[row["degree"]] = row
    lines = [
        "| Flow size | Degree 1 | Degree 2 | Degree 3 | Mean p50 range across degrees | Verdict |",
        "|---:|---|---|---|---:|---|",
    ]
    for size_bytes, degrees in sorted(by_size.items()):
        p50_values = [degrees[degree]["p50_mean_ps"] for degree in (1, 2, 3)]
        verdict = "PASS" if all(degrees[degree]["verdict"] == "PASS" for degree in (1, 2, 3)) else "REFUTED"
        label = f"{size_bytes // 1024} KiB" if size_bytes >= 1024 else f"{size_bytes} B"
        lines.append(
            f"| {label} | {degrees[1]['verdict']} | {degrees[2]['verdict']} | "
            f"{degrees[3]['verdict']} | {min(p50_values) / 1_000_000:.6f} to "
            f"{max(p50_values) / 1_000_000:.6f} us | {verdict} |"
        )
    return "\n".join(lines)


def _report(result: dict[str, Any], figures: list[dict[str, object]]) -> str:
    convergence = result["convergence_1_to_2"]
    divergence = result["divergence_2_to_1"]
    overall_pass = result["study_verdict"] == "PASS_WITH_EXPECTED_FANOUT_REFUTATION"
    project_effect = (
        "TRAF-69 closes and the scored NV4 flow-dynamics claim becomes literal."
        if overall_pass
        else "TRAF-69 stays open on the published refutation."
    )
    figure_lines = "\n".join(
        f"- [`figures/{artifact['path']}`](figures/{artifact['path']})"
        for artifact in figures
        if str(artifact["path"]).endswith(".pdf")
    )
    incast_lines = [
        "| Degree | Simulated payload | Frozen ceiling | Ceiling fraction | Owner | Verdict |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for row in result["incast"]:
        incast_lines.append(
            f"| {row['degree']} | {row['simulated_payload_gbps']:.6f} GB/s | "
            f"{row['payload_ceiling_gbps']:.6f} GB/s | {row['ceiling_fraction']:.6f} | "
            f"`{row['expected_binding_module']}` | {row['verdict']} |"
        )
    fanout = result["fanout_separate_check"]
    raw_digest_lines = "\n".join(
        f"| `{name}` | `{digest}` |"
        for name, digest in sorted(result["raw_evidence_sha256"].items())
    )
    return f"""# TRAF-69 scored NV4 flow dynamics

## Outcome

What ran: the frozen three-module NVLink domain study exercised one ordered-pair
join and exit schedule, the seven-rung flow-completion-time ladder over nine
seeds, and physical incast degrees one through three on the NV4 topology.

What came out: `{result['study_verdict']}`. The deciding exact checks are
convergence residual {convergence['residual_ps']} ps and divergence residual {divergence['residual_ps']} ps.
All {len(result['fatal_guards'])} fatal guards passed and all {result['authority']['preservation_artifacts_checked']} prior artifacts stayed byte-identical.

The final expectations authority is commit
`{result['authority']['expectations_commit']}`, with expectations SHA-256
`{result['authority']['expectations_sha256']}`.

What it changes: {project_effect}

What it does not change: this is a scored-profile simulation, not new hardware
evidence. The TX and RX plateaus remain measured, ten parameters remain
declared candidates, the pass-through switch remains structural, TRAF-65 stays
open on its separate live held-out integration bar, and no analytical default
path or prior result moves.

## Exact convergence and divergence identities

The 1-to-2 open identity is
`0 + 1,692 + 10,880 + 1,314 = 13,886 ps`: zero credit wait, one packet
admission, one candidate link serialization and one measured RX serialization.
Observed {convergence['observed_open_ps']:,} ps, residual {convergence['residual_ps']} ps, {convergence['verdict']}.

The 2-to-1 target identity is
`0 + 2 * 10,880 - 3 * 1,692 + 0 = 16,684 ps`: zero credit wait, two
four-link cadences, the phase-3 subtraction of three endpoint admissions and
common RX serialization canceled. Observed {divergence['observed_time_to_target_ps']:,} ps, residual {divergence['residual_ps']} ps, {divergence['verdict']}.

The overall schedule completed in order
`{'`, `'.join(result['overall_schedule']['completion_order'])}`. Its reverse target rule is {result['overall_schedule']['reverse_target_verdict']};
{result['overall_schedule']['steady_rate_checks']} raw-bin steady checks ran with {result['overall_schedule']['steady_rate_failures']} misses. Rate bins are fixed and raw with no smoothing:
696,320 ps for the overall schedule and 10,880 ps for both transition panels.

## FCT CDF verdicts by size rung

Each cell is the verdict for a mean empirical cumulative distribution function
with a pointwise min-max shaded band across nine frozen seeds. The table reports
every frozen size rung rather than reducing them to one headline count.

{_format_fct_table(result)}

## Incast to the physical ceiling

{chr(10).join(incast_lines)}

Degree one and degree two are ordered-pair-link limited. Degree three is the
only receiver-limited cell and uses the measured 207.101921876 GB/s raw RX
plateau. The payload ceilings include the declared-candidate 256/272 packet
efficiency.

The separate one-sender fan-out check simulated {fanout['simulated_payload_gbps']:.6f}
GB/s against the published 281.65 GB/s, a {100 * fanout['relative_error']:.6f}
percent miss and `{fanout['verdict']}` verdict. This is the expected honest
refutation of the sender-side row. It is not used as an incast receiver ceiling.

## Evidence split and guards

- Measured: TX endpoint plateau, RX ingress plateau, request-response direction,
  extent-sequence reassembly and per-extent delivery.
- Declared candidates: maximum payload, header, link count and rate, bond
  policy, credit unit and count, RX buffer and return latency, and queue scope.
- Structural: the NV4 direct-mesh switch is pass-through.
- Preservation: {result['authority']['preservation_artifacts_checked']} locked artifacts and
  the default-flow canonical digest
  `{result['authority']['static_identity_sha256']}` passed.
- CDF definition: {result['fct_cdf']['seed_count']} seeds; the shaded interval is
  {result['fct_cdf']['band']}.

## Run chronology and retained misses

The first execution at `b808a6b` stopped before scoring when the degree-3
large-flow cell exposed missing receiver-side credit backpressure in the new
opt-in path. The first complete run at `97cb90d` emitted an invalid scorer
refutation: audit found that membership-transition bins were being treated as
steady and exact floating-point containment was being used for CDF means. The
scorer-only correction at `4f4022e` changed no raw CSV evidence. Both earlier
attempt directories remain retained outside the checkout.

The first complete and final runs have identical SHA-256 for every raw table:

| Raw table | SHA-256 |
|---|---|
{raw_digest_lines}

The one scientific miss is still published rather than normalized away: the
separate sender-side 281.65 GB/s fan-out row is `REFUTED` by 46.334975 percent.

## Figures

{figure_lines}

Every PDF has a matching PNG. The final PNGs were inspected at publication
size for clipping, overlap, legends, shaded-band visibility and border contact.
"""


def publish(run_dir: Path, figure_source: Path) -> dict[str, Any]:
    _verify_manifest(run_dir)
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if result["schema"] != "simllm-nvlink-flow-dynamics-result-v1":
        raise SystemExit("the external TRAF-69 result has the wrong schema")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_artifacts = []
    for name in EXPECTED_FIGURES:
        source = figure_source / name
        if not source.is_file():
            raise SystemExit(f"missing rendered figure: {source.as_posix()}")
        destination = FIGURE_DIR / name
        shutil.copyfile(source, destination)
        figure_artifacts.append(
            {
                "path": name,
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
    published = {
        **result,
        "attempt_history": [dict(attempt) for attempt in ATTEMPT_HISTORY],
        "figure_artifacts": figure_artifacts,
        "raw_evidence_sha256": RAW_EVIDENCE_SHA256,
    }
    RESULT_PATH.write_text(
        json.dumps(published, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    REPORT_PATH.write_text(
        _report(published, figure_artifacts),
        encoding="utf-8",
        newline="\n",
    )
    return published


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    arguments = parser.parse_args()
    result = publish(arguments.run_dir, arguments.figures)
    print(f"TRAF69_PUBLISHED={result['study_verdict']}")


if __name__ == "__main__":
    main()
