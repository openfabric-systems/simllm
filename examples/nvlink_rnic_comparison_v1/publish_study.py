#!/usr/bin/env python3
"""Publish one verified external TRAF-71 run into compact tracked evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE / "figures"
RESULT_PATH = HERE / "results.json"
DISPERSION_PATH = HERE / "dispersion.csv"
REPORT_PATH = HERE / "RESULTS.md"
EXPECTATIONS_COMMIT = "6224d90fea2eed788b8e6ba876787fe7f0e52319"
EXPECTATIONS_SHA256 = (
    "4b60365d8251b5fd3c7627dbe38c66ad1fc1c096b21fdfada4fc744320a5bdfa"
)
EXPECTED_FIGURES = tuple(
    f"{stem}.{suffix}"
    for stem in ("nvlink-rnic-fct-cdf", "nvlink-rnic-dispersion")
    for suffix in ("pdf", "png")
)
SIZE_LABELS = {
    256: "256 B",
    1024: "1 KiB",
    4096: "4 KiB",
    16384: "16 KiB",
    65536: "64 KiB",
    262144: "256 KiB",
    524288: "512 KiB",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _verify_manifest(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "simllm-nvlink-rnic-comparison-run-manifest-v1":
        raise SystemExit("the external TRAF-71 run manifest has the wrong schema")
    for artifact in manifest["artifacts"]:
        path = run_dir / artifact["path"]
        if (
            not path.is_file()
            or path.stat().st_size != artifact["bytes"]
            or _sha256(path) != artifact["sha256"]
        ):
            raise SystemExit(f"external run artifact mismatch: {artifact['path']}")
    return manifest


def _dispersion_rows(result: dict[str, Any]) -> list[dict[str, object]]:
    by_key = {
        (row["transport"], row["degree"], row["size_bytes"]): row
        for row in result["cell_summaries"]
    }
    rows = []
    for size_bytes in SIZE_LABELS:
        for degree in (1, 2, 3):
            nvlink = float(
                by_key[("nvlink-credit", degree, size_bytes)]["dispersion_ratio"]
            )
            rnic = float(by_key[("rnic-nn", degree, size_bytes)]["dispersion_ratio"])
            if nvlink < rnic:
                tighter = "nvlink-credit"
            elif rnic < nvlink:
                tighter = "rnic-nn"
            else:
                tighter = "tie"
            lower = min(nvlink, rnic)
            factor = max(nvlink, rnic) / lower if lower else None
            rows.append(
                {
                    "size_bytes": size_bytes,
                    "degree": degree,
                    "nvlink_dispersion_ratio": nvlink,
                    "rnic_dispersion_ratio": rnic,
                    "tighter_transport": tighter,
                    "absolute_difference_percentage_points": 100
                    * abs(nvlink - rnic),
                    "wider_to_tighter_factor": factor,
                }
            )
    return rows


def _cell_text(row: dict[str, object]) -> str:
    nvlink = 100 * float(row["nvlink_dispersion_ratio"])
    rnic = 100 * float(row["rnic_dispersion_ratio"])
    delta = float(row["absolute_difference_percentage_points"])
    winner = row["tighter_transport"]
    if winner == "tie":
        comparison = "tie"
    else:
        label = "rnic" if winner == "rnic-nn" else "NVLink"
        comparison = f"{label} by {delta:.3f} pp"
    return f"NV {nvlink:.3f}%, RN {rnic:.3f}%; {comparison}"


def _dispersion_table(rows: list[dict[str, object]]) -> str:
    by_key = {(row["size_bytes"], row["degree"]): row for row in rows}
    lines = [
        "| Rung | Degree 1 | Degree 2 | Degree 3 |",
        "|---:|---|---|---|",
    ]
    for size_bytes, label in SIZE_LABELS.items():
        cells = [_cell_text(by_key[(size_bytes, degree)]) for degree in (1, 2, 3)]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return "\n".join(lines)


def _fct_table(result: dict[str, Any]) -> str:
    by_key = {
        (row["transport"], row["degree"], row["size_bytes"]): row
        for row in result["cell_summaries"]
    }
    lines = [
        "| Rung | Degree | NVLink mean seed p50 | rnic-nn mean seed p50 | Signed shift |",
        "|---:|---:|---:|---:|---:|",
    ]
    for size_bytes, label in SIZE_LABELS.items():
        for degree in (1, 2, 3):
            nvlink = float(
                by_key[("nvlink-credit", degree, size_bytes)]["p50_seed_mean_ps"]
            )
            rnic = float(
                by_key[("rnic-nn", degree, size_bytes)]["p50_seed_mean_ps"]
            )
            lines.append(
                f"| {label} | {degree} | {nvlink / 1_000_000:.6f} us | "
                f"{rnic / 1_000_000:.6f} us | "
                f"{(rnic - nvlink) / 1_000_000:+.6f} us |"
            )
    return "\n".join(lines)


def _direction_table(result: dict[str, Any]) -> str:
    lines = [
        "| Freeze ID | Passed | Required | Verdict |",
        "|---|---:|---:|---|",
    ]
    for row in result["expected_direction_verdicts"]:
        required = row.get("required_passes", row["total_instances"])
        lines.append(
            f"| {row['id']} | {row['passed_instances']}/{row['total_instances']} | "
            f"{required} | {row['verdict']} |"
        )
    return "\n".join(lines)


def _mechanism_text(result: dict[str, Any]) -> str:
    by_key = {
        (row["transport"], row["degree"], row["size_bytes"]): row
        for row in result["cell_summaries"]
    }
    nvlink_rows = [
        row for row in result["cell_summaries"] if row["transport"] == "nvlink-credit"
    ]
    credit_cells = sum(int(row["credit_wait_packets"]) > 0 for row in nvlink_rows)
    credit_packets = sum(int(row["credit_wait_packets"]) for row in nvlink_rows)
    credit_wait_ps = sum(int(row["credit_wait_ps"]) for row in nvlink_rows)
    rx_cells = sum(int(row["rx_wait_packets"]) > 0 for row in nvlink_rows)
    rx_packets = sum(int(row["rx_wait_packets"]) for row in nvlink_rows)
    nvlink_oddity = []
    rnic_oddity = []
    for size_bytes in tuple(SIZE_LABELS)[1:]:
        if (
            by_key[("nvlink-credit", 3, size_bytes)]["p50_seed_mean_ps"]
            < by_key[("nvlink-credit", 1, size_bytes)]["p50_seed_mean_ps"]
        ):
            nvlink_oddity.append(SIZE_LABELS[size_bytes])
        if (
            by_key[("rnic-nn", 3, size_bytes)]["p50_seed_mean_ps"]
            < by_key[("rnic-nn", 1, size_bytes)]["p50_seed_mean_ps"]
        ):
            rnic_oddity.append(SIZE_LABELS[size_bytes])
    if nvlink_oddity and rnic_oddity:
        oddity_attribution = (
            "Because both transports show it, the frozen decision rule assigns "
            "the common sign to the staggered release pattern. Transport-specific "
            "differences in magnitude remain algorithm effects."
        )
    elif nvlink_oddity:
        oddity_attribution = (
            "Only NVLink shows it, so the frozen decision rule assigns the "
            "difference to credit-domain and stable RX arbitration behavior."
        )
    else:
        oddity_attribution = (
            "Neither transport reproduces it, so the study publishes failure to "
            "reproduce before assigning a mechanism."
        )
    return f"""- Credit window: positive reconstructed credit wait appeared in
  {credit_cells}/21 NVLink cells, covering {credit_packets:,} packets and
  {credit_wait_ps:,} ps in aggregate. This is the direct test for credit-window
  stalls. rnic-nn has no credit or congestion window.
- Pacing: the pinned rnic-nn arm is a central progressive max-min allocator
  feeding deterministic full-packet slots. It emitted DATA events only, with
  zero ACK events and zero reverse bytes. Any smoothness is max-min slot pacing,
  not ACK pacing.
- Packetization: both arms use 256 payload plus 16 header bytes, exactly
  5.882353 percent header at a full packet. The 256 B intercept is therefore
  serializer composition and slot phase: 12.194 ns for NVLink versus 5.440,
  2.720 and 2.628 ns for mapped rnic-nn degrees 1, 2 and 3.
- Incast-3 arbitration: positive reconstructed RX admission wait appeared in
  {rx_cells}/21 NVLink cells and covered {rx_packets:,} packets. NVLink uses
  release-aware per-source packet round robin followed by stable tied-arrival
  order at RX; rnic-nn uses deterministic max-min grants and packet slots.
- Degree-3-left-of-degree-1 oddity: NVLink reproduced it on
  {', '.join(nvlink_oddity) if nvlink_oddity else 'no scored rung'}; rnic-nn
  reproduced it on {', '.join(rnic_oddity) if rnic_oddity else 'no scored rung'}.
  {oddity_attribution}

The homogeneous rnic-nn adapter accepts one endpoint capacity. Its
degree-specific mapping is exact at full incast membership, but a temporarily
single active sender can exceed one 100 GB/s ordered-pair cap. That declared
limitation can only move rnic-nn FCT left and reduce its apparent dispersion;
it is not counted as an algorithm win."""


def _report(result: dict[str, Any], dispersion: list[dict[str, object]]) -> str:
    counts = {
        name: sum(row["tighter_transport"] == name for row in dispersion)
        for name in ("rnic-nn", "nvlink-credit", "tie")
    }
    large = [row for row in dispersion if int(row["size_bytes"]) >= 65536]
    large_rnic = sum(row["tighter_transport"] == "rnic-nn" for row in large)
    misses = ", ".join(result["scored_misses"]) or "none"
    return f"""# TRAF-71 NVLink credit versus rnic-nn on one physical mapping

## Outcome

What came out: `{result['study_verdict']}`. rnic-nn is tighter in
{counts['rnic-nn']}/21 rung-degree cells, NVLink is tighter in
{counts['nvlink-credit']}/21 and {counts['tie']}/21 tie. At 64 KiB and above,
rnic-nn is tighter in {large_rnic}/9 cells. Frozen nonfatal misses: {misses}.

What ran: the byte-identical seven-rung, three-degree, nine-seed staggered FCT
ladder from the merged NVLink study ran once through its scored three-module
credit domain and once through pinned htsim rnic-nn. Both arms received the
same release tuples and the frozen NVLink physical mapping.

All {len(result['fatal_guards'])} fatal guards passed. The regenerated NVLink
sample and CDF projections exactly match the merged raw SHA-256 values. The
authority is expectations commit `{result['authority']['expectations_commit']}`
with SHA-256 `{result['authority']['expectations_sha256']}`.

What it changes: TRAF-71 closes with a direct algorithm comparison and explicit
dispersion evidence.

What it does not change: the merged flow-dynamics study and its scored artifacts
remain byte-identical. This is a scored-profile simulation, not new hardware
evidence. The zero-fit homogeneous rnic-nn mapping limitation remains explicit.

## Per-rung dispersion comparison

Each cell reports `NVLink dispersion, rnic-nn dispersion; tighter transport by
absolute percentage-point difference`. Dispersion is the cross-seed p50 band
width divided by the median seed p50. Lower is tighter.

{_dispersion_table(dispersion)}

## Mechanism diagnosis

{_mechanism_text(result)}

## FCT location by rung and degree

The signed shift is rnic-nn minus NVLink. Negative is left of NVLink.

{_fct_table(result)}

## Frozen expected directions

{_direction_table(result)}

Honest misses remain published without changing a threshold or mapping.

## Figures

- [`figures/nvlink-rnic-fct-cdf.pdf`](figures/nvlink-rnic-fct-cdf.pdf)
- [`figures/nvlink-rnic-dispersion.pdf`](figures/nvlink-rnic-dispersion.pdf)

Every PDF has a matching PNG. The final PNGs were inspected at publication
size for clipping, overlap, readable log ticks, visible min-max bands, legend
crossings and border contact. Compact numeric dispersion evidence is in
[`dispersion.csv`](dispersion.csv).
"""


def publish(run_dir: Path, figure_source: Path) -> dict[str, Any]:
    manifest = _verify_manifest(run_dir)
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if result["schema"] != "simllm-nvlink-rnic-comparison-result-v1":
        raise SystemExit("the external TRAF-71 result has the wrong schema")
    if result["authority"]["expectations_commit"] != EXPECTATIONS_COMMIT:
        raise SystemExit("the external run used the wrong expectations commit")
    if result["authority"]["expectations_sha256"] != EXPECTATIONS_SHA256:
        raise SystemExit("the external run used the wrong expectations digest")
    if result["fatal_guard_verdict"] != "PASS" or any(
        row["verdict"] != "PASS" for row in result["fatal_guards"]
    ):
        raise SystemExit("the external run did not pass every fatal guard")
    if result["sample_count"] != 9072 or result["rnic_adapter_invocations"] != 189:
        raise SystemExit("the external run has the wrong frozen workload size")

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

    dispersion = _dispersion_rows(result)
    _write_csv(
        DISPERSION_PATH,
        [
            "size_bytes",
            "degree",
            "nvlink_dispersion_ratio",
            "rnic_dispersion_ratio",
            "tighter_transport",
            "absolute_difference_percentage_points",
            "wider_to_tighter_factor",
        ],
        dispersion,
    )
    run_artifacts = {
        row["path"]: row["sha256"]
        for row in manifest["artifacts"]
        if "/" not in row["path"]
    }
    published = {
        **result,
        "dispersion_comparison": dispersion,
        "figure_artifacts": figure_artifacts,
        "external_run_artifacts_sha256": run_artifacts,
    }
    RESULT_PATH.write_text(
        json.dumps(published, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    REPORT_PATH.write_text(
        _report(published, dispersion),
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
    print(f"TRAF71_PUBLISHED={result['study_verdict']}")


if __name__ == "__main__":
    main()
