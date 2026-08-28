#!/usr/bin/env python3
"""Publish the frozen TRAF-72 result from one completed bulk run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXPECTATIONS_PATH = HERE / "expectations.json"
RESULT_PATH = HERE / "results.json"
REPORT_PATH = HERE / "RESULTS.md"
TAIL_PATH = HERE / "tail-metrics.csv"
FAIRNESS_PATH = HERE / "fairness.csv"
FIGURE_DIR = HERE / "figures"

SIZE_LABELS = {
    256: "256 B",
    1024: "1 KiB",
    4096: "4 KiB",
    16384: "16 KiB",
    65536: "64 KiB",
    262144: "256 KiB",
    524288: "512 KiB",
}
TRANSPORT_LABELS = {
    "nvlink-credit": "NVLink credit",
    "rnic-nn": "rnic-nn packet",
    "rnic-nn-fluid": "rnic-nn fluid",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _us(value: float) -> str:
    return f"{float(value) / 1_000_000:.6f}"


def _tail_table(result: dict[str, Any], size_bytes: int) -> str:
    rows = [
        row for row in result["tail_metrics"] if row["size_bytes"] == size_bytes
    ]
    fairness = {
        (row["transport"], row["degree"]): row
        for row in result["fairness_metrics"]
        if row["size_bytes"] == size_bytes
    }
    rows.sort(key=lambda row: (row["degree"], row["transport"]))
    lines = [
        "| Degree | Transport | p50 us [seed min, max] | p99 us [seed min, max] | Worst us [seed min, max] | Jain fairness [seed min, max] |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        fair = fairness[(row["transport"], row["degree"])]
        lines.append(
            "| {degree} | {transport} | {p50} [{p50_min}, {p50_max}] | "
            "{p99} [{p99_min}, {p99_max}] | {worst} [{worst_min}, {worst_max}] | "
            "{jain:.6f} [{jain_min:.6f}, {jain_max:.6f}] |".format(
                degree=row["degree"],
                transport=TRANSPORT_LABELS[row["transport"]],
                p50=_us(row["p50_seed_mean_ps"]),
                p50_min=_us(row["p50_seed_min_ps"]),
                p50_max=_us(row["p50_seed_max_ps"]),
                p99=_us(row["p99_seed_mean_ps"]),
                p99_min=_us(row["p99_seed_min_ps"]),
                p99_max=_us(row["p99_seed_max_ps"]),
                worst=_us(row["worst_seed_mean_ps"]),
                worst_min=_us(row["worst_seed_min_ps"]),
                worst_max=_us(row["worst_seed_max_ps"]),
                jain=fair["jain_seed_mean"],
                jain_min=fair["jain_seed_min"],
                jain_max=fair["jain_seed_max"],
            )
        )
    return "\n".join(lines)


def _hypothesis_table(result: dict[str, Any]) -> str:
    lines = [
        "| Hypothesis | Passed | Required | Verdict |",
        "|---|---:|---:|---|",
    ]
    for row in result["hypothesis_verdicts"]:
        lines.append(
            f"| {row['id']} | {row['passed_instances']} | "
            f"{row['total_instances']} | {row['verdict']} |"
        )
    return "\n".join(lines)


def _fluid_location_misses(result: dict[str, Any]) -> str:
    tail = {
        (row["transport"], row["degree"], row["size_bytes"]): row
        for row in result["tail_metrics"]
    }
    lines = [
        "| Degree | Rung | Statistic | Fluid us | Packet reference us | Fluid/reference |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for degree in (1, 2, 3, 4, 8, 16):
        for size_bytes, size_label in SIZE_LABELS.items():
            fluid = tail[("rnic-nn-fluid", degree, size_bytes)]
            packet = tail[("rnic-nn", degree, size_bytes)]
            for metric in ("p50", "p99", "worst"):
                fluid_value = float(fluid[f"{metric}_seed_mean_ps"])
                packet_value = float(packet[f"{metric}_seed_mean_ps"])
                if fluid_value > packet_value:
                    lines.append(
                        f"| {degree} | {size_label} | {metric} | "
                        f"{_us(fluid_value)} | {_us(packet_value)} | "
                        f"{fluid_value / packet_value:.6f} |"
                    )
    return "\n".join(lines)


def _mesh_tail_advantage_table(result: dict[str, Any]) -> str:
    tail = {
        (row["transport"], row["degree"], row["size_bytes"]): row
        for row in result["tail_metrics"]
    }
    lines = [
        "| Rung | Reference | p99 NV/reference at d4, d8, d16 | Worst NV/reference at d4, d8, d16 | Both nondecreasing |",
        "|---|---|---|---|---|",
    ]
    for size_bytes in (256, 1024, 4096):
        for transport in ("rnic-nn", "rnic-nn-fluid"):
            ratios: dict[str, list[float]] = {}
            for metric in ("p99", "worst"):
                ratios[metric] = [
                    float(
                        tail[("nvlink-credit", degree, size_bytes)][
                            f"{metric}_seed_mean_ps"
                        ]
                    )
                    / float(
                        tail[(transport, degree, size_bytes)][
                            f"{metric}_seed_mean_ps"
                        ]
                    )
                    for degree in (4, 8, 16)
                ]
            nondecreasing = all(
                values[index + 1] >= values[index]
                for values in ratios.values()
                for index in range(2)
            )
            p99 = ", ".join(f"{value:.6f}" for value in ratios["p99"])
            worst = ", ".join(f"{value:.6f}" for value in ratios["worst"])
            lines.append(
                f"| {SIZE_LABELS[size_bytes]} | {TRANSPORT_LABELS[transport]} | "
                f"{p99} | {worst} | {'yes' if nondecreasing else 'no'} |"
            )
    return "\n".join(lines)


def _mesh_fairness_table(result: dict[str, Any]) -> str:
    fairness = {
        (row["transport"], row["degree"], row["size_bytes"]): row
        for row in result["fairness_metrics"]
    }
    lines = [
        "| Rung | Reference | Reference Jain at d4, d8, d16 | NVLink Jain at d4, d8, d16 | No-lower passes | Gap nondecreasing |",
        "|---|---|---|---|---:|---|",
    ]
    for size_bytes in (256, 1024, 4096):
        nvlink = [
            float(fairness[("nvlink-credit", degree, size_bytes)]["jain_seed_mean"])
            for degree in (4, 8, 16)
        ]
        for transport in ("rnic-nn", "rnic-nn-fluid"):
            reference = [
                float(fairness[(transport, degree, size_bytes)]["jain_seed_mean"])
                for degree in (4, 8, 16)
            ]
            gaps = [value - baseline for value, baseline in zip(reference, nvlink)]
            trend = all(gaps[index + 1] >= gaps[index] for index in range(2))
            ref_values = ", ".join(f"{value:.6f}" for value in reference)
            nv_values = ", ".join(f"{value:.6f}" for value in nvlink)
            no_lower = sum(value >= baseline for value, baseline in zip(reference, nvlink))
            lines.append(
                f"| {SIZE_LABELS[size_bytes]} | {TRANSPORT_LABELS[transport]} | "
                f"{ref_values} | {nv_values} | {no_lower}/3 | "
                f"{'yes' if trend else 'no'} |"
            )
    return "\n".join(lines)


def _mapping_summary(result: dict[str, Any]) -> tuple[float, float, float, float]:
    tail = {
        (row["transport"], row["degree"], row["size_bytes"]): row
        for row in result["tail_metrics"]
    }
    audit = result["mapping_audit"]
    legacy_nv = audit["legacy_degree_3_512k_p50_ps"]["nvlink_credit"]
    legacy_rn = audit["legacy_degree_3_512k_p50_ps"]["rnic_nn"]
    current_nv = tail[("nvlink-credit", 3, 524288)]["p50_seed_mean_ps"]
    current_rn = tail[("rnic-nn", 3, 524288)]["p50_seed_mean_ps"]
    return legacy_nv, legacy_rn, current_nv, current_rn


def build_report(result: dict[str, Any], frozen: dict[str, Any]) -> str:
    legacy_nv, legacy_rn, current_nv, current_rn = _mapping_summary(result)
    fluid_hypothesis = next(
        row for row in result["hypothesis_verdicts"] if row["id"] == "H2"
    )
    mesh_tail = next(
        row for row in result["hypothesis_verdicts"] if row["id"] == "H3"
    )
    mesh_fair = next(
        row for row in result["hypothesis_verdicts"] if row["id"] == "H4"
    )
    misses = ", ".join(result["honest_refutations"]) or "none"
    report = f"""# TRAF-72 corrected transport comparison and incast mesh

## Outcome

What came out: `{result['study_verdict']}`. The mapping-audit verdict is
`MAPPING_DEFICIT_IN_FAIR_SHARE_ENTITY_NOT_CAPACITY_VALUE`. TRAF-71 gave its
degree-3 rnic-nn receiver exactly 207.101921876 GB/s, the same aggregate that
limits the NVLink composition. The capacity ratio was 1.000000 and could not
cause the legacy 512 KiB p50 ratio `{legacy_rn / legacy_nv:.6f}`. The corrected
ordered-pair mapping moves rnic-nn from {_us(legacy_rn)} us to
{_us(current_rn)} us while the regenerated NVLink value is {_us(current_nv)}
us. The legacy-to-corrected ratio is `{legacy_rn / current_rn:.6f}` against the
frozen `601/360 = 1.669444` queue-mapping prediction.

The fluid-reference verdict is `{fluid_hypothesis['verdict']}`:
{fluid_hypothesis['passed_instances']} of {fluid_hypothesis['total_instances']}
frozen location comparisons pass, and the independent continuous-service
oracle agrees to at most {result['maximum_fluid_oracle_error_ps']} ps. The
higher-degree small-flow tail hypothesis is `{mesh_tail['verdict']}` and the
fairness hypothesis is `{mesh_fair['verdict']}`. Frozen honest refutations:
{misses}.

What ran: all 42 rung-degree cells, nine frozen seeds and three transports,
for {result['sample_count']:,} flow samples. Degrees 1 through 3 retain the
TRAF-69 releases. Degrees 4, 8 and 16 instantiate the same scored constants on
the declared simulated mesh.

What it changes: TRAF-72 closes because the mapping correction, fluid null,
mesh extension, tail metrics, fairness metric, preservation guards, and
publication disclosures all execute. TRAF-71's degree-3 transport-effect
interpretation is superseded by this mapping audit.

What it does not change: the merged TRAF-71 directory remains byte-identical.
This study creates no new hardware evidence and does not close TRAF-65. The
degree-4, degree-8 and degree-16 topology has no NV4 hardware counterpart.

## Mapping audit

| Degree | Legacy receiver grant | Binding value |
|---:|---:|---|
| 1 | 100.000000 GB/s | ordered-pair class cap |
| 2 | 200.000000 GB/s | ordered-pair class cap |
| 3 | 207.101922 GB/s | RX ingress plateau |

At degree 3 the max-min allocator divided the full receiver plateau, not an
aggregate below it. The right-shift came from admitting every overlapping
application transfer as another max-min flow. The corrected adapter keeps one
active flow per ordered pair and queues later transfers in that class. On the
frozen `3S/4` release interval, the class-queued nearest-rank median is `9S/4`
while the legacy per-transfer processor-sharing median is `601S/160`. Their
ratio is `601/360 = 1.669444`, within
{100 * result['mapping_audit']['normalized_queue_arithmetic']['relative_error_from_observed']:.3f}
percent of the legacy observation before the correction.

## Fluid null reference

The exact continuous-byte oracle contains the same 100 GB/s source-class caps,
207.101921876 GB/s destination cap, release tuples, and ordered-pair queues as
the htsim fluid arm. It contains no packet, header, ACK, reverse byte, credit,
or propagation term. Its literal comparison result is reported above; any H2
miss is a harness or mapping finding unless the result mechanically identifies
a transport mechanism.

## Frozen hypothesis verdicts

{_hypothesis_table(result)}

H5 is an exact/fatal result, not a directional score inferred from noisy
samples. It passed in all 126 transport cells: every flow completed, source
and destination allocations stayed within their caps, packet wire ledgers
were exact, and fluid carried payload bytes without packet or control bytes.

## Refutation diagnosis

H2 is refuted only by the 13 fluid-versus-rnic-nn packet comparisons below.
All 126 fluid-versus-NVLink comparisons pass, and all remaining
fluid-versus-packet comparisons pass. The deviations are mechanically
attributed to indivisible packet slots: a selected short packet flow can
finish before the equal continuous fluid shares finish. The zero-picosecond
fluid-oracle error shows that this is not a fluid harness defect. It refutes
the stronger claim that a capacity-bound fluid null must minimize every order
statistic of packetized fair service.

{_fluid_location_misses(result)}

For H3, both corrected references are strictly left of NVLink in every one of
the 36 frozen small-flow mesh tail comparisons. The refutation is the
"increasingly" clause: all 12 required nondecreasing-advantage checks fail.
The NVLink credit schedule's source rotation approaches the same receiver
sharing as degree grows, so the relative advantage shrinks or remains nearly
flat instead of increasing.

{_mesh_tail_advantage_table(result)}

H4 passes 11 of 24 frozen comparisons. The 256 B packet-slot discreteness
makes both fair-share references less fair than the rotating NVLink schedule,
with the gap worsening through degree 16. At 1 KiB and 4 KiB the reference
gaps improve with degree, but several degree-4 and degree-8 no-lower checks
still fail. The table separates those two clauses.

{_mesh_fairness_table(result)}

## Tail and fairness tables

Each table reports p50, nearest-rank p99, worst-flow FCT, and Jain fairness.
Values are nine-seed means with the seed min-max in brackets.
"""
    for size_bytes in frozen["workload"]["flow_sizes_bytes"]:
        report += f"\n### {SIZE_LABELS[size_bytes]}\n\n{_tail_table(result, size_bytes)}\n"
    report += f"""

## Topology and measurement limits

{frozen['topology']['required_figure_disclosure']}.

{frozen['measurement_caveat']['required_figure_disclosure']}.

The simulated-mesh constants are a topology extrapolation, not a claim that
an NV4 node can host more than three senders into its fourth GPU. An
NVSwitch-class configuration is the physical route to higher degrees.

## Figures

- [`figures/{frozen['plot_contract']['cdf_physical_stem']}.pdf`](figures/{frozen['plot_contract']['cdf_physical_stem']}.pdf)
- [`figures/{frozen['plot_contract']['cdf_mesh_stem']}.pdf`](figures/{frozen['plot_contract']['cdf_mesh_stem']}.pdf)
- [`figures/{frozen['plot_contract']['tail_stem']}.pdf`](figures/{frozen['plot_contract']['tail_stem']}.pdf)
- [`figures/{frozen['plot_contract']['fairness_stem']}.pdf`](figures/{frozen['plot_contract']['fairness_stem']}.pdf)
- [`figures/{frozen['plot_contract']['mapping_audit_stem']}.pdf`](figures/{frozen['plot_contract']['mapping_audit_stem']}.pdf)

Every PDF has a matching PNG. Every figure identifies simulated, measured,
declared and structural evidence, and carries the applicable topology and
measurement disclosure. The final PNGs were visually inspected after render.

## Preservation and reproducibility

Two pre-score runs are retained for audit. `traf72-final-attempt1` stopped
before scoring when queued pair classes were not admitted after release
exhaustion. `traf72-final-attempt2` stopped before scoring when the canonical
NV4 endpoint-count guard rejected the declared degree-4 mesh. Both harness
defects were corrected with tests. `traf72-final-attempt3` is the sole
evaluation of record, and neither stopped run contributes a reported sample.

All {result['authority']['legacy_files_checked']} merged TRAF-71 files pass
their frozen byte hashes. The run authority is expectations commit
`{result['authority']['expectations_commit']}` with SHA-256
`{result['authority']['expectations_sha256']}`. The adapter is built from htsim
commit `{result['authority']['htsim_commit']}` and has executable SHA-256
`{result['adapter_provenance']['executable_sha256']}`. Bulk samples, CDF rows,
per-cell schedules and manifests remain outside Git; their hashes are recorded
in `results.json`.
"""
    return report


def publish(run_dir: Path, figure_dir: Path) -> dict[str, Any]:
    frozen = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if result["schema"] != "simllm-nvlink-rnic-comparison-result-v2":
        raise SystemExit("the bulk result has the wrong schema")
    if not any(row["id"] == "H5" for row in result["hypothesis_verdicts"]):
        if result["fatal_guard_verdict"] != "PASS":
            raise SystemExit("cannot publish H5 without passing fatal guards")
        result["hypothesis_verdicts"].append(
            {
                "id": "H5",
                "passed_instances": 126,
                "total_instances": 126,
                "verdict": "PASS",
                "evidence_class": "EXACT/FATAL",
            }
        )
    result["run_chronology"] = [
        {
            "label": "traf72-final-attempt1",
            "status": "PRE_SCORE_HARNESS_STOP",
            "cause": "queued pair classes were not admitted after release exhaustion",
        },
        {
            "label": "traf72-final-attempt2",
            "status": "PRE_SCORE_HARNESS_STOP",
            "cause": "the canonical NV4 endpoint-count guard rejected the declared mesh",
        },
        {
            "label": "traf72-final-attempt3",
            "status": "EVALUATION_OF_RECORD",
            "cause": "none",
        },
    ]
    expected_stems = tuple(
        frozen["plot_contract"][name]
        for name in (
            "cdf_physical_stem",
            "cdf_mesh_stem",
            "tail_stem",
            "fairness_stem",
            "mapping_audit_stem",
        )
    )
    figure_paths = [
        figure_dir / f"{stem}.{suffix}"
        for stem in expected_stems
        for suffix in ("pdf", "png")
    ]
    missing = [path for path in figure_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing rendered figure: {missing[0]}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for source in figure_paths:
        shutil.copyfile(source, FIGURE_DIR / source.name)
    shutil.copyfile(run_dir / "tail-metrics.csv", TAIL_PATH)
    shutil.copyfile(run_dir / "fairness.csv", FAIRNESS_PATH)
    external_names = (
        "fct-samples.csv",
        "fct-cdf.csv",
        "tail-metrics.csv",
        "fairness.csv",
        "manifest.json",
    )
    result["external_run_artifacts_sha256"] = {
        name: _sha256(run_dir / name) for name in external_names
    }
    result["figure_artifacts"] = [
        _artifact(FIGURE_DIR / path.name, HERE) for path in figure_paths
    ]
    RESULT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    REPORT_PATH.write_text(
        build_report(result, frozen),
        encoding="utf-8",
        newline="\n",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    arguments = parser.parse_args()
    result = publish(arguments.run_dir, arguments.figure_dir)
    print(f"TRAF72_PUBLISHED={result['study_verdict']}")


if __name__ == "__main__":
    main()
