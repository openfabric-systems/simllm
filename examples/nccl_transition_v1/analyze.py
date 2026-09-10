"""Audit dense hardware captures and evaluate the frozen relation families."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from model_residuals import residuals
from observe_results import audit_observer
from run_campaign import make_plan

CHOICES = ("algo", "proto", "#channels", "#warps", "kernelVariant")


def grid_values(config, name):
    grid = config[name]
    return list(range(grid["start_bytes"], grid["end_bytes"] + 1, grid["step_bytes"]))


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo])


def extract_rows(data, run):
    """Return explicit measurement rows; diagnostic time remains labeled."""
    if data["nccl_version"] != 23102:
        raise ValueError("F3: runtime NCCL version is not 2.31.2")
    rows = []
    if run["lane"] == "legacy":
        for item in data["collectives"]:
            if item["op"] != "allreduce" or item["status"] != "measured":
                raise ValueError("F1: unexpected inherited-harness operation")
            if item["allreduce_mismatching_ranks"] != 0:
                raise ValueError("F2: inherited correctness probe failed")
            rows.append(
                {
                    "width": item["width"],
                    "bytes": item["bytes"],
                    "time_us": item["time_us"],
                    "busbw_gbps": item["busbw_gbps"],
                }
            )
    else:
        if data["version"] != 4 or data["out_of_bounds"]["count"] != 0:
            raise ValueError("F2: benchmark schema or full validation failed")
        if any(error.strip() for error in data["errors"]):
            raise ValueError("F2: benchmark error record present")
        for item in data["results"]:
            if item["type"] != "float" or item["redop"] != "sum":
                raise ValueError("F1: operation contract changed")
            for placement in ("out_of_place", "in_place"):
                if item[placement]["nwrong"] != 0:
                    raise ValueError("F2: full output validation failed")
            metric = item["out_of_place"]
            row = {
                "width": run["width"],
                "bytes": item["size"],
                "time_us": metric["time"],
                "busbw_gbps": metric["bus_bw"],
            }
            if run["lane"] == "diagnostic":
                row.update(item["tuning"])
            elif "tuning" in item:
                raise ValueError("F6: instrumented result in the timing lane")
            rows.append(row)
    for row in rows:
        if not math.isfinite(row["time_us"]) or row["time_us"] <= 0:
            raise ValueError("F1: nonpositive or nonfinite time")
    return rows


def audit_campaign(directory, config):
    raw = directory / "raw"
    manifest = json.loads((raw / "campaign.json").read_text())
    arch = manifest["architecture"]
    failures, missing_controls, records = [], [], []
    expected_keys = {(r["lane"], r["arm"], r["width"], r["repeat"]) for r in make_plan(config)}
    actual_keys = [(r["lane"], r["arm"], r["width"], r["repeat"]) for r in manifest["runs"]]
    if len(set(actual_keys)) != len(actual_keys) or set(actual_keys) != expected_keys:
        failures.append("F1: process inventory incomplete or duplicated")
    if not (directory / "completed.txt").exists():
        failures.append("F1: job completion marker absent")
    if (directory / "nccl_tests_commit.txt").read_text().strip() != config["nccl_tests_commit"]:
        failures.append("F3: upstream pin differs")
    if (
        hashlib.sha256((directory / "expectations.json").read_bytes()).hexdigest()
        != manifest["expectations_sha256"]
    ):
        failures.append("F3: expectations checksum differs")
    if json.loads((directory / "expectations.json").read_text()) != config:
        failures.append("F3: staged expectations differ from analysis")
    for phase in ("before", "after"):
        file = directory / (f"processes_{phase}.csv")
        if (
            not file.exists()
            or len([s for s in file.read_text().splitlines()[1:] if s.strip()]) != 0
        ):
            failures.append("F4: foreign GPU process check failed or missing: " + phase)
    identities = list(csv.DictReader((directory / "gpu_identity.csv").open()))
    if len(identities) != 4 or any(arch.upper() not in str(row).upper() for row in identities):
        failures.append("F3: architecture or four-GPU identity differs")
    provenance = json.loads((directory / "legacy_dense.provenance.json").read_text())
    if not provenance["timing_block_identical"]:
        failures.append("F6: inherited timing block changed")
    min_floor_ratio = math.inf
    max_bandwidth_arithmetic_error = 0.0
    for run in manifest["runs"]:
        try:
            if run["returncode"] != 0:
                if run["arm"] != "auto" or run["lane"] == "diagnostic":
                    missing_controls.append(
                        {
                            "lane": run["lane"],
                            "arm": run["arm"],
                            "width": run["width"],
                            "repeat": run["repeat"],
                            "returncode": run["returncode"],
                        }
                    )
                    continue
                raise ValueError("F1: primary process failed")
            payload = (raw / run["output"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != run["output_sha256"]:
                raise ValueError("F3: raw checksum differs")
            data = json.loads(payload)
            if run["lane"] != "legacy":
                grid = config[run["grid"]]
                diagnostic = run["lane"] == "diagnostic"
                expected_config = {
                    "nthreads": run["width"],
                    "ngpus": 1,
                    "minimum_bytes": grid["start_bytes"],
                    "maximum_bytes": grid["end_bytes"],
                    "step_bytes": grid["step_bytes"],
                    "validation": 1,
                    "aggregated_iterations": 1,
                    "graph": run["graph"],
                    "warmup_iters": config["diagnostic_warmup" if diagnostic else "timing_warmup"],
                    "iterations": config[
                        "diagnostic_iterations" if diagnostic else "timing_iterations"
                    ],
                }
                if any(data["config"].get(k) != v for k, v in expected_config.items()):
                    raise ValueError("F1: benchmark configuration differs from freeze")
                if any(
                    r["actual_iterations"] != expected_config["iterations"] for r in data["results"]
                ):
                    raise ValueError("F1: benchmark iteration count differs from freeze")
            elif data["collective_warmup_iters"] != config["legacy_warmup"] or any(
                r["timed_iters"] != config["legacy_iterations"] for r in data["collectives"]
            ):
                raise ValueError("F1: inherited timing iteration counts changed")
            rows = extract_rows(data, run)
            expected_widths = config["widths"] if run["lane"] == "legacy" else [run["width"]]
            expected = {(w, b) for w in expected_widths for b in grid_values(config, run["grid"])}
            actual = [(r["width"], r["bytes"]) for r in rows]
            if set(actual) != expected or len(set(actual)) != len(actual):
                raise ValueError("F1: payload/width inventory differs")
            for row in rows:
                endpoint = 2 * (row["width"] - 1) / row["width"] * row["bytes"]
                floor_us = endpoint / (config["egress_gbps"][arch] * 1000)
                ratio = row["time_us"] / floor_us
                min_floor_ratio = min(min_floor_ratio, ratio)
                if ratio < 1 - config["floor_tolerance_fraction"]:
                    raise ValueError("F5: aggregate NVLink serialization floor violated")
                calculated = endpoint / (row["time_us"] * 1000)
                error = abs(row["busbw_gbps"] / calculated - 1)
                max_bandwidth_arithmetic_error = max(error, max_bandwidth_arithmetic_error)
                if error > 0.0001:
                    raise ValueError("F5: bandwidth arithmetic disagrees")
                records.append(
                    dict(
                        architecture=arch,
                        lane=run["lane"],
                        arm=run["arm"],
                        repeat=run["repeat"],
                        sequence=run["sequence"],
                        **row,
                    )
                )
        except (ValueError, KeyError, OSError) as error:
            failures.append(run["output"] + ": " + str(error))
    audit = {
        "architecture": arch,
        "job_id": directory.name,
        "status": "void" if failures else "valid",
        "fatal_findings": failures,
        "missing_controls": missing_controls,
        "process_records": len(manifest["runs"]),
        "minimum_aggregate_floor_ratio": min_floor_ratio,
        "maximum_bandwidth_arithmetic_relative_error": max_bandwidth_arithmetic_error,
        "freeze": manifest["freeze"],
        "implementation": manifest["implementation"],
        "library_sha256": (directory / "library.sha256").read_text().split()[0],
        "gpu_identities": identities,
    }
    return records, audit


def summarize(records):
    grouped = defaultdict(list)
    tuning = []
    for row in records:
        if row["lane"] == "diagnostic":
            # Remove profiler-influenced time from exported tuning summaries.
            tuning.append({k: v for k, v in row.items() if k not in ("time_us", "busbw_gbps")})
        else:
            key = (row["architecture"], row["lane"], row["arm"], row["width"], row["bytes"])
            grouped[key].append(row["time_us"])
    points = []
    for key, values in sorted(grouped.items()):
        med = statistics.median(values)
        q1, q3 = quantile(values, 0.25), quantile(values, 0.75)
        points.append(
            dict(
                zip(("architecture", "lane", "arm", "width", "bytes"), key),
                median_us=med,
                min_us=min(values),
                max_us=max(values),
                q1_us=q1,
                q3_us=q3,
                relative_iqr=(q3 - q1) / med,
                repetitions=len(values),
            )
        )
    return points, tuning


def relation_results(points, tuning, config, repo):
    lookup = {(p["architecture"], p["lane"], p["arm"], p["width"], p["bytes"]): p for p in points}
    choices = {(p["architecture"], p["arm"], p["width"], p["bytes"]): p for p in tuning}

    def point(arch, width, size, arm="auto", lane="timing"):
        return lookup.get((arch, lane, arm, width, size))

    results = []
    for arch in sorted({p["architecture"] for p in points}):
        old = json.loads(
            (
                repo / (f"examples/{arch}_hardware_envelope_v1/measurements/lane_b_result.json")
            ).read_text()
        )
        historical = {
            (r["width"], r["bytes"]): r["time_us"]
            for r in old["collectives"]
            if r["op"] == "allreduce" and r["status"] == "measured"
        }
        for width in config["widths"]:
            result = {"architecture": arch, "width": width}
            historical_checks, agreement, graph_checks, transitions = [], [], [], []
            for size in (512 * 1024, 1024 * 1024, 2048 * 1024):
                p = point(arch, width, size, lane="legacy")
                if p:
                    error = p["median_us"] / historical[width, size] - 1
                    historical_checks.append(
                        {
                            "bytes": size,
                            "old_us": historical[width, size],
                            "new_us": p["median_us"],
                            "error_fraction": error,
                            "holds": abs(error) <= 0.1,
                        }
                    )
            for size in grid_values(config, "dense_grid"):
                legacy, independent = (
                    point(arch, width, size, lane="legacy"),
                    point(arch, width, size),
                )
                if legacy and independent:
                    delta = independent["median_us"] - legacy["median_us"]
                    agreement.append(
                        {
                            "bytes": size,
                            "delta_us": delta,
                            "error_fraction": delta / legacy["median_us"],
                            "holds": abs(delta) <= max(2, 0.1 * legacy["median_us"]),
                        }
                    )
            for size in grid_values(config, "control_grid"):
                graph, normal = point(arch, width, size, "auto_graph"), point(arch, width, size)
                if graph and normal:
                    delta = graph["median_us"] - normal["median_us"]
                    graph_checks.append(
                        {
                            "bytes": size,
                            "delta_us": delta,
                            "error_fraction": delta / normal["median_us"],
                            "holds": abs(delta) <= max(2, 0.1 * normal["median_us"]),
                        }
                    )
            previous = None
            for size in grid_values(config, "dense_grid"):
                current = choices.get((arch, "auto", width, size))
                if current and previous:
                    changed = [k for k in CHOICES if current[k] != previous[k]]
                    if changed:
                        before = {k: previous[k] for k in CHOICES}
                        after = {k: current[k] for k in CHOICES}
                        low = (
                            max(b for b in grid_values(config, "control_grid") if b < size)
                            if size > 512 * 1024
                            else None
                        )
                        high = next(
                            (b for b in grid_values(config, "control_grid") if b >= size), None
                        )
                        transition = {
                            "last_old_bytes": previous["bytes"],
                            "first_new_bytes": size,
                            "changed": changed,
                            "before": before,
                            "after": after,
                            "controls": [],
                        }
                        if low and high:
                            a, b = point(arch, width, low), point(arch, width, high)
                            if a and b:
                                step = b["median_us"] / a["median_us"] - 1
                                noise = max(a["relative_iqr"], b["relative_iqr"])
                                transition.update(
                                    control_low_bytes=low,
                                    control_high_bytes=high,
                                    automatic_step_fraction=step,
                                    relative_iqr=noise,
                                )
                                for arm in sorted(
                                    {p["arm"] for p in tuning} - {"auto", "auto_graph"}
                                ):
                                    c, d = (
                                        point(arch, width, low, arm),
                                        point(arch, width, high, arm),
                                    )
                                    tc, td = (
                                        choices.get((arch, arm, width, low)),
                                        choices.get((arch, arm, width, high)),
                                    )
                                    if not all((c, d, tc, td)):
                                        continue
                                    held = [
                                        k
                                        for k in changed
                                        if tc[k] == td[k] and tc[k] in (before[k], after[k])
                                    ]
                                    fixed_step = d["median_us"] / c["median_us"] - 1
                                    reduction = 1 - fixed_step / step if step > 0 else None
                                    stable = max(noise, c["relative_iqr"], d["relative_iqr"]) <= 0.1
                                    compatible = tc["algo"] == td["algo"] and tc["algo"] in (
                                        before["algo"],
                                        after["algo"],
                                    )
                                    supported = bool(
                                        held
                                        and compatible
                                        and stable
                                        and step > max(0.05, 2 * noise)
                                        and reduction >= 0.5
                                    )
                                    transition["controls"].append(
                                        {
                                            "arm": arm,
                                            "held_changed_fields": held,
                                            "fixed_step_fraction": fixed_step,
                                            "excess_step_reduction": reduction,
                                            "stable": stable,
                                            "compatible_algorithm": compatible,
                                            "supports_intervention": supported,
                                        }
                                    )
                        transitions.append(transition)
                previous = current
            window = (768, 1280) if width == 2 else (1536, 2560)
            result.update(
                H1_historical=historical_checks,
                H2_harness=agreement,
                H3_observer_available=all(
                    (arch, "auto", width, size) in choices
                    for size in grid_values(config, "dense_grid")
                ),
                H3_window_contains_transition=any(
                    window[0] * 1024 <= t["first_new_bytes"] <= window[1] * 1024
                    for t in transitions
                ),
                transitions=transitions,
                H5_graph=graph_checks,
            )
            for check in graph_checks:
                normal = choices.get((arch, "auto", width, check["bytes"]))
                graph = choices.get((arch, "auto_graph", width, check["bytes"]))
                check["same_observed_choice"] = (
                    all(normal[k] == graph[k] for k in CHOICES) if normal and graph else None
                )
            results.append(result)
    return results


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", type=Path, nargs="+")
    parser.add_argument("--expectations", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observers", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    config = json.loads(args.expectations.read_text())
    records, audits = [], []
    for directory in args.campaigns:
        rows, audit = audit_campaign(directory, config)
        records.extend(rows)
        audits.append(audit)
    points, summary_tuning = summarize(records)
    observer_choices, observer_audits = [], []
    timing_directories = {
        json.loads((d / "raw/campaign.json").read_text())["architecture"]: d for d in args.campaigns
    }
    for directory in args.observers:
        arch = json.loads((directory / "observer/observer.json").read_text())["architecture"]
        choices, observer_audit = audit_observer(directory, timing_directories[arch], config)
        observer_choices.extend(choices)
        observer_audits.append(observer_audit)
    tuning = [r for r in observer_choices if not r["in_place"] and r["lane"] == "diagnostic"]
    choice_lookup = {
        (r["architecture"], r["lane"], r["arm"], r["width"], r["bytes"], r["in_place"]): r
        for r in observer_choices
    }
    inherited_choice_comparison = []
    for row in observer_choices:
        if row["lane"] != "legacy":
            continue
        other = choice_lookup.get(
            (row["architecture"], "diagnostic", "auto", row["width"], row["bytes"], False)
        )
        inherited_choice_comparison.append(
            {
                "architecture": row["architecture"],
                "width": row["width"],
                "bytes": row["bytes"],
                "same_choice": other is not None and all(row[k] == other[k] for k in CHOICES),
            }
        )
    relations = relation_results(points, tuning, config, args.repo)
    residual_rows, residual_summary = residuals(points, args.repo)
    report = {
        "schema": "simllm-nccl-transition-analysis-v1",
        "audits": audits,
        "observer_audits": observer_audits,
        "inherited_choice_comparison": inherited_choice_comparison,
        "unavailable_original_summary_choices": sum(
            r.get("algo") in ("N/A", "", None) for r in summary_tuning
        ),
        "relation_families": relations,
        "unchanged_model_descriptive_residuals": residual_summary,
        "unstable_points": [
            p for p in points if p["relative_iqr"] > config["unstable_iqr_fraction"]
        ],
        "rows": len(records),
        "timing_points": len(points),
        "diagnostic_points": len(tuning),
        "missing_architectures": sorted(
            set(config["architectures"]) - {a["architecture"] for a in audits}
        ),
    }
    args.output.mkdir(exist_ok=True, parents=True)
    write_csv(
        args.output / "timing_repetitions.csv", [r for r in records if r["lane"] != "diagnostic"]
    )
    write_csv(args.output / "timing_summary.csv", points)
    write_csv(args.output / "tuning.csv", tuning)
    write_csv(args.output / "callback_choices.csv", observer_choices)
    write_csv(args.output / "model_residuals.csv", residual_rows)
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("relation_families", "unstable_points", "inherited_choice_comparison")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
