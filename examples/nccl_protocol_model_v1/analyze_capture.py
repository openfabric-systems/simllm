"""Check independent capture integrity, locked predictions and isolated controls."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simllm.traffic.collective_protocol import NcclRingProtocolModel

LIBRARY = {
    "a100": "dba12e429fe11268b895d0531ba96a7f679f35227d5b1ec77c5febbcd02281bd",
    "gh200": "1dbd9a78c092f7b20e597793ca21622644ba1d1baba8a82292605e808f276dd9",
}


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_capture(directory, config, model_bytes):
    arch = directory.parent.name
    require(arch in LIBRARY, "unknown architecture")
    require((directory / "completed.txt").exists(), "capture has not completed")
    require(
        (directory / "library.sha256").read_text().split()[0] == LIBRARY[arch],
        "library identity changed",
    )
    require(
        (directory / "bin/models.json").read_bytes() == model_bytes, "candidate parameters changed"
    )
    for phase in ("before", "after"):
        require(not csv_rows(directory / f"processes_{phase}.csv"), "foreign GPU process")
    identities = csv_rows(directory / "gpu_identity.csv")
    require(
        len(identities) == 4 and all(arch.upper() in str(row).upper() for row in identities),
        "GPU architecture differs",
    )
    topology = (directory / "topology.txt").read_text()
    require(
        topology.count("NV4" if arch == "a100" else "NV6") == 12, "direct mesh topology differs"
    )
    raw = directory / "raw"
    manifest = json.loads((raw / "manifest.json").read_text())
    plan = json.loads((raw / "plan.json").read_text())
    require(
        len(manifest) == len(plan) == (15 if config.get("fresh_only", False) else 260),
        "process inventory is incomplete",
    )
    timing, methods, visibility = [], [], []
    for run in manifest:
        require(run["returncode"] == 0, "capture process failed")
        path = raw / run["result"]
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == run["sha256"], "raw checksum differs"
        )
        base = {"architecture": arch, "repeat": run["repeat"], "process_index": run["index"]}
        kind = run["kind"]
        if kind == "visibility":
            rows = csv_rows(path)
            require(
                {r["variant"] for r in rows} == {"embedded_ready", "separate_fenced"},
                "missing probe variant",
            )
            for row in rows:
                require(
                    int(row["mismatches"]) == 0 and int(row["exchanges"]) == 4096,
                    "visibility correctness failed",
                )
                require(
                    float(row["rtt_ns"]) > 0 and math.isfinite(float(row["rtt_ns"])), "invalid RTT"
                )
                visibility.append(
                    {**base, "variant": row["variant"], "rtt_ns": float(row["rtt_ns"])}
                )
        elif kind == "method":
            rows = csv_rows(path)
            require(
                [int(r["bytes"]) for r in rows] == config["method_payloads_bytes"],
                "method payload inventory differs",
            )
            for row in rows:
                require(int(row["mismatching_ranks"]) == 0, "method data mismatch")
                for field in ("width", "warmup", "timed", "persistent", "rotate"):
                    key = "warm" if field == "warmup" else field
                    require(int(row[field]) == run[key], "method configuration differs")
                for timer in ("event", "wall"):
                    value = float(row[timer + "_us"])
                    require(value > 0 and math.isfinite(value), "invalid timer")
                    methods.append(
                        {
                            **base,
                            "lane": timer,
                            "width": run["width"],
                            "bytes": int(row["bytes"]),
                            "warm": run["warm"],
                            "timed": run["timed"],
                            "persistent": run["persistent"],
                            "rotate": run["rotate"],
                            "time_us": value,
                        }
                    )
        else:
            payload = json.loads(path.read_text())
            require(payload["nccl_version"] == 23102, "runtime NCCL version differs")
            if kind == "fresh_legacy":
                require(payload["collective_warmup_iters"] == 5, "inherited warmup differs")
                rows = payload["collectives"]
                require(
                    {(r["width"], r["bytes"]) for r in rows}
                    == set(itertools.product(config["widths"], config["fresh_payloads_bytes"])),
                    "inherited inventory differs",
                )
                for row in rows:
                    require(
                        row["allreduce_mismatching_ranks"] == 0 and row["timed_iters"] == 20,
                        "inherited correctness or iterations failed",
                    )
                    timing.append(
                        {
                            **base,
                            "lane": "legacy",
                            "width": row["width"],
                            "bytes": row["bytes"],
                            "time_us": row["time_us"],
                        }
                    )
            else:
                require(
                    payload["out_of_bounds"]["count"] == 0 and not any(payload["errors"]),
                    "benchmark correctness failed",
                )
                warm = config["reference_warmup"] if kind == "fresh_reference" else run["warm"]
                timed = (
                    config["reference_iterations"] if kind == "fresh_reference" else run["timed"]
                )
                expected = {
                    "nthreads": run["width"],
                    "ngpus": 1,
                    "warmup_iters": warm,
                    "iterations": timed,
                    "validation": 1,
                    "aggregated_iterations": 1,
                    "graph": 0,
                }
                require(
                    all(payload["config"][k] == v for k, v in expected.items()),
                    "benchmark configuration differs",
                )
                rows = payload["results"]
                expected_sizes = (
                    config["fresh_payloads_bytes"]
                    if kind == "fresh_reference"
                    else list(range(run["grid"][0], run["grid"][1] + 1, run["grid"][2]))
                )
                require(
                    [r["size"] for r in rows] == expected_sizes,
                    "benchmark payload inventory differs",
                )
                for row in rows:
                    require(
                        row["out_of_place"]["nwrong"] == 0
                        and row["in_place"]["nwrong"] == 0
                        and row["actual_iterations"] == timed,
                        "benchmark row invalid",
                    )
                    entry = {
                        **base,
                        "lane": "timing" if kind == "fresh_reference" else "reference",
                        "width": run["width"],
                        "bytes": row["size"],
                        "time_us": row["out_of_place"]["time"],
                    }
                    if kind == "fresh_reference":
                        timing.append(entry)
                    elif row["size"] in config["method_payloads_bytes"]:
                        methods.append(
                            {**entry, "warm": warm, "timed": timed, "persistent": -1, "rotate": -1}
                        )
    for row in timing:
        require(row["time_us"] > 0 and math.isfinite(row["time_us"]), "non-finite timing")
        floor = (
            2
            * (row["width"] - 1)
            * row["bytes"]
            / row["width"]
            / ((100 if arch == "a100" else 150) * 1e9 * (1 if row["width"] == 2 else 3))
            * 1e6
        )
        require(row["time_us"] >= floor, "application byte floor violated")
    return (
        timing,
        methods,
        visibility,
        {
            "architecture": arch,
            "job_id": directory.name,
            "node": (directory / "hostname.txt").read_text().strip(),
            "candidate_commit": (directory / "candidate_commit.txt").read_text().strip(),
            "processes": len(manifest),
            "integrity": "valid",
            "gpu_uuids": [r[" uuid"] for r in identities],
        },
    )


def summarize(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    output = []
    for values, group in sorted(groups.items()):
        require(
            len(group) == 5 and {r["repeat"] for r in group} == set(range(5)),
            "missing or duplicate repetition",
        )
        numbers = [r["time_us"] for r in group]
        q1, median, q3 = map(float, np.quantile(numbers, [0.25, 0.5, 0.75]))
        output.append(
            {**dict(zip(keys, values, strict=True)), "median_us": median, "q1_us": q1, "q3_us": q3}
        )
    return output


def comparisons(methods):
    keys = ("architecture", "lane", "width", "bytes", "warm", "timed", "persistent", "rotate")
    rows = summarize(methods, keys)
    index = {tuple(row[k] for k in keys): row for row in rows}
    contrasts = []
    arms = {
        "warmup_only": (20, 20, 0, 0),
        "iterations_only": (5, 100, 0, 0),
        "workers_only": (5, 20, 1, 0),
        "rotation_only": (5, 20, 0, 1),
        "joint": (20, 100, 1, 1),
    }
    for arch, width, size in sorted({(r["architecture"], r["width"], r["bytes"]) for r in rows}):
        base = index[(arch, "event", width, size, 5, 20, 0, 0)]
        reference = index[(arch, "reference", width, size, 20, 100, -1, -1)]
        gap = abs(base["median_us"] - reference["median_us"])
        for arm, settings in arms.items():
            control = index[(arch, "event", width, size, *settings)]
            residual = abs(control["median_us"] - reference["median_us"])
            pooled = math.hypot(base["q3_us"] - base["q1_us"], control["q3_us"] - control["q1_us"])
            effect = abs(control["median_us"] - base["median_us"])
            qualifies = gap > 0 and residual <= gap / 2 and effect > 2 * pooled
            contrasts.append(
                {
                    "architecture": arch,
                    "width": width,
                    "bytes": size,
                    "control": arm,
                    "baseline_us": base["median_us"],
                    "reference_us": reference["median_us"],
                    "controlled_us": control["median_us"],
                    "baseline_gap_us": gap,
                    "controlled_gap_us": residual,
                    "effect_us": effect,
                    "twice_pooled_iqr_us": 2 * pooled,
                    "qualifies": qualifies,
                    "diagnostic_tiny": size < 524288,
                }
            )
    return rows, contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expectations", type=Path, default=Path(__file__).parent / "expectations.json"
    )
    args = parser.parse_args()
    config = json.loads(args.expectations.read_text())
    model_bytes = args.models.read_bytes()
    models = [NcclRingProtocolModel.from_json(v) for v in json.loads(model_bytes)]
    timing, methods, visibility, provenance = [], [], [], []
    for directory in args.captures:
        a, b, c, d = read_capture(directory, config, model_bytes)
        timing += a
        methods += b
        visibility += c
        provenance.append(d)
    summary = summarize(timing, ("architecture", "lane", "width", "bytes"))
    index = {(m.architecture, m.width): m for m in models}
    predictions = []
    for row in summary:
        e = index[(row["architecture"], row["width"])].predict(row["bytes"])
        measured = row["median_us"] * 1e6
        model = index[(row["architecture"], row["width"])]
        target = (
            e.reference_ps
            if model.center_method == "event" and row["lane"] == "timing"
            else e.central_ps
        )
        predictions.append(
            {
                **row,
                "measured_us": row["median_us"],
                "protocol": e.geometry.protocol,
                "model_us": target / 1e6,
                "reference_us": e.reference_ps / 1e6,
                "lower_us": e.lower_ps / 1e6,
                "upper_us": e.upper_ps / 1e6,
                "signed_error_pct": 100 * (target / measured - 1),
                "covered": e.lower_ps <= measured <= e.upper_ps,
                "band_width_fraction": (e.upper_ps - e.lower_ps) / measured,
            }
        )
    curves = []
    for key in sorted({(r["architecture"], r["width"], r["lane"]) for r in predictions}):
        selected = [r for r in predictions if (r["architecture"], r["width"], r["lane"]) == key]
        worst = max(selected, key=lambda r: abs(r["signed_error_pct"]))
        curves.append(
            {
                "architecture": key[0],
                "width": key[1],
                "lane": key[2],
                "points": len(selected),
                "covered": sum(r["covered"] for r in selected),
                "worst_center_error_pct": worst["signed_error_pct"],
                "worst_payload_bytes": worst["bytes"],
                "maximum_band_width_fraction": max(r["band_width_fraction"] for r in selected),
            }
        )
    method_summary, contrasts = comparisons(methods)
    probe = []
    for arch in sorted({r["architecture"] for r in visibility}):
        for variant in ("embedded_ready", "separate_fenced"):
            values = [
                r["rtt_ns"]
                for r in visibility
                if r["architecture"] == arch and r["variant"] == variant
            ]
            require(len(values) == 5, "missing probe repeat")
            q1, median, q3 = map(float, np.quantile(values, [0.25, 0.5, 0.75]))
            probe.append(
                {
                    "architecture": arch,
                    "variant": variant,
                    "median_ns": median,
                    "q1_ns": q1,
                    "q3_ns": q3,
                }
            )
    result = {
        "schema": "simllm-nccl-protocol-validation-v1",
        "integrity": "valid",
        "provenance": provenance,
        "fresh_comparison": curves,
        "uncovered_points": [r for r in predictions if not r["covered"]],
        "visibility_probe": probe,
        "method_contrasts": contrasts,
        "parameter_sha256": hashlib.sha256(model_bytes).hexdigest(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, rows in [
        ("timing_repetitions", timing),
        ("predictions", predictions),
        ("method_repetitions", methods),
        ("method_summary", method_summary),
        ("method_contrasts", contrasts),
        ("visibility_repetitions", visibility),
    ]:
        write_csv(args.output / (name + ".csv"), rows)
    print(json.dumps({"fresh_comparison": curves, "visibility_probe": probe}, indent=2))


if __name__ == "__main__":
    main()
