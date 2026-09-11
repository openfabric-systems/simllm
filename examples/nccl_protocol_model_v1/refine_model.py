"""Apply frozen source chooser and protocol-specific method identification."""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_study import read_rows, write_csv

from simllm.traffic.collective_protocol import MIB, PROTOCOLS, NcclRingProtocolModel


def chooser_costs(root, arch, width):
    job = "204703" if arch == "a100" else "204704"
    observations = {p: [] for p in PROTOCOLS}
    for path in sorted(
        (root / arch / job / "raw").glob("*diagnostic_Ring_*_w" + str(width) + "_r0.nccl.*.log")
    ):
        if "_cta" in path.name:
            continue
        pending = {}
        for line in path.read_text().splitlines():
            if "[0] NCCL INFO" not in line:
                continue
            key = line.split("NCCL INFO")[0]
            if "Input:" in line:
                match = re.search(
                    r"\.func = AllReduce,.*\.dataType = 7, \.nBytes = (\d+), \.numPipesOps = (\d+)",
                    line,
                )
                pending[key] = None if not match or match[2] != "1" else int(match[1])
            if "Best tuning {" in line and pending.get(key) is not None:
                match = re.search(
                    r"timeUs = ([\d.]+),.*\.algo = RING, \.proto = (LL128|LL|SIMPLE),", line
                )
                if match:
                    observations[match[2]].append((pending[key], float(match[1])))
                pending[key] = None
    result = []
    audit = []
    for protocol in PROTOCOLS:
        rows = sorted(set(observations[protocol]))
        if len(rows) < 2:
            raise ValueError("missing source chooser observations")
        x = np.array([[1, size / MIB] for size, _ in rows])
        y = np.array([value for _, value in rows])
        fit = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = float(np.max(np.abs(x @ fit - y)))
        if residual > 2e-4 or min(fit) < 0:
            raise ValueError("source chooser is not the frozen linear operator")
        result.append((protocol, round(fit[0] * 1e6), round(fit[1] * 1e6)))
        audit.append(
            {
                "protocol": protocol,
                "distinct_software_predictions": len(rows),
                "maximum_reconstruction_error_us": residual,
            }
        )
    return tuple(result), audit


def starts(model):
    candidates = {model.payload_min_bytes}
    costs = model.choice_costs
    for i, (_, a, b) in enumerate(costs):
        for _, c, d in costs[i + 1 :]:
            if b == d:
                continue
            point = (c - a) * MIB / (b - d)
            for offset in (-32, -16, 0, 16, 32):
                value = int(point) // 16 * 16 + offset
                if model.payload_min_bytes <= value <= model.payload_max_bytes:
                    candidates.add(value)
    result = []
    for size in sorted(candidates):
        protocol = model.protocol(size)
        if not result or result[-1][1] != protocol:
            result.append((size, protocol))
    return tuple(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--known", type=Path, required=True)
    parser.add_argument("--first-validation", type=Path, required=True)
    parser.add_argument("--known-captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = [NcclRingProtocolModel.from_json(v) for v in json.loads(args.base.read_text())]
    known = read_rows(args.known / "timing_summary.csv")
    callbacks = read_rows(args.known / "tuning.csv")
    fresh = read_rows(args.first_validation / "predictions.csv")
    for row in fresh:
        row["arm"] = "auto"
        row["relative_iqr"] = (row["q3_us"] - row["q1_us"]) / row["median_us"]
    models = []
    audits = []
    predictions = []
    for model in base:
        costs, source = chooser_costs(args.known_captures, model.architecture, model.width)
        candidate = replace(
            model,
            model_id=model.model_id.removesuffix("-v1") + "-v2",
            choice_costs=costs,
            selection_resolution_bytes=16,
            center_method="event",
            method_intercept_ps=0,
            method_physical_fraction_ppm=0,
            protocol_radius_ppm=(),
        )
        candidate = replace(candidate, protocol_starts=starts(candidate))
        selected = [
            r
            for r in known + fresh
            if r["architecture"] == model.architecture
            and r["width"] == model.width
            and r["lane"] in ("timing", "legacy")
            and r["arm"] in ("auto", "Ring_LL", "Ring_LL128", "Ring_Simple")
        ]
        automatic = [r for r in selected if r["arm"] == "auto"]
        # Pair each capture independently, so the method estimate does not
        # mistake a node-to-node shift for a timing-boundary effect.
        pairs = []
        for source_rows in (known, fresh):
            index = {
                (r["lane"], r["bytes"]): r
                for r in source_rows
                if r["architecture"] == model.architecture
                and r["width"] == model.width
                and r["arm"] == "auto"
                and r["lane"] in ("timing", "legacy")
            }
            for lane, size in sorted(index):
                if lane == "timing":
                    pairs.append(
                        (
                            size,
                            index[("legacy", size)]["median_us"]
                            - index[("timing", size)]["median_us"],
                        )
                    )
        offsets = []
        for protocol in PROTOCOLS:
            rows = [(size, delta) for size, delta in pairs if candidate.protocol(size) == protocol]
            if rows:
                matrix = np.array(
                    [[1, candidate.predict(size).physical_floor_ps / 1e6] for size, _ in rows]
                )
                target = np.array([delta for _, delta in rows])
                fitted, _ = nnls(matrix, target)
                offsets.append((protocol, round(fitted[0] * 1e6), round(fitted[1] * 1e6)))
            else:
                offsets.append((protocol, 0, 0))
        candidate = replace(candidate, protocol_method_costs=tuple(offsets))
        radii = []
        for protocol in PROTOCOLS:
            residuals = []
            spreads = []
            for row in selected:
                selected_protocol = (
                    candidate.protocol(row["bytes"])
                    if row["arm"] == "auto"
                    else row["arm"].split("_", 1)[1].upper()
                )
                if selected_protocol != protocol:
                    continue
                e = candidate.predict(row["bytes"], protocol=protocol)
                target = e.reference_ps if row["lane"] == "timing" else e.central_ps
                residuals.append(abs(row["median_us"] * 1e6 - target) / e.reference_ps)
                spreads.append(row["relative_iqr"])
            radii.append(
                (protocol, int(np.ceil((max(residuals) + float(np.quantile(spreads, 0.9))) * 1e6)))
            )
        candidate = replace(candidate, protocol_radius_ppm=tuple(radii))
        selected_callbacks = [
            r
            for r in callbacks
            if r["architecture"] == model.architecture
            and r["width"] == model.width
            and r["arm"] == "auto"
        ]
        mismatches = [
            r["bytes"] for r in selected_callbacks if candidate.protocol(r["bytes"]) != r["proto"]
        ]
        if mismatches:
            raise ValueError(f"chooser projection mismatch: {mismatches}")
        models.append(candidate)
        for row in automatic:
            e = candidate.predict(row["bytes"])
            target = e.reference_ps if row["lane"] == "timing" else e.central_ps
            predictions.append(
                {
                    "architecture": model.architecture,
                    "width": model.width,
                    "lane": row["lane"],
                    "bytes": row["bytes"],
                    "measured_us": row["median_us"],
                    "model_us": target / 1e6,
                    "reference_us": e.reference_ps / 1e6,
                    "lower_us": e.lower_ps / 1e6,
                    "upper_us": e.upper_ps / 1e6,
                    "signed_error_pct": 100 * (target / (row["median_us"] * 1e6) - 1),
                    "covered": e.lower_ps <= row["median_us"] * 1e6 <= e.upper_ps,
                    "band_width_fraction": (e.upper_ps - e.lower_ps) / (row["median_us"] * 1e6),
                }
            )
        audits.append(
            {
                "architecture": model.architecture,
                "width": model.width,
                "source_reconstruction": source,
                "automatic_selection_cells": len(selected_callbacks),
                "source_selection_mismatches": mismatches,
                "method_cells": len(pairs),
                "calibration_cells": len(selected),
                "work_costs_unchanged": candidate.services == model.services
                and candidate.startup_ps == model.startup_ps,
                "software_choice_costs": costs,
                "method_costs": offsets,
                "protocol_radius_ppm": radii,
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "models.json").write_text(
        json.dumps([m.to_json() for m in models], indent=2) + "\n"
    )
    (args.output / "refinement.json").write_text(
        json.dumps(
            {
                "expectations_commit": "14070e87",
                "base_parameter_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(),
                "chronology": "All comparisons here are development calibration; second-grid validation remains separate.",
                "audits": audits,
            },
            indent=2,
        )
        + "\n"
    )
    write_csv(args.output / "development_predictions.csv", predictions)
    for model in models:
        for lane in ("timing", "legacy"):
            rows = [
                r
                for r in predictions
                if r["architecture"] == model.architecture
                and r["width"] == model.width
                and r["lane"] == lane
            ]
            print(
                model.architecture,
                model.width,
                lane,
                "covered",
                sum(r["covered"] for r in rows),
                len(rows),
                "error",
                max(abs(r["signed_error_pct"]) for r in rows),
                "width",
                max(r["band_width_fraction"] for r in rows),
            )


if __name__ == "__main__":
    main()
