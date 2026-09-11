"""Calibrate frozen protocol work costs and evaluate withheld dense payloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simllm.traffic.collective_protocol import (
    MIB,
    PROTOCOLS,
    NcclProtocolService,
    NcclRingProtocolModel,
    ring_geometry,
)


def read_rows(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("width", "bytes", "repeat", "#channels", "#warps"):
            if key in row:
                row[key] = int(row[key])
        for key in ("median_us", "q1_us", "q3_us", "relative_iqr"):
            if key in row:
                row[key] = float(row[key])
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def is_anchor(row, config):
    return (row["bytes"] - config["known_anchor_start_bytes"]) % config[
        "automatic_fit_stride_bytes"
    ] == 0


def selection_starts(tuning, arch, width):
    auto = sorted(
        (
            r
            for r in tuning
            if r["architecture"] == arch and r["width"] == width and r["arm"] == "auto"
        ),
        key=lambda r: r["bytes"],
    )
    result = []
    for row in auto:
        if not result or row["proto"] != result[-1][1]:
            result.append((row["bytes"], row["proto"]))
    return tuple(result)


def geometry_audit(tuning):
    disagreements, checked = [], 0
    for row in tuning:
        if row["algo"] != "RING":
            continue
        arm = row["arm"]
        fixed = int(arm.rsplit("cta", 1)[1]) if "_cta" in arm else None
        geometry = ring_geometry(row["bytes"], row["width"], row["proto"], fixed_channels=fixed)
        checked += 1
        if (geometry.channels, geometry.warps) != (row["#channels"], row["#warps"]):
            disagreements.append(
                {
                    "architecture": row["architecture"],
                    "width": row["width"],
                    "arm": arm,
                    "bytes": row["bytes"],
                    "expected": [row["#channels"], row["#warps"]],
                    "modeled": [geometry.channels, geometry.warps],
                }
            )
    return {
        "checked_callback_cells": checked,
        "disagreements": disagreements,
        "status": "valid" if not disagreements else "source_projection_mismatch",
    }


def fit_model(arch, width, config, points, tuning, floors):
    blank = NcclRingProtocolModel(
        model_id=f"{arch}-nccl-2.31-ring-w{width}-v1",
        architecture=arch,
        width=width,
        payload_min_bytes=config["payload_min_bytes"],
        payload_max_bytes=config["payload_max_bytes"],
        startup_ps=round(floors[(arch, width)] * 1e6),
        peer_rate_bytes_per_second=(100 if arch == "a100" else 150) * 10**9,
        endpoint_rate_bytes_per_second=(300 if arch == "a100" else 450) * 10**9,
        protocol_starts=selection_starts(tuning, arch, width),
        selection_resolution_bytes=config["protocol_selection_resolution_bytes"],
        services=tuple(NcclProtocolService(p, 0, 0, 0, 0) for p in PROTOCOLS),
    )
    selected = [r for r in points if r["architecture"] == arch and r["width"] == width]
    calibration = [
        r
        for r in selected
        if r["lane"] == "timing"
        and (
            r["arm"] in config["fixed_protocol_arms"] or r["arm"] == "auto" and is_anchor(r, config)
        )
    ]
    inherited = [
        r for r in selected if r["lane"] == "legacy" and r["arm"] == "auto" and is_anchor(r, config)
    ]
    services, fit_details = [], []
    for protocol in PROTOCOLS:
        rows = [
            r
            for r in calibration
            if (
                blank.protocol(r["bytes"])
                if r["arm"] == "auto"
                else r["arm"].split("_", 1)[1].upper()
            )
            == protocol
        ]
        x, y, physical = [], [], []
        for row in rows:
            estimate = blank.predict(row["bytes"], protocol=protocol)
            geometry = estimate.geometry
            x.append(
                [
                    geometry.maximum_channel_encoded_bytes / MIB,
                    geometry.instruction_rounds,
                    geometry.synchronization_slices,
                    geometry.nonempty_publications,
                ]
            )
            y.append(row["median_us"] - blank.startup_ps / 1e6)
            physical.append(estimate.physical_floor_ps / 1e6)
        matrix = np.array(x)
        identify_startup = protocol == "LL"
        if identify_startup:
            matrix = np.column_stack((np.ones(len(rows)), matrix))
        active = np.zeros(matrix.shape[1], dtype=bool)
        for column in range(matrix.shape[1]):
            candidate = active.copy()
            candidate[column] = True
            if np.linalg.matrix_rank(matrix[:, candidate]) > int(active.sum()):
                active = candidate
        floor = np.array(physical)
        target = np.array(y) + (blank.startup_ps / 1e6 if identify_startup else 0)
        data_start = int(identify_startup)

        def residual(
            values,
            matrix=matrix,
            active=active,
            identify_startup=identify_startup,
            data_start=data_start,
            floor=floor,
            target=target,
            rows=rows,
        ):
            coefficients = np.zeros(matrix.shape[1])
            coefficients[active] = values
            startup = coefficients[0] if identify_startup else 0
            work = (
                matrix[:, data_start : data_start + 2] @ coefficients[data_start : data_start + 2]
            )
            synchronization = matrix[:, data_start + 2 :] @ coefficients[data_start + 2 :]
            return (startup + np.maximum(floor, work) + synchronization - target) / np.array(
                [r["median_us"] for r in rows]
            )

        seeds = [np.full(int(active.sum()), 1e-6)]
        for index, column in enumerate(np.flatnonzero(active)):
            seed = np.full(int(active.sum()), 1e-6)
            seed[index] = max(1e-6, float(np.median(target) / np.median(matrix[:, column])))
            seeds.append(seed)
        fits = [
            least_squares(
                residual,
                seed,
                bounds=(0, np.inf),
                max_nfev=3000,
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-11,
            )
            for seed in seeds
        ]
        best = min(fits, key=lambda fit: float(fit.fun @ fit.fun))
        fitted = np.zeros(matrix.shape[1])
        fitted[active] = best.x
        norm = np.linalg.norm(best.fun)
        if identify_startup:
            blank = replace(blank, startup_ps=round(fitted[0] * 1e6))
        cost = NcclProtocolService(protocol, *(round(v * 1e6) for v in fitted[data_start:]))
        services.append(cost)
        fit_details.append(
            {
                "protocol": protocol,
                "calibration_cells": len(rows),
                "active_columns": int(active.sum()),
                "matrix_rank": int(np.linalg.matrix_rank(matrix[:, active])),
                "fractional_residual_l2": float(norm),
                "costs": asdict(cost),
            }
        )
    model = replace(blank, services=tuple(services))
    reference = {r["bytes"]: r for r in calibration if r["arm"] == "auto"}
    mx, my = [], []
    for row in inherited:
        mx.append([1, model.predict(row["bytes"]).physical_floor_ps / 1e6])
        my.append(row["median_us"] - reference[row["bytes"]]["median_us"])
    allowance, _ = nnls(np.array(mx), np.array(my))
    model = replace(
        model,
        method_intercept_ps=round(allowance[0] * 1e6),
        method_physical_fraction_ppm=round(allowance[1] * 1e6),
    )
    residuals, spreads, manifest = [], [], []
    grouped_residuals = {p: [] for p in PROTOCOLS}
    grouped_spreads = {p: [] for p in PROTOCOLS}
    for row in calibration + inherited:
        protocol = None if row["arm"] == "auto" else row["arm"].split("_", 1)[1].upper()
        estimate = model.predict(row["bytes"], protocol=protocol)
        prediction = estimate.reference_ps
        if row["lane"] == "legacy":
            prediction += estimate.method_allowance_ps
        residuals.append(abs(row["median_us"] * 1e6 - prediction) / estimate.reference_ps)
        spreads.append(row["relative_iqr"])
        grouped_residuals[estimate.geometry.protocol].append(residuals[-1])
        grouped_spreads[estimate.geometry.protocol].append(spreads[-1])
        manifest.append(
            {
                k: row[k]
                for k in (
                    "architecture",
                    "width",
                    "lane",
                    "arm",
                    "bytes",
                    "median_us",
                    "relative_iqr",
                )
            }
        )
    radii = tuple(
        (
            p,
            int(
                np.ceil(
                    (max(grouped_residuals[p]) + float(np.quantile(grouped_spreads[p], 0.9))) * 1e6
                )
            ),
        )
        for p in PROTOCOLS
    )
    model = replace(model, protocol_radius_ppm=radii)
    return model, {
        "architecture": arch,
        "width": width,
        "parameter_fits": fit_details,
        "maximum_calibration_fractional_residual": max(residuals),
        "p90_calibration_relative_iqr": float(np.quantile(spreads, 0.9)),
        "calibration_fingerprint": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "calibration_cells": manifest,
    }


def evaluate(models, points, config):
    rows, summaries = [], []
    for model in models:
        selected = [
            r
            for r in points
            if r["architecture"] == model.architecture
            and r["width"] == model.width
            and r["arm"] == "auto"
            and r["lane"] in ("timing", "legacy")
        ]
        for point in selected:
            prediction = model.predict(point["bytes"])
            measured = point["median_us"] * 1e6
            rows.append(
                {
                    "architecture": model.architecture,
                    "width": model.width,
                    "lane": point["lane"],
                    "bytes": point["bytes"],
                    "partition": "calibration" if is_anchor(point, config) else "withheld",
                    "protocol": prediction.geometry.protocol,
                    "channels": prediction.geometry.channels,
                    "measured_us": point["median_us"],
                    "q1_us": point["q1_us"],
                    "q3_us": point["q3_us"],
                    "physical_floor_us": prediction.physical_floor_ps / 1e6,
                    "reference_us": prediction.reference_ps / 1e6,
                    "model_us": prediction.central_ps / 1e6,
                    "lower_us": prediction.lower_ps / 1e6,
                    "upper_us": prediction.upper_ps / 1e6,
                    "signed_error_pct": 100 * (prediction.central_ps / measured - 1),
                    "covered": prediction.lower_ps <= measured <= prediction.upper_ps,
                    "band_width_fraction": (prediction.upper_ps - prediction.lower_ps) / measured,
                    "encoded_endpoint_floor_bytes": prediction.geometry.encoded_endpoint_floor_bytes,
                    "inline_flag_floor_bytes": prediction.geometry.inline_flag_floor_bytes,
                    "counter_store_bytes": prediction.geometry.counter_store_bytes_per_rank,
                    "instruction_rounds": prediction.geometry.instruction_rounds,
                    "nonempty_publications": prediction.geometry.nonempty_publications,
                    "composite_publication_us": prediction.composite_publication_ps / 1e6,
                }
            )
        for lane in ("timing", "legacy"):
            for partition in ("all", "withheld"):
                subset = [
                    r
                    for r in rows
                    if r["architecture"] == model.architecture
                    and r["width"] == model.width
                    and r["lane"] == lane
                    and (partition == "all" or r["partition"] == partition)
                ]
                worst = max(subset, key=lambda r: abs(r["signed_error_pct"]))
                summaries.append(
                    {
                        "architecture": model.architecture,
                        "width": model.width,
                        "lane": lane,
                        "partition": partition,
                        "points": len(subset),
                        "covered": sum(r["covered"] for r in subset),
                        "worst_center_error_pct": worst["signed_error_pct"],
                        "worst_payload_bytes": worst["bytes"],
                        "maximum_band_width_fraction": max(
                            r["band_width_fraction"] for r in subset
                        ),
                    }
                )
    return rows, summaries


def physical_checks(models):
    rows = []
    for model in models:
        for size in (524288, 2097152):
            baseline = model.predict(size)
            for numerator, denominator in ((1, 2), (1, 1), (2, 1)):
                changed = replace(
                    model,
                    peer_rate_bytes_per_second=model.peer_rate_bytes_per_second
                    * numerator
                    // denominator,
                    endpoint_rate_bytes_per_second=model.endpoint_rate_bytes_per_second
                    * numerator
                    // denominator,
                ).predict(size)
                exact = abs(
                    changed.physical_floor_ps * numerator - baseline.physical_floor_ps * denominator
                ) <= max(numerator, denominator)
                assert exact and changed.lower_ps >= changed.physical_floor_ps
                rows.append(
                    {
                        "architecture": model.architecture,
                        "width": model.width,
                        "bytes": size,
                        "rate_scale": numerator / denominator,
                        "physical_floor_ps": changed.physical_floor_ps,
                        "central_ps": changed.central_ps,
                    }
                )
            for rtt in (0, 250000, 1000000):
                changed = replace(model, visibility_rtt_ps=rtt).predict(size)
                assert changed.attributed_rtt_ps == changed.geometry.nonempty_publications * rtt
                assert changed.residual_poll_fence_ps >= 0
                expected = baseline.reference_ps + max(
                    0, changed.attributed_rtt_ps - baseline.composite_publication_ps
                )
                assert changed.reference_ps == expected
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    config = json.loads((Path(__file__).parent / "expectations.json").read_text())
    points = read_rows(args.data / "timing_summary.csv")
    tuning = read_rows(args.data / "tuning.csv")
    old = json.loads(
        (args.repo / "examples/collective_regime_curve_v1/validation.json").read_text()
    )
    floors = {(r["machine"], r["width"]): r["floor_us"] for r in old["curves"]}
    audit = geometry_audit(tuning)
    if audit["disagreements"]:
        raise ValueError(f"source projection disagreement: {audit['disagreements'][:5]}")
    models, calibration = [], []
    for arch in config["architectures"]:
        for width in config["widths"]:
            model, detail = fit_model(arch, width, config, points, tuning, floors)
            models.append(model)
            calibration.append(detail)
    rows, summaries = evaluate(models, points, config)
    physics = physical_checks(models)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "simllm-nccl-protocol-study-v1",
        "expectations_commit": "0c64d2c2",
        "chronology": "Retrospective calibration and withheld regression; existing timing captures predate this model.",
        "source_geometry": audit,
        "calibration": calibration,
        "dense_comparison": summaries,
        "uncovered_points": [r for r in rows if not r["covered"]],
        "integrity_status": "valid",
        "physical_sweep": physics,
    }
    (args.output / "models.json").write_text(
        json.dumps([m.to_json() for m in models], indent=2) + "\n"
    )
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(args.output / "predictions.csv", rows)
    print(
        json.dumps(
            {"geometry_cells": audit["checked_callback_cells"], "summaries": summaries}, indent=2
        )
    )


if __name__ == "__main__":
    main()
