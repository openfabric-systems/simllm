"""Run the frozen single-switch TM1 packet study through its native binary.

Raw cases, commands, binary/source hashes and outputs stay in --out. The
study is component evidence, independent of GPU metrics or RDMA RC transport.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
FREEZE = "33a44d6d"
AMENDMENT = "3d75af59"
POLICIES = ("quota_dd", "quota_prbs", "prbs", "unpaced")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def catalog() -> list[dict[str, str | int | float]]:
    """Frozen core sweep plus isolated topology and accounting sensitivities."""
    cases: list[dict[str, str | int | float]] = []
    core = itertools.product(
        POLICIES, ("4x25", "1x100"), (1500, 9000), (8, 16, 32, 64),
        (8, 64), (0.90, 0.95), (1, 7, 31),
    )
    for policy, profile, mtu, window, tick, fraction, seed in core:
        cases.append({
            "family": "core", "policy": policy, "source": profile, "receiver": profile,
            "mtu": mtu, "window_us": window, "tick_ns": tick, "fraction": fraction, "seed": seed,
        })
    for policy, mtu, routing, seed in itertools.product(
        POLICIES, (1500, 9000), ("collision", "hash"), (1, 7, 31),
    ):
        cases.append({
            "family": "routing", "policy": policy, "source": "4x25", "receiver": "4x25",
            "mtu": mtu, "routing": routing, "seed": seed,
        })
    for policy, mtu, overhead, mask, unavailable, admission in itertools.product(
        ("quota_dd", "prbs", "unpaced"), (1500, 9000), (0, 64), (1, 3),
        (0, 4096), ("static", "dynamic"),
    ):
        cases.append({
            "family": "accounting", "policy": policy, "source": "4x25", "receiver": "4x25",
            "mtu": mtu, "internal_overhead": overhead, "xpe_mask": mask,
            "unavailable_cells": unavailable, "admission": admission,
        })
    for policy, profile, mtu in itertools.product(
        ("quota_dd", "prbs"), ("4x25", "1x100"), (1500, 9000),
    ):
        cases.append({
            "family": "rate", "policy": policy, "source": profile, "receiver": profile,
            "mtu": mtu, "rate_scale": 0.5,
        })
    return cases


def run_case(binary: Path, out: Path, index: int, case: dict) -> dict[str, str]:
    command = [str(binary)]
    for name, value in case.items():
        if name != "family":
            command.extend(["--" + name.replace("_", "-"), str(value)])
    result = subprocess.run(command, text=True, capture_output=True, timeout=120, check=False)
    stem = out / f"case-{index:04d}"
    stem.with_suffix(".csv").write_text(result.stdout)
    stem.with_suffix(".stderr").write_text(result.stderr)
    stem.with_suffix(".json").write_text(json.dumps(
        {"command": command, "case": case, "returncode": result.returncode}, indent=2,
    ) + "\n")
    if result.returncode:
        raise RuntimeError(f"case {index} failed: {result.stderr.strip()}")
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    if len(rows) != 1:
        raise RuntimeError(f"case {index}: expected exactly one native summary row")
    return dict(case_id=str(index), family=case["family"], **rows[0])


def guards(row: dict[str, str]) -> list[str]:
    """Fatal invariants stay separate from scored behavioral comparisons."""
    errors = []
    for field in ("conservation", "allocation_constraints"):
        if row[field] != "1":
            errors.append(field)
    if int(row["pending_packets"]):
        errors.append("pending packets after drain")
    if int(row["generated_payload_bytes"]) != 4 * int(row["payload_per_sender"]):
        errors.append("requested payload generation")
    payload = int(row["payload_per_sender"])
    packets_per_sender = (payload + int(row["mtu"]) - 65) // (int(row["mtu"]) - 64)
    if int(row["generated_wire_bytes"]) != 4 * (payload + 102 * packets_per_sender):
        errors.append("independent packetization wire oracle")
    if int(row["generated_packets"]) != (
        int(row["delivered_packets"]) + int(row["dropped_packets"])
    ):
        errors.append("packet conservation")
    if int(row["generated_payload_bytes"]) != (
        int(row["delivered_payload_bytes"]) + int(row["dropped_payload_bytes"])
    ):
        errors.append("payload conservation")
    if int(row["completion_ps"]) < int(row["receiver_floor_ps"]):
        errors.append("receiver serialization floor")
    if row["admission"] == "static" and int(row["data_max_cells"]) > 1260:
        errors.append("static DATA cell cap")
    if int(row["xpe_max_cells"]) > 20165 - int(row["unavailable_cells"]):
        errors.append("XPE advertised capacity")
    if row["policy"].startswith("quota"):
        if row["complete_payload"] != "1" or int(row["dropped_packets"]):
            errors.append("calendar complete delivery")
        if int(row.get("quota_overspend_bytes", "0")):
            errors.append("quota ledger overspend")
    return errors


def analytic_checks(rows: list[dict[str, str]]) -> list[dict]:
    """Unscored geometry checks, with loaded timing as a separate observation."""
    checks = []
    fixed_forward_ps = 650000
    for half in (r for r in rows if r["family"] == "rate"):
        full = next(r for r in rows if r["family"] == "core"
                    and r["policy"] == half["policy"] and r["source"] == half["source"]
                    and r["mtu"] == half["mtu"] and r["window_us"] == "32"
                    and r["tick_ns"] == "8" and r["seed"] == "1"
                    and float(r["fraction"]) == 0.95)
        full_serial = int(full["receiver_floor_ps"]) - fixed_forward_ps
        half_serial = int(half["receiver_floor_ps"]) - fixed_forward_ps
        checks.append({
            "family": "rate scaling of pure wire floor", "case": half["case_id"],
            "observed_ratio": half_serial / full_serial,
            "expected": "exactly 2; fixed forward delay remains 650 ns",
            "pass": half_serial == 2 * full_serial,
            "loaded_makespan_ratio_diagnostic": (
                int(half["completion_ps"]) / int(full["completion_ps"])
            ),
        })
    for profile in ("4x25", "1x100"):
        group = [r for r in rows if r["family"] == "core"
                 and r["source"] == profile and r["policy"] == "quota_dd"
                 and r["window_us"] == "32" and r["tick_ns"] == "8"
                 and r["seed"] == "1" and float(r["fraction"]) == 0.95]
        small = next(r for r in group if r["mtu"] == "1500")
        jumbo = next(r for r in group if r["mtu"] == "9000")
        checks.append({
            "family": "MTU wire cost", "ports": profile,
            "small_wire_bytes": int(small["generated_wire_bytes"]),
            "jumbo_wire_bytes": int(jumbo["generated_wire_bytes"]),
            "expected": "jumbo lowers wire overhead at identical requested payload",
            "pass": int(jumbo["generated_wire_bytes"]) < int(small["generated_wire_bytes"]),
        })
    for collision in (r for r in rows if r["family"] == "routing"
                      and r["routing"] == "collision" and r["policy"].startswith("quota")):
        checks.append({
            "family": "one collided lane capacity", "case": collision["case_id"],
            "observed_wire_bps": int(collision["allocated_wire_bps"]),
            "expected": "23.75 Gb/s allocated on the only addressed 25G egress",
            "pass": int(collision["allocated_wire_bps"]) == 23750000000,
        })
    return checks


def summarize(rows: list[dict[str, str]]) -> str:
    text = [
        "# Single-switch TM1 simulation results", "",
        f"Frozen expectations: {FREEZE}; pre-run clarification: {AMENDMENT}.",
        "Packet-level synthetic UDP component evidence; no hardware calibration,",
        "distributed control deployment, RC/DCQCN performance or GPU metrics.", "",
        f"Executed {len(rows)} native configurations. Fatal guards are separate",
        "from behavioral comparisons. Raw CSV and manifests stay outside Git.", "",
        "## Candidate configurations", "",
        "Four senders each provide 1 MiB to one receiver. Balanced explicit lanes,",
        "95 percent wire allocation, W=32 us, tick=8 ns, seed=1, static DATA cap.", "",
        "| Ports | IP MTU | Policy | Makespan us | Payload Gb/s | Max DATA cells | Drops |",
        "| :-- | --: | :-- | --: | --: | --: | --: |",
    ]
    for row in rows:
        if (row["family"] == "core" and row["window_us"] == "32"
                and row["tick_ns"] == "8" and float(row["fraction"]) == 0.95
                and row["seed"] == "1"):
            elapsed = int(row["completion_ps"])
            goodput = int(row["delivered_payload_bytes"]) * 8000 / elapsed
            text.append(
                f"| {row['source']} | {row['mtu']} | {row['policy']} | "
                f"{elapsed / 1e6:.6f} | {goodput:.4f} | {row['data_max_cells']} | "
                f"{row['dropped_packets']} |"
            )
    text.extend([
        "", "Dropped-run makespans describe the last delivered packet and are not",
        "completion times for the original workload. Their goodput alone cannot",
        "rank a policy against a loss-free calendar.", "",
        "## Window and tick tradeoff", "",
        "All three seeds are included in these ranges, using four 25G lanes and",
        "95 percent wire allocation. Lower completion is preferable only if all",
        "bytes arrive and the proposed queue cap remains respected.", "",
        "| MTU | Policy | W us | Tick ns | Makespan range us | Max cells | Max drops |",
        "| --: | :-- | --: | --: | :-- | --: | --: |",
    ])
    for mtu, policy, window, tick in itertools.product(
        (1500, 9000), ("quota_dd", "quota_prbs", "prbs"), (8, 16, 32, 64), (8, 64),
    ):
        group = [r for r in rows if r["family"] == "core"
                 and r["source"] == "4x25" and int(r["mtu"]) == mtu
                 and r["policy"] == policy and int(r["window_us"]) == window
                 and int(r["tick_ns"]) == tick and float(r["fraction"]) == 0.95]
        times = [int(r["completion_ps"]) / 1e6 for r in group]
        text.append(
            f"| {mtu} | {policy} | {window} | {tick} | {min(times):.6f} to "
            f"{max(times):.6f} | {max(int(r['data_max_cells']) for r in group)} | "
            f"{max(int(r['dropped_packets']) for r in group)} |"
        )
    text.extend([
        "", "## Routing and accounting", "",
        "| Family | Configurations | Dropped configurations | Largest DATA cells |",
        "| :-- | --: | --: | --: |",
    ])
    for family in ("core", "routing", "accounting", "rate"):
        group = [r for r in rows if r["family"] == family]
        text.append(f"| {family} | {len(group)} | "
                    f"{sum(int(r['dropped_packets']) > 0 for r in group)} | "
                    f"{max(int(r['data_max_cells']) for r in group)} |")
    text.extend([
        "", "XPE masks and internal overhead are uncalibrated sensitivities.",
        "Dynamic alpha admission has no 256 KiB static cap in this profile.",
        "Buffered residence includes processing eligibility, while queue wait",
        "starts at processing-ready eligibility. Packet",
        "latency starts at granted/lottery emission and excludes earlier",
        "application waiting, declaration setup and receiver resequencing.", "",
    ])
    return "\n".join(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend-root", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    cases = catalog()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(
            lambda pair: run_case(binary, args.out, pair[0], pair[1]), enumerate(cases),
        ))
    with (args.out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failures = [{"case": row["case_id"], "errors": guards(row)} for row in rows if guards(row)]
    geometry_checks = analytic_checks(rows)
    geometry_failures = [check for check in geometry_checks if not check["pass"]]
    manifest = {
        "schema": "asu-tm1-component-study-v1", "freeze": FREEZE, "amendment": AMENDMENT,
        "binary_sha256": sha256(binary), "runner_sha256": sha256(Path(__file__)),
        "expectation_sha256": sha256(HERE / "expectations.md"),
        "amendment_sha256": sha256(HERE / "expectations_amendment.md"),
        "configurations": len(cases), "fatal_guard_failures": failures,
        "unscored_analytic_families": sorted({c["family"] for c in geometry_checks}),
        "unscored_analytic_instances": len(geometry_checks),
        "analytic_failures": geometry_failures,
        "status": "void" if failures else "valid",
        "evidence": "synthetic UDP packet-level component, oracle preinstalled calendar",
    }
    if args.backend_root is not None:
        source_root = args.backend_root.resolve(strict=True)
        sources = sorted((source_root / "htsim/sim/datacenter").glob("ns_tm1*"))
        sources.append(source_root / "htsim/sim/datacenter/CMakeLists.txt")
        manifest["native_source_sha256"] = {
            str(path.relative_to(source_root)): sha256(path) for path in sources if path.is_file()
        }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "analytic_checks.json").write_text(
        json.dumps(geometry_checks, indent=2) + "\n"
    )
    (args.out / "RESULTS.md").write_text(summarize(rows))
    print(json.dumps({"status": manifest["status"], "configurations": len(cases),
                          "fatal_guard_failures": len(failures)}, sort_keys=True))
    if failures:
        raise SystemExit("Study void: inspect manifest.json before interpreting comparisons")
    if geometry_failures:
        raise SystemExit("Analytic consistency failed: inspect analytic_checks.json")


if __name__ == "__main__":
    main()
