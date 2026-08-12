"""Exercise the frozen HTSIM-2 rnic-cn trace-flag expectations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

PROFILE = "rnic-cn"
RANKS = 8
PAYLOAD_BYTES = 262144
LINK_RATES_BPS = (400000000000, 200000000000)
BIN_WIDTHS_PS = (1000000, 500000)
QUEUE_TRACE_MAX_ROWS = 4000000

GOODPUT_CSV_FLAG = "-rnic_cn_goodput_trace_csv"
GOODPUT_BIN_FLAG = "-rnic_cn_goodput_trace_bin_ps"
STATE_CSV_FLAG = "-rnic_cn_state_trace_csv"
QUEUE_CSV_FLAG = "-rnic_cn_queue_trace_csv"
QUEUE_MAX_ROWS_FLAG = "-rnic_cn_queue_trace_max_rows"

MAKESPAN_RATIO_RANGE = (1.5, 2.05)
EFFECTIVE_RATE_FLOOR_SLACK = 3


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _required_executable(environment_name: str) -> Path:
    raw = os.environ.get(environment_name)
    if not raw:
        raise ValueError(f"{environment_name} must name an executable")
    path = Path(raw).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"{environment_name} is not an executable file: {path}")
    return path.resolve()


def _serialization_floor_ps(link_bps: int) -> int:
    return PAYLOAD_BYTES * 8 * 10**12 // link_bps


def _bin_ceiling_bytes(bin_width_ps: int, link_bps: int) -> int:
    return bin_width_ps * link_bps // (8 * 10**12)


def _write_ring_goal(path: Path) -> Path:
    from simllm.goal import GoalTrace

    trace = GoalTrace(RANKS)
    for rank in range(RANKS):
        peer = (rank + 1) % RANKS
        trace.rank(rank).send(PAYLOAD_BYTES, to=peer, tag=0)
        trace.rank(peer).recv(PAYLOAD_BYTES, source=rank, tag=0)
    return trace.write(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _run(
    out: Path,
    name: str,
    goal_bin: Path,
    binary: Path,
    link_bps: int,
    bin_width_ps: int | None,
) -> dict[str, Any]:
    from simllm.backends.htsim_rnic import HtsimRnicConfig, run_htsim_rnic

    # Every run writes the one shared completion path so that the manifest a
    # run prints carries no run-specific option string. F1 compares manifests
    # for the same GOAL and the same options, which a per-run CSV name would
    # have made impossible.
    completion_csv = out / "completion.csv"
    extra: dict[str, str] = {}
    if bin_width_ps is not None:
        extra = {
            GOODPUT_CSV_FLAG: str(out / f"{name}.goodput.csv"),
            GOODPUT_BIN_FLAG: str(bin_width_ps),
            STATE_CSV_FLAG: str(out / f"{name}.state.csv"),
            QUEUE_CSV_FLAG: str(out / f"{name}.queue.csv"),
            QUEUE_MAX_ROWS_FLAG: str(QUEUE_TRACE_MAX_ROWS),
        }
    config = HtsimRnicConfig(
        goal_bin=goal_bin,
        profile=PROFILE,
        linkspeed_bps=link_bps,
        completion_csv=completion_csv,
        extra_flags=extra,
    )
    result = run_htsim_rnic(config, binary=binary, timeout_s=3600)
    record: dict[str, Any] = {
        "bin_width_ps": bin_width_ps,
        "completion_csv_bytes": completion_csv.read_bytes(),
        "completion_rows": _read_csv(completion_csv),
        "link_bps": link_bps,
        "manifest": sorted(result.manifest),
        "name": name,
    }
    completion_csv.replace(out / f"{name}.completion.csv")
    if bin_width_ps is not None:
        record["goodput_rows"] = _read_csv(out / f"{name}.goodput.csv")
        record["state_rows"] = _read_csv(out / f"{name}.state.csv")
        record["queue_rows"] = _read_csv(out / f"{name}.queue.csv")
    return record


def _makespan_ps(record: dict[str, Any]) -> int:
    return max(int(row["completion_time_ps"]) for row in record["completion_rows"])


#: The one observation line a traced run adds. Nothing else may differ.
TRACE_MANIFEST_PREFIX = "[RNIC manifest] rnic_cn_goodput_trace_rows="


def _non_trace_manifest(record: dict[str, Any]) -> list[str]:
    return [line for line in record["manifest"] if not line.startswith(TRACE_MANIFEST_PREFIX)]


def _check_only(args: argparse.Namespace, executables: dict[str, Path]) -> None:
    plan = {
        "artifacts_created": False,
        "bin_widths_ps": list(BIN_WIDTHS_PS),
        "effective_rate_floor_slack": EFFECTIVE_RATE_FLOOR_SLACK,
        "executables": {name: str(path) for name, path in executables.items()},
        "flags": [
            GOODPUT_CSV_FLAG,
            GOODPUT_BIN_FLAG,
            STATE_CSV_FLAG,
            QUEUE_CSV_FLAG,
            QUEUE_MAX_ROWS_FLAG,
        ],
        "goal": {"payload_bytes": PAYLOAD_BYTES, "pattern": "ring", "ranks": RANKS},
        "link_rates_bps": list(LINK_RATES_BPS),
        "makespan_ratio_range": list(MAKESPAN_RATIO_RANGE),
        "out": str(args.out),
        "profile": PROFILE,
        "raw_relation_evaluation": "csv rows parsed before any relation is read",
        "serialization_floor_ps": {
            str(rate): _serialization_floor_ps(rate) for rate in LINK_RATES_BPS
        },
    }
    print(json.dumps(plan, indent=2, sort_keys=True))


def _goodput_totals(record: dict[str, Any]) -> dict[int, int]:
    totals: dict[int, int] = {}
    for row in record["goodput_rows"]:
        flow_id = int(row["flow_id"])
        totals[flow_id] = totals.get(flow_id, 0) + int(row["delivered_payload_bytes"])
    return totals


def _fatal_checks(
    baseline: dict[str, Any],
    plain: dict[str, Any],
    traced: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "baseline_completion_csv_identical": (
            baseline["completion_csv_bytes"] == plain["completion_csv_bytes"]
        ),
        "baseline_manifest_identical": baseline["manifest"] == plain["manifest"],
    }
    for key, record in traced.items():
        label = f"{key[0]}bps_{key[1]}ps"
        checks[f"traced_completion_identical_{label}"] = (
            record["completion_csv_bytes"] == plain["completion_csv_bytes"]
            if record["link_bps"] == plain["link_bps"]
            else True
        )
        checks[f"traced_manifest_identical_{label}"] = (
            _non_trace_manifest(record) == _non_trace_manifest(plain)
            if record["link_bps"] == plain["link_bps"]
            else True
        )
        payloads = {
            int(row["flow_id"]): int(row["payload_bytes"]) for row in record["completion_rows"]
        }
        checks[f"goodput_conserved_{label}"] = _goodput_totals(record) == payloads
        ports: dict[tuple[str, str], dict[str, int]] = {}
        final_buffered: dict[tuple[str, str], int] = {}
        for row in record["queue_rows"]:
            port = (row["switch_id"], row["egress_id"])
            counts = ports.setdefault(port, {"Dequeued": 0, "Dropped": 0, "Enqueued": 0})
            if row["transition"] in counts:
                counts[row["transition"]] += 1
            final_buffered[port] = int(row["egress_buffered_bytes"])
        checks[f"queue_port_conserved_{label}"] = all(
            counts["Enqueued"] == counts["Dequeued"] + counts["Dropped"]
            for counts in ports.values()
        )
        checks[f"queue_port_drained_{label}"] = all(
            buffered == 0 for buffered in final_buffered.values()
        )
        state_flows = {int(row["flow_id"]) for row in record["state_rows"]}
        checks[f"state_covers_flows_{label}"] = state_flows == set(payloads)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    executables = {
        "htsim_rnic": _required_executable("SIMLLM_HTSIM_RNIC"),
        "htsim_rnic_baseline": _required_executable("SIMLLM_HTSIM_RNIC_BASELINE"),
        "txt2bin": _required_executable("SIMLLM_TXT2BIN"),
    }
    if args.check_only:
        _check_only(args, executables)
        return
    if args.out.exists():
        parser.error("--out must not exist")
    args.out.mkdir(parents=True)

    from simllm.goal import to_binary

    goal_bin = to_binary(
        _write_ring_goal(args.out / "ring.goal"),
        args.out / "ring.bin",
        tool=executables["txt2bin"],
    )

    reference_rate = LINK_RATES_BPS[0]
    baseline = _run(
        args.out, "baseline", goal_bin, executables["htsim_rnic_baseline"], reference_rate, None
    )
    plain = _run(args.out, "plain", goal_bin, executables["htsim_rnic"], reference_rate, None)
    traced = {
        (rate, width): _run(
            args.out,
            f"traced-{rate}-{width}",
            goal_bin,
            executables["htsim_rnic"],
            rate,
            width,
        )
        for rate in LINK_RATES_BPS
        for width in BIN_WIDTHS_PS
    }

    fatal = _fatal_checks(baseline, plain, traced)
    fatal_failures = [name for name, passed in fatal.items() if not passed]

    coarse = BIN_WIDTHS_PS[0]
    s1 = []
    for rate in LINK_RATES_BPS:
        record = traced[(rate, coarse)]
        completions = {
            int(row["flow_id"]): int(row["completion_time_ps"])
            for row in record["completion_rows"]
        }
        last_bin: dict[int, tuple[int, int]] = {}
        for row in record["goodput_rows"]:
            flow_id = int(row["flow_id"])
            bounds = (int(row["bin_start_ps"]), int(row["bin_end_ps"]))
            if flow_id not in last_bin or bounds[0] > last_bin[flow_id][0]:
                last_bin[flow_id] = bounds
        for flow_id, completion_ps in sorted(completions.items()):
            start, end = last_bin.get(flow_id, (-1, -1))
            s1.append(
                {
                    "bin_end_ps": end,
                    "bin_start_ps": start,
                    "completion_time_ps": completion_ps,
                    "genuine_risk": True,
                    "instance": f"link={rate} flow={flow_id}",
                    "passed": start <= completion_ps < end,
                }
            )

    s2 = []
    for (rate, width), record in sorted(traced.items()):
        ceiling = _bin_ceiling_bytes(width, rate)
        worst_bytes = max(
            (int(row["delivered_payload_bytes"]) for row in record["goodput_rows"]), default=0
        )
        worst_bps = max((int(row["goodput_bps"]) for row in record["goodput_rows"]), default=0)
        s2.append(
            {
                "ceiling_bytes": ceiling,
                "genuine_risk": True,
                "instance": f"link={rate} bin={width}",
                "passed": worst_bytes <= ceiling and worst_bps <= rate,
                "worst_bin_bytes": worst_bytes,
                "worst_goodput_bps": worst_bps,
            }
        )

    s3 = []
    for rate in LINK_RATES_BPS:
        coarse_record = traced[(rate, BIN_WIDTHS_PS[0])]
        fine_record = traced[(rate, BIN_WIDTHS_PS[1])]
        coarse_starts = {int(row["bin_start_ps"]) for row in coarse_record["goodput_rows"]}
        fine_starts = {int(row["bin_start_ps"]) for row in fine_record["goodput_rows"]}
        s3.append(
            {
                "coarse_bin_starts": len(coarse_starts),
                "fine_bin_starts": len(fine_starts),
                "genuine_risk": True,
                "instance": f"link={rate}",
                "passed": (
                    _goodput_totals(coarse_record) == _goodput_totals(fine_record)
                    and len(fine_starts) >= len(coarse_starts)
                ),
            }
        )

    fast = traced[(LINK_RATES_BPS[0], coarse)]
    slow = traced[(LINK_RATES_BPS[1], coarse)]

    def _enqueued(record: dict[str, Any]) -> int:
        return sum(1 for row in record["queue_rows"] if row["transition"] == "Enqueued")

    def _peak_backlog(record: dict[str, Any]) -> int:
        return max(
            (int(row["egress_backlog_bytes"]) for row in record["queue_rows"]), default=0
        )

    s4 = [
        {
            "fast": _enqueued(fast),
            "genuine_risk": True,
            "instance": "enqueued rows are rate invariant",
            "passed": _enqueued(fast) == _enqueued(slow),
            "slow": _enqueued(slow),
        },
        {
            "fast": _peak_backlog(fast),
            "genuine_risk": True,
            "instance": "peak backlog does not fall when the rate halves",
            "passed": _peak_backlog(slow) >= _peak_backlog(fast),
            "slow": _peak_backlog(slow),
        },
    ]

    def _rate_values(record: dict[str, Any], column: str) -> list[int]:
        return sorted({int(row[column]) for row in record["state_rows"] if row[column]})

    fast_configured = _rate_values(fast, "configured_rate_bps")
    slow_configured = _rate_values(slow, "configured_rate_bps")
    fast_effective = _rate_values(fast, "effective_rate_bps")
    slow_effective = _rate_values(slow, "effective_rate_bps")
    s5 = [
        {
            "fast": fast_configured,
            "genuine_risk": True,
            "instance": "configured rate halves exactly",
            "passed": bool(fast_configured)
            and slow_configured == [value // 2 for value in fast_configured],
            "slow": slow_configured,
        },
        {
            "fast": fast_effective,
            "genuine_risk": True,
            "instance": "effective rate halves within the floor slack",
            "passed": bool(fast_effective)
            and len(fast_effective) == len(slow_effective)
            and all(
                2 * low <= high <= 2 * low + EFFECTIVE_RATE_FLOOR_SLACK
                for low, high in zip(slow_effective, fast_effective, strict=True)
            ),
            "slow": slow_effective,
        },
    ]

    fast_makespan = _makespan_ps(fast)
    slow_makespan = _makespan_ps(slow)
    ratio = slow_makespan / fast_makespan
    s6 = [
        {
            "fast_makespan_ps": fast_makespan,
            "genuine_risk": True,
            "instance": "makespan scaling",
            "passed": (
                MAKESPAN_RATIO_RANGE[0] <= ratio <= MAKESPAN_RATIO_RANGE[1]
                and fast_makespan > _serialization_floor_ps(LINK_RATES_BPS[0])
                and slow_makespan > _serialization_floor_ps(LINK_RATES_BPS[1])
            ),
            "ratio": ratio,
            "slow_makespan_ps": slow_makespan,
        }
    ]

    families = {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5, "S6": s6}
    summary = {
        "authored_against_htsim_commit": "fc4400e4ca619223481536632074045cb6af2756",
        "fatal_unscored": {
            "checks": fatal,
            "failures": fatal_failures,
            "valid": not fatal_failures,
        },
        "observed_simllm_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "schema": "simllm-rnic-cn-trace-study-v1",
        "scored": {
            name: {
                "genuine_risk_passed": sum(row["passed"] for row in rows),
                "genuine_risk_total": len(rows),
                "instances": rows,
            }
            for name, rows in families.items()
        },
    }
    (args.out / "summary.json").write_bytes(_canonical(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    scored_rows = [row for rows in families.values() for row in rows]
    if fatal_failures or not all(row["passed"] for row in scored_rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
