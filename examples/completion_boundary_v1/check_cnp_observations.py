"""Verify frozen external-controller observation controls from native CTest output."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

FREEZE = "d6ba3d37b65f73e26c4364e98c18beebc99e0b61"
POINTS = {(n, b, observed) for n in (4, 8) for b in (65536, 131072) for observed in (0, 1)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(log: str) -> dict:
    """Return a valid comparison or a void record with its first fatal finding."""
    report = {"freeze": FREEZE, "log_sha256": hashlib.sha256(log.encode()).hexdigest(),
              "status": "void", "fatal_findings": [], "comparisons": []}
    try:
        require("100% tests passed, 0 tests failed out of 8" in log, "native matrix gate is incomplete")
        rows = {point: [] for point in POINTS}
        boundaries = {}
        for line in log.splitlines():
            match = re.search(r"BACK71_(FLOW|BOUNDARY) (.*)$", line)
            if not match:
                continue
            pairs = [field.split("=", 1) for field in match[2].split()]
            require(len(dict(pairs)) == len(pairs), "duplicate record key")
            record = {key: int(value) for key, value in pairs}
            point = tuple(record.pop(key) for key in ("senders", "payload", "observations"))
            require(point in POINTS, "unfrozen configuration")
            if match[1] == "FLOW":
                rows[point].append(record)
            else:
                require(point not in boundaries, "duplicate boundary")
                boundaries[point] = record
        require(set(boundaries) == POINTS, "missing configuration")
        for point in sorted(POINTS):
            n, payload, observed = point
            records, boundary = rows[point], boundaries[point]
            require(len(records) == n, "lost or duplicated completion")
            require({r["source"] for r in records} == set(range(n)), "source inventory changed")
            require(len({r["flow"] for r in records}) == n, "native flow identity duplicated")
            for row in records:
                require(row["flow"] == (row["source"] << 32) | (100 + row["source"]),
                        "native flow identity changed")
                require(row["destination"] == 63 and row["bytes"] == payload and row["tag"] == 9,
                        "message inventory changed")
                require(row["start_ps"] == 0 and row["completion_ps"] > 0, "invalid flow timing")
            floor = n * payload * 20 + 2_000_000
            require(boundary["floor_ps"] == floor, "physical floor changed")
            last = max(row["completion_ps"] for row in records)
            require(boundary["last_completion_ps"] == last, "completion boundary changed")
            require(floor <= last <= boundary["quiescence_ps"] <= 1_000_000_000,
                    "physical floor or quiescence budget violated")
            require(0 < boundary["callbacks"] <= 100_000, "callback budget violated")
            if not observed:
                require(boundary["packet_events"] == boundary["control_events"] == 0,
                        "disabled observation path emitted events")
            else:
                require(all(boundary[key] > 0 for key in
                            ("packet_events", "control_events", "ecn", "cnp", "rates",
                             "pfc_frames", "pauses", "resumes")), "physical observation coverage missing")
        comparisons = []
        for n in (4, 8):
            for payload in (65536, 131072):
                off, on = rows[n, payload, 0], rows[n, payload, 1]
                require(off == on, f"completion identity/order/timestamp changed: {n}, {payload}")
                comparisons.append({"senders": n, "payload_bytes": payload,
                                    "off": boundaries[n, payload, 0], "on": boundaries[n, payload, 1],
                                    "completion_rows": off,
                                    "maximum_absolute_completion_difference_ps": 0})
        report.update(status="valid", configurations=8, paired_comparisons=4,
                      exact_completion_row_pairs=sum(len(c["completion_rows"]) for c in comparisons),
                      comparisons=comparisons)
    except (ValueError, KeyError, TypeError) as error:
        report["fatal_findings"].append(str(error))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = check(args.log.read_text())
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "comparisons"}, sort_keys=True))
    return 0 if report["status"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
