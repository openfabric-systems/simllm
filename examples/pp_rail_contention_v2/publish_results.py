"""Publish a compact projection of an already completed pipeline queue study.

The original result and full per-cell trace audits remain in external storage.
This helper preserves their verdict and measured relations without rescoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def project(source: Path):
    raw = source.read_bytes()
    result = json.loads(raw)
    if result.get("schema") != "pp-queue-attribution-v1":
        raise ValueError("expected the complete pp-queue-attribution-v1 result")
    if result.get("artifact_kind") is not None:
        raise ValueError("input must be the complete result, not a prior projection")
    result["artifact_kind"] = "compact-public-projection"
    result["bulk_results_sha256"] = sha256(raw)
    result["projection_scope"] = {
        "unchanged": "verdict, configurations, metrics, exact oracles, relations and diagnostics",
        "trace_retained": "counts, queue work, receiver storage peaks, completion witnesses and trigger attempts",
        "trace_external": "complete packet inventories and switch visits, per-egress diagnostics",
        "audit_digest": "SHA-256 of the complete per-cell trace_audit.json bytes",
    }
    for row in result["physical_configurations"]:
        audit = row.pop("trace_audit")
        if audit is None:
            row["audit_projection"] = None
            continue
        audit_file = source.parent / row["cell"] / "trace-on" / "trace_audit.json"
        audit_raw = audit_file.read_bytes()
        if json.loads(audit_raw) != audit:
            raise ValueError(f"full result and retained audit disagree: {row['cell']}")
        row["trace_audit_sha256"] = sha256(audit_raw)
        audit["schema"] = "pp-queue-audit-public-projection-v1"
        audit.pop("egresses")
        for flow in audit["pp_flows"]:
            packets = flow.pop("packets")
            final = next(packet for packet in packets
                         if packet["lifecycle_id"] == flow["completion_packet_lifecycle_id"])
            trigger = next(packet for packet in packets
                           if packet["lifecycle_id"] == flow["completion_trigger_lifecycle_id"])
            flow["completion_packet"] = {key: value for key, value in final.items() if key != "visits"}
            flow["delivery_trigger_attempts"] = [
                {key: value for key, value in packet.items() if key != "visits"}
                for packet in packets if packet["packet_index"] == trigger["packet_index"]
            ]
            flow["physical_packet_count"] = len(packets)
            flow["packet_outcomes"] = {
                field: dict(sorted(Counter(str(packet[field]) for packet in packets).items()))
                for field in ("attempt", "terminal", "admission")
            }
        row["audit_projection"] = audit
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = project(args.source)
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Published {result['verdict']} without changing acceptance")


if __name__ == "__main__":
    main()
