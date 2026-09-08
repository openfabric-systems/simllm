"""Publish the compact arrival-study evidence without changing its interpretation."""

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
    if result.get("schema") != "pp-arrival-regimes-v1" or result.get("artifact_kind") is not None:
        raise ValueError("expected a complete pp-arrival-regimes-v1 result")
    result.update(artifact_kind="compact-public-projection", bulk_results_sha256=sha256(raw))
    result["projection_scope"] = {
        "unchanged": "verdict, configurations, metrics, exact controls and behavioral relations",
        "trace_retained": "counts, queue work, storage peaks, completion witnesses and all trigger attempts",
        "trace_external": "complete packet inventories, switch visits and per-egress diagnostics",
        "audit_digest": "SHA-256 of the complete per-cell trace_audit.json bytes",
        "prior_evidence": "hashes of the predecessor full result and twelve retained full trace audits",
    }
    for row in result["physical_configurations"]:
        audit = row.pop("trace_audit")
        if audit is None:
            row["audit_projection"] = None
            continue
        path = source.parent / row["cell"] / "trace-on/trace_audit.json"
        audit_raw = path.read_bytes()
        if json.loads(audit_raw) != audit:
            raise ValueError(f"complete result and retained audit disagree: {row['cell']}")
        row["trace_audit_sha256"] = sha256(audit_raw)
        audit["schema"] = "pp-arrival-audit-public-projection-v1"
        audit.pop("egresses")
        for flow in audit["pp_flows"]:
            packets = flow.pop("packets")
            indexed = {packet["lifecycle_id"]: packet for packet in packets}
            if len(indexed) != len(packets):
                raise ValueError("complete audit contains duplicate packet lifecycles")
            completion = indexed[flow["completion_packet_lifecycle_id"]]
            trigger = indexed[flow["completion_trigger_lifecycle_id"]]
            flow["completion_packet"] = {key: value for key, value in completion.items() if key != "visits"}
            flow["delivery_trigger_attempts"] = [
                {key: value for key, value in packet.items() if key != "visits"}
                for packet in packets if packet["packet_index"] == trigger["packet_index"]]
            flow["physical_packet_count"] = len(packets)
            flow["packet_outcomes"] = {
                field: dict(sorted(Counter(str(packet[field]) for packet in packets).items()))
                for field in ("attempt", "terminal", "admission")}
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
