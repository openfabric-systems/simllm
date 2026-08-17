#!/usr/bin/env python3
"""Package the captured series into the byte-locked reference dataset.

Copies, per captured cell, the destination and source per-chunk series
(gzipped deterministically with mtime 0), the cell summary, and the
per-rank metadata into dataset/<cell>/, then writes MANIFEST.json with the
SHA-256 and byte size of every packaged file. The manifest hash list is
what RESULTS.md cites as the calibration reference.

Deterministic: same inputs give byte-identical outputs.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import shutil

CELL_DIRS = {
    "s1-stream": "s1-stream",
    "i2-incast": "i2-incast",
    "i3-incast": "i3-incast",
    "j3-join": "j3-join",
    "j2x-join": "j2x-join",
    "i4-incast": "i4-incast",
    "x4-incast": "x4-incast",
    "mx-gh2a": "mx-pair",
    "mx-a2gh": "mx-pair",
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gzip_deterministic(src, dst):
    with open(src, "rb") as f:
        data = f.read()
    with open(dst, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(data)


def parse_counters(path):
    """tag -> {iface: (rx, tx)} from a byte_counters.txt."""
    with open(path) as f:
        text = f.read()
    snaps = {}
    for m in re.finditer(
            r"epoch_ns=\d+ tag=(\S+)\n"
            r"((?:iface=\S+ rx_bytes=\d+ tx_bytes=\d+\n)+)", text):
        snaps[m.group(1)] = {
            i: (int(rx), int(tx))
            for i, rx, tx in re.findall(
                r"iface=(\S+) rx_bytes=(\d+) tx_bytes=(\d+)", m.group(2))
        }
    return snaps


def port_tx_deltas_homogeneous(job_dir):
    """Per-node per-iface before-to-after deltas for a homogeneous cell."""
    out = {}
    for cf in sorted(glob.glob(os.path.join(job_dir, "node_*",
                                            "byte_counters.txt"))):
        node = os.path.basename(os.path.dirname(cf))
        snaps = parse_counters(cf)
        if "before" in snaps and "after" in snaps:
            b, a = snaps["before"], snaps["after"]
            out[node] = {
                i: {"tx_delta_bytes": a[i][1] - b[i][1],
                    "rx_delta_bytes": a[i][0] - b[i][0]}
                for i in sorted(b) if i in a
            }
    return out


def port_tx_deltas_mx(job_dir, phase):
    """Per-side per-iface deltas around one mx phase from side.txt files."""
    out = {}
    for sf in sorted(glob.glob(os.path.join(job_dir, "*", "side.txt"))):
        side = os.path.basename(os.path.dirname(sf))
        with open(sf) as f:
            text = f.read()
        snaps = {}
        for m in re.finditer(
                r"counters tag=(\S+) epoch_ns=\d+\n"
                r"((?:iface=\S+ rx_bytes=\d+ tx_bytes=\d+\n)+)", text):
            snaps[m.group(1)] = {
                i: (int(rx), int(tx))
                for i, rx, tx in re.findall(
                    r"iface=(\S+) rx_bytes=(\d+) tx_bytes=(\d+)",
                    m.group(2))
            }
        b, a = snaps.get(f"before_{phase}"), snaps.get(f"after_{phase}")
        if b and a:
            out[side] = {
                i: {"tx_delta_bytes": a[i][1] - b[i][1],
                    "rx_delta_bytes": a[i][0] - b[i][0]}
                for i in sorted(b) if i in a
            }
    return out


def copy_guard_evidence(job_dir, out_dir, cell):
    """Copy the files the analyzer re-derives every guard verdict from."""
    # Homogeneous layout: transport summary, per-rank guards, samplers.
    tp = os.path.join(job_dir, "transport_summary.txt")
    if os.path.exists(tp):
        shutil.copyfile(tp, os.path.join(out_dir, "transport_summary.txt"))
    for gf in sorted(glob.glob(os.path.join(job_dir, "rank_*",
                                            "guards_*.txt"))):
        rank_dir = os.path.basename(os.path.dirname(gf))
        dst_dir = os.path.join(out_dir, "guards", rank_dir)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile(gf, os.path.join(dst_dir, os.path.basename(gf)))
    # Heterogeneous layout: each side's side.txt carries the guard
    # evidence, the counters snapshots and the transport section.
    for sf in sorted(glob.glob(os.path.join(job_dir, "*", "side.txt"))):
        side_dir = os.path.basename(os.path.dirname(sf))
        dst_dir = os.path.join(out_dir, side_dir)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile(sf, os.path.join(dst_dir, "side.txt"))
    # Source pinning: the in-run hash record of the harness that produced
    # the series.
    sp = os.path.join(job_dir, "source.sha256")
    if os.path.exists(sp):
        shutil.copyfile(sp, os.path.join(out_dir, "source.sha256"))
    # NCCL interface-selection lines from the debug logs (homogeneous) or
    # the per-side logs (mx), a frozen deliverable of the study.
    lines = set()
    for lg in (glob.glob(os.path.join(job_dir, "nccl_debug*.log"))
               + glob.glob(os.path.join(job_dir, "*", "nccl_debug*.log"))):
        with open(lg, errors="replace") as f:
            for line in f:
                if ("NET/Socket : Using" in line
                        or "Using network" in line
                        or "via NET/" in line):
                    lines.add(line.split("NCCL INFO ")[-1].strip())
    if lines:
        with open(os.path.join(out_dir, "nccl_interfaces.txt"), "w") as f:
            for line in sorted(lines):
                f.write(line + "\n")
    # Established-socket samples (homogeneous cells carry a 1 Hz sampler).
    counts = []
    for smp in sorted(glob.glob(os.path.join(job_dir, "node_*",
                                             "sampler_run.txt"))):
        node = os.path.basename(os.path.dirname(smp))
        with open(smp) as f:
            vals = [int(x) for x in
                    re.findall(r"tcp_established=(\d+)", f.read())]
        if vals:
            counts.append({"node": node, "samples": len(vals),
                           "min": min(vals), "max": max(vals)})
    if counts:
        with open(os.path.join(out_dir, "socket_counts.json"), "w") as f:
            json.dump(counts, f, indent=1, sort_keys=True)


def newest_complete_job(base, cell):
    """Latest job dir that has a summary for the cell, or None."""
    best = None
    for job_dir in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.exists(os.path.join(job_dir, f"{cell}_summary.json")):
            best = job_dir
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-root", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument(
        "--bulk-dir",
        default=None,
        help="where source-side series go; when given, only destination-side"
        " series and metadata land under --dataset-dir (the repo-tracked"
        " set) while *_src series are written here and still listed in the"
        " manifest under bulk/",
    )
    args = ap.parse_args()

    manifest = {}
    for cell, subdir in CELL_DIRS.items():
        base = os.path.join(args.capture_root, "capture", subdir)
        job_dir = newest_complete_job(base, cell)
        if job_dir is None:
            continue
        out_dir = os.path.join(args.dataset_dir, cell)
        os.makedirs(out_dir, exist_ok=True)
        job_id = os.path.basename(job_dir)
        with open(os.path.join(out_dir, "job_id.txt"), "w") as f:
            f.write(job_id + "\n")
        bulk_out = None
        if args.bulk_dir is not None:
            bulk_out = os.path.join(args.bulk_dir, cell)
            os.makedirs(bulk_out, exist_ok=True)
        for src in sorted(glob.glob(os.path.join(job_dir, f"{cell}_*"))):
            name = os.path.basename(src)
            if name.endswith("_src.csv") and bulk_out is not None:
                gzip_deterministic(src, os.path.join(bulk_out, name + ".gz"))
            elif name.endswith(".csv"):
                dst = os.path.join(out_dir, name + ".gz")
                gzip_deterministic(src, dst)
            elif name.endswith(".json"):
                dst = os.path.join(out_dir, name)
                shutil.copyfile(src, dst)
            else:
                continue
        # The mixed cell's matrix probe rides along with either mx cell.
        if cell.startswith("mx-"):
            mp = os.path.join(job_dir, "mx_matrix.json")
            if os.path.exists(mp):
                shutil.copyfile(mp, os.path.join(out_dir, "mx_matrix.json"))
        copy_guard_evidence(job_dir, out_dir, cell)
        if cell.startswith("mx-"):
            deltas = port_tx_deltas_mx(job_dir, cell.split("-", 1)[1])
        else:
            deltas = port_tx_deltas_homogeneous(job_dir)
        if deltas:
            with open(os.path.join(out_dir, "port_tx_deltas.json"),
                      "w") as f:
                json.dump(deltas, f, indent=1, sort_keys=True)

    for path in sorted(glob.glob(os.path.join(args.dataset_dir, "**", "*"), recursive=True)):
        if os.path.isfile(path):
            rel = os.path.relpath(path, args.dataset_dir)
            if rel == "MANIFEST.json":
                continue
            manifest[rel] = {
                "sha256": sha256_of(path),
                "bytes": os.path.getsize(path),
            }
    if args.bulk_dir is not None:
        for path in sorted(glob.glob(os.path.join(args.bulk_dir, "**", "*"), recursive=True)):
            if os.path.isfile(path):
                rel = os.path.join("bulk", os.path.relpath(path, args.bulk_dir))
                manifest[rel] = {
                    "sha256": sha256_of(path),
                    "bytes": os.path.getsize(path),
                }
    with open(os.path.join(args.dataset_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    total = sum(m["bytes"] for m in manifest.values())
    print(f"packaged {len(manifest)} files, {total / 1e6:.2f} MB total")
    for rel in sorted(manifest):
        print(f"  {manifest[rel]['bytes']:>10} {manifest[rel]['sha256'][:16]} {rel}")


if __name__ == "__main__":
    main()
