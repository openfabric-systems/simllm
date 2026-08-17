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
import shutil

CELL_DIRS = {
    "s1-stream": "s1-stream",
    "i2-incast": "i2-incast",
    "i3-incast": "i3-incast",
    "j3-join": "j3-join",
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
        for src in sorted(glob.glob(os.path.join(job_dir, f"{cell}_*"))):
            name = os.path.basename(src)
            if name.endswith(".csv"):
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

    for path in sorted(glob.glob(os.path.join(args.dataset_dir, "**", "*"), recursive=True)):
        if os.path.isfile(path):
            rel = os.path.relpath(path, args.dataset_dir)
            if rel == "MANIFEST.json":
                continue
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
