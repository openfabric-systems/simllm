"""Byte lock over the Merlin fabric flow capture reference dataset.

The dataset under examples/merlin_fabric_flow_capture_v1/dataset/ is the
calibration reference the wave-19 Slingshot comparison consumes; its
RESULTS.md pins every file through dataset/MANIFEST.json. These tests
enforce the lock in CI: every manifest-listed tracked file must exist with
the exact recorded size and SHA-256, no unmanifested file may appear in
the tracked tree, and a negative control proves the checker detects a
mutation rather than passing vacuously. Files under the manifest's bulk/
prefix live outside the repository by design and are not checked here.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

DATASET = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples"
    / "merlin_fabric_flow_capture_v1"
    / "dataset"
)
MANIFEST = DATASET / "MANIFEST.json"


def _verify(manifest: dict, root: pathlib.Path) -> list[str]:
    problems = []
    for rel, meta in manifest.items():
        if rel.startswith("bulk/"):
            continue
        path = root / rel
        if not path.exists():
            problems.append(f"missing {rel}")
            continue
        data = path.read_bytes()
        if len(data) != meta["bytes"]:
            problems.append(f"size mismatch {rel}")
        if hashlib.sha256(data).hexdigest() != meta["sha256"]:
            problems.append(f"hash mismatch {rel}")
    return problems


def test_every_manifest_file_locked():
    manifest = json.loads(MANIFEST.read_text())
    assert _verify(manifest, DATASET) == []


def test_no_unmanifested_file_in_tracked_tree():
    manifest = json.loads(MANIFEST.read_text())
    tracked = {rel for rel in manifest if not rel.startswith("bulk/")}
    on_disk = {
        p.relative_to(DATASET).as_posix()
        for p in DATASET.rglob("*")
        if p.is_file()
    }
    on_disk.discard("MANIFEST.json")
    assert on_disk == tracked


def test_mutation_is_detected():
    """Negative control: a corrupted manifest entry must fail the check."""
    manifest = json.loads(MANIFEST.read_text())
    rel = next(k for k in sorted(manifest) if not k.startswith("bulk/"))
    corrupted = dict(manifest)
    corrupted[rel] = dict(corrupted[rel], sha256="0" * 64)
    assert _verify(corrupted, DATASET) != []
    truncated = dict(manifest)
    truncated[rel] = dict(truncated[rel], bytes=manifest[rel]["bytes"] + 1)
    assert _verify(truncated, DATASET) != []
