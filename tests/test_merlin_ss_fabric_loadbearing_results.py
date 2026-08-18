"""Byte lock over the TRAF-51 load-bearing recalibration's tracked results.

The tracked summaries under examples/merlin_ss_fabric_loadbearing_v1/results/
are the scored output of the wave-21 load-bearing recalibration
(recalibration_summary.json) and the run-identity record of the simulation
matrix that produced it (run_manifest.json), pinned through the study's
results/MANIFEST.json. These tests enforce the lock in CI: every
manifest-listed file must exist with the exact recorded size and SHA-256, no
unmanifested file may appear in the results tree, the run identity inside
the locked bytes must match the frozen binary hash, submodule pin and
dataset manifest hash, the recorded topology hashes must equal the tracked
instance bytes (FG-1's machine-enforcement clause), and a negative control
proves the checker detects a mutation rather than passing vacuously.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

RESULTS = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples"
    / "merlin_ss_fabric_loadbearing_v1"
    / "results"
)
MANIFEST = RESULTS / "MANIFEST.json"

FROZEN_BINARY_SHA256 = (
    "662416918457542c4fcab18aebd99f43f00cd80b5518ed78947a65e65620ce92")
FROZEN_SUBMODULE_PIN = "1dcbfec36a33753bf978cf6323bade1a6645fe4f"
FROZEN_DATASET_MANIFEST_SHA256 = (
    "67f898a04dd0e6787d4d50d6139f99f3c507963c6c937a9505434cf2d9dca002")


def _verify(manifest: dict, root: pathlib.Path) -> list[str]:
    problems = []
    for rel, meta in manifest.items():
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
    assert _verify(manifest, RESULTS) == []


def test_no_unmanifested_file_in_results_tree():
    manifest = json.loads(MANIFEST.read_text())
    on_disk = {
        p.relative_to(RESULTS).as_posix()
        for p in RESULTS.rglob("*")
        if p.is_file()
    }
    on_disk.discard("MANIFEST.json")
    assert on_disk == set(manifest)


def test_locked_run_identity_matches_the_freeze():
    summary = json.loads((RESULTS / "recalibration_summary.json").read_text())
    run = json.loads((RESULTS / "run_manifest.json").read_text())
    assert summary["binary_sha256"] == FROZEN_BINARY_SHA256
    assert run["binary_sha256"] == FROZEN_BINARY_SHA256
    assert run["submodule_head"] == FROZEN_SUBMODULE_PIN
    assert summary["dataset_manifest_sha256"] == FROZEN_DATASET_MANIFEST_SHA256
    assert summary["void"] is False
    assert summary["fatal_failures"] == []


def test_run_manifest_topology_hashes_match_the_tracked_instances():
    """FG-1's topology clause, machine-enforced (the wave-19 correction 4
    lesson): the locked run manifest records the SHA-256 of every instance
    the runs consumed, and those hashes must equal the tracked bytes."""
    run = json.loads((RESULTS / "run_manifest.json").read_text())
    study = RESULTS.parent
    tracked = {
        "v1": study.parent
        / "merlin_ss_fabric_calibration_v1"
        / "merlin_a100_singleswitch_v1.topo",
        "x4buf32": study / "merlin_a100_singleswitch_v1_x4buf32.topo",
        "x4buf64": study / "merlin_a100_singleswitch_v1_x4buf64.topo",
    }
    for key, path in tracked.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert run["topologies"][key] == digest


def test_scored_rows_carry_the_frozen_inventory():
    summary = json.loads((RESULTS / "recalibration_summary.json").read_text())
    by_id = {row["id"]: row for row in summary["rows"]}
    assert sorted(by_id) == ["BE-1", "BE-2", "BE-3", "BE-4", "BE-5",
                             "EX-1", "EX-2", "ST-1"]
    classes = {row_id: row["class"] for row_id, row in by_id.items()}
    assert classes == {"EX-1": "exact", "EX-2": "exact", "BE-1": "behavioral",
                       "BE-2": "behavioral", "BE-3": "behavioral",
                       "BE-4": "behavioral", "BE-5": "behavioral",
                       "ST-1": "structural"}


def test_mutation_is_detected():
    """Negative control: a corrupted manifest entry must fail the check."""
    manifest = json.loads(MANIFEST.read_text())
    rel = next(iter(sorted(manifest)))
    corrupted = dict(manifest)
    corrupted[rel] = dict(corrupted[rel], sha256="0" * 64)
    assert _verify(corrupted, RESULTS) != []
    truncated = dict(manifest)
    truncated[rel] = dict(truncated[rel], bytes=manifest[rel]["bytes"] + 1)
    assert _verify(truncated, RESULTS) != []
