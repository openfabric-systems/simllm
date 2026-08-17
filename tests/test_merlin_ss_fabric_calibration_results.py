"""Byte lock over the TRAF-51 calibration study's tracked results.

The tracked summaries under examples/merlin_ss_fabric_calibration_v1/results/
are the scored output of the wave-19 Slingshot calibration comparison
(calibration_summary.json) and the run-identity record of the simulation
matrix that produced it (run_manifest.json), pinned through the study's
results/MANIFEST.json. These tests enforce the lock in CI: every
manifest-listed file must exist with the exact recorded size and SHA-256, no
unmanifested file may appear in the results tree, the run identity inside the
locked bytes must match the frozen binary hash and submodule pin, and a
negative control proves the checker detects a mutation rather than passing
vacuously.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

RESULTS = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples"
    / "merlin_ss_fabric_calibration_v1"
    / "results"
)
MANIFEST = RESULTS / "MANIFEST.json"

FROZEN_BINARY_SHA256 = (
    "5075021a6af762e914d782a1c69c1633d9b084f767ac82d9b6931bd33f69f787")
FROZEN_SUBMODULE_PIN = "89b7a5a8caa16185c257899a4200935de0cc69c4"
FROZEN_DATASET_MANIFEST_SHA256 = (
    "a6b7e61e294d87d76ce69ee7042e15c2eade99bbc8789e296377615d2bd4af88")


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
    summary = json.loads((RESULTS / "calibration_summary.json").read_text())
    run = json.loads((RESULTS / "run_manifest.json").read_text())
    assert summary["binary_sha256"] == FROZEN_BINARY_SHA256
    assert run["binary_sha256"] == FROZEN_BINARY_SHA256
    assert run["submodule_head"] == FROZEN_SUBMODULE_PIN
    assert summary["dataset_manifest_sha256"] == FROZEN_DATASET_MANIFEST_SHA256
    assert summary["void"] is False
    assert summary["fatal_failures"] == []


def test_run_manifest_topology_hashes_match_the_tracked_instances():
    """FG-1's topology clause, machine-enforced (review correction 4).

    The locked run manifest records the SHA-256 of every topology file the
    runs consumed; the two Merlin instances are tracked in the study
    directory, so the recorded hashes must equal the tracked bytes' hashes.
    """
    run = json.loads((RESULTS / "run_manifest.json").read_text())
    study = RESULTS.parent
    for name in (
        "merlin_a100_singleswitch_v1.topo",
        "merlin_a100_singleswitch_v1_100g.topo",
    ):
        tracked = hashlib.sha256((study / name).read_bytes()).hexdigest()
        assert run["topologies"][name] == tracked


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
