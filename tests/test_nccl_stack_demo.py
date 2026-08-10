from __future__ import annotations

import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[1]
STUDY = ROOT / "examples" / "nccl_stack_v1"


def test_nccl_stack_study_reproduces_reviewed_results(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(STUDY / "run_nccl_stack_v1.py"),
            "--out",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "run configurations: 3" in completed.stdout
    assert "scored behavioral relation families: 5/5 pass" in completed.stdout
    assert "scored behavioral relation instances: 35/35 pass" in completed.stdout
    assert "fatal unscored structural invariants: 10/10 hold" in completed.stdout
    generated = tmp_path / "results.csv"
    assert generated.read_bytes() == (STUDY / "results.csv").read_bytes()

    with generated.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert Counter(row["evidence_class"] for row in rows) == {
        "run_configuration": 3,
        "behavioral_relation": 35,
        "structural_invariant": 10,
    }
    behavioral = [row for row in rows if row["evidence_class"] == "behavioral_relation"]
    assert {row["family"] for row in behavioral} == {
        "inter_node_event_sequence",
        "intra_node_event_sequence",
        "planner_chunk_count",
        "planner_ring_step_structure",
        "planner_channel_assignment",
    }
    assert all(row["status"] == "PASS" for row in behavioral)
    assert all(
        row["status"] == "HOLD"
        for row in rows
        if row["evidence_class"] == "structural_invariant"
    )
