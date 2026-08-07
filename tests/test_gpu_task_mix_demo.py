import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[1]
DEMO = ROOT / "examples" / "gpu_task_mix"


def test_gpu_task_mix_demo_reproduces_reviewed_artifacts(tmp_path):
    command = [
        sys.executable,
        str(DEMO / "run_gpu_task_mix.py"),
        "--out",
        str(tmp_path),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "distinct run configurations: 38" in completed.stdout
    assert "replay invocations: 46" in completed.stdout
    assert "scored exact-oracle rows: 36/36 pass" in completed.stdout
    assert "scored behavioral relation families: 6/6 pass" in completed.stdout
    assert "scored behavioral relation instances: 17/17 pass" in completed.stdout
    assert "unscored structural invariants: 21/21 hold" in completed.stdout
    assert "historical ledger rows: 2 superseded failures retained" in completed.stdout
    for name in ("results.csv", "nccl_convergence.csv", "diagnostics.csv"):
        assert (tmp_path / name).read_bytes() == (DEMO / name).read_bytes()

    with (tmp_path / "results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert Counter(row["evidence_class"] for row in rows) == {
        "exact_oracle": 36,
        "behavioral_relation": 17,
        "structural_invariant": 21,
        "historical_ledger": 2,
    }
    assert len(
        {
            row["family"]
            for row in rows
            if row["evidence_class"] == "behavioral_relation"
        }
    ) == 6
    assert all(
        row["status"] == "PASS"
        for row in rows
        if row["evidence_class"] != "historical_ledger"
    )
    history = [row for row in rows if row["evidence_class"] == "historical_ledger"]
    assert [
        (row["check"], row["expected"], row["measured"], row["status"])
        for row in history
    ] == [
        ("H-D2", "328", "329", "HISTORICAL_FAIL"),
        ("H-D3", "132", "243", "HISTORICAL_FAIL"),
    ]
