from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "task_progress.py"
LEDGER = REPO_ROOT / "docs" / "task-ledger.json"

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task_progress import OWNERS, PREFIXES, collect, render


def test_readme_pro_progress_block_is_current():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_no_id_is_both_closed_and_open():
    per_module, closed, _ = collect()
    open_all = set().union(*per_module.values())
    assert not (open_all & closed)


def test_every_owned_prefix_has_a_module():
    owned = {prefix for _, prefixes in OWNERS for prefix in prefixes}
    assert owned == set(PREFIXES)


def test_ledger_ids_are_well_formed_and_unique():
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    closed = ledger["closed"]
    assert len(closed) == len(set(closed))
    for identifier in closed:
        prefix, _, number = identifier.rpartition("-")
        assert prefix in PREFIXES
        assert number.isdigit()
    assert set(ledger.get("retracted", ())) <= set(closed)


def test_every_open_id_belongs_to_its_owning_module():
    per_module, _, _ = collect()
    owners = {name: prefixes for name, prefixes in OWNERS}
    for module, identifiers in per_module.items():
        if module not in owners:
            assert not identifiers, f"{module}.md carries task IDs but owns none"
            continue
        for identifier in identifiers:
            prefix = identifier.rsplit("-", 1)[0]
            assert prefix in owners[module], (
                f"{identifier} is registered in {module}.md, which does not own {prefix}"
            )


def test_render_is_deterministic():
    assert render() == render()
