"""Lock the framework identities used by the pin-bump re-extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SUITES = REPOSITORY / "offline" / "calibration" / "suites"
HISTORICAL = SUITES / "transformer-dag-v1" / "suite.json"
PIN_BUMP = SUITES / "transformer-dag-v1-frameworks-2026-08-24" / "suite.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pin_bump_suite_changes_only_framework_identity() -> None:
    historical = _load(HISTORICAL)
    current = _load(PIN_BUMP)

    assert hashlib.sha256(PIN_BUMP.read_bytes()).hexdigest() == (
        "1282207c5ad8c8c0d0bea586bca7a39197a997dc165c3ad730a9847ddedd9dce"
    )
    assert current["frameworks"] == [
        {
            "id": "vllm",
            "version": "0.27.1",
            "source_commit": "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac",
        },
        {
            "id": "sglang",
            "version": "0.5.19.dev345+gbfeae4e79",
            "source_commit": "bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3",
            "source_tree": "9ffe149f40e1cd5bff7dadc6806ad1927d312e69",
        },
    ]
    current["frameworks"] = historical["frameworks"]
    assert current == historical
