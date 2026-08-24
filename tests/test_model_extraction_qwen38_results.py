from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/model_extraction_qwen38_v1/RESULTS.md"
COVERAGE = ROOT / "docs/design/calibration-coverage.md"


def test_qwen38_results_publish_a_blocked_total_rejection() -> None:
    text = RESULTS.read_text(encoding="utf-8")

    assert "f95d05a9bc0defa7171e371bcd2b2ad03db46954" in text
    assert "zero complete inventories" in text
    assert "No file was added under `offline/calibration/model-inventories/`" in text
    assert "COMP-54 remains open" in text
    assert "COMP-62 remains open" in text
    assert "Local weight-byte and weight-hash verification is intentionally not" in text


def test_qwen38_coverage_changes_only_the_model_state_claim() -> None:
    rows = [
        line
        for line in COVERAGE.read_text(encoding="utf-8").splitlines()
        if line.startswith("| Qwen3.8-27B |")
    ]

    assert len(rows) == 1
    assert "model_extraction_qwen38_v1" in rows[0]
    assert "zero complete inventories" in rows[0]
    assert "COMP-62" in rows[0]
    assert "COMP-54 stays open" in rows[0]
