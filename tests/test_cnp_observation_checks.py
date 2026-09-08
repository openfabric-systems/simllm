"""Adversarial checks for the external-controller comparison evidence."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "examples/completion_boundary_v1/check_cnp_observations.py"
spec = importlib.util.spec_from_file_location("cnp_observation_checks", PATH)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def log_fixture():
    lines = ["100% tests passed, 0 tests failed out of 8"]
    for n, payload, observed in sorted(checks.POINTS):
        prefix = f"senders={n} payload={payload} observations={observed}"
        floor = n * payload * 20 + 2_000_000
        for rank in range(n):
            lines.append(f"BACK71_FLOW {prefix} source={rank} destination=63 bytes={payload} tag=9 "
                         f"flow={(rank << 32) | (100 + rank)} start_ps=0 completion_ps={floor + rank}")
        control = " ".join(f"{key}={observed}" for key in
                           ("packet_events", "control_events", "ecn", "cnp", "rates",
                            "pfc_frames", "pauses", "resumes"))
        lines.append(f"BACK71_BOUNDARY {prefix} floor_ps={floor} last_completion_ps={floor + n - 1} "
                     f"quiescence_ps={floor + n} callbacks=100 {control}")
    return "\n".join(lines)


def test_complete_comparison_keeps_evidence_classes_separate():
    result = checks.check(log_fixture())
    assert result["status"] == "valid"
    assert result["configurations"] == 8 and result["paired_comparisons"] == 4
    assert result["exact_completion_row_pairs"] == 24 and result["fatal_findings"] == []
    assert "behavioral_score" not in result


@pytest.mark.parametrize("fault", ("lost", "duplicate", "one-ps", "order", "floor", "budget", "coverage"))
def test_corrupt_evidence_is_void_without_a_pass_fraction(fault):
    lines = log_fixture().splitlines()
    selected = [i for i, line in enumerate(lines) if "observations=1" in line and "BACK71_FLOW" in line]
    index = selected[0]
    if fault == "lost":
        lines.pop(index)
    elif fault == "duplicate":
        lines.append(lines[index])
    elif fault == "one-ps":
        before = lines[index].rsplit("=", 1)[1]
        lines[index] = lines[index].rsplit("=", 1)[0] + "=" + str(int(before) + 1)
    elif fault == "order":
        lines[index], lines[index + 1] = lines[index + 1], lines[index]
    elif fault == "floor":
        lines = [line.replace("floor_ps=7242880", "floor_ps=7242879") for line in lines]
    elif fault == "budget":
        lines = [line.replace("callbacks=100", "callbacks=100001") for line in lines]
    else:
        lines = [line.replace("cnp=1", "cnp=0") for line in lines]
    result = checks.check("\n".join(lines))
    assert result["status"] == "void" and result["fatal_findings"]
    assert result["comparisons"] == []
    assert "exact_completion_row_pairs" not in result
