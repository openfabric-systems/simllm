import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTATIONS = (
    REPO_ROOT / "examples" / "rnic_live_v1" / "tier_a_expectations.json"
)
CHECKER = REPO_ROOT / "examples" / "rnic_live_v1" / "tier_a_acceptance.py"
RUN_ROOT = Path("/data3/yifeng/simllm-dev/wave2-runs")

_SPEC = importlib.util.spec_from_file_location("tier_a_acceptance", CHECKER)
assert _SPEC is not None and _SPEC.loader is not None
tier_a = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = tier_a
_SPEC.loader.exec_module(tier_a)


def _expectations() -> dict:
    return json.loads(EXPECTATIONS.read_text(encoding="utf-8"))


def _d_row(offset_ps: int) -> dict:
    return {
        "wqes": [
            {
                "eligible_at_ps": offset_ps,
                "port_tx_at_ps": offset_ps,
                "terminal_at_ps": offset_ps + 81_920,
                "cqe_visible_at_ps": offset_ps + 81_920,
                "polled_at_ps": offset_ps + 81_920,
            }
        ],
        "jct_ps": offset_ps + 81_920,
    }


def test_frozen_tier_a_manifest_validates():
    tier_a._validate_expectations(_expectations())


def test_same_d_predicate_accepts_composition_and_rejects_bypass():
    expectations = _expectations()
    low = _d_row(0)
    tier_a._validate_d_pair(low, _d_row(1_000), expectations, "positive")

    with pytest.raises(tier_a.AcceptanceError, match="D-additivity"):
        tier_a._validate_d_pair(low, _d_row(0), expectations, "bypass")


@pytest.mark.parametrize("factory", ["fake", "htsim"])
def test_registered_tier_a_command_contract_is_parse_only(factory):
    run_dir = RUN_ROOT / "pytest" / "tier_a" / factory
    producer_name = (
        "simllm_rnic_tier_a" if factory == "fake" else "htsim_rnic_tier_a"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--factory",
            factory,
            "--producer",
            str(run_dir / "build" / producer_name),
            "--run-dir",
            str(run_dir),
            "--check-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Tier A command contract valid" in completed.stdout
    assert not run_dir.exists()


def test_raw_schema_rejects_producer_verdict_fields():
    expectations = _expectations()
    raw = {key: None for key in expectations["raw_observation_keys"]}
    raw["pass"] = True

    with pytest.raises(tier_a.AcceptanceError, match="unexpected=.*pass"):
        tier_a._require_exact_keys(
            raw, expectations["raw_observation_keys"], "raw observations"
        )
