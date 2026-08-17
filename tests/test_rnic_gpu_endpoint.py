"""Byte-identity and closed-form locks for the BACK-46 GPU fabric endpoint.

The locked artifact set is every accepted artifact the second device could have
perturbed plus this study's own measured rows: the BACK-10 PCIe matrix, the
BACK-19 host-memory rows, the BACK-20 submission rows, the BACK-18 device rows
and captured native-test counts, the BACK-8 session record, the RNIC work-queue
rows, and `examples/rnic_gpu_endpoint_v1/results.csv`. Two independent locks
carry that, and neither is claimed to do the other's job:

- this module locks the tracked bytes of all of them, and proves the lock is
  mutation sensitive by flipping one byte of a copied tree and requiring the
  same guard to reject it;
- the study command registry rebuilds the native library and re-derives both
  the accepted rows and this study's rows from source, with a --check mode that
  compares without writing, so a C++ change that perturbs either fails there,
  and the full native CTest suite carries the compiled-in assertions.

The module also locks the study's frozen staging literals against an
independently derived rational closed form, with a negative control that
perturbs the literal and requires the registry check to fail.
"""

from __future__ import annotations

import hashlib
import importlib.util
from fractions import Fraction
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = REPO_ROOT / "examples" / "rnic_gpu_endpoint_v1" / "run_study.py"
SPEC = importlib.util.spec_from_file_location("rnic_gpu_endpoint_study", STUDY_PATH)
assert SPEC is not None
assert SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


#: The study's own rows are locked by their own constant, so the guarded set is
#: the accepted artifacts plus that one file.
STUDY_RESULTS = "examples/rnic_gpu_endpoint_v1/results.csv"
LOCKED_ARTIFACTS = (*study.FROZEN_ARTIFACT_DIGESTS, STUDY_RESULTS)


def _expected_digest(relative: str) -> str:
    if relative == STUDY_RESULTS:
        return study.STUDY_RESULTS_DIGEST
    return study.FROZEN_ARTIFACT_DIGESTS[relative]


def _mirror_artifacts(root: Path) -> None:
    for relative in LOCKED_ARTIFACTS:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO_ROOT / relative).read_bytes())


def _configure_run_root(monkeypatch, root: Path) -> Path:
    out = root / "runs" / "w17b" / "back46"
    monkeypatch.setenv(study.RUN_ROOT_ENV, str(root / "runs"))
    return out


@pytest.mark.parametrize("relative", LOCKED_ARTIFACTS)
def test_locked_artifacts_keep_their_frozen_bytes(relative: str):
    digest = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()

    assert digest == _expected_digest(relative)


def test_artifact_guard_accepts_the_tracked_tree(monkeypatch, tmp_path: Path):
    _mirror_artifacts(tmp_path)
    out = _configure_run_root(monkeypatch, tmp_path)
    monkeypatch.setattr(study, "REPO_ROOT", tmp_path)

    study._validate_registry(out)


@pytest.mark.parametrize("relative", LOCKED_ARTIFACTS)
def test_artifact_guard_rejects_a_single_flipped_byte(
    monkeypatch,
    tmp_path: Path,
    relative: str,
):
    _mirror_artifacts(tmp_path)
    out = _configure_run_root(monkeypatch, tmp_path)
    monkeypatch.setattr(study, "REPO_ROOT", tmp_path)
    mutated = bytearray((tmp_path / relative).read_bytes())
    mutated[-2] = mutated[-2] ^ 0x01
    (tmp_path / relative).write_bytes(bytes(mutated))

    with pytest.raises(AssertionError, match="digest drifted"):
        study._validate_registry(out)


def test_frozen_staging_literals_match_an_independent_closed_form():
    for (payload, lanes), frozen in study.FROZEN_STAGING_PS.items():
        # One byte costs 8 * 130 / (32 GT/s * lanes * 128) seconds on the
        # modeled generation-5 link, and a maximum-payload-aligned posted write
        # adds one 24-byte header per 256-byte TLP.
        rate = Fraction(8 * 130 * 1_000_000, 32_000 * lanes * 128)
        link_bytes = payload + 24 * (payload // 256)
        exact = link_bytes * rate
        expected = -((-exact.numerator) // exact.denominator)

        assert frozen == expected
        assert frozen > payload * rate


def test_registry_rejects_a_perturbed_staging_literal(monkeypatch, tmp_path: Path):
    _mirror_artifacts(tmp_path)
    out = _configure_run_root(monkeypatch, tmp_path)
    monkeypatch.setattr(study, "REPO_ROOT", tmp_path)
    perturbed = dict(study.FROZEN_STAGING_PS)
    perturbed[(4096, 16)] = perturbed[(4096, 16)] + 1
    monkeypatch.setattr(study, "FROZEN_STAGING_PS", perturbed)

    with pytest.raises(AssertionError, match="disagrees with the closed"):
        study._validate_registry(out)


def test_registry_requires_the_external_run_root(monkeypatch, tmp_path: Path):
    _mirror_artifacts(tmp_path)
    monkeypatch.setattr(study, "REPO_ROOT", tmp_path)
    monkeypatch.delenv(study.RUN_ROOT_ENV, raising=False)

    with pytest.raises(RuntimeError, match=study.RUN_ROOT_ENV):
        study._validate_registry(tmp_path / "elsewhere")


def test_registry_rejects_an_output_outside_the_run_root(
    monkeypatch,
    tmp_path: Path,
):
    _mirror_artifacts(tmp_path)
    monkeypatch.setattr(study, "REPO_ROOT", tmp_path)
    monkeypatch.setenv(study.RUN_ROOT_ENV, str(tmp_path / "runs"))

    with pytest.raises(ValueError, match="must remain under"):
        study._validate_registry(tmp_path / "outside")


def test_row_schema_and_sweep_are_frozen():
    assert study.ARMS == ("host_bounce", "gpu_direct")
    assert study.PAYLOAD_SIZES == (4096, 16384)
    assert study.LANE_COUNTS == (8, 16)
    assert len(study.ROW_FIELDS) == len(set(study.ROW_FIELDS))
    assert study.ROW_FIELDS[:3] == ("arm", "payload_bytes", "lane_count")
    assert (study.HOST_ENDPOINT, study.NIC_ENDPOINT, study.GPU_ENDPOINT) == (
        4000,
        4001,
        4002,
    )


def test_regenerated_off_path_inventory_covers_the_native_row_artifacts():
    assert set(study.REGENERATED_ARTIFACTS) <= set(study.FROZEN_ARTIFACT_DIGESTS)
    assert set(study.REGENERATED_ARTIFACTS) == {
        "examples/rnic_hostmem_v1/results.csv",
        "examples/rnic_submission_v1/results.csv",
    }


def test_published_scored_form_excludes_the_entailed_family():
    assert study.SCORED_FAMILIES == (
        "bounce_penalty_equals_staging",
        "staging_closed_form",
        "gpu_completer_charge",
    )
    assert study.RETAINED_ENTAILED_FAMILIES == ("arm_ordering",)
    assert "arm_ordering" not in study.SCORED_FAMILIES
