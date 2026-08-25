"""Independent C++17 checks for the calibration ASCII canonical subset."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from simllm.calibration.canonical import canonical_bytes, strict_json_loads

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests" / "native" / "calibration_canonical"


@pytest.fixture(scope="session")
def canonical_verifier(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the standalone verifier without fetching any dependency."""

    if shutil.which("cmake") is None:
        pytest.skip("CMake is required for the native canonical conformance test")
    build = tmp_path_factory.mktemp("calibration-canonical-native")
    subprocess.run(
        ["cmake", "-S", str(SOURCE), "-B", str(build)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--config", "Release", "--parallel", "2"],
        check=True,
        capture_output=True,
        text=True,
    )
    names = ("calibration_canonical_verify", "calibration_canonical_verify.exe")
    candidates = [candidate for name in names for candidate in build.rglob(name)]
    if len(candidates) != 1:
        raise AssertionError(f"expected one native verifier, found {candidates!r}")
    return candidates[0]


VALID_VECTORS = (
    (
        '{ "z": [3, true, null], "a": "x" }',
        '{"a":"x","z":[3,true,null]}',
        "9a06baa4c7d1e7144a93a0aec6540e9c9fadaa4c037dddc927222700af36c8ad",
    ),
    (
        (
            '{"negative":-999999999999999999999999999999999999999,'
            '"integer":1234567890123456789012345678901234567890}'
        ),
        (
            '{"integer":1234567890123456789012345678901234567890,'
            '"negative":-999999999999999999999999999999999999999}'
        ),
        "259566ac95b73b2d6006ab8682fd4d9ff9837cd7773fdd97cd54a9c6e3d3abb2",
    ),
    (
        '{"zero":0,"empty":{},"array":[{"b":1,"a":2},[],false]}',
        '{"array":[{"a":2,"b":1},[],false],"empty":{},"zero":0}',
        "6cf1595f38eb78fdc500534cf4d33f335dca9cd9e4a956794645c4234745e3b3",
    ),
    (
        r'{"escaped":"quote\" slash\/ backslash\\","controls":"\b\f\n\r\t\u0001"}',
        r'{"controls":"\b\f\n\r\t\u0001","escaped":"quote\" slash/ backslash\\"}',
        "6459a6dbb85ebdcc6ddfda5786416f0defe5b0db707b929a6fe2807b0a1e39ac",
    ),
    (
        '["", {}, [], true, false, null, -1, 0, 1]',
        '["",{},[],true,false,null,-1,0,1]',
        "c025d84643051697d191ce03df1ffb86b59450015183ffffa94cc291dab8fea2",
    ),
)


def _run(verifier: Path, tmp_path: Path, raw: str) -> subprocess.CompletedProcess[str]:
    source = tmp_path / "input.json"
    source.write_bytes(raw.encode("utf-8"))
    return subprocess.run(
        [str(verifier), str(source)],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(("raw", "expected", "digest"), VALID_VECTORS)
def test_native_ascii_vectors_match_fixed_goldens_and_python(
    canonical_verifier: Path,
    tmp_path: Path,
    raw: str,
    expected: str,
    digest: str,
) -> None:
    completed = _run(canonical_verifier, tmp_path, raw)
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout.splitlines() == [expected, digest]
    assert hashlib.sha256(expected.encode("ascii")).hexdigest() == digest
    parsed = strict_json_loads(raw)
    assert canonical_bytes(parsed) == expected.encode("ascii")


@pytest.mark.parametrize(
    "raw",
    (
        '{"a":1,"a":2}',
        r'{"a":1,"\u0061":2}',
        '{"value":1.5}',
        '{"value":1e3}',
        '{"value":-0}',
        '{"value":01}',
        r'{"value":"\u0080"}',
        r'{"value":"\ud800"}',
        '{"value":"é"}',
        '{"value":true} trailing',
        '[1,]',
    ),
)
def test_native_ascii_subset_rejects_invalid_or_out_of_subset_input(
    canonical_verifier: Path,
    tmp_path: Path,
    raw: str,
) -> None:
    completed = _run(canonical_verifier, tmp_path, raw)
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert completed.stderr.startswith("error: ")


def test_native_verifier_has_no_external_build_dependency() -> None:
    cmake = (SOURCE / "CMakeLists.txt").read_text(encoding="utf-8")
    source = (SOURCE / "canonical_verify.cpp").read_text(encoding="utf-8")
    forbidden = ("FetchContent", "find_package", "OpenSSL", "ICU", "nlohmann")
    for name in forbidden:
        assert name not in cmake
        assert name not in source
