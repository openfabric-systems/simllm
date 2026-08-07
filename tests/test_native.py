from pathlib import Path

import simllm._native as native


def test_windows_cmake_candidates_cover_exe_and_multiconfig(monkeypatch, tmp_path):
    monkeypatch.setattr(native, "_WINDOWS", True)
    candidates = native.cmake_binary_candidates(
        tmp_path / "build with spaces",
        "htsim_rnic",
        subdirectory="datacenter",
    )

    assert tmp_path / "build with spaces" / "datacenter" / "htsim_rnic.exe" in candidates
    assert (
        tmp_path
        / "build with spaces"
        / "datacenter"
        / "Release"
        / "htsim_rnic.exe"
        in candidates
    )


def test_windows_discovery_accepts_exe_without_posix_execute_bit(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(native, "_WINDOWS", True)
    monkeypatch.delenv("SIMLLM_TEST_BINARY", raising=False)
    binary = tmp_path / "directory with spaces" / "tool.exe"
    binary.parent.mkdir()
    binary.touch()

    assert native.find_native_binary(
        "SIMLLM_TEST_BINARY", "definitely-not-on-path", [binary]
    ) == Path(binary)


def test_environment_override_precedes_build_candidates(monkeypatch, tmp_path):
    monkeypatch.setattr(native, "_WINDOWS", True)
    configured = tmp_path / "configured.exe"
    candidate = tmp_path / "candidate.exe"
    configured.touch()
    candidate.touch()
    monkeypatch.setenv("SIMLLM_TEST_BINARY", str(configured))

    assert native.find_native_binary(
        "SIMLLM_TEST_BINARY", "definitely-not-on-path", [candidate]
    ) == configured
