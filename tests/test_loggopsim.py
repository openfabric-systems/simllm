"""Exact argument and parse oracles for the LogGOPSim invocation helper."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from simllm.backends import loggopsim
from simllm.backends.loggopsim import (
    LOGGOPSIM_DECLARED_EVIDENCE_SOURCE,
    LogGopsimConfig,
    build_loggopsim_command,
    derive_loggp_params,
    find_loggopsim,
    parse_loggopsim_stdout,
    run_loggopsim,
)

# Captured from the unmodified tool. The per-host block appears at 16 ranks or
# fewer outside batch mode; batch mode prints only the maximum instead.
PER_HOST_STDOUT = """The schedule will be valid after this simulation!
size: 2 (1 CPUs, 1 NICs); L=2500, o=1500 g=1000, G=6.000000, O=0, P=2, S=65535
PERFORMANCE: Processes: 2 \t Events: 3 \t Time: 0 s \t Speed: inf ev/s
Average FCT is 41943.040000
Times:
Host 0: 6297950
Host 1: 6293950
"""

BATCH_STDOUT = """The schedule will be valid after this simulation!
size: 2 (1 CPUs, 1 NICs); L=2500, o=1500 g=1000, G=6.000000, O=0, P=2, S=65535
PERFORMANCE: Processes: 2 \t Events: 3 \t Time: 0 s \t Speed: inf ev/s
Average FCT is 41943.040000
Maximum finishing time at host 0: 6297950 (0.00629795 s)
"""


def _executable(path: Path) -> Path:
    path.write_text("", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_command_matches_the_frozen_option_grammar():
    cfg = LogGopsimConfig(goal_bin=Path("trace.bin"))
    assert build_loggopsim_command(Path("LogGOPSim"), cfg) == [
        "LogGOPSim",
        "-f", "trace.bin",
        "-L", "2500",
        "-o", "1500",
        "-g", "1000",
        "-G", "6.0",
        "-O", "0",
        "-S", "65535",
        "-n", "LogGP",
    ]


def test_batch_mode_and_optional_files_render_after_the_model_parameters():
    cfg = LogGopsimConfig(
        goal_bin=Path("trace.bin"),
        latency_ns=1,
        overhead_ns=2,
        message_gap_ns=3,
        byte_gap_ns=3.5,
        byte_overhead_ns=4,
        rendezvous_threshold_bytes=5,
        network_type="simple",
        network_file=Path("fabric.dot"),
        batch_mode=True,
        comm_dependency_file=Path("deps.csv"),
    )
    assert build_loggopsim_command(Path("LogGOPSim"), cfg) == [
        "LogGOPSim",
        "-f", "trace.bin",
        "-L", "1",
        "-o", "2",
        "-g", "3",
        "-G", "3.5",
        "-O", "4",
        "-S", "5",
        "-n", "simple",
        "--network-file", "fabric.dot",
        "-b",
        "--comm-dep-file", "deps.csv",
    ]


def test_extra_flags_render_valueless_and_valued_options():
    cfg = LogGopsimConfig(
        goal_bin=Path("trace.bin"),
        extra_flags={"--progress": "", "--qstat": "prefix"},
    )
    assert build_loggopsim_command(Path("LogGOPSim"), cfg)[-3:] == [
        "--progress",
        "--qstat",
        "prefix",
    ]


def test_integer_gap_renders_as_a_double_literal():
    cfg = LogGopsimConfig(goal_bin=Path("trace.bin"), byte_gap_ns=6)
    assert "6.0" in build_loggopsim_command(Path("LogGOPSim"), cfg)


def test_an_exact_gap_string_is_preserved_in_the_argument_vector():
    cfg = LogGopsimConfig(
        goal_bin=Path("trace.bin"),
        byte_gap_ns=0.02,
        byte_gap_ns_string="0.020000000000000000",
    )
    argv = build_loggopsim_command(Path("LogGOPSim"), cfg)
    assert argv[argv.index("-G") + 1] == "0.020000000000000000"


def test_ideal_parameter_derivation_preserves_shortest_decimal_and_evidence():
    rate_400 = derive_loggp_params(
        rate_bits_per_second=400_000_000_000,
        latency_ns=123,
    )
    rate_200 = derive_loggp_params(
        rate_bits_per_second=200_000_000_000,
        latency_ns=123,
    )
    assert rate_400.exact_g_string == "0.02"
    assert rate_200.exact_g_string == "0.04"
    assert rate_400.to_json()["L"]["value"] == 123
    assert rate_400.to_json()["o"]["value"] == 0
    assert rate_400.to_json()["g"]["value"] == 0
    assert rate_400.to_json()["O"]["value"] == 0
    assert {
        parameter["evidence_source"]
        for parameter in rate_400.to_json().values()
    } == {LOGGOPSIM_DECLARED_EVIDENCE_SOURCE}


def test_parses_the_per_host_output_shape():
    result = parse_loggopsim_stdout(PER_HOST_STDOUT)
    assert result.rank_count == 2
    assert result.cpu_count == 1
    assert result.nic_count == 1
    assert result.host_finish_ps == {0: 6297950000, 1: 6293950000}
    assert result.max_finish_host == 0
    assert result.average_fct_ns == pytest.approx(41943.04)
    assert result.unmatched_queue_diagnostics == ()


def test_parses_the_batch_output_shape():
    result = parse_loggopsim_stdout(BATCH_STDOUT)
    assert result.rank_count == 2
    assert result.host_finish_ps == {}
    assert result.max_finish_host == 0
    assert result.job_completion_time_ps() == 6297950000


def test_nanosecond_to_picosecond_conversion_is_exactly_one_thousand():
    per_host = parse_loggopsim_stdout(PER_HOST_STDOUT)
    batch = parse_loggopsim_stdout(BATCH_STDOUT)
    assert per_host.job_completion_time_ps() == 6297950 * 1000
    assert batch.job_completion_time_ps() == 6297950 * 1000


def test_unparseable_average_fct_stays_absent():
    stdout = BATCH_STDOUT.replace("Average FCT is 41943.040000", "Average FCT is -nan")
    assert parse_loggopsim_stdout(stdout).average_fct_ns is None


def test_missing_banner_and_missing_finish_are_rejected():
    with pytest.raises(RuntimeError, match="banner"):
        parse_loggopsim_stdout("Times:\nHost 0: 1\n")
    banner_only = BATCH_STDOUT.splitlines()[1] + "\n"
    with pytest.raises(RuntimeError, match="finishing time"):
        parse_loggopsim_stdout(banner_only)


def test_unmatched_queue_diagnostics_are_collected(tmp_path, monkeypatch):
    stdout = BATCH_STDOUT.replace(
        "Average FCT is",
        "unexpected queue on host 1 contains 2 elements!\nAverage FCT is",
    )
    parsed = parse_loggopsim_stdout(stdout)
    assert parsed.unmatched_queue_diagnostics == (
        "unexpected queue on host 1 contains 2 elements!",
    )

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    _Completed.stdout = stdout
    monkeypatch.setattr(loggopsim.subprocess, "run", lambda *a, **k: _Completed)
    with pytest.raises(RuntimeError, match="unmatched messages"):
        run_loggopsim(
            LogGopsimConfig(goal_bin=tmp_path / "trace.bin"),
            binary=_executable(tmp_path / "LogGOPSim"),
        )


def test_discovery_prefers_the_environment_variable(tmp_path, monkeypatch):
    binary = _executable(tmp_path / "LogGOPSim")
    monkeypatch.setenv("SIMLLM_LOGGOPSIM", str(binary))
    assert find_loggopsim() == binary


def test_absent_executable_names_the_environment_variable(monkeypatch):
    monkeypatch.delenv("SIMLLM_LOGGOPSIM", raising=False)
    monkeypatch.setattr(loggopsim, "find_loggopsim", lambda: None)
    with pytest.raises(FileNotFoundError, match="SIMLLM_LOGGOPSIM"):
        run_loggopsim(LogGopsimConfig(goal_bin=Path("trace.bin")))


def test_configuration_rejects_an_unknown_network_model():
    with pytest.raises(ValueError, match="network_type"):
        LogGopsimConfig(goal_bin=Path("trace.bin"), network_type="loggp")


def test_simple_network_model_requires_a_topology_file():
    with pytest.raises(ValueError, match="network_file"):
        LogGopsimConfig(goal_bin=Path("trace.bin"), network_type="simple")


def test_nonzero_exit_is_reported_with_the_captured_error(tmp_path, monkeypatch):
    class _Failed:
        returncode = 3
        stdout = ""
        stderr = "Couldn't parse command line arguments!"

    monkeypatch.setattr(loggopsim.subprocess, "run", lambda *a, **k: _Failed)
    with pytest.raises(RuntimeError, match="exit 3"):
        run_loggopsim(
            LogGopsimConfig(goal_bin=tmp_path / "trace.bin"),
            binary=_executable(tmp_path / "LogGOPSim"),
        )
