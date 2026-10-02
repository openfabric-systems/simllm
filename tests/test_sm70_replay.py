from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from simllm.calibration.sm70_replay import (
    FRAMEWORK_PIN,
    GPGPU_SIM_PIN,
    LockedFile,
    Sm70OfflineReplay,
    Sm70ReplayRequest,
    Sm70SimulatorBinding,
    file_sha256,
    parse_serial_statistics,
)
from simllm.compute import GPU_ENVELOPES
from simllm.compute.volta import V100_PCIE_FP16


def _locked(path: Path, text: str) -> LockedFile:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return LockedFile(path, file_sha256(path))


def _inputs(tmp_path, *, count=1, clock=1000):
    traces = tuple(_locked(tmp_path / f"kernel-{index}.traceg",
                           f"-kernel name = k{index}\n-binary version = 70\n"
                           "-cuda stream id = 0\n# test header only, no executable trace\n")
                   for index in range(count))
    request = Sm70ReplayRequest(
        _locked(tmp_path / "kernelslist.g", "\n".join(trace.path.name for trace in traces)),
        traces, (_locked(tmp_path / "test.cubin", "test-only-code-object-fixture"),),
        capture_evidence_id="unit-test-fixture-never-simulator-evidence",
    )
    binding = Sm70SimulatorBinding(
        tmp_path / "framework", tmp_path / "gpgpu-sim",
        _locked(tmp_path / "accel-sim.out", "unit-test-not-executable"),
        _locked(tmp_path / "gpgpu-sim/configs/tested-cfgs/SM7_QV100/gpgpusim.config",
                f"-gpgpu_clock_domains {clock}:1000:1000:1000\n"),
        _locked(tmp_path / "framework/gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config",
                "unit-test-config"),
    )
    return request, binding


def _output(count, cycles):
    return "".join(f"kernel_name = k{i}\ngpu_sim_cycle = {cycles}\n"
                   f"gpu_tot_sim_cycle = {(i+1)*cycles}\n" for i in range(count)) + (
        "GPGPU-Sim: *** exit detected ***\n"
    )


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("clock", [1000, 2000])
@pytest.mark.parametrize("cycles", [10, 20])
def test_serial_cycle_and_clock_oracle(tmp_path, monkeypatch, count, clock, cycles):
    request, binding = _inputs(tmp_path, count=count, clock=clock)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "git":
            pin = FRAMEWORK_PIN if "framework" in argv[2] else GPGPU_SIM_PIN
            return SimpleNamespace(stdout=pin if "rev-parse" in argv else "")
        kwargs["stdout"].write(_output(count, cycles))
        assert argv[-4:] == ["-gpgpu_max_concurrent_kernel", "1",
                             "-gpgpu_concurrent_kernel_sm", "0"]
        assert kwargs["check"] is True
        return SimpleNamespace()

    monkeypatch.setattr("simllm.calibration.sm70_replay.subprocess.run", fake_run)
    observation = Sm70OfflineReplay(enabled=True).replay(request, binding,
                                                       output_path=tmp_path / "result.log")
    assert observation.duration_ps == count * cycles * 10**12 // (clock * 10**6)
    assert observation.source == "accel-sim" and observation.status == "candidate"
    assert len(calls) == 5
    assert len(observation.kernels) == count
    assert observation.simulator_profile_id == "QV100-SASS"
    assert observation.config_sha256 == binding.config.sha256


def test_disabled_replay_never_invokes_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr("simllm.calibration.sm70_replay.subprocess.run",
                        lambda *args, **kwargs: pytest.fail("disabled subprocess"))
    request, binding = _inputs(tmp_path)
    with pytest.raises(ValueError, match="disabled"):
        Sm70OfflineReplay().replay(request, binding, output_path=tmp_path / "result")
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("bad", ["", "kernel_name = k0\ngpu_sim_cycle = 10\n",
                                _output(1, 0), _output(1, 10) + "break due to reaching"])
def test_incomplete_or_nonpositive_statistics_fail_closed(bad):
    with pytest.raises(ValueError):
        parse_serial_statistics(bad)


def test_hash_mismatch_fails_before_source_or_simulator_subprocess(tmp_path, monkeypatch):
    request, binding = _inputs(tmp_path)
    request.traces[0].path.write_text("corrupt")
    monkeypatch.setattr("simllm.calibration.sm70_replay.subprocess.run",
                        lambda *args, **kwargs: pytest.fail("invalid input subprocess"))
    with pytest.raises(ValueError, match="hash mismatch"):
        Sm70OfflineReplay(enabled=True).replay(request, binding, output_path=tmp_path / "result")


def test_v100_envelope_is_explicit_fp16_and_nameplate():
    gpu = V100_PCIE_FP16
    assert gpu.peak_flops == 112e12 and gpu.mem_bandwidth == 900e9
    assert "v100" not in GPU_ENVELOPES
    assert gpu.name not in GPU_ENVELOPES


def test_freeze_names_both_independent_axes_and_no_capture_claim():
    root = Path(__file__).resolve().parents[1]
    freeze = json.loads((root / "offline/calibration/suites/sm70-serial-v1/expectations.json").read_text())
    assert freeze["parameters"]["kernel_counts"] == [1, 2, 4]
    assert freeze["parameters"]["core_clock_hz"] == [10**9, 2 * 10**9]
    assert freeze["execution_scope"] == "offline-adapter-tests-and-cuda-compilation"


def test_unlocked_config_dependency_is_rejected(tmp_path):
    from dataclasses import replace

    from simllm.calibration.sm70_replay import _configuration_closure
    _, binding = _inputs(tmp_path)
    config = _locked(binding.config.path, "-inter_config_file unexpected.icnt\n")
    with pytest.raises(ValueError, match="not hash locked"):
        _configuration_closure(replace(binding, config=config))
    auxiliary = _locked(config.path.parent / "unexpected.icnt", "locked fixture")
    _configuration_closure(replace(binding, config=config, auxiliary_files=(auxiliary,)))


def test_statistics_accept_enabled_deadlock_detection_flag():
    assert parse_serial_statistics("gpgpu_deadlock_detect 1\n" + _output(1, 10))[0].cycles == 10


def test_probe_refuses_nonfinite_values_before_error_reduction():
    source = (Path(__file__).resolve().parents[1] /
              "tools/compute_capture/sm70_serial_probe.cu").read_text()
    assert "!std::isfinite(result[row + column * rows])" in source
    assert "!std::isfinite(sum[column])" in source


@pytest.mark.parametrize("field,value", [("architecture", "sm75"),
                                        ("kernel_concurrency", 2),
                                        ("kernel_concurrency", True)])
def test_unsupported_capture_is_refused_before_subprocess(tmp_path, monkeypatch, field, value):
    from dataclasses import replace

    request, binding = _inputs(tmp_path)
    monkeypatch.setattr("simllm.calibration.sm70_replay.subprocess.run",
                        lambda *args, **kwargs: pytest.fail("unsupported input subprocess"))
    with pytest.raises(ValueError, match="SM70|concurrency"):
        Sm70OfflineReplay(enabled=True).replay(replace(request, **{field: value}), binding,
                                               output_path=tmp_path / "result")


def test_source_pin_mismatch_is_refused_before_simulator(tmp_path, monkeypatch):
    request, binding = _inputs(tmp_path)

    def wrong_source(argv, **kwargs):
        assert argv[0] == "git"
        return SimpleNamespace(stdout="wrong-pin" if "rev-parse" in argv else "")

    monkeypatch.setattr("simllm.calibration.sm70_replay.subprocess.run", wrong_source)
    with pytest.raises(ValueError, match="source pin"):
        Sm70OfflineReplay(enabled=True).replay(request, binding, output_path=tmp_path / "result")


def test_capture_stream_disagreement_fails_before_simulator(tmp_path, monkeypatch):
    from dataclasses import replace

    request, binding = _inputs(tmp_path, count=2)
    changed = _locked(request.traces[1].path,
                      request.traces[1].path.read_text().replace("stream id = 0", "stream id = 1"))
    request = replace(request, traces=(request.traces[0], changed))

    def pinned_source(argv, **kwargs):
        assert argv[0] == "git"
        pin = FRAMEWORK_PIN if "framework" in argv[2] else GPGPU_SIM_PIN
        return SimpleNamespace(stdout=pin if "rev-parse" in argv else "")

    monkeypatch.setattr("simllm.calibration.sm70_replay.subprocess.run", pinned_source)
    with pytest.raises(ValueError, match="one CUDA stream"):
        Sm70OfflineReplay(enabled=True).replay(request, binding, output_path=tmp_path / "result")


def test_changed_input_during_replay_invalidates_retained_output(tmp_path, monkeypatch):
    request, binding = _inputs(tmp_path)

    def mutating_run(argv, **kwargs):
        if argv[0] == "git":
            pin = FRAMEWORK_PIN if "framework" in argv[2] else GPGPU_SIM_PIN
            return SimpleNamespace(stdout=pin if "rev-parse" in argv else "")
        kwargs["stdout"].write(_output(1, 10))
        binding.config.path.write_text("changed during replay")
        return SimpleNamespace()

    monkeypatch.setattr("simllm.calibration.sm70_replay.subprocess.run", mutating_run)
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="hash mismatch"):
        Sm70OfflineReplay(enabled=True).replay(request, binding, output_path=output)
    assert output.is_file()
