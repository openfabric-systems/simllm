from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from simllm.calibration.protocols import (
    CalibrationCompiler,
    HardwareCollector,
    OfflineKernelSimulator,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class _Collector:
    def doctor(self, request, /):
        return {"request": request}

    def capture(self, request, graph, /):
        return (request, graph)


class _Simulator:
    def supports(self, request, binding, /):
        return (request, binding)

    def replay(self, request, trace_ref, /):
        return (request, trace_ref)


class _Compiler:
    def fit(self, evidence_manifest, /):
        return ("fit", evidence_manifest)

    def compile(self, fit, /):
        return ("model", fit)

    def verify(self, evidence_manifest, fit, model, /):
        return (evidence_manifest, fit, model)


def test_protocols_accept_structural_implementations() -> None:
    collector = _Collector()
    simulator = _Simulator()
    compiler = _Compiler()

    assert isinstance(collector, HardwareCollector)
    assert collector.capture("request", "graph") == ("request", "graph")
    assert isinstance(simulator, OfflineKernelSimulator)
    assert simulator.replay("request", "trace") == ("request", "trace")
    assert isinstance(compiler, CalibrationCompiler)
    assert compiler.verify("manifest", "fit", "model") == (
        "manifest",
        "fit",
        "model",
    )


def test_incomplete_objects_do_not_satisfy_protocols() -> None:
    class DoctorOnly:
        def doctor(self, request, /):
            return request

    class SupportsOnly:
        def supports(self, request, binding, /):
            return request, binding

    class FitOnly:
        def fit(self, evidence_manifest, /):
            return evidence_manifest

    assert not isinstance(DoctorOnly(), HardwareCollector)
    assert not isinstance(SupportsOnly(), OfflineKernelSimulator)
    assert not isinstance(FitOnly(), CalibrationCompiler)


def test_protocol_module_loads_no_hardware_or_simulator_runtime() -> None:
    source = (
        "import sys; import simllm.calibration.protocols; "
        "forbidden=('torch','cupy','cuda','rocm','rocprofiler','accel_sim'); "
        "assert not any(name == prefix or name.startswith(prefix + '.') "
        "for name in sys.modules for prefix in forbidden)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
