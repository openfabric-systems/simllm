"""Optional, hash-locked serial replay of captured Volta SASS kernels.

This adapter launches an explicitly configured official offline simulator. It
does not fetch dependencies, manufacture traces or promote an observation to
silicon calibration. Default serving imports never load or invoke it.
"""

from __future__ import annotations

import gzip
import hashlib
import re
import subprocess
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

FRAMEWORK_PIN = "3016c658f810bdae9a14bf4534ee99e9945eedae"
GPGPU_SIM_PIN = "6c3cf4ff32110908386d605a7034fc67666a92de"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LockedFile:
    path: Path
    sha256: str

    def verify(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_file():
            raise ValueError("a replay input must name an existing regular file")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("a replay input needs a lowercase SHA-256 digest")
        if file_sha256(self.path) != self.sha256:
            raise ValueError(f"replay file hash mismatch: {self.path.name}")


@dataclass(frozen=True)
class Sm70ReplayRequest:
    trace_list: LockedFile
    traces: tuple[LockedFile, ...]
    code_objects: tuple[LockedFile, ...]
    architecture: str = "sm70"
    kernel_concurrency: int = 1
    capture_evidence_id: str = ""


@dataclass(frozen=True)
class Sm70SimulatorBinding:
    framework_root: Path
    gpgpu_sim_root: Path
    executable: LockedFile
    config: LockedFile
    trace_config: LockedFile
    auxiliary_files: tuple[LockedFile, ...] = ()
    profile_id: str = "QV100-SASS"


@dataclass(frozen=True)
class ReplayCapability:
    supported: bool
    reason: str


@dataclass(frozen=True)
class SerialKernelObservation:
    kernel_name: str
    cycles: int


@dataclass(frozen=True)
class Sm70ReplayObservation:
    kernels: tuple[SerialKernelObservation, ...]
    core_clock_hz: int
    duration_ps: int
    input_sha256: tuple[str, ...]
    output_sha256: str
    capture_evidence_id: str
    simulator_profile_id: str
    config_sha256: str
    trace_config_sha256: str
    executable_sha256: str
    framework_pin: str = FRAMEWORK_PIN
    gpgpu_sim_pin: str = GPGPU_SIM_PIN
    source: str = "accel-sim"
    status: str = "candidate"


def parse_serial_statistics(text: str) -> tuple[SerialKernelObservation, ...]:
    if "GPGPU-Sim: *** exit detected ***" not in text:
        raise ValueError("simulator output has no complete exit marker")
    if "break due to reaching" in text or re.search(r"DEADLOCK:|deadlock detected", text):
        raise ValueError("simulator stopped before complete replay")
    names = re.findall(r"^kernel_name\s*=\s*(\S+)\s*$", text, re.MULTILINE)
    cycles = [int(value) for value in re.findall(
        r"^gpu_sim_cycle\s*=\s*(\d+)\s*$", text, re.MULTILINE,
    )]
    totals = [int(value) for value in re.findall(
        r"^gpu_tot_sim_cycle\s*=\s*(\d+)\s*$", text, re.MULTILINE,
    )]
    if not names or len(names) != len(cycles) or len(names) != len(totals):
        raise ValueError("serial kernel statistics are absent or incomplete")
    cumulative = 0
    for cycle_count, total in zip(cycles, totals, strict=True):
        cumulative += cycle_count
        if cycle_count <= 0 or total != cumulative:
            raise ValueError("serial cycle counters do not conserve total service")
    return tuple(SerialKernelObservation(name, count)
                 for name, count in zip(names, cycles, strict=True))


def _git_identity(root: Path, expected: str) -> None:
    actual = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if actual != expected or dirty:
        raise ValueError("offline simulator source pin is mismatched or modified")


def _trace_headers(request: Sm70ReplayRequest) -> tuple[str, ...]:
    commands = request.trace_list.path.read_text().splitlines()
    listed = [line.strip() for line in commands if line.strip().startswith("kernel-")]
    expected = [trace.path.name for trace in request.traces]
    if listed != expected or len(set(listed)) != len(listed):
        raise ValueError("trace list must close the ordered captured kernel files")
    if any(line.strip() and not (line.strip().startswith("kernel-") or
                                line.strip().startswith("MemcpyHtoD,"))
           for line in commands):
        raise ValueError("unsupported trace-list command")
    names, streams = [], set()
    for trace in request.traces:
        if trace.path.parent.resolve() != request.trace_list.path.parent.resolve():
            raise ValueError("captured traces must be siblings of the trace list")
        opener = gzip.open if trace.path.suffix == ".gz" else open
        with opener(trace.path, "rt") as source:
            header = []
            for _ in range(64):
                line = source.readline()
                if not line or line.startswith("#"):
                    break
                header.append(line)
        text = "".join(header)
        isa = re.search(r"^-binary version = (\d+)$", text, re.MULTILINE)
        stream = re.search(r"^-cuda stream id = (\d+)$", text, re.MULTILINE)
        name = re.search(r"^-kernel name = (\S+)$", text, re.MULTILINE)
        if not isa or isa.group(1) != "70" or not stream or not name:
            raise ValueError("captured input is not a complete SM70 kernel header")
        names.append(name.group(1))
        streams.add(stream.group(1))
    if len(streams) != 1:
        raise ValueError("serial capture must use exactly one CUDA stream")
    return tuple(names)


def _configuration_closure(binding: Sm70SimulatorBinding) -> None:
    config_path = binding.gpgpu_sim_root / "configs/tested-cfgs/SM7_QV100/gpgpusim.config"
    trace_path = binding.framework_root / "gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config"
    if binding.profile_id != "QV100-SASS" or binding.config.path.resolve() != config_path.resolve():
        raise ValueError("only the pinned QV100-SASS profile is supported")
    if binding.trace_config.path.resolve() != trace_path.resolve():
        raise ValueError("trace configuration must belong to the pinned QV100-SASS profile")
    active = [line.split("#", 1)[0].strip() for line in binding.config.path.read_text().splitlines()]
    locked = {entry.path.resolve(): entry for entry in binding.auxiliary_files}
    for line in active:
        if not line:
            continue
        parts = line.split()
        option = parts[0]
        if option == "-inter_config_file":
            if len(parts) != 2:
                raise ValueError("interconnect configuration reference is malformed")
            dependency = (binding.config.path.parent / parts[1]).resolve()
            if dependency not in locked:
                raise ValueError("interconnect configuration dependency is not hash locked")
        elif ("file" in option or "path" in option or "include" in option) and option not in {
            "-enable_ptx_file_line_stats", "-gpgpu_reg_file_port_throughput",
        }:
            raise ValueError(f"unsupported configuration dependency option: {option}")


class Sm70OfflineReplay:
    def __init__(self, *, enabled: bool = False, timeout_seconds: int = 3600) -> None:
        if type(enabled) is not bool or type(timeout_seconds) is not int:
            raise TypeError("enabled and timeout must be a bool and integer")
        if timeout_seconds <= 0:
            raise ValueError("timeout must be positive")
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds

    def supports(self, request: Sm70ReplayRequest,
                 binding: Sm70SimulatorBinding) -> ReplayCapability:
        if not self.enabled:
            return ReplayCapability(False, "optional offline replay is disabled")
        if not isinstance(request, Sm70ReplayRequest) or not isinstance(binding, Sm70SimulatorBinding):
            return ReplayCapability(False, "typed request and simulator binding are required")
        if request.architecture != "sm70" or type(request.kernel_concurrency) is not int:
            return ReplayCapability(False, "only explicit SM70 serial replay is supported")
        if request.kernel_concurrency != 1:
            return ReplayCapability(False, "kernel concurrency must be one")
        if binding.profile_id != "QV100-SASS":
            return ReplayCapability(False, "only the pinned QV100-SASS simulator profile is supported")
        if not request.capture_evidence_id.strip() or not request.traces or not request.code_objects:
            return ReplayCapability(False, "captured trace and code-object evidence is required")
        return ReplayCapability(True, "serial SM70 capture is eligible for validation")

    def replay(self, request: Sm70ReplayRequest, binding: Sm70SimulatorBinding,
               *, output_path: Path) -> Sm70ReplayObservation:
        capability = self.supports(request, binding)
        if not capability.supported:
            raise ValueError(capability.reason)
        if output_path.exists():
            raise ValueError("replay output must not overwrite retained evidence")
        files = (binding.executable, binding.config, binding.trace_config,
                 *binding.auxiliary_files, request.trace_list, *request.traces,
                 *request.code_objects)
        for locked in files:
            locked.verify()
        _git_identity(binding.framework_root, FRAMEWORK_PIN)
        _git_identity(binding.gpgpu_sim_root, GPGPU_SIM_PIN)
        _configuration_closure(binding)
        names = _trace_headers(request)
        clock = re.search(r"^-gpgpu_clock_domains\s+([0-9.]+):", binding.config.path.read_text(),
                          re.MULTILINE)
        if not clock:
            raise ValueError("simulator configuration does not declare its core clock")
        clock_hz = Decimal(clock.group(1)) * 1_000_000
        if clock_hz <= 0 or clock_hz != int(clock_hz):
            raise ValueError("simulator core clock must be a positive integer Hz")
        argv = [str(binding.executable.path.resolve()), "-trace",
                str(request.trace_list.path.resolve()), "-config", str(binding.config.path.resolve()),
                "-config", str(binding.trace_config.path.resolve()),
                "-power_simulation_enabled", "0", "-gpgpu_max_concurrent_kernel", "1",
                "-gpgpu_concurrent_kernel_sm", "0"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as destination:
            subprocess.run(argv, cwd=binding.config.path.parent, stdout=destination,
                           stderr=subprocess.STDOUT, check=True, timeout=self.timeout_seconds)
        for locked in files:
            locked.verify()
        kernels = parse_serial_statistics(output_path.read_text())
        if tuple(kernel.kernel_name for kernel in kernels) != names:
            raise ValueError("replayed kernel identities disagree with captured inputs")
        cycles = sum(kernel.cycles for kernel in kernels)
        hz = int(clock_hz)
        return Sm70ReplayObservation(kernels, hz, (cycles * 10**12 + hz - 1) // hz,
                                    tuple(locked.sha256 for locked in files),
                                    file_sha256(output_path), request.capture_evidence_id,
                                    binding.profile_id, binding.config.sha256,
                                    binding.trace_config.sha256, binding.executable.sha256)
