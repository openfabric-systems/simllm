"""Lifetime-owned subprocess execution for native simulator invocations."""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LAUNCHER = Path(__file__).with_name("_child_exec.py")
_MARKER_DIR_ENV = "SIMLLM_CHILD_LIFETIME_MARKER_DIR"
_MARKER_NONCE_ENV = "SIMLLM_CHILD_LIFETIME_RUN_NONCE"
_MARKER_NONCE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_HANDSHAKE = "G"
_TERMINATION_GRACE_S = 1.0
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_ERROR_ACCESS_DENIED = 5


if os.name == "nt":
    from ctypes import wintypes

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("read_operation_count", ctypes.c_ulonglong),
            ("write_operation_count", ctypes.c_ulonglong),
            ("other_operation_count", ctypes.c_ulonglong),
            ("read_transfer_count", ctypes.c_ulonglong),
            ("write_transfer_count", ctypes.c_ulonglong),
            ("other_transfer_count", ctypes.c_ulonglong),
        ]

    class _JobBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("per_process_user_time_limit", ctypes.c_longlong),
            ("per_job_user_time_limit", ctypes.c_longlong),
            ("limit_flags", wintypes.DWORD),
            ("minimum_working_set_size", ctypes.c_size_t),
            ("maximum_working_set_size", ctypes.c_size_t),
            ("active_process_limit", wintypes.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority_class", wintypes.DWORD),
            ("scheduling_class", wintypes.DWORD),
        ]

    class _JobExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("basic_limit_information", _JobBasicLimitInformation),
            ("io_info", _IoCounters),
            ("process_memory_limit", ctypes.c_size_t),
            ("job_memory_limit", ctypes.c_size_t),
            ("peak_process_memory_used", ctypes.c_size_t),
            ("peak_job_memory_used", ctypes.c_size_t),
        ]


class _WindowsJob:
    """One Windows Job Object whose close kills all assigned descendants."""

    def __init__(self, handle: int) -> None:
        self._handle = handle
        self._lock = threading.Lock()

    @classmethod
    def create(cls) -> _WindowsJob:
        if os.name != "nt":
            raise RuntimeError("Windows Job Objects are available only on Windows")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        job = cls(int(handle))
        information = _JobExtendedLimitInformation()
        information.basic_limit_information.limit_flags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not kernel32.SetInformationJobObject(
            wintypes.HANDLE(job._handle),
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            job.close()
            raise error
        return job

    def assign(self, process: subprocess.Popen[str]) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        process_handle = wintypes.HANDLE(int(process._handle))  # type: ignore[attr-defined]
        if not kernel32.AssignProcessToJobObject(
            wintypes.HANDLE(self._handle), process_handle
        ):
            error_number = ctypes.get_last_error()
            detail = (
                "the current process job rejected nested assignment"
                if error_number == _ERROR_ACCESS_DENIED
                else "Job Object assignment failed"
            )
            raise OSError(error_number, f"{detail}: {ctypes.WinError(error_number)}")

    def terminate(self) -> None:
        with self._lock:
            if not self._handle:
                return
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
            kernel32.TerminateJobObject.restype = wintypes.BOOL
            if not kernel32.TerminateJobObject(wintypes.HANDLE(self._handle), 1):
                error_number = ctypes.get_last_error()
                if error_number:
                    raise ctypes.WinError(error_number)

    def close(self) -> None:
        with self._lock:
            if not self._handle:
                return
            handle = self._handle
            self._handle = 0
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        if not kernel32.CloseHandle(wintypes.HANDLE(handle)):
            raise ctypes.WinError(ctypes.get_last_error())


@dataclass
class _OwnedChild:
    process: subprocess.Popen[str]
    command: tuple[str, ...]
    process_group_id: int | None
    windows_job: _WindowsJob | None


class _OwnedChildRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._children: dict[int, _OwnedChild] = {}
        self._previous_handlers: dict[int, Any] = {}
        self._handlers_installed = False
        self._atexit_registered = False
        self._handling_signal = False

    def prepare(self) -> None:
        if os.name not in {"nt", "posix"}:
            raise RuntimeError(
                f"owned simulator processes are unsupported on platform {sys.platform!r}"
            )
        with self._lock:
            if not self._atexit_registered:
                atexit.register(self.cleanup_all)
                self._atexit_registered = True
            if os.name != "posix" or self._handlers_installed:
                return
            if threading.current_thread() is not threading.main_thread():
                if sys.platform.startswith("linux"):
                    return
                raise RuntimeError(
                    "install simulator child signal handlers from the main thread "
                    "before launching from a worker"
                )
            signals = [signal.SIGINT, signal.SIGTERM]
            if hasattr(signal, "SIGHUP"):
                signals.append(signal.SIGHUP)
            for signum in signals:
                self._previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)
            self._handlers_installed = True

    def register(self, child: _OwnedChild) -> None:
        with self._lock:
            if child.process.pid in self._children:
                raise RuntimeError(f"child PID {child.process.pid} is already registered")
            self._children[child.process.pid] = child

    def unregister(self, child: _OwnedChild) -> None:
        with self._lock:
            current = self._children.get(child.process.pid)
            if current is child:
                self._children.pop(child.process.pid)
        if child.windows_job is not None:
            child.windows_job.close()

    @staticmethod
    def _signal_posix(child: _OwnedChild, signum: int) -> None:
        if child.process.poll() is not None:
            return
        group_id = child.process_group_id
        if group_id is None or group_id != child.process.pid:
            raise RuntimeError("owned POSIX child has an invalid process-group identity")
        try:
            actual_group = os.getpgid(child.process.pid)
        except ProcessLookupError:
            return
        if actual_group != group_id:
            raise RuntimeError(
                f"refusing to signal child {child.process.pid}: process group changed"
            )
        try:
            os.killpg(group_id, signum)
        except ProcessLookupError:
            pass

    @staticmethod
    def _terminate_windows(child: _OwnedChild) -> None:
        if child.process.poll() is None and child.windows_job is not None:
            child.windows_job.terminate()

    def terminate(self, child: _OwnedChild) -> None:
        self._terminate_many((child,))

    def _terminate_many(self, children: Sequence[_OwnedChild]) -> None:
        active = tuple(child for child in children if child.process.poll() is None)
        for child in active:
            if os.name == "posix":
                self._signal_posix(child, signal.SIGTERM)
            elif os.name == "nt":
                self._terminate_windows(child)
        deadline = time.monotonic() + _TERMINATION_GRACE_S
        while any(child.process.poll() is None for child in active):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        for child in active:
            if child.process.poll() is not None:
                continue
            if os.name == "posix":
                self._signal_posix(child, signal.SIGKILL)
            elif os.name == "nt":
                self._terminate_windows(child)
        for child in active:
            try:
                child.process.wait(timeout=_TERMINATION_GRACE_S)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass

    def cleanup_all(self) -> None:
        with self._lock:
            children = tuple(self._children.values())
        self._terminate_many(children)
        for child in children:
            self.unregister(child)

    def _handle_signal(self, signum: int, _frame: object) -> None:
        with self._lock:
            if self._handling_signal:
                return
            self._handling_signal = True
            previous = self._previous_handlers.get(signum, signal.SIG_DFL)
        cleanup_error: Exception | None = None
        try:
            self.cleanup_all()
        except (OSError, RuntimeError) as error:
            cleanup_error = error
        with self._lock:
            self._handling_signal = False
        if previous == signal.SIG_IGN:
            if cleanup_error is not None:
                raise cleanup_error
            return
        if callable(previous):
            previous(signum, _frame)
            if cleanup_error is not None:
                raise cleanup_error
            return
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


_REGISTRY = _OwnedChildRegistry()


def prepare_owned_child_runtime() -> None:
    """Install the main-thread cleanup boundary before worker submission."""

    _REGISTRY.prepare()


def cleanup_owned_children() -> None:
    """Idempotently terminate and reap all currently registered children."""

    _REGISTRY.cleanup_all()


def _linux_start_time(pid: int) -> str | None:
    if not sys.platform.startswith("linux"):
        return None
    stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
    closing = stat.rfind(")")
    if closing < 0:
        raise ValueError(f"cannot parse procfs identity for PID {pid}")
    fields = stat[closing + 2 :].split()
    if len(fields) <= 19:
        raise ValueError(f"incomplete procfs identity for PID {pid}")
    return fields[19]


def _write_marker(
    process: subprocess.Popen[str],
    command: Sequence[str],
    environment: Mapping[str, str] | None,
) -> None:
    marker_environment = os.environ if environment is None else environment
    raw_directory = marker_environment.get(_MARKER_DIR_ENV)
    raw_nonce = marker_environment.get(_MARKER_NONCE_ENV)
    if raw_directory is None and raw_nonce is None:
        return
    if not raw_directory or not raw_nonce:
        raise ValueError(
            f"{_MARKER_DIR_ENV} and {_MARKER_NONCE_ENV} must be configured together"
        )
    if _MARKER_NONCE_RE.fullmatch(raw_nonce) is None:
        raise ValueError(f"{_MARKER_NONCE_ENV} contains an invalid run nonce")
    directory = Path(raw_directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"child marker directory does not exist: {directory}")
    command_digest = hashlib.sha256("\0".join(command).encode("utf-8")).hexdigest()
    marker = {
        "child_pid": process.pid,
        "command_sha256": command_digest,
        "owner_pid": os.getpid(),
        "run_nonce": raw_nonce,
        "schema": "simllm-child-lifetime-marker-v1",
        "start_time_token": _linux_start_time(process.pid),
    }
    path = directory / f"{raw_nonce}-{process.pid}.json"
    temporary = directory / (
        f".{raw_nonce}-{process.pid}-{threading.get_ident()}.tmp"
    )
    temporary.write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _popen_environment(environment: Mapping[str, str] | None) -> Mapping[str, str] | None:
    if environment is None:
        return None
    return dict(environment)


def _communicate_unmanaged(
    command: tuple[str, ...],
    *,
    timeout_s: float,
    environment: Mapping[str, str] | None,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_popen_environment(environment),
    )
    try:
        _write_marker(process, command, environment)
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    except BaseException:
        process.kill()
        process.wait()
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def run_owned_process(
    command: Sequence[str],
    *,
    timeout_s: float,
    environment: Mapping[str, str] | None = None,
    unsafe_unmanaged: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one command with owned descendants and exact captured text output."""

    argv = tuple(str(item) for item in command)
    if not argv:
        raise ValueError("owned process command must not be empty")
    if timeout_s <= 0:
        raise ValueError("owned process timeout must be positive")
    if unsafe_unmanaged:
        return _communicate_unmanaged(
            argv,
            timeout_s=timeout_s,
            environment=environment,
        )

    _REGISTRY.prepare()
    job = _WindowsJob.create() if os.name == "nt" else None
    launcher = (sys.executable, str(_LAUNCHER), str(os.getpid()), "--", *argv)
    popen_options: dict[str, Any] = {
        "env": _popen_environment(environment),
        "stderr": subprocess.PIPE,
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "text": True,
    }
    if os.name == "posix":
        popen_options["start_new_session"] = True
    process: subprocess.Popen[str] | None = None
    child: _OwnedChild | None = None
    try:
        process = subprocess.Popen(launcher, **popen_options)
        if job is not None:
            job.assign(process)
        child = _OwnedChild(
            process=process,
            command=argv,
            process_group_id=process.pid if os.name == "posix" else None,
            windows_job=job,
        )
        _REGISTRY.register(child)
        _write_marker(process, argv, environment)
        if process.stdin is None:
            raise RuntimeError("owned child launcher has no handshake stream")
        process.stdin.write(_HANDSHAKE)
        process.stdin.flush()
        process.stdin.close()
        process.stdin = None
        stdout, stderr = process.communicate(timeout=timeout_s)
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
    except subprocess.TimeoutExpired as error:
        if child is not None:
            _REGISTRY.terminate(child)
        elif process is not None:
            process.kill()
            process.wait()
        raise subprocess.TimeoutExpired(
            argv,
            timeout_s,
            output=error.output,
            stderr=error.stderr,
        ) from None
    except BaseException:
        if child is not None:
            _REGISTRY.terminate(child)
        elif process is not None:
            process.kill()
            process.wait()
        raise
    finally:
        if child is not None:
            _REGISTRY.unregister(child)
        elif job is not None:
            job.close()
