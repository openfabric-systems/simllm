"""Lifetime-owned subprocess execution for native simulator invocations."""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import json
import math
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
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
_LAST_WINDOWS_JOB_DIAGNOSTICS_LOCK = threading.Lock()
_LAST_WINDOWS_JOB_DIAGNOSTICS: dict[str, object] = {}


def _publish_windows_job_diagnostics(values: Mapping[str, object]) -> None:
    global _LAST_WINDOWS_JOB_DIAGNOSTICS

    with _LAST_WINDOWS_JOB_DIAGNOSTICS_LOCK:
        _LAST_WINDOWS_JOB_DIAGNOSTICS = dict(values)


def _windows_job_diagnostics_for_test() -> dict[str, object]:
    """Return the most recent Job Object transition record for test failures."""

    with _LAST_WINDOWS_JOB_DIAGNOSTICS_LOCK:
        return dict(_LAST_WINDOWS_JOB_DIAGNOSTICS)


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
        self._lock = threading.RLock()
        self._diagnostics: dict[str, object] = {
            "created_handle": handle,
            "platform": sys.platform,
        }
        self._record(handle_open_after_create=self._handle_is_open(handle)[0])

    @staticmethod
    def _handle_is_open(handle: int) -> tuple[bool, int]:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetHandleInformation.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel32.GetHandleInformation.restype = wintypes.BOOL
        flags = wintypes.DWORD()
        ctypes.set_last_error(0)
        succeeded = bool(
            kernel32.GetHandleInformation(
                wintypes.HANDLE(handle), ctypes.byref(flags)
            )
        )
        return succeeded, ctypes.get_last_error()

    @staticmethod
    def _process_job_membership(
        process_handle: int, job_handle: int | None
    ) -> tuple[bool, bool, int]:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.IsProcessInJob.argtypes = (
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.BOOL),
        )
        kernel32.IsProcessInJob.restype = wintypes.BOOL
        membership = wintypes.BOOL()
        ctypes.set_last_error(0)
        succeeded = bool(
            kernel32.IsProcessInJob(
                wintypes.HANDLE(process_handle),
                wintypes.HANDLE(job_handle) if job_handle is not None else None,
                ctypes.byref(membership),
            )
        )
        return succeeded, bool(membership.value), ctypes.get_last_error()

    def _record(self, **values: object) -> None:
        with self._lock:
            self._diagnostics.update(values)
            snapshot = dict(self._diagnostics)
        _publish_windows_job_diagnostics(snapshot)

    def note(self, phase: str, process: subprocess.Popen[str]) -> None:
        """Capture handle and launcher state at a lifecycle boundary."""

        with self._lock:
            handle = self._handle
        handle_open, handle_error = self._handle_is_open(handle)
        self._record(
            **{
                f"handle_error_{phase}": handle_error,
                f"handle_open_{phase}": handle_open,
                f"launcher_returncode_{phase}": process.poll(),
            }
        )

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
        ctypes.set_last_error(0)
        configured = bool(
            kernel32.SetInformationJobObject(
                wintypes.HANDLE(job._handle),
                _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(information),
                ctypes.sizeof(information),
            )
        )
        configuration_error = ctypes.get_last_error()
        job._record(
            set_information_last_error=configuration_error,
            set_information_result=configured,
        )
        if not configured:
            error = ctypes.WinError(configuration_error)
            job.close()
            raise error
        return job

    def assign(self, process: subprocess.Popen[str]) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.argtypes = ()
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        raw_process_handle = int(process._handle)  # type: ignore[attr-defined]
        process_handle = wintypes.HANDLE(raw_process_handle)
        raw_owner_handle = int(kernel32.GetCurrentProcess())
        owner_query, owner_in_job, owner_query_error = self._process_job_membership(
            raw_owner_handle, None
        )
        before_query, before_in_job, before_query_error = (
            self._process_job_membership(raw_process_handle, None)
        )
        ctypes.set_last_error(0)
        assigned = bool(
            kernel32.AssignProcessToJobObject(
                wintypes.HANDLE(self._handle), process_handle
            )
        )
        assignment_error = ctypes.get_last_error()
        after_query, after_in_owned_job, after_query_error = (
            self._process_job_membership(raw_process_handle, self._handle)
        )
        handle_open, handle_error = self._handle_is_open(self._handle)
        self._record(
            assign_last_error=assignment_error,
            assign_result=assigned,
            handle_error_after_assign=handle_error,
            handle_open_after_assign=handle_open,
            launcher_any_job_before_assign=before_in_job,
            launcher_any_job_query_error=before_query_error,
            launcher_any_job_query_result=before_query,
            launcher_owned_job_after_assign=after_in_owned_job,
            launcher_owned_job_query_error=after_query_error,
            launcher_owned_job_query_result=after_query,
            owner_any_job=owner_in_job,
            owner_any_job_query_error=owner_query_error,
            owner_any_job_query_result=owner_query,
        )
        if not assigned:
            error_number = assignment_error
            detail = (
                "the current process job rejected nested assignment"
                if error_number == _ERROR_ACCESS_DENIED
                else "Job Object assignment failed"
            )
            raise OSError(error_number, f"{detail}: {ctypes.WinError(error_number)}")

    def terminate(self) -> None:
        with self._lock:
            if not self._handle:
                self._record(terminate_skipped_closed_handle=True)
                return
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
            kernel32.TerminateJobObject.restype = wintypes.BOOL
            ctypes.set_last_error(0)
            terminated = bool(
                kernel32.TerminateJobObject(wintypes.HANDLE(self._handle), 1)
            )
            error_number = ctypes.get_last_error()
            self._record(
                terminate_last_error=error_number,
                terminate_result=terminated,
            )
            if not terminated and error_number:
                raise ctypes.WinError(error_number)

    def close(self) -> None:
        with self._lock:
            if not self._handle:
                self._record(close_repeated=True)
                return
            handle = self._handle
            open_before, error_before = self._handle_is_open(handle)
            self._record(
                handle_error_before_close=error_before,
                handle_open_before_close=open_before,
            )
            self._handle = 0
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        ctypes.set_last_error(0)
        closed = bool(kernel32.CloseHandle(wintypes.HANDLE(handle)))
        close_error = ctypes.get_last_error()
        open_after, error_after = self._handle_is_open(handle)
        self._record(
            close_last_error=close_error,
            close_result=closed,
            handle_error_after_close=error_after,
            handle_open_after_close=open_after,
        )
        if not closed:
            raise ctypes.WinError(close_error)


@dataclass
class _OwnedChild:
    process: subprocess.Popen[str] | subprocess.Popen[bytes]
    command: tuple[str, ...]
    process_group_id: int | None
    windows_job: _WindowsJob | None
    kill_before_reap: bool = False


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
        status = child.process.returncode if child.kill_before_reap else child.process.poll()
        if status is not None:
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
        for child in children:
            if not child.kill_before_reap:
                continue
            if os.name == "posix":
                self._signal_posix(child, signal.SIGKILL)
            elif child.windows_job is not None:
                child.windows_job.terminate()
            child.process.wait(timeout=_TERMINATION_GRACE_S)
        active = tuple(child for child in children
                       if not child.kill_before_reap and child.process.poll() is None)
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
            job.note("after_popen", process)
            job.assign(process)
            job.note("after_assign", process)
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
        if job is not None:
            job.note("before_communicate", process)
        stdout, stderr = process.communicate(timeout=timeout_s)
        if job is not None:
            job.note("after_communicate", process)
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
    except subprocess.TimeoutExpired as error:
        if job is not None and process is not None:
            job.note("timeout_observed", process)
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
            if job is not None:
                job.note("before_unregister", child.process)
            _REGISTRY.unregister(child)
        elif job is not None:
            job.close()


class OwnedBinaryReadError(EOFError):
    """An incomplete read retaining every byte received before EOF."""

    def __init__(self, partial: bytes) -> None:
        super().__init__("owned child ended a binary response early")
        self.partial = partial


class OwnedBinaryProcess:
    """An owned binary stream with one deadline for its entire lifetime."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_s: float,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.command = tuple(str(item) for item in command)
        if not self.command:
            raise ValueError("owned process command must not be empty")
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("owned process timeout must be finite and positive")
        if os.name == "posix" and not all(
            hasattr(os, name) for name in ("waitid", "WNOWAIT", "WEXITED", "P_PID")
        ):
            raise RuntimeError("owned binary streams require POSIX waitid/WNOWAIT or Windows Jobs")
        self.timeout_s = timeout_s
        self._deadline = time.monotonic() + timeout_s
        self._lock = threading.RLock()
        self._closed = False
        self._failure: BaseException | None = None
        self._stderr = bytearray()
        self._tasks: queue.Queue[tuple[Callable[[], Any], queue.Queue[Any]] | None] = (
            queue.Queue()
        )
        self._timer: threading.Timer | None = None
        self._threads: list[threading.Thread] = []
        _REGISTRY.prepare()
        job = _WindowsJob.create() if os.name == "nt" else None
        process = None
        self._child: _OwnedChild | None = None
        try:
            process = subprocess.Popen(
                (sys.executable, str(_LAUNCHER), str(os.getpid()), "--", *self.command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                env=_popen_environment(environment),
                start_new_session=os.name == "posix",
            )
            if job is not None:
                job.note("after_popen", process)
                job.assign(process)
                job.note("after_assign", process)
            self._child = _OwnedChild(
                process, self.command, process.pid if os.name == "posix" else None, job,
                kill_before_reap=True,
            )
            _REGISTRY.register(self._child)
            _write_marker(process, self.command, environment)
            for target in (self._work, self._drain_stderr):
                thread = threading.Thread(target=target, daemon=True)
                self._threads.append(thread)
                thread.start()
            self._timer = threading.Timer(max(0, self._deadline - time.monotonic()),
                                          self._expire)
            self._timer.daemon = True
            self._timer.start()
            self.write(_HANDSHAKE.encode("ascii"))
        except BaseException:
            if self._child is not None:
                self.abort()
            else:
                if process is not None:
                    process.kill()
                    process.wait()
                if job is not None:
                    job.close()
            raise

    @property
    def pid(self) -> int:
        assert self._child is not None
        return self._child.process.pid

    @property
    def stderr(self) -> bytes:
        with self._lock:
            return bytes(self._stderr)

    def _expire(self) -> None:
        self._failure = subprocess.TimeoutExpired(self.command, self.timeout_s)
        self.abort()

    def _work(self) -> None:
        while (task := self._tasks.get()) is not None:
            function, result = task
            try:
                result.put((True, function()))
            except BaseException as error:  # noqa: BLE001 (forward to the waiting owner)
                result.put((False, error))

    def _drain_stderr(self) -> None:
        assert self._child is not None and self._child.process.stderr is not None
        try:
            while data := self._child.process.stderr.read(65536):
                with self._lock:
                    room = (1 << 20) - len(self._stderr)
                    self._stderr.extend(data[:room])
                if len(data) > room:
                    self._failure = RuntimeError("owned child stderr exceeds one mebibyte")
                    self.abort()
                    return
        except (OSError, ValueError):
            return

    def _call(self, function: Callable[[], Any]) -> Any:
        try:
            if self._failure is not None:
                raise self._failure
            if self._closed:
                raise RuntimeError("owned binary process is closed")
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(self.command, self.timeout_s)
            result: queue.Queue[Any] = queue.Queue(maxsize=1)
            self._tasks.put((function, result))
            try:
                succeeded, value = result.get(timeout=remaining)
            except queue.Empty:
                raise subprocess.TimeoutExpired(self.command, self.timeout_s) from None
            if self._failure is not None:
                raise self._failure
            if not succeeded:
                raise value
            return value
        except BaseException:
            self.abort()
            raise

    def write(self, data: bytes) -> None:
        def write_all() -> None:
            assert self._child is not None and self._child.process.stdin is not None
            view = memoryview(data)
            while view:
                written = self._child.process.stdin.write(view)
                if not written:
                    raise BrokenPipeError("owned child closed its input")
                view = view[written:]
        self._call(write_all)

    def read_exact(self, size: int) -> bytes:
        if type(size) is not int or size < 0:
            raise ValueError("binary read size must be a nonnegative integer")

        def read_all() -> bytes:
            assert self._child is not None and self._child.process.stdout is not None
            data = bytearray()
            while len(data) < size:
                chunk = self._child.process.stdout.read(size - len(data))
                if not chunk:
                    raise OwnedBinaryReadError(bytes(data))
                data.extend(chunk)
            return bytes(data)
        return self._call(read_all)

    def finish(self) -> int:
        """Close input, require clean output EOF, and reap the successful child."""
        def finish_io() -> int:
            assert self._child is not None
            process = self._child.process
            assert process.stdin is not None and process.stdout is not None
            process.stdin.close()
            if process.stdout.read(1):
                raise RuntimeError("owned child wrote trailing protocol bytes")
            if os.name == "posix":
                # Keep the leader unreaped until its group has been killed. Its PID
                # then still authenticates descendants even after early leader exit.
                os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
                _REGISTRY.terminate(self._child)
                assert process.returncode is not None
                status = process.returncode
            else:
                status = process.wait(timeout=max(0, self._deadline - time.monotonic()))
            self._threads[1].join(timeout=max(0, self._deadline - time.monotonic()))
            if self._threads[1].is_alive():
                raise subprocess.TimeoutExpired(self.command, self.timeout_s)
            return status
        try:
            return self._call(finish_io)
        finally:
            self.abort()

    def abort(self) -> None:
        """Idempotently stop the owned process and release its ownership record."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()
            if self._child is not None:
                try:
                    _REGISTRY.terminate(self._child)
                finally:
                    _REGISTRY.unregister(self._child)
            self._tasks.put(None)
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=0.1)
        if self._child is not None:
            for pipe in (self._child.process.stdin, self._child.process.stdout,
                         self._child.process.stderr):
                if pipe is not None:
                    pipe.close()

    def __enter__(self) -> OwnedBinaryProcess:  # noqa: PYI034 (Python 3.10 compatibility)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.abort()
