import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from simllm.backends import _child_process
from simllm.backends._child_process import (
    OwnedBinaryProcess,
    _windows_job_diagnostics_for_test,
    _WindowsJob,
    cleanup_owned_children,
    run_owned_process,
)
from simllm.backends.step_sink import HtsimPersistentStepSink, HtsimStepSinkConfig
from simllm.compute import ModelDims


def _pid_is_live(pid: int) -> bool:
    if os.name == "posix":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        kernel32.GetExitCodeProcess.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        try:
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return exit_code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    raise RuntimeError(f"unsupported test platform: {sys.platform}")


def _wait_until_not_live(pid: int, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while _pid_is_live(pid):
        if time.monotonic() >= deadline:
            pytest.fail(f"process {pid} remained live after {timeout_s} seconds")
        time.sleep(0.02)


def _wait_for_marker(directory: Path, timeout_s: float = 5.0) -> dict[str, object]:
    """Wait for exactly one marker that also parses as complete JSON.

    A marker becomes visible in the directory before its contents are
    written, so treating the glob hit as readiness can read an empty or
    partial file. Keep polling until it parses.
    """

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        markers = list(directory.glob("*.json"))
        if len(markers) == 1:
            try:
                return json.loads(markers[0].read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        time.sleep(0.02)
    pytest.fail("owned child marker was not published")


def _wait_for_pid_file(path: Path, timeout_s: float = 5.0) -> int:
    """Wait for the PID file to exist AND to hold a complete integer.

    Creating the file and writing its contents are two steps, so a reader
    that treats existence as readiness can observe it empty or partial. The
    window is wide enough on Windows to fail a run, so keep polling until
    the content parses instead of trusting the first sighting.
    """

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                return int(path.read_text(encoding="ascii").strip())
            except ValueError:
                pass
        time.sleep(0.02)
    pytest.fail(f"target PID file was not published: {path}")


def _windows_failure_context(
    completed: subprocess.CompletedProcess[str] | None,
    target_pid_file: Path,
) -> str:
    target_pid = (
        target_pid_file.read_text(encoding="ascii")
        if target_pid_file.is_file()
        else None
    )
    return json.dumps(
        {
            "completed_returncode": (
                completed.returncode if completed is not None else None
            ),
            "job": _windows_job_diagnostics_for_test(),
            "target_pid": target_pid,
        },
        sort_keys=True,
    )


def test_owned_process_preserves_captured_output_and_status(tmp_path):
    target_pid_file = tmp_path / "normal-target.pid"
    code = (
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "marker = os.environ.get('SIMLLM_TEST_TARGET_PID_FILE')\n"
        "if marker:\n"
        "    Path(marker).write_text(str(os.getpid()), encoding='ascii')\n"
        "sys.stdout.write('stdout line 1\\nstdout line 2')\n"
        "sys.stderr.write('stderr line')\n"
        "raise SystemExit(7)\n"
    )
    command = [sys.executable, "-c", code]
    direct_environment = os.environ.copy()
    direct_environment.pop("SIMLLM_TEST_TARGET_PID_FILE", None)
    owned_environment = direct_environment.copy()
    owned_environment["SIMLLM_TEST_TARGET_PID_FILE"] = str(target_pid_file)
    direct = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=direct_environment,
    )
    owned = run_owned_process(
        command,
        timeout_s=5.0,
        environment=owned_environment,
    )
    context = _windows_failure_context(owned, target_pid_file)
    if os.name == "nt":
        print(f"WINDOWS_CHILD_DIAGNOSTICS={context}")

    _wait_for_pid_file(target_pid_file)
    assert owned.returncode == direct.returncode == 7, context
    assert owned.stdout == direct.stdout, context
    assert owned.stderr == direct.stderr, context
    if os.name == "nt":
        diagnostics = _windows_job_diagnostics_for_test()
        assert diagnostics["assign_result"] is True, context
        assert diagnostics["launcher_owned_job_after_assign"] is True, context
        assert diagnostics["handle_open_after_assign"] is True, context
        assert diagnostics["handle_open_before_communicate"] is True, context
        assert diagnostics["handle_open_after_communicate"] is True, context
        assert diagnostics["handle_open_before_close"] is True, context
        assert diagnostics["handle_open_after_close"] is False, context
    cleanup_owned_children()
    cleanup_owned_children()


def test_timeout_terminates_reaps_and_allows_repeat_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMLLM_CHILD_LIFETIME_MARKER_DIR", str(tmp_path))
    monkeypatch.setenv("SIMLLM_CHILD_LIFETIME_RUN_NONCE", "timeout-control")
    target_pid_file = tmp_path / "timeout-target.pid"
    code = (
        "import os\n"
        "import time\n"
        "from pathlib import Path\n"
        f"Path({str(target_pid_file)!r}).write_text("
        "str(os.getpid()), encoding='ascii')\n"
        "time.sleep(30)\n"
    )
    command = [sys.executable, "-c", code]

    completed = None
    try:
        completed = run_owned_process(command, timeout_s=1.0)
    except subprocess.TimeoutExpired:
        pass
    context = _windows_failure_context(completed, target_pid_file)
    if os.name == "nt":
        print(f"WINDOWS_CHILD_DIAGNOSTICS={context}")
    if completed is not None:
        pytest.fail(f"owned process returned before timeout: {context}")

    marker = _wait_for_marker(tmp_path)
    child_pid = int(marker["child_pid"])
    target_pid = _wait_for_pid_file(target_pid_file)
    _wait_until_not_live(child_pid)
    _wait_until_not_live(target_pid)
    if os.name == "nt":
        diagnostics = _windows_job_diagnostics_for_test()
        assert diagnostics["assign_result"] is True, context
        assert diagnostics["launcher_owned_job_after_assign"] is True, context
        assert diagnostics["handle_open_timeout_observed"] is True, context
        assert diagnostics["terminate_result"] is True, context
        assert diagnostics["handle_open_before_close"] is True, context
        assert diagnostics["handle_open_after_close"] is False, context
    cleanup_owned_children()
    cleanup_owned_children()


def test_persistent_sink_close_is_idempotent(tmp_path):
    sink = HtsimPersistentStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=ModelDims(
                num_layers=1,
                hidden_size=64,
                intermediate_size=128,
                num_heads=1,
                num_kv_heads=1,
                head_size=64,
                vocab_size=256,
                dtype_bytes=2,
            ),
            workdir=tmp_path,
        ),
        max_workers=1,
    )

    sink.close()
    sink.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object control")
def test_windows_job_creation_failure_does_not_release_child(tmp_path, monkeypatch):
    executed = tmp_path / "executed"

    def fail_job_creation():
        raise OSError("injected Job Object failure")

    monkeypatch.setattr(_WindowsJob, "create", staticmethod(fail_job_creation))
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(executed)!r}).touch()",
    ]
    with pytest.raises(OSError, match="injected Job Object failure"):
        run_owned_process(command, timeout_s=5.0)
    assert not executed.exists()


@pytest.mark.skipif(os.name not in {"nt", "posix"}, reason="unsupported platform")
def test_owner_termination_kills_only_the_registered_child(tmp_path):
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    target_pid_file = tmp_path / "owner-kill-target.pid"
    environment = os.environ.copy()
    environment["SIMLLM_CHILD_LIFETIME_MARKER_DIR"] = str(marker_dir)
    environment["SIMLLM_CHILD_LIFETIME_RUN_NONCE"] = "owner-kill-control"
    sleep_code = "import time; time.sleep(30)"
    child_code = (
        "import os, time; "
        "from pathlib import Path; "
        f"Path({str(target_pid_file)!r}).write_text("
        "str(os.getpid()), encoding='ascii'); "
        "time.sleep(30)"
    )
    owner_code = (
        "import sys; "
        "from simllm.backends._child_process import run_owned_process; "
        "run_owned_process([sys.executable, '-c', "
        + repr(child_code)
        + "], timeout_s=60.0)"
    )
    owner = subprocess.Popen([sys.executable, "-c", owner_code], env=environment)
    sentinel = subprocess.Popen([sys.executable, "-c", sleep_code])
    try:
        marker = _wait_for_marker(marker_dir)
        child_pid = int(marker["child_pid"])
        target_pid = _wait_for_pid_file(target_pid_file)
        assert int(marker["owner_pid"]) == owner.pid
        if os.name == "posix":
            os.kill(owner.pid, signal.SIGTERM)
        else:
            owner.terminate()
        owner.wait(timeout=5.0)
        _wait_until_not_live(child_pid)
        _wait_until_not_live(target_pid)
        assert sentinel.poll() is None
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5.0)
        if sentinel.poll() is None:
            sentinel.terminate()
            try:
                sentinel.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                sentinel.kill()
                sentinel.wait(timeout=5.0)


def test_owned_binary_stream_preserves_immediate_first_frame_and_stderr():
    code = (
        "import os, sys\n"
        "for size in (65537, 5):\n"
        "    data = bytearray()\n"
        "    while len(data) < size:\n"
        "        data.extend(os.read(0, size - len(data)))\n"
        "    os.write(2, b'diagnostic' * 20000)\n"
        "    sys.stdout.buffer.write(data)\n"
        "    sys.stdout.buffer.flush()\n"
        "assert os.read(0, 1) == b''\n"
    )
    process = OwnedBinaryProcess((sys.executable, "-c", code), timeout_s=10)
    with process:
        first = b"\x00\xff" * 32768 + b"x"
        process.write(first)
        assert process.read_exact(len(first)) == first
        process.write(b"again")
        assert process.read_exact(5) == b"again"
        assert process.finish() == 0
    assert process.stderr == b"diagnostic" * 40000
    _wait_until_not_live(process.pid)
    process.abort()


def test_binary_stream_eof_reaps_child_and_preserves_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMLLM_CHILD_LIFETIME_MARKER_DIR", str(tmp_path))
    monkeypatch.setenv("SIMLLM_CHILD_LIFETIME_RUN_NONCE", "stream-eof")
    with OwnedBinaryProcess((sys.executable, "-c", "print('x', end='')"),
                            timeout_s=5) as process, pytest.raises(EOFError):
        process.read_exact(4)
    marker = _wait_for_marker(tmp_path)
    assert marker["child_pid"] == process.pid
    _wait_until_not_live(process.pid)


def test_binary_stream_watchdog_applies_while_caller_is_idle():
    process = OwnedBinaryProcess(
        (sys.executable, "-c", "import time; time.sleep(30)"), timeout_s=0.3)
    time.sleep(0.5)
    with pytest.raises(subprocess.TimeoutExpired):
        process.read_exact(1)
    _wait_until_not_live(process.pid)


def test_binary_stream_blocked_write_is_bounded():
    with OwnedBinaryProcess(
        (sys.executable, "-c", "import time; time.sleep(30)"), timeout_s=0.3
    ) as process, pytest.raises(subprocess.TimeoutExpired):
        process.write(b"x" * (1 << 20))
    _wait_until_not_live(process.pid)


def test_binary_stream_rejects_trailing_output():
    code = "import os; os.read(0, 1); os.write(1, b'extra')"
    with (OwnedBinaryProcess((sys.executable, "-c", code), timeout_s=5) as process,
          pytest.raises(RuntimeError, match="trailing protocol bytes")):
        process.finish()
    _wait_until_not_live(process.pid)


@pytest.mark.parametrize("allowance", [1, 2])
@pytest.mark.parametrize("idle", [3, 6])
def test_exchange_budget_is_fixed_and_excludes_prior_client_idle(monkeypatch, allowance, idle):
    now, aborted = [0], []
    process = OwnedBinaryProcess.__new__(OwnedBinaryProcess)
    process._deadline, process._io_deadline, process._io_owner = 30, None, None
    process._active_io_calls = 0
    process._closed, process._failure = False, None
    process._lock, process.command = threading.RLock(), ("controlled-clock",)
    process.abort = lambda: aborted.append(True)
    monkeypatch.setattr(_child_process, "time", SimpleNamespace(monotonic=lambda: now[0]))
    now[0] = idle
    with process.io_deadline(allowance):
        assert process._io_deadline == idle + allowance
        now[0] += allowance / 4
        deadline = process._io_deadline
        with pytest.raises(RuntimeError, match="overlap or nest"), process.io_deadline(10):
            pytest.fail("nested exchange was entered")
        assert process._io_deadline == deadline
        now[0] += allowance / 4
    assert process._io_deadline is None and process._io_owner is None and not aborted
    now[0] = 29.5
    with process.io_deadline(allowance):
        assert process._io_deadline == 30
    assert process._deadline == 30


def test_exchange_deadline_is_cumulative_across_multiple_reads_and_reaps():
    code = "import os,time; os.write(1,b'R'); os.read(0,1); time.sleep(.2); os.write(1,b'A'); time.sleep(.6); os.write(1,b'B')"
    with OwnedBinaryProcess((sys.executable, "-c", code), timeout_s=10) as process:
        assert process.read_exact(1) == b"R"
        with pytest.raises(subprocess.TimeoutExpired), process.io_deadline(.5):
            process.write(b"x")
            assert process.read_exact(1) == b"A"
            process.read_exact(1)
        _wait_until_not_live(process.pid)
        with pytest.raises(RuntimeError, match="closed"), process.io_deadline(1):
            pytest.fail("timed-out exchange was retried")


def test_exchange_rejects_other_threads_without_aborting_the_owner():
    code = "import os; os.write(1,b'R'); os.write(1,os.read(0,1)); assert os.read(0,1)==b''"
    with OwnedBinaryProcess((sys.executable, "-c", code), timeout_s=10) as process:
        assert process.read_exact(1) == b"R"
        errors = []

        def intrude():
            for action in (lambda: process.write(b"wrong"), lambda: process.io_deadline(1).__enter__()):
                try:
                    action()
                except RuntimeError as error:
                    errors.append(str(error))

        with process.io_deadline(2):
            thread = threading.Thread(target=intrude)
            thread.start()
            thread.join(timeout=1)
            assert not thread.is_alive() and len(errors) == 2
            process.write(b"x")
            assert process.read_exact(1) == b"x"
        with process.io_deadline(2):
            assert process.finish() == 0


def test_exchange_error_survives_abort_failure(monkeypatch):
    process = OwnedBinaryProcess((sys.executable, "-c", "import time; time.sleep(30)"), timeout_s=10)
    abort, original = process.abort, ValueError("first error")

    def fail_abort():
        abort()
        raise OSError("cleanup error")

    monkeypatch.setattr(process, "abort", fail_abort)
    with pytest.raises(ValueError) as error, process.io_deadline(1):
        raise original
    assert error.value is original and str(process._cleanup_failure) == "cleanup error"
    _wait_until_not_live(process.pid)


def test_exchange_cannot_begin_over_an_unscoped_inflight_read():
    code = "import os,time; os.write(1,b'R'); time.sleep(.5); os.write(1,b'X'); assert os.read(0,1)==b''"
    with OwnedBinaryProcess((sys.executable, "-c", code), timeout_s=10) as process:
        assert process.read_exact(1) == b"R"
        received = []
        thread = threading.Thread(target=lambda: received.append(process.read_exact(1)))
        thread.start()
        deadline = time.monotonic() + 1
        while not process._active_io_calls and time.monotonic() < deadline:
            time.sleep(.001)
        assert process._active_io_calls == 1
        with pytest.raises(RuntimeError, match="overlap or nest"), process.io_deadline(1):
            pytest.fail("inflight I/O was given a replacement deadline")
        thread.join(timeout=2)
        assert not thread.is_alive() and received == [b"X"]
        assert process.finish() == 0


def test_exchange_allowance_cannot_extend_the_child_lifetime():
    with OwnedBinaryProcess((sys.executable, "-c", "import time; time.sleep(30)"), timeout_s=.3) as process:
        with pytest.raises(subprocess.TimeoutExpired), process.io_deadline(10):
            process.read_exact(1)
        _wait_until_not_live(process.pid)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object control")
def test_binary_stream_keeps_windows_job_open_until_finish():
    code = "import os; os.write(1, os.read(0, 1)); assert os.read(0, 1) == b''"
    with OwnedBinaryProcess((sys.executable, "-c", code), timeout_s=5) as process:
        process.write(b"x")
        assert process.read_exact(1) == b"x"
        diagnostics = _windows_job_diagnostics_for_test()
        assert diagnostics["assign_result"] is True
        assert diagnostics["handle_open_after_assign"] is True
        assert process.finish() == 0
    assert _windows_job_diagnostics_for_test()["handle_open_after_close"] is False


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object control")
def test_binary_stream_job_creation_failure_never_releases_child(tmp_path, monkeypatch):
    executed = tmp_path / "stream-executed"

    def fail_job_creation():
        raise OSError("injected Job Object failure")

    monkeypatch.setattr(_WindowsJob, "create", staticmethod(fail_job_creation))
    with pytest.raises(OSError, match="injected Job Object failure"):
        OwnedBinaryProcess((sys.executable, "-c",
                            f"from pathlib import Path; Path({str(executed)!r}).touch()"),
                           timeout_s=5)
    assert not executed.exists()
