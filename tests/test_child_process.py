import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from simllm.backends._child_process import (
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
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        markers = list(directory.glob("*.json"))
        if len(markers) == 1:
            return json.loads(markers[0].read_text(encoding="utf-8"))
        time.sleep(0.02)
    pytest.fail("owned child marker was not published")


def _wait_for_pid_file(path: Path, timeout_s: float = 5.0) -> int:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            return int(path.read_text(encoding="ascii"))
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
