"""Minimal handshake launcher for one lifetime-owned simulator process."""

from __future__ import annotations

import ctypes
import os
import signal
import sys

_PR_SET_PDEATHSIG = 1
_HANDSHAKE = b"G"


def _arm_linux_parent_death(expected_parent_pid: int) -> bool:
    """Arm SIGKILL on parent death and close the pre-arm race."""

    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = (
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    prctl.restype = ctypes.c_int
    if prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return os.getppid() == expected_parent_pid


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        print("owned child launcher: expected PARENT_PID -- COMMAND...", file=sys.stderr)
        return 125
    try:
        expected_parent_pid = int(sys.argv[1])
    except ValueError:
        print("owned child launcher: parent PID must be an integer", file=sys.stderr)
        return 125
    command = sys.argv[3:]
    if expected_parent_pid <= 0 or not command:
        print("owned child launcher: invalid parent or empty command", file=sys.stderr)
        return 125

    try:
        if sys.platform.startswith("linux") and not _arm_linux_parent_death(
            expected_parent_pid
        ):
            return 125
        if sys.stdin.buffer.read(1) != _HANDSHAKE:
            return 125
        if os.getppid() != expected_parent_pid:
            return 125
        os.execvpe(command[0], command, os.environ)
    except OSError as error:
        print(f"owned child launcher: {error}", file=sys.stderr)
        return 126
    return 126


if __name__ == "__main__":
    raise SystemExit(main())
