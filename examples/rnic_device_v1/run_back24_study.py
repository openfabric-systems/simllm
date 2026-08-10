"""Run the frozen BACK-24 RNIC device rejection study."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(
    "/data3/yifeng/simllm-dev/wave2-runs/"
    "codex/back8_session_records/back24"
)
INVALID_TERMINALS = (
    "unknown_token",
    "duplicate_token",
    "cross_wqe",
)
FUTURE_EVENT_TIMES_PS = (110, 1010)
CONTINUATION_TIME_PS = 20
EXPECTED_EXCEPTIONS = {
    "unknown_token": (
        "std::logic_error",
        "unknown or duplicate RNIC network token",
    ),
    "duplicate_token": (
        "std::logic_error",
        "unknown or duplicate RNIC network token",
    ),
    "cross_wqe": (
        "std::logic_error",
        "RNIC network token/WQE mismatch",
    ),
}
SNAPSHOT_SURFACES = (
    "wqe_records",
    "counters",
    "evidence",
    "port_ledger",
    "next_event_time",
    "pending_physical_work",
    "fatal",
    "occupied_sq_entries",
    "completion_queue_depth",
    "unpublished_wqe_count",
    "pcie_state",
)
FROZEN_ARTIFACT_DIGESTS = {
    "results.csv": (
        "7a0b8423d0a99de9538047f307bb7fd2f20c8d19bd408ef90fe02199da868934"
    ),
    "native_tests.csv": (
        "969963477314bfb723770556a02e4f038c7220820d522ae60dfa8c80744a202d"
    ),
}


def _validate_registry(out: Path) -> None:
    cells = {
        (terminal, event_time_ps)
        for terminal in INVALID_TERMINALS
        for event_time_ps in FUTURE_EVENT_TIMES_PS
    }
    if len(cells) != 6:
        raise AssertionError("BACK-24 registry must contain six unique cells")
    if set(EXPECTED_EXCEPTIONS) != set(INVALID_TERMINALS):
        raise AssertionError("every invalid terminal needs an exception identity")
    if len(SNAPSHOT_SURFACES) != 11 or len(set(SNAPSHOT_SURFACES)) != 11:
        raise AssertionError("BACK-24 snapshot inventory drifted")
    if set(FROZEN_ARTIFACT_DIGESTS) != {"results.csv", "native_tests.csv"}:
        raise AssertionError("BACK-24 frozen artifact inventory drifted")
    if CONTINUATION_TIME_PS != 20:
        raise AssertionError("BACK-24 continuation time drifted")
    for name, expected_digest in FROZEN_ARTIFACT_DIGESTS.items():
        artifact = Path(__file__).with_name(name)
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise AssertionError(
                f"frozen artifact digest drifted for {name}: {digest}"
            )
    data_root = Path("/data3/yifeng").resolve()
    try:
        out.resolve().relative_to(data_root)
    except ValueError as error:
        raise ValueError("study output must remain under /data3/yifeng") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the frozen registry without creating outputs",
    )
    arguments = parser.parse_args()
    _validate_registry(arguments.out)
    if arguments.check_only:
        print("BACK-24 study registry check passed; no results produced")
        return
    raise RuntimeError("BACK-24 result mode lands only after the expectation freeze")


if __name__ == "__main__":
    main()
