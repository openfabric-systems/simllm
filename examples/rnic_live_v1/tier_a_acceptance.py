"""Run the frozen Tier A checker over raw native observations."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_EXPECTATIONS = STUDY_DIR / "tier_a_expectations.json"
RUN_ROOT = Path("/data3/yifeng/simllm-dev/wave2-runs")


class AcceptanceError(RuntimeError):
    """Raised when raw observations violate the frozen gate."""


@dataclass(frozen=True)
class CheckSummary:
    factory: str
    scored_family_instances: dict[str, int]
    exact_oracle_rows: int
    fatal_invariant_families: dict[str, bool]
    wrapper_bypass_rejected: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "simllm-rnic-tier-a-summary-v1",
            "factory": self.factory,
            "scored_family_instances": self.scored_family_instances,
            "exact_oracle_rows": self.exact_oracle_rows,
            "fatal_invariant_families": self.fatal_invariant_families,
            "wrapper_bypass_rejected": self.wrapper_bypass_rejected,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceError(message)


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    _require(isinstance(value, list), f"{name} must be an array")
    return value


def _require_int(value: Any, name: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{name} must be an integer",
    )
    return value


def _require_keys(value: dict[str, Any], keys: list[str], name: str) -> None:
    missing = sorted(set(keys) - value.keys())
    _require(not missing, f"{name} is missing keys: {missing}")


def _require_exact_keys(
    value: dict[str, Any], keys: list[str], name: str
) -> None:
    expected = set(keys)
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    _require(
        not missing and not unexpected,
        f"{name} keys differ: missing={missing}, unexpected={unexpected}",
    )


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AcceptanceError(f"cannot parse {name} {path}: {error}") from error
    return _require_dict(value, name)


def _expected_service_ps(payload_bytes: int, link_rate_gbps: int) -> int:
    numerator = payload_bytes * 8 * 1000
    _require(link_rate_gbps > 0, "link rate must be positive")
    _require(
        numerator % link_rate_gbps == 0,
        "frozen wire service must be an exact integer number of ps",
    )
    return numerator // link_rate_gbps


def _expected_grid(
    payloads: list[int], rates: list[int], doorbells: list[int]
) -> set[tuple[int, int, int]]:
    return set(product(payloads, rates, doorbells))


def _validate_expectations(expectations: dict[str, Any]) -> None:
    _require(
        expectations.get("schema") == "simllm-rnic-tier-a-expectations-v1",
        "unsupported Tier A expectations schema",
    )
    _require(
        expectations.get("observation_schema")
        == "simllm-rnic-tier-a-observations-v1",
        "unsupported Tier A observation schema",
    )
    _require(
        expectations.get("factories") == ["fake", "htsim"],
        "factory list must preserve the frozen fake-to-htsim seam",
    )

    single = _require_dict(expectations.get("single_wqe"), "single_wqe")
    _require(
        single
        == {
            "payload_bytes": [4096, 1048576],
            "link_rate_gbps": [200, 400],
            "doorbell_service_ps": [0, 1000],
            "wqe_count": 1,
            "port_capacity": 1,
        },
        "single-WQE matrix differs from the frozen grid",
    )
    fifo = _require_dict(expectations.get("fifo"), "fifo")
    _require(
        fifo
        == {
            "payload_bytes": 4096,
            "link_rate_gbps": [200, 400],
            "doorbell_service_ps": [0, 1000],
            "wqe_count": 2,
            "doorbell_count": 1,
            "port_capacity": 1,
        },
        "FIFO matrix differs from the frozen grid",
    )
    _require(
        expectations.get("network_fixture")
        == {
            "data_header_bytes": 0,
            "propagation_delay_ps": 0,
            "control_frames": False,
            "congestion": False,
        },
        "network fixture differs from the frozen exact model",
    )
    additivity = _require_dict(
        expectations.get("d_additivity"), "d_additivity"
    )
    _require(additivity.get("delta_ps") == 1000, "D delta must be 1000 ps")
    _require(
        additivity.get("fields")
        == [
            "eligible_at_ps",
            "port_tx_at_ps",
            "terminal_at_ps",
            "cqe_visible_at_ps",
            "polled_at_ps",
        ],
        "D-additivity field list differs from the frozen contract",
    )
    families = _require_dict(
        expectations.get("behavioral_families"), "behavioral_families"
    )
    _require(
        families
        == {
            "d_additivity": 4,
            "inverse_rate_serialization": 4,
            "two_wqe_fifo": 4,
        },
        "behavioral family counts differ from the frozen contract",
    )
    _require(
        expectations.get("controlled_drop")
        == {
            "payload_bytes": 4096,
            "link_rate_gbps": 400,
            "doorbell_service_ps": 0,
            "signaled": False,
            "drop_location": "fabric",
            "drop_reason": "injected",
        },
        "controlled-drop fixture differs from the frozen contract",
    )
    _require(
        expectations.get("wrapper_bypass_control")
        == {
            "payload_bytes": 4096,
            "link_rate_gbps": 400,
            "doorbell_service_ps": [0, 1000],
        },
        "wrapper-bypass fixture differs from the frozen contract",
    )
    _require(
        expectations.get("terminal_controls")
        == {
            "payload_bytes": 4096,
            "link_rate_gbps": 400,
            "doorbell_service_ps": 0,
            "wqe_count": 2,
            "port_capacity": 2,
            "kinds": ["duplicate", "unknown", "cross_wqe"],
            "invalid_event_time_ps": 200000,
            "clock_probe_time_ps": 150000,
            "exception_type": "std::invalid_argument",
            "exception_messages": {
                "duplicate": "terminal token already consumed",
                "unknown": "terminal token was never issued",
                "cross_wqe": "terminal token does not belong to WQE",
            },
            "clock_probe_exception_type": "",
            "clock_probe_changes": 0,
            "control_keys": [
                "kind",
                "invalid_event_time_ps",
                "exception_type",
                "exception_message",
                "clock_probe_time_ps",
                "clock_probe_exception_type",
                "clock_probe_changes",
                "before",
                "after",
            ],
            "snapshot_required_keys": [
                "caller_time_ps",
                "device_records",
                "device_counters",
                "device_evidence",
                "port_issued",
                "port_terminals",
                "port_live_tokens",
                "occupied_sq_entries",
                "completion_queue_depth",
                "unpublished_wqes",
                "has_pending_physical_work",
            ],
            "snapshot_record_keys": [
                "wqe_id",
                "state",
                "network_token",
                "network_accepted_at_ps",
                "network_outcome_at_ps",
                "completion_status",
            ],
        },
        "terminal-control schema differs from the frozen contract",
    )
    _require(
        expectations.get("authority_control")
        == {
            "exception_type": "std::invalid_argument",
            "exception_message": (
                "structural and bypass authorities are mutually exclusive"
            ),
            "attempt_keys": [
                "exception_type",
                "exception_message",
                "before",
                "after",
            ],
            "snapshot_keys": [
                "native_session_constructed",
                "native_posts",
                "legacy_ledger_constructed",
                "legacy_posts",
                "legacy_mutations",
            ],
        },
        "authority-control schema differs from the frozen contract",
    )
    _require(
        expectations.get("raw_observation_keys")
        == [
            "schema",
            "factory",
            "single_wqe",
            "fifo",
            "wrapper_bypass_control",
            "authority_cases",
            "terminal_controls",
            "controlled_drop",
        ],
        "raw observation schema differs from the frozen contract",
    )
    _require(
        expectations.get("raw_cell_keys")
        == [
            "payload_bytes",
            "link_rate_gbps",
            "doorbell_service_ps",
            "authority",
            "device",
            "port",
            "wqes",
            "cqe_order",
            "jct_ps",
        ],
        "raw cell schema differs from the frozen contract",
    )
    _require(
        expectations.get("raw_drop_cell_keys")
        == [
            "payload_bytes",
            "link_rate_gbps",
            "doorbell_service_ps",
            "authority",
            "device",
            "port",
            "wqes",
            "cqe_order",
            "jct_ps",
            "signaled",
            "all_cqe_statuses",
            "evidence",
        ],
        "raw drop-cell schema differs from the frozen contract",
    )
    _require(
        expectations.get("raw_wqe_keys")
        == [
            "ordinal",
            "wqe_id",
            "eligible_at_ps",
            "network_accepted_at_ps",
            "port_tx_at_ps",
            "terminal_kind",
            "terminal_at_ps",
            "cqe_status",
            "cqe_visible_at_ps",
            "polled_at_ps",
        ],
        "raw WQE schema differs from the frozen contract",
    )
    _require(
        expectations.get("raw_device_keys")
        == [
            "counters",
            "has_pending_physical_work",
            "occupied_sq_entries",
            "completion_queue_depth",
            "unpublished_wqes",
            "fatal",
        ],
        "raw device schema differs from the frozen contract",
    )
    _require(
        expectations.get("raw_counter_keys")
        == [
            "posted_wqes",
            "network_accepted",
            "network_delivered",
            "network_dropped",
        ],
        "raw counter schema differs from the frozen contract",
    )
    _require(
        expectations.get("raw_port_keys")
        == ["issued", "terminals", "live_tokens"],
        "raw port schema differs from the frozen contract",
    )
    _require(
        expectations.get("raw_issued_keys")
        == [
            "token",
            "wqe_id",
            "accepted_at_ps",
            "port_tx_at_ps",
            "payload_bytes",
        ],
        "raw issued-token schema differs from the frozen contract",
    )
    _require(
        expectations.get("raw_terminal_keys")
        == ["token", "wqe_id", "kind", "at_ps"],
        "raw terminal schema differs from the frozen contract",
    )
    for payload, rate in product(
        single["payload_bytes"], single["link_rate_gbps"]
    ):
        _expected_service_ps(payload, rate)


def _validate_run_path(path: Path, name: str) -> Path:
    _require(path.is_absolute(), f"{name} must be absolute")
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(RUN_ROOT)
    except ValueError as error:
        raise AcceptanceError(
            f"{name} must be under the external wave-2 run root {RUN_ROOT}"
        ) from error
    return resolved


def _cell_key(cell: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(cell["payload_bytes"]),
        int(cell["link_rate_gbps"]),
        int(cell["doorbell_service_ps"]),
    )


def _validate_authority(
    authority: dict[str, Any], native_posts: int, name: str
) -> None:
    expected = {
        "mode": "structural",
        "native_session_constructed": 1,
        "native_posts": native_posts,
        "legacy_ledger_constructed": 0,
        "legacy_posts": 0,
        "legacy_mutations": 0,
    }
    for key in expected:
        if key != "mode":
            _require_int(authority.get(key), f"{name}.authority.{key}")
    _require(authority == expected, f"{name} violates structural authority")


def _validate_token_ledger(
    cell: dict[str, Any], expectations: dict[str, Any], name: str
) -> None:
    port = _require_dict(cell["port"], f"{name}.port")
    _require_exact_keys(
        port, expectations["raw_port_keys"], f"{name}.port"
    )
    issued = _require_list(port.get("issued"), f"{name}.port.issued")
    terminals = _require_list(port.get("terminals"), f"{name}.port.terminals")
    live = _require_list(port.get("live_tokens"), f"{name}.port.live_tokens")

    issued_rows = [_require_dict(row, "issued token") for row in issued]
    for row in issued_rows:
        _require_exact_keys(
            row, expectations["raw_issued_keys"], "issued token"
        )
    issued_tokens = [int(row["token"]) for row in issued_rows]
    terminal_rows = [_require_dict(row, "terminal") for row in terminals]
    for row in terminal_rows:
        _require_exact_keys(
            row, expectations["raw_terminal_keys"], "terminal"
        )
    terminal_tokens = [int(row["token"]) for row in terminal_rows]
    live_tokens = [int(token) for token in live]
    _require(all(token > 0 for token in issued_tokens), f"{name} issued token zero")
    _require(
        len(issued_tokens) == len(set(issued_tokens)),
        f"{name} recycled an issued token",
    )
    _require(
        not (set(terminal_tokens) | set(live_tokens)) - set(issued_tokens),
        f"{name} terminal or live token was never issued",
    )
    _require(
        not set(terminal_tokens) & set(live_tokens),
        f"{name} token is both terminal and live",
    )
    terminal_counts = Counter(terminal_tokens)
    _require(
        all(count == 1 for count in terminal_counts.values()),
        f"{name} token has more than one terminal",
    )
    _require(
        len(issued_tokens) == len(terminals) + len(live_tokens),
        f"{name} violates issued = terminal + live",
    )
    issued_wqes = {
        int(row["token"]): int(row["wqe_id"])
        for row in issued_rows
    }
    cell_wqe_rows = [
        _require_dict(row, "cell WQE")
        for row in _require_list(cell["wqes"], f"{name}.wqes")
    ]
    cell_wqes = {int(row["wqe_id"]): row for row in cell_wqe_rows}
    _require(
        len(cell_wqes) == len(cell_wqe_rows),
        f"{name} repeats a native WQE id",
    )
    _require(
        len(issued_rows) == len(cell_wqes)
        and Counter(issued_wqes.values()) == Counter(cell_wqes),
        f"{name} must issue exactly one token for each frozen WQE",
    )
    _require(
        len(terminal_rows) == len(cell_wqes)
        and Counter(int(row["wqe_id"]) for row in terminal_rows)
        == Counter(cell_wqes),
        f"{name} must terminate exactly one token for each frozen WQE",
    )
    for issued_row in issued_rows:
        wqe = cell_wqes[int(issued_row["wqe_id"])]
        _require(
            issued_row["accepted_at_ps"] == wqe["network_accepted_at_ps"]
            and issued_row["port_tx_at_ps"] == wqe["port_tx_at_ps"]
            and issued_row["payload_bytes"] == cell["payload_bytes"],
            f"{name} port issue and native WQE projections disagree",
        )
    for terminal in terminal_rows:
        _require(
            terminal.get("kind") in {"delivered", "dropped"},
            f"{name} has a nonterminal terminal kind",
        )
        token = int(terminal["token"])
        _require(
            int(terminal["wqe_id"]) == issued_wqes[token],
            f"{name} terminal crossed WQE identity",
        )
        wqe = cell_wqes[int(terminal["wqe_id"])]
        _require(
            terminal.get("kind") == wqe["terminal_kind"]
            and terminal.get("at_ps") == wqe["terminal_at_ps"],
            f"{name} port and native terminal projections disagree",
        )

    device = _require_dict(cell["device"], f"{name}.device")
    _require_exact_keys(
        device, expectations["raw_device_keys"], f"{name}.device"
    )
    counters = _require_dict(device.get("counters"), f"{name}.device.counters")
    _require_exact_keys(
        counters,
        expectations["raw_counter_keys"],
        f"{name}.device.counters",
    )
    delivered = sum(row["kind"] == "delivered" for row in terminal_rows)
    dropped = sum(row["kind"] == "dropped" for row in terminal_rows)
    _require(
        counters.get("network_accepted") == len(issued_tokens),
        f"{name} device accepted count disagrees with port",
    )
    _require(
        counters.get("posted_wqes") == len(cell_wqes),
        f"{name} device post count disagrees with WQEs",
    )
    _require(
        counters.get("network_delivered") == delivered,
        f"{name} device delivered count disagrees with port",
    )
    _require(
        counters.get("network_dropped") == dropped,
        f"{name} device dropped count disagrees with port",
    )
    _require(not live_tokens, f"{name} retained live tokens at quiescence")
    _require(
        device.get("has_pending_physical_work") is False
        and device.get("occupied_sq_entries") == 0
        and device.get("completion_queue_depth") == 0
        and device.get("unpublished_wqes") == 0
        and device.get("fatal") is False,
        f"{name} did not reach validated physical quiescence",
    )


def _validate_cell_shape(
    cell: dict[str, Any],
    expectations: dict[str, Any],
    wqe_count: int,
    name: str,
    *,
    key_schema: str = "raw_cell_keys",
) -> None:
    _require_exact_keys(
        cell,
        expectations[key_schema],
        name,
    )
    wqes = _require_list(cell["wqes"], f"{name}.wqes")
    _require(len(wqes) == wqe_count, f"{name} has the wrong WQE count")
    for index, value in enumerate(wqes):
        wqe = _require_dict(value, f"{name}.wqes[{index}]")
        _require_exact_keys(
            wqe,
            expectations["raw_wqe_keys"],
            f"{name}.wqes[{index}]",
        )
        _require(wqe["ordinal"] == index, f"{name} changed WQE ordinal order")
    _validate_authority(
        _require_dict(cell["authority"], f"{name}.authority"),
        wqe_count,
        name,
    )
    _validate_token_ledger(cell, expectations, name)


def _validate_single_exact(cell: dict[str, Any], name: str) -> None:
    payload, rate, doorbell = _cell_key(cell)
    service = _expected_service_ps(payload, rate)
    wqe = _require_dict(cell["wqes"][0], f"{name}.wqe")
    expected_terminal = doorbell + service
    _require(
        wqe["eligible_at_ps"] == doorbell,
        f"{name} eligibility differs from D",
    )
    _require(
        wqe["port_tx_at_ps"] == doorbell,
        f"{name} fake TX issue differs from D",
    )
    _require(
        wqe["network_accepted_at_ps"] == doorbell,
        f"{name} network acceptance differs from D",
    )
    _require(
        wqe["terminal_kind"] == "delivered"
        and wqe["terminal_at_ps"] == expected_terminal,
        f"{name} terminal differs from D + L",
    )
    _require(
        wqe["cqe_status"] == "success"
        and wqe["cqe_visible_at_ps"] == expected_terminal
        and wqe["polled_at_ps"] == expected_terminal,
        f"{name} CQE boundary differs from the terminal",
    )
    _require(
        cell["cqe_order"] == [wqe["wqe_id"]]
        and cell["jct_ps"] == expected_terminal,
        f"{name} completion order or JCT differs from the exact oracle",
    )


def _d_pair_errors(
    low: dict[str, Any], high: dict[str, Any], expectations: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    delta = expectations["d_additivity"]["delta_ps"]
    low_wqe = low["wqes"][0]
    high_wqe = high["wqes"][0]
    for field in expectations["d_additivity"]["fields"]:
        observed = high_wqe[field] - low_wqe[field]
        if observed != delta:
            errors.append(f"{field} delta {observed}, expected {delta}")
    jct_delta = high["jct_ps"] - low["jct_ps"]
    if jct_delta != delta:
        errors.append(f"jct_ps delta {jct_delta}, expected {delta}")
    low_service = low_wqe["terminal_at_ps"] - low_wqe["port_tx_at_ps"]
    high_service = high_wqe["terminal_at_ps"] - high_wqe["port_tx_at_ps"]
    if low_service != high_service:
        errors.append(
            f"network service changed from {low_service} to {high_service}"
        )
    return errors


def _validate_d_pair(
    low: dict[str, Any],
    high: dict[str, Any],
    expectations: dict[str, Any],
    name: str,
) -> None:
    errors = _d_pair_errors(low, high, expectations)
    _require(not errors, f"D-additivity {name}: {errors}")


def _validate_fifo_timing(cell: dict[str, Any], name: str) -> None:
    payload, rate, doorbell = _cell_key(cell)
    service = _expected_service_ps(payload, rate)
    w0 = _require_dict(cell["wqes"][0], f"{name}.w0")
    w1 = _require_dict(cell["wqes"][1], f"{name}.w1")
    _require(
        w0["eligible_at_ps"] == doorbell
        and w1["eligible_at_ps"] == doorbell,
        f"{name} FIFO eligibility differs from D",
    )
    _require(
        w0["port_tx_at_ps"] == doorbell
        and w0["terminal_at_ps"] == doorbell + service,
        f"{name} W0 violates the FIFO equation",
    )
    _require(
        w1["port_tx_at_ps"] == doorbell + service
        and w1["terminal_at_ps"] == doorbell + 2 * service,
        f"{name} W1 violates the FIFO equation",
    )
    _require(
        w1["port_tx_at_ps"] - w1["eligible_at_ps"] == service,
        f"{name} W1 wait differs from L",
    )
    _require(
        cell["jct_ps"] == doorbell + 2 * service,
        f"{name} JCT violates the FIFO timing relation",
    )


def _validate_fifo_structural(cell: dict[str, Any], name: str) -> None:
    w0 = _require_dict(cell["wqes"][0], f"{name}.w0")
    w1 = _require_dict(cell["wqes"][1], f"{name}.w1")
    for wqe in (w0, w1):
        _require(
            wqe["network_accepted_at_ps"] == wqe["port_tx_at_ps"],
            f"{name} fake acceptance and TX probe disagree",
        )
        _require(
            wqe["terminal_kind"] == "delivered"
            and wqe["cqe_status"] == "success"
            and wqe["cqe_visible_at_ps"] == wqe["terminal_at_ps"]
            and wqe["polled_at_ps"] == wqe["terminal_at_ps"],
            f"{name} terminal and CQE boundaries disagree",
        )
    _require(
        cell["cqe_order"] == [w0["wqe_id"], w1["wqe_id"]],
        f"{name} CQE order violates native ordered retirement",
    )


def _validate_authority_cases(
    observations: dict[str, Any], expectations: dict[str, Any]
) -> None:
    cases = _require_dict(observations.get("authority_cases"), "authority_cases")
    _require_exact_keys(
        cases,
        ["structural", "bypass", "dual_attempt"],
        "authority_cases",
    )
    structural = _require_dict(cases.get("structural"), "structural authority")
    bypass = _require_dict(cases.get("bypass"), "bypass authority")
    dual = _require_dict(cases.get("dual_attempt"), "dual authority attempt")
    frozen = expectations["authority_control"]
    _require_exact_keys(
        dual, frozen["attempt_keys"], "dual authority attempt"
    )
    for name, snapshot in (("structural", structural), ("bypass", bypass)):
        for key, value in snapshot.items():
            _require_int(value, f"{name} authority.{key}")
    _require(
        structural
        == {
            "native_session_constructed": 1,
            "native_posts": 1,
            "legacy_ledger_constructed": 0,
            "legacy_posts": 0,
            "legacy_mutations": 0,
        },
        "standalone structural authority case is not exclusive",
    )
    _require(
        bypass
        == {
            "native_session_constructed": 0,
            "native_posts": 0,
            "legacy_ledger_constructed": 1,
            "legacy_posts": 1,
            "legacy_mutations": 1,
        },
        "standalone bypass authority case is not exclusive",
    )
    before = _require_dict(dual["before"], "dual authority state before")
    after = _require_dict(dual["after"], "dual authority state after")
    _require_exact_keys(
        before, frozen["snapshot_keys"], "dual authority state before"
    )
    _require_exact_keys(
        after, frozen["snapshot_keys"], "dual authority state after"
    )
    for name, snapshot in (("before", before), ("after", after)):
        for key, value in snapshot.items():
            _require_int(value, f"dual authority {name}.{key}")
    zero_state = {key: 0 for key in frozen["snapshot_keys"]}
    _require(
        dual["exception_type"] == frozen["exception_type"]
        and dual["exception_message"] == frozen["exception_message"]
        and before == zero_state
        and after == before,
        "dual authority attempt did not throw before state mutation",
    )
def _validate_terminal_snapshot(
    snapshot: dict[str, Any], expectations: dict[str, Any], name: str
) -> None:
    frozen = expectations["terminal_controls"]
    _require_exact_keys(snapshot, frozen["snapshot_required_keys"], name)
    service = _expected_service_ps(
        frozen["payload_bytes"], frozen["link_rate_gbps"]
    )
    _require(
        _require_int(snapshot["caller_time_ps"], f"{name}.caller_time_ps")
        == service,
        f"{name} caller time differs from the completed control fixture",
    )

    records = [
        _require_dict(row, f"{name}.device_records[]")
        for row in _require_list(snapshot["device_records"], f"{name}.records")
    ]
    _require(
        len(records) == frozen["wqe_count"],
        f"{name} has the wrong record count",
    )
    record_ids: set[int] = set()
    record_tokens: dict[int, int] = {}
    for record in records:
        _require_exact_keys(
            record, frozen["snapshot_record_keys"], f"{name}.record"
        )
        wqe_id = _require_int(record["wqe_id"], f"{name}.record.wqe_id")
        token = _require_int(
            record["network_token"], f"{name}.record.network_token"
        )
        accepted_at = _require_int(
            record["network_accepted_at_ps"],
            f"{name}.record.network_accepted_at_ps",
        )
        outcome_at = _require_int(
            record["network_outcome_at_ps"],
            f"{name}.record.network_outcome_at_ps",
        )
        record_ids.add(wqe_id)
        record_tokens[token] = wqe_id
        _require(
            record["state"] == "completed"
            and accepted_at == 0
            and outcome_at == service
            and record["completion_status"] == "success",
            f"{name} record differs from the completed control fixture",
        )
    _require(
        len(record_ids) == frozen["wqe_count"]
        and len(record_tokens) == frozen["wqe_count"]
        and all(token > 0 for token in record_tokens),
        f"{name} record identities are not unique",
    )

    counters = _require_dict(
        snapshot["device_counters"], f"{name}.device_counters"
    )
    _require_exact_keys(
        counters, expectations["raw_counter_keys"], f"{name}.device_counters"
    )
    for key, value in counters.items():
        _require_int(value, f"{name}.device_counters.{key}")
    _require(
        counters
        == {
            "posted_wqes": 2,
            "network_accepted": 2,
            "network_delivered": 2,
            "network_dropped": 0,
        },
        f"{name} device counters differ from the control fixture",
    )
    _require(
        _require_list(snapshot["device_evidence"], f"{name}.device_evidence")
        == [],
        f"{name} unexpectedly contains device evidence",
    )

    issued = [
        _require_dict(row, f"{name}.port_issued[]")
        for row in _require_list(snapshot["port_issued"], f"{name}.port_issued")
    ]
    terminals = [
        _require_dict(row, f"{name}.port_terminals[]")
        for row in _require_list(
            snapshot["port_terminals"], f"{name}.port_terminals"
        )
    ]
    _require(
        len(issued) == frozen["wqe_count"]
        and len(terminals) == frozen["wqe_count"],
        f"{name} port ledger has the wrong cardinality",
    )
    for row in issued:
        _require_exact_keys(
            row, expectations["raw_issued_keys"], f"{name}.issued"
        )
        token = _require_int(row["token"], f"{name}.issued.token")
        wqe_id = _require_int(row["wqe_id"], f"{name}.issued.wqe_id")
        accepted_at = _require_int(
            row["accepted_at_ps"], f"{name}.issued.accepted_at_ps"
        )
        tx_at = _require_int(
            row["port_tx_at_ps"], f"{name}.issued.port_tx_at_ps"
        )
        payload = _require_int(
            row["payload_bytes"], f"{name}.issued.payload_bytes"
        )
        _require(
            record_tokens.get(token) == wqe_id
            and accepted_at == 0
            and tx_at == 0
            and payload == frozen["payload_bytes"],
            f"{name} issue ledger disagrees with the device records",
        )
    for row in terminals:
        _require_exact_keys(
            row, expectations["raw_terminal_keys"], f"{name}.terminal"
        )
        token = _require_int(row["token"], f"{name}.terminal.token")
        wqe_id = _require_int(row["wqe_id"], f"{name}.terminal.wqe_id")
        terminal_at = _require_int(row["at_ps"], f"{name}.terminal.at_ps")
        _require(
            record_tokens.get(token) == wqe_id
            and row["kind"] == "delivered"
            and terminal_at == service,
            f"{name} terminal ledger disagrees with the device records",
        )
    occupied = _require_int(
        snapshot["occupied_sq_entries"], f"{name}.occupied_sq_entries"
    )
    cq_depth = _require_int(
        snapshot["completion_queue_depth"],
        f"{name}.completion_queue_depth",
    )
    unpublished = _require_int(
        snapshot["unpublished_wqes"], f"{name}.unpublished_wqes"
    )
    _require(
        _require_list(
            snapshot["port_live_tokens"], f"{name}.port_live_tokens"
        )
        == []
        and occupied == 0
        and cq_depth == 0
        and unpublished == 0
        and snapshot["has_pending_physical_work"] is False,
        f"{name} control fixture is not quiescent",
    )


def _validate_terminal_controls(
    observations: dict[str, Any], expectations: dict[str, Any]
) -> None:
    raw_controls = _require_list(
        observations.get("terminal_controls"), "terminal_controls"
    )
    controls = [
        _require_dict(control, "terminal control")
        for control in raw_controls
    ]
    frozen = expectations["terminal_controls"]
    expected = frozen["kinds"]
    _require(
        len(controls) == len(expected)
        and Counter(control.get("kind") for control in controls)
        == Counter(expected),
        "terminal control inventory differs from the frozen set",
    )
    for control in controls:
        kind = control.get("kind")
        _require_exact_keys(
            control,
            frozen["control_keys"],
            f"{kind} terminal control",
        )
        before = _require_dict(control["before"], "terminal state before")
        after = _require_dict(control["after"], "terminal state after")
        _validate_terminal_snapshot(
            before, expectations, f"{kind} terminal state before"
        )
        _validate_terminal_snapshot(
            after, expectations, f"{kind} terminal state after"
        )
        _require(
            control["invalid_event_time_ps"]
            == frozen["invalid_event_time_ps"]
            and control["exception_type"] == frozen["exception_type"]
            and control["exception_message"]
            == frozen["exception_messages"][kind]
            and control["clock_probe_time_ps"]
            == frozen["clock_probe_time_ps"]
            and control["clock_probe_exception_type"]
            == frozen["clock_probe_exception_type"]
            and control["clock_probe_changes"]
            == frozen["clock_probe_changes"]
            and before == after,
            f"{kind} terminal call did not throw before state mutation",
        )


def _validate_drop_case(
    cell: dict[str, Any], expectations: dict[str, Any]
) -> None:
    expected = expectations["controlled_drop"]
    _require(
        _cell_key(cell)
        == (
            expected["payload_bytes"],
            expected["link_rate_gbps"],
            expected["doorbell_service_ps"],
        ),
        "controlled-drop configuration differs from the freeze",
    )
    _validate_cell_shape(
        cell,
        expectations,
        1,
        "controlled_drop",
        key_schema="raw_drop_cell_keys",
    )
    wqe = _require_dict(cell["wqes"][0], "controlled_drop.wqe")
    cqe_statuses = _require_list(cell.get("all_cqe_statuses"), "drop CQEs")
    evidence = _require_list(cell.get("evidence"), "drop evidence")
    _require(
        cell.get("signaled") is False
        and wqe["terminal_kind"] == "dropped"
        and wqe["cqe_status"] == "transport_error"
        and cqe_statuses == ["transport_error"]
        and "success" not in cqe_statuses,
        "controlled drop did not produce the sole error CQE",
    )
    _require(
        evidence
        == [
            {
                "kind": "network_drop",
                "drop_location": expected["drop_location"],
                "drop_reason": expected["drop_reason"],
                "wqe_id": wqe["wqe_id"],
            }
        ],
        "controlled drop evidence differs from the frozen oracle",
    )


def check_observations(
    observations: dict[str, Any], expectations: dict[str, Any], factory: str
) -> CheckSummary:
    _require_exact_keys(
        observations,
        expectations["raw_observation_keys"],
        "raw observations",
    )
    _require(
        observations.get("schema") == expectations["observation_schema"],
        "raw observations use the wrong schema",
    )
    _require(observations.get("factory") == factory, "factory echo mismatch")
    single_rows = [
        _require_dict(row, "single-WQE row")
        for row in _require_list(observations.get("single_wqe"), "single_wqe")
    ]
    single_by_key = {_cell_key(row): row for row in single_rows}
    single_grid = expectations["single_wqe"]
    expected_single = _expected_grid(
        single_grid["payload_bytes"],
        single_grid["link_rate_gbps"],
        single_grid["doorbell_service_ps"],
    )
    _require(
        len(single_rows) == len(single_by_key)
        and set(single_by_key) == expected_single,
        "single-WQE observations do not cover the frozen grid exactly",
    )

    for key, cell in sorted(single_by_key.items()):
        name = f"single{key}"
        _validate_cell_shape(cell, expectations, 1, name)
        _validate_single_exact(cell, name)

    d_instances = 0
    inverse_instances = 0
    for payload, rate in product(
        single_grid["payload_bytes"], single_grid["link_rate_gbps"]
    ):
        _validate_d_pair(
            single_by_key[(payload, rate, 0)],
            single_by_key[(payload, rate, 1000)],
            expectations,
            str((payload, rate)),
        )
        d_instances += 1
    for payload, doorbell in product(
        single_grid["payload_bytes"], single_grid["doorbell_service_ps"]
    ):
        slow = single_by_key[(payload, 200, doorbell)]["wqes"][0]
        fast = single_by_key[(payload, 400, doorbell)]["wqes"][0]
        slow_service = slow["terminal_at_ps"] - slow["port_tx_at_ps"]
        fast_service = fast["terminal_at_ps"] - fast["port_tx_at_ps"]
        _require(
            slow_service == 2 * fast_service,
            f"inverse-rate serialization {(payload, doorbell)} failed",
        )
        inverse_instances += 1

    fifo_rows = [
        _require_dict(row, "FIFO row")
        for row in _require_list(observations.get("fifo"), "fifo")
    ]
    fifo_by_key = {_cell_key(row): row for row in fifo_rows}
    fifo_grid = expectations["fifo"]
    expected_fifo = _expected_grid(
        [fifo_grid["payload_bytes"]],
        fifo_grid["link_rate_gbps"],
        fifo_grid["doorbell_service_ps"],
    )
    _require(
        len(fifo_rows) == len(fifo_by_key) and set(fifo_by_key) == expected_fifo,
        "FIFO observations do not cover the frozen grid exactly",
    )
    fifo_instances = 0
    for key, cell in sorted(fifo_by_key.items()):
        name = f"fifo{key}"
        _validate_cell_shape(cell, expectations, 2, name)
        _validate_fifo_structural(cell, name)
        _validate_fifo_timing(cell, name)
        fifo_instances += 1

    negative = _require_list(
        observations.get("wrapper_bypass_control"),
        "wrapper_bypass_control",
    )
    _require(len(negative) == 2, "wrapper-bypass control must contain one D pair")
    negative_rows = [
        _require_dict(row, "wrapper-bypass row") for row in negative
    ]
    for index, row in enumerate(negative_rows):
        _validate_cell_shape(
            row,
            expectations,
            1,
            f"wrapper_bypass_control[{index}]",
        )
    negative_by_d = {
        int(row["doorbell_service_ps"]): row for row in negative_rows
    }
    _require(set(negative_by_d) == {0, 1000}, "wrapper-bypass D pair is incomplete")
    negative_config = expectations["wrapper_bypass_control"]
    _require(
        {
            (
                int(row["payload_bytes"]),
                int(row["link_rate_gbps"]),
                int(row["doorbell_service_ps"]),
            )
            for row in negative_rows
        }
        == {
            (
                negative_config["payload_bytes"],
                negative_config["link_rate_gbps"],
                doorbell,
            )
            for doorbell in negative_config["doorbell_service_ps"]
        },
        "wrapper-bypass control differs from its frozen configuration",
    )
    d_fields = expectations["d_additivity"]["fields"]
    low_negative = negative_by_d[0]
    high_negative = negative_by_d[1000]
    low_wqe = low_negative["wqes"][0]
    high_wqe = high_negative["wqes"][0]
    _require(
        all(high_wqe[field] - low_wqe[field] == 0 for field in d_fields)
        and high_negative["jct_ps"] - low_negative["jct_ps"] == 0
        and high_wqe["terminal_at_ps"] - high_wqe["port_tx_at_ps"]
        == low_wqe["terminal_at_ps"] - low_wqe["port_tx_at_ps"],
        "wrapper-bypass control is not the frozen zero-D mutant",
    )
    positive_base = single_by_key[(4096, 400, 0)]
    positive_wqe = positive_base["wqes"][0]
    _require(
        all(low_wqe[field] == positive_wqe[field] for field in d_fields)
        and low_negative["jct_ps"] == positive_base["jct_ps"],
        "wrapper-bypass baseline differs from the accepted D=0 row",
    )
    negative_rejected = False
    try:
        _validate_d_pair(
            negative_by_d[0],
            negative_by_d[1000],
            expectations,
            "wrapper-bypass control",
        )
    except AcceptanceError:
        negative_rejected = True
    _require(
        negative_rejected,
        "wrapper-bypass mutant passed the D-additivity checker",
    )

    _validate_authority_cases(observations, expectations)
    _validate_terminal_controls(observations, expectations)
    _validate_drop_case(
        _require_dict(observations.get("controlled_drop"), "controlled_drop"),
        expectations,
    )

    family_counts = {
        "d_additivity": d_instances,
        "inverse_rate_serialization": inverse_instances,
        "two_wqe_fifo": fifo_instances,
    }
    _require(
        family_counts == expectations["behavioral_families"],
        "scored family counts differ from the freeze",
    )
    return CheckSummary(
        factory=factory,
        scored_family_instances=family_counts,
        exact_oracle_rows=len(single_rows),
        fatal_invariant_families={
            "authority_exclusivity": True,
            "token_conservation": True,
            "quiescence": True,
            "terminal_atomicity": True,
            "controlled_drop": True,
            "fifo_ordering": True,
            "wrapper_bypass_sensitivity": True,
        },
        wrapper_bypass_rejected=True,
    )


def _producer_command(
    producer: Path,
    factory: str,
    expectations: Path,
    observations: Path,
) -> list[str]:
    return [
        str(producer),
        "--factory",
        factory,
        "--expectations",
        str(expectations),
        "--observations",
        str(observations),
    ]


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True, choices=("fake", "htsim"))
    parser.add_argument("--producer", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--expectations", type=Path, default=DEFAULT_EXPECTATIONS
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    expectations_path = arguments.expectations.resolve(strict=True)
    expectations = _load_json(expectations_path, "expectations")
    _validate_expectations(expectations)
    _require(
        arguments.factory in expectations["factories"],
        "factory is absent from the frozen contract",
    )
    run_dir = _validate_run_path(arguments.run_dir, "run directory")
    producer = _validate_run_path(arguments.producer, "producer")
    _require(
        producer.is_relative_to(run_dir),
        "producer must reside under its run directory",
    )
    observations_path = run_dir / "raw_observations.json"
    summary_path = run_dir / "summary.json"

    command = _producer_command(
        producer,
        arguments.factory,
        expectations_path,
        observations_path,
    )
    if arguments.check_only:
        print(
            "Tier A command contract valid: "
            + json.dumps(command, separators=(",", ":"))
        )
        return

    _require(producer.is_file(), f"producer does not exist: {producer}")
    _require(
        not observations_path.exists() and not summary_path.exists(),
        "run directory already contains Tier A observations or summary; "
        "select a fresh run directory",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    observations = _load_json(observations_path, "raw observations")
    summary = check_observations(
        observations, expectations, arguments.factory
    ).as_dict()
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
