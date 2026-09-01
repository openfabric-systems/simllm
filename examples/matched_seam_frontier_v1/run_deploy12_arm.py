#!/usr/bin/env python3
"""Run the frozen DEPLOY-12 LogGOPSim third arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from examples.matched_seam_frontier_v1 import run_study as base
from simllm.backends import (
    LogGopsimConfig,
    build_loggopsim_command,
    run_loggopsim,
)
from simllm.deploy import weak_dominance_pareto

EXPECTATIONS_PATH = STUDY_DIR / "expectations_deploy12.md"
RESULT_PATH = STUDY_DIR / "deploy12_record.json"
CSV_PATH = STUDY_DIR / "deploy12_results.csv"
PROTECTED_RECORD_PATH = STUDY_DIR / "record.json"

SCHEMA = "simllm-matched-seam-deploy12-record-v1"
EVALUATION_SCHEMA = "simllm-matched-seam-deploy12-evaluation-v1"
EXPECTATIONS_COMMIT = "2db8595ab869f500d6da2b0690d977dd11093ff6"
EXPECTATIONS_SHA256 = (
    "ed784f7514fe766c509b02ed591391370129b84c63cc51552e278f5fcee44812"
)
PINNED_LOGGOPSIM_SHA256 = (
    "7e0f13ee3c87a20e9d2e94dbbd74c46075fd03df2f1b04d1ed9739c43ee0a2bf"
)
EXACT_G_STRING = "0.02"
LATENCY_NS = 2_000
RENDEZVOUS_THRESHOLD_BYTES = (1 << 63) - 1
LOGGOPSIM_ENV = "SIMLLM_LOGGOPSIM"
BULK_ROOT_ENV = "SIMLLM_DEPLOY12_BULK_ROOT"

PROTECTED_PRIOR_SHA256 = {
    "RESULTS.md": "fa1170277fa8f3b9f1a14df353add3dbd4e8e490aeb4847748dd2120d4434e62",
    "expectations.md": "fc5af307fee560fc7050011543e18e1cf77030d0aa6a13e6c5a014cb159a5726",
    "expectations_v2.md": "fe403500575d674a25c8b7c6c59eb41aec65fce6cc29024609fa995b29585f35",
    "external_adjustments.json": "c6778a81cdc6078ce74f06733e4bce9d99a92b4ab3eccba4a83d14e7d063a09e",
    "figure_addendum.md": "cc4dcb8c82bbcd5e542457b56d91ddf172af2cbe05e6bac5c865535dcc307762",
    "plot_publication.py": "a98514cb985a9980a679357285a11dbe52418e774a55d69a6c9f30ba9ddda53d",
    "plot_study.py": "d4fe430f1fede23bcbcbb21834d98a51d3563c4b4e4c21dc887c7b8c837a7e4f",
    "record.json": "bddd7cb040a3c0f0ec8afd7ea836d873fa22cad2131f98ff36e38da5441b2d50",
    "results.csv": "4113ab2413084b7da957de60002abc4a4f8530bbb89837a5a5f73b9852f4448d",
    "run_study.py": "242b5f1ae46ac18ac2cb474ad6fa24acc4dba21c4b8ff1d6683137163fec3182",
    "study_config.json": "64c8e16de53e194e98f5ca7c9b27d533d4c7f7ca32311841a62e3c6cece21f17",
    "figures/matched-seam-frontier-publication.pdf": (
        "511a0fb869d3397a87664d28c6b0c1d5adc17738dd84543973f66c7fcfd764cb"
    ),
    "figures/matched-seam-frontier-publication.png": (
        "d79b5099cbbfeed9e4272a64d7007512ed1889a08fc3438c9f2eef41a28986d1"
    ),
    "figures/matched-seam-frontier.pdf": (
        "4ecc3bf2822f916bfd53107b55d1344406efea01fd0b1ad7a417019391712dbb"
    ),
    "figures/matched-seam-frontier.png": (
        "852378a01d3c9e0aeab74423259afe86b456dca0b193e27c23e48256322069c4"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_hashes() -> dict[str, str]:
    return {
        relative: _sha256(STUDY_DIR / relative)
        for relative in PROTECTED_PRIOR_SHA256
    }


def _portable_argv(argv: list[str], packet_root: Path) -> list[str]:
    portable = list(argv)
    portable[0] = "LogGOPSim"
    goal_index = portable.index("-f") + 1
    portable[goal_index] = Path(portable[goal_index]).relative_to(
        packet_root
    ).as_posix()
    return portable


def _loggopsim_config(goal_binary: Path) -> LogGopsimConfig:
    return LogGopsimConfig(
        goal_bin=goal_binary,
        latency_ns=LATENCY_NS,
        overhead_ns=0,
        message_gap_ns=0,
        byte_gap_ns=float(EXACT_G_STRING),
        byte_gap_ns_string=EXACT_G_STRING,
        byte_overhead_ns=0,
        rendezvous_threshold_bytes=RENDEZVOUS_THRESHOLD_BYTES,
        network_type="LogGP",
    )


def _network_arm(
    *,
    mode: str,
    packet_cells: list[dict[str, Any]],
    packet_root: Path | None = None,
    loggopsim: Path | None = None,
) -> dict[str, Any]:
    if mode not in {"priced", "bypass"}:
        raise ValueError("third-arm mode must be 'priced' or 'bypass'")
    if mode == "bypass":
        return {
            "mode": "bypass",
            "invocation_count": 0,
            "cells": [
                {
                    "configuration_id": cell["configuration_id"],
                    "decode_tp": int(str(cell["configuration_id"]).rsplit("tp", 1)[1]),
                    "network_service_ps": 0,
                }
                for cell in packet_cells
            ],
        }
    if packet_root is None or loggopsim is None:
        raise ValueError("priced third arm requires packet_root and loggopsim")

    cells = []
    for packet_cell in packet_cells:
        artifacts = packet_cell["artifacts"]
        goal_path = packet_root / str(artifacts["goal"]["path"])
        goal_binary = packet_root / str(artifacts["goal_binary"]["path"])
        goal_sha256 = _sha256(goal_path)
        goal_binary_sha256 = _sha256(goal_binary)
        config = _loggopsim_config(goal_binary)
        argv = build_loggopsim_command(loggopsim, config)
        result = run_loggopsim(config, binary=loggopsim)
        network_service_ps = result.max_finish_ps - base.PCIE_SUBMISSION_PS
        if network_service_ps <= 0:
            raise AssertionError("LogGOPSim network service is not positive")
        decode_tp = int(str(packet_cell["configuration_id"]).rsplit("tp", 1)[1])
        flow_count = int(packet_cell["flow_count"])
        serial_ceiling_ps = (
            base.KV_BYTES * 8 * base.PICOSECONDS_PER_SECOND // base.LINKSPEED_BPS
            + flow_count * LATENCY_NS * 1_000
        )
        cells.append(
            {
                "configuration_id": packet_cell["configuration_id"],
                "decode_tp": decode_tp,
                "aggregate_kv_bytes": int(packet_cell["aggregate_kv_bytes"]),
                "flow_count": flow_count,
                "goal_sha256": goal_sha256,
                "goal_binary_sha256": goal_binary_sha256,
                "packet_goal_sha256": artifacts["goal"]["sha256"],
                "packet_goal_binary_sha256": artifacts["goal_binary"]["sha256"],
                "argv": _portable_argv(argv, packet_root),
                "raw_max_finish_ps": result.max_finish_ps,
                "control_prefix_ps": base.PCIE_SUBMISSION_PS,
                "network_service_ps": network_service_ps,
                "sender_floor_ps": (
                    base.KV_BYTES
                    * 8
                    * base.PICOSECONDS_PER_SECOND
                    // (base.PREFILL_TP * base.LINKSPEED_BPS)
                ),
                "serial_ceiling_ps": serial_ceiling_ps,
                "max_finish_host": result.max_finish_host,
                "rank_count": result.rank_count,
                "quiescent": result.quiescent,
            }
        )
    return {"mode": "priced", "invocation_count": len(cells), "cells": cells}


def _project_arm(
    ideal_points: list[dict[str, Any]],
    network_service_by_tp: dict[int, int],
    *,
    arm_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = base._load_json(base.CONFIG_PATH)
    prefill_factor = Fraction.from_float(
        float.fromhex(config["composition"]["prefill_rate_matching_degradation_hex"])
    )
    points = []
    for ideal in ideal_points:
        decode_tp = int(ideal["configuration"]["decode_tp"])
        network_service_ps = network_service_by_tp[decode_tp]
        prefill_request_ps = int(ideal["prefill_request_ps"]) + network_service_ps
        prefill_capacity = (
            Fraction(
                int(ideal["configuration"]["prefill_workers"])
                * base.PICOSECONDS_PER_SECOND,
                prefill_request_ps,
            )
            * prefill_factor
        )
        decode_capacity = base._fraction(ideal["decode_capacity_requests_per_second"])
        request_capacity = min(prefill_capacity, decode_capacity)
        y_value = (
            request_capacity
            * base.OUTPUT_TOKENS
            / int(ideal["configuration"]["used_gpus"])
        )
        if network_service_ps == 0:
            point = dict(ideal)
            if point != ideal:
                raise AssertionError("bypass projection changed the unpriced point")
        else:
            point = {
                **ideal,
                "evidence_class": "MEASURED-EXTERNAL + DECLARED-LOGGOP",
                "prefill_request_ps": prefill_request_ps,
                "prefill_capacity_requests_per_second": base._fraction_json(
                    prefill_capacity
                ),
                "request_capacity_per_second": base._fraction_json(request_capacity),
                "capacity_limiter": (
                    "prefill" if prefill_capacity <= decode_capacity else "decode"
                ),
                "y_tokens_per_second_per_gpu": base._fraction_json(y_value),
                "third_arm_term": {
                    "arm": arm_name,
                    "duration_ps": network_service_ps,
                    "evidence_class": "DECLARED-LOGGOP",
                    "source": f"loggopsim:tp4-to-tp{decode_tp}",
                },
            }
        points.append(point)
    frontier = list(
        weak_dominance_pareto(
            points,
            coordinate=base._point_xy,
            identity=lambda point: str(point["candidate_key"]),
        )
    )
    return points, frontier


def _decomposition(
    *,
    base_evaluation: dict[str, Any],
    priced_arm: dict[str, Any],
    bypass_arm: dict[str, Any],
) -> dict[str, Any]:
    families = base_evaluation["families"]
    ideal_points = families["F"]["ideal_points"]
    packet_points = families["F"]["packet_points"]
    priced_services = {
        int(cell["decode_tp"]): int(cell["network_service_ps"])
        for cell in priced_arm["cells"]
    }
    bypass_services = {
        int(cell["decode_tp"]): int(cell["network_service_ps"])
        for cell in bypass_arm["cells"]
    }
    priced_points, priced_frontier = _project_arm(
        ideal_points, priced_services, arm_name="loggopsim-priced"
    )
    bypass_points, bypass_frontier = _project_arm(
        ideal_points, bypass_services, arm_name="explicit-bypass"
    )
    packet_by_row = {int(point["row"]): point for point in packet_points}
    priced_by_row = {int(point["row"]): point for point in priced_points}
    rows = []
    for unpriced in ideal_points:
        row_number = int(unpriced["row"])
        priced = priced_by_row[row_number]
        packet = packet_by_row[row_number]
        decode_tp = int(unpriced["configuration"]["decode_tp"])
        unpriced_y = base._fraction(unpriced["y_tokens_per_second_per_gpu"])
        priced_y = base._fraction(priced["y_tokens_per_second_per_gpu"])
        packet_y = base._fraction(packet["y_tokens_per_second_per_gpu"])
        priced_penalty = unpriced_y / priced_y
        residual_penalty = priced_y / packet_y
        total_penalty = unpriced_y / packet_y
        if priced_penalty * residual_penalty != total_penalty:
            raise AssertionError("three-arm quotient decomposition lost exactness")
        packet_service_ps = int(packet["packet_term"]["duration_ps"])
        loggopsim_service_ps = priced_services[decode_tp]
        rows.append(
            {
                "row": row_number,
                "candidate_id": unpriced["candidate_id"],
                "decode_tp": decode_tp,
                "capacity_limiter": {
                    "unpriced": unpriced["capacity_limiter"],
                    "loggopsim_priced": priced["capacity_limiter"],
                    "packet": packet["capacity_limiter"],
                },
                "network_service_ps": {
                    "unpriced": 0,
                    "loggopsim_priced": loggopsim_service_ps,
                    "packet": packet_service_ps,
                },
                "prefill_request_ps": {
                    "unpriced": int(unpriced["prefill_request_ps"]),
                    "loggopsim_priced": int(priced["prefill_request_ps"]),
                    "packet": int(packet["prefill_request_ps"]),
                },
                "y_tokens_per_second_per_gpu": {
                    "unpriced": unpriced["y_tokens_per_second_per_gpu"],
                    "loggopsim_priced": priced["y_tokens_per_second_per_gpu"],
                    "packet": packet["y_tokens_per_second_per_gpu"],
                },
                "priced_penalty": base._fraction_json(priced_penalty),
                "residual_penalty": base._fraction_json(residual_penalty),
                "total_packet_penalty": base._fraction_json(total_penalty),
                "multiplicative_identity_holds": True,
                "network_residual_ps": packet_service_ps - loggopsim_service_ps,
                "frontier_visible_residual": residual_penalty > 1,
            }
        )
    visible = [row for row in rows if row["frontier_visible_residual"]]
    maximum_visible = max(
        (base._fraction(row["residual_penalty"]) for row in visible),
        default=Fraction(1),
    )
    maximum_network_residual_ps = max(
        int(row["network_residual_ps"]) for row in rows
    )
    return {
        "evidence_class": "UNSCORED-DECOMPOSITION",
        "rows": rows,
        "priced_points": priced_points,
        "priced_frontier": priced_frontier,
        "bypass_points": bypass_points,
        "bypass_frontier": bypass_frontier,
        "frontier_visible_residual_survives": bool(visible),
        "frontier_visible_rows": [int(row["row"]) for row in visible],
        "maximum_residual_penalty": base._fraction_json(maximum_visible),
        "maximum_network_residual_ps": maximum_network_residual_ps,
    }


def _expected_argv(cell: dict[str, Any]) -> bool:
    argv = list(cell["argv"])
    expected = {
        "-L": "2000",
        "-o": "0",
        "-g": "0",
        "-G": "0.02",
        "-O": "0",
        "-S": "9223372036854775807",
        "-n": "LogGP",
    }
    return argv[0] == "LogGOPSim" and all(
        flag in argv and argv[argv.index(flag) + 1] == value
        for flag, value in expected.items()
    )


def _full_evaluation_worker(
    *,
    packet_root: Path,
    txt2bin: Path,
    htsim_rnic: Path,
    loggopsim: Path,
    external_python: Path,
) -> dict[str, Any]:
    hashes_before = _protected_hashes()
    base_evaluation = base._full_evaluation_worker(
        packet_root=packet_root,
        txt2bin=txt2bin,
        htsim_rnic=htsim_rnic,
        external_python=external_python,
    )
    packet_cells = base_evaluation["families"]["M"]["packet_cells"]
    priced_arm = _network_arm(
        mode="priced",
        packet_cells=packet_cells,
        packet_root=packet_root,
        loggopsim=loggopsim,
    )
    bypass_arm = _network_arm(mode="bypass", packet_cells=packet_cells)
    decomposition = _decomposition(
        base_evaluation=base_evaluation,
        priced_arm=priced_arm,
        bypass_arm=bypass_arm,
    )
    hashes_after = _protected_hashes()
    protected_record = base._load_json(PROTECTED_RECORD_PATH)
    expected_tools = protected_record["native_tools"]
    observed_tools = {
        "htsim_rnic": _sha256(htsim_rnic),
        "txt2bin": _sha256(txt2bin),
        "loggopsim": _sha256(loggopsim),
    }
    rows = list(base_evaluation["rows"])
    rows.extend(
        (
            base._fatal_row(
                "FG-A",
                hashes_before == hashes_after == PROTECTED_PRIOR_SHA256,
                "the void first publication and corrected nonvoid publication set are byte-identical before and after",
            ),
            base._fatal_row(
                "FG-B",
                observed_tools
                == {
                    "htsim_rnic": expected_tools["htsim_rnic"]["sha256"],
                    "txt2bin": expected_tools["txt2bin"]["sha256"],
                    "loggopsim": PINNED_LOGGOPSIM_SHA256,
                }
                and all(_expected_argv(cell) for cell in priced_arm["cells"]),
                "native tool hashes and every exact LogGOP argv match the frozen declarations",
            ),
            base._fatal_row(
                "FG-C",
                all(
                    cell["goal_sha256"] == cell["packet_goal_sha256"]
                    and cell["goal_binary_sha256"]
                    == cell["packet_goal_binary_sha256"]
                    for cell in priced_arm["cells"]
                ),
                "each LogGOP-priced cell consumes the packet cell's exact GOAL text and binary",
            ),
            base._fatal_row(
                "FG-D",
                all(base_evaluation["fatal_guards_without_fg6"].values()),
                "all inherited corrected-run fatal guards hold with original scored families and bands",
            ),
            base._fatal_row(
                "FG-E",
                base._is_ancestor(EXPECTATIONS_COMMIT),
                "the DEPLOY-12 expectation freeze precedes this implementation and execution",
            ),
            base._fatal_row(
                "FG-G",
                bypass_arm["invocation_count"] == 0
                and all(
                    int(cell["network_service_ps"]) == 0
                    for cell in bypass_arm["cells"]
                )
                and decomposition["bypass_points"]
                == protected_record["families"]["F"]["ideal_points"]
                and decomposition["bypass_frontier"]
                == protected_record["families"]["F"]["ideal_frontier"],
                "the explicit bypass starts no LogGOPSim process and reproduces the corrected unpriced points and frontier byte for byte",
            ),
            base._fatal_row(
                "FG-H",
                all(
                    int(cell["sender_floor_ps"])
                    <= int(cell["network_service_ps"])
                    <= int(cell["serial_ceiling_ps"])
                    for cell in priced_arm["cells"]
                ),
                "every LogGOP-priced network cell lies inside its frozen physical floor and ceiling",
            ),
        )
    )
    for row in decomposition["rows"]:
        rows.append(
            base._unscored_row(
                f"T-{int(row['row']):02d}",
                f"residual_penalty={base._fraction(row['residual_penalty'])!s}",
                "exact quotient",
                (
                    f"unpriced={row['network_service_ps']['unpriced']} ps; "
                    f"loggopsim={row['network_service_ps']['loggopsim_priced']} ps; "
                    f"packet={row['network_service_ps']['packet']} ps; "
                    f"network_residual={row['network_residual_ps']} ps"
                ),
                family="T",
                evidence_class="UNSCORED-DECOMPOSITION",
            )
        )
    return {
        "schema": EVALUATION_SCHEMA,
        "base": base_evaluation,
        "priced_arm": priced_arm,
        "bypass_arm": bypass_arm,
        "decomposition": decomposition,
        "protected_hashes": {"before": hashes_before, "after": hashes_after},
        "native_tools": observed_tools,
        "rows": rows,
        "fatal_guards_without_fg_f": {
            row["id"]: row["passed"] for row in rows if row["kind"] == "fatal"
        },
        "family_tallies_without_wall_time": base._family_tallies(rows),
    }


def _worker_command(
    python: Path,
    *,
    packet_root: Path,
    txt2bin: Path,
    htsim_rnic: Path,
    loggopsim: Path,
    external_python: Path,
) -> list[str]:
    return [
        os.fspath(python),
        os.fspath(Path(__file__).resolve()),
        "--worker",
        "evaluation",
        "--packet-root",
        os.fspath(packet_root),
        "--txt2bin",
        os.fspath(txt2bin),
        "--htsim-rnic",
        os.fspath(htsim_rnic),
        "--loggopsim",
        os.fspath(loggopsim),
        "--external-python",
        os.fspath(external_python),
    ]


def _run_evaluation(
    *,
    attempt: Path,
    repetition: int,
    txt2bin: Path,
    htsim_rnic: Path,
    loggopsim: Path,
    external_python: Path,
) -> tuple[dict[str, Any], bytes]:
    evaluation_root = attempt / f"evaluation-run-{repetition}"
    packet_root = evaluation_root / "packet"
    evaluation_root.mkdir(parents=True, exist_ok=False)
    command = _worker_command(
        Path(sys.executable),
        packet_root=packet_root,
        txt2bin=txt2bin,
        htsim_rnic=htsim_rnic,
        loggopsim=loggopsim,
        external_python=external_python,
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    base._write_new(
        attempt / f"evaluation-run-{repetition}.stdout.json",
        completed.stdout.encode(),
    )
    base._write_new(
        attempt / f"evaluation-run-{repetition}.stderr.txt",
        completed.stderr.encode(),
    )
    if completed.returncode:
        raise RuntimeError(
            f"evaluation worker {repetition} failed with status "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise TypeError("evaluation worker did not return a JSON object")
    return value, completed.stdout.encode()


def _coordinator(
    *,
    bulk_root: Path,
    external_venv: Path,
    txt2bin: Path,
    htsim_rnic: Path,
    loggopsim: Path,
    write_tracked: bool,
) -> dict[str, Any]:
    for name, path in (
        (base.EXTERNAL_VENV_ENV, external_venv),
        (base.TXT2BIN_ENV, txt2bin),
        (base.HTSIM_ENV, htsim_rnic),
        (LOGGOPSIM_ENV, loggopsim),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    external_python = next(
        (
            path
            for path in (
                external_venv / "bin/python",
                external_venv / "Scripts/python.exe",
            )
            if path.is_file()
        ),
        None,
    )
    if external_python is None:
        raise FileNotFoundError(
            f"{base.EXTERNAL_VENV_ENV} has no Python interpreter"
        )
    attempt, attempt_number = base._new_attempt(bulk_root)
    started = time.monotonic()
    runs = [
        _run_evaluation(
            attempt=attempt,
            repetition=repetition,
            txt2bin=txt2bin,
            htsim_rnic=htsim_rnic,
            loggopsim=loggopsim,
            external_python=external_python,
        )
        for repetition in (1, 2)
    ]
    evaluations = [value for value, _ in runs]
    evaluation_bytes = [payload for _, payload in runs]
    deterministic = evaluation_bytes[0] == evaluation_bytes[1]
    evaluation_hashes = [
        hashlib.sha256(payload).hexdigest() for payload in evaluation_bytes
    ]
    rows = list(evaluations[0]["rows"])
    rows.append(
        base._fatal_row(
            "FG-F",
            deterministic,
            "two complete fresh-process scored evaluation JSON payloads are byte-identical; elapsed_seconds and W-1 are excluded by name",
        )
    )
    elapsed_seconds = time.monotonic() - started
    rows.append(
        base._scored_row(
            "W",
            "W-1",
            elapsed_seconds <= base.WALL_CEILING_SECONDS,
            expected=f"<= {base.WALL_CEILING_SECONDS:.0f}",
            observed=f"{elapsed_seconds:.6f}",
            units="seconds",
            evidence_class="WALL",
            detail="two complete fresh-process scored evaluations with priced and bypass third arms",
        )
    )
    failed_guards = [
        row["id"] for row in rows if row["kind"] == "fatal" and not row["passed"]
    ]
    decomposition = evaluations[0]["decomposition"]
    record = {
        "schema": SCHEMA,
        "study": "matched_seam_frontier_v1 DEPLOY-12 third arm",
        "run_state": "void" if failed_guards else "nonvoid",
        "voiding_guards": failed_guards,
        "attempt": f"attempt-{attempt_number:04d}",
        "bulk_evidence": f"${{{BULK_ROOT_ENV}}}/attempt-{attempt_number:04d}",
        "run_commit": base._git_output("rev-parse", "HEAD"),
        "freeze": {
            "commit": EXPECTATIONS_COMMIT,
            "path": EXPECTATIONS_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": _sha256(EXPECTATIONS_PATH),
        },
        "protected_corrected_record_sha256": _sha256(PROTECTED_RECORD_PATH),
        "native_tools": {
            "htsim_rnic": {
                "filename": htsim_rnic.name,
                "sha256": _sha256(htsim_rnic),
            },
            "txt2bin": {"filename": txt2bin.name, "sha256": _sha256(txt2bin)},
            "loggopsim": {
                "filename": "LogGOPSim",
                "sha256": _sha256(loggopsim),
            },
        },
        "machine": {
            "architecture": platform.machine(),
            "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
            "system": platform.system(),
        },
        "fatal_guards": {
            row["id"]: row["passed"] for row in rows if row["kind"] == "fatal"
        },
        "family_tallies": base._family_tallies(rows),
        "families": {
            **evaluations[0]["base"]["families"],
            "T": decomposition,
        },
        "priced_arm": evaluations[0]["priced_arm"],
        "bypass_arm": evaluations[0]["bypass_arm"],
        "rows": rows,
        "determinism": {
            "comparison": "byte-for-byte complete scored evaluation JSON",
            "fresh_processes": 2,
            "evaluation_sha256": evaluation_hashes,
            "excluded_by_name": ["elapsed_seconds", "W-1"],
            "equal": deterministic,
        },
        "elapsed_seconds": elapsed_seconds,
        "disposition": {
            "frontier_visible_residual_survives": decomposition[
                "frontier_visible_residual_survives"
            ],
            "frontier_visible_rows": decomposition["frontier_visible_rows"],
            "maximum_residual_penalty": decomposition[
                "maximum_residual_penalty"
            ],
            "maximum_network_residual_ps": decomposition[
                "maximum_network_residual_ps"
            ],
        },
        "reporting_rule": (
            "fatal guards, inherited scored families, wall time, and the "
            "unscored third-arm decomposition remain separate evidence classes"
        ),
    }
    base._write_new(
        attempt / "record.json",
        (json.dumps(record, indent=2, sort_keys=True) + "\n").encode(),
    )
    csv_payload = base._csv_bytes(rows)
    base._write_new(attempt / "results.csv", csv_payload)
    if write_tracked:
        base._write_json(RESULT_PATH, record)
        CSV_PATH.write_bytes(csv_payload)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("evaluation",))
    parser.add_argument("--packet-root", type=Path)
    parser.add_argument("--external-python", type=Path)
    parser.add_argument("--bulk-root", type=Path)
    parser.add_argument("--external-venv", type=Path)
    parser.add_argument("--txt2bin", type=Path)
    parser.add_argument("--htsim-rnic", type=Path)
    parser.add_argument("--loggopsim", type=Path)
    parser.add_argument("--write-tracked", action="store_true")
    args = parser.parse_args()
    if args.worker == "evaluation":
        missing = [
            name
            for name, value in (
                ("--packet-root", args.packet_root),
                ("--external-python", args.external_python),
                ("--txt2bin", args.txt2bin),
                ("--htsim-rnic", args.htsim_rnic),
                ("--loggopsim", args.loggopsim),
            )
            if value is None
        ]
        if missing:
            parser.error("evaluation worker missing " + ", ".join(missing))
        result = _full_evaluation_worker(
            packet_root=args.packet_root,
            txt2bin=args.txt2bin,
            htsim_rnic=args.htsim_rnic,
            loggopsim=args.loggopsim,
            external_python=args.external_python,
        )
        print(json.dumps(result, sort_keys=True))
        return
    bulk_root = args.bulk_root or (
        Path(os.environ[BULK_ROOT_ENV]) if BULK_ROOT_ENV in os.environ else None
    )
    external_venv = args.external_venv or (
        Path(os.environ[base.EXTERNAL_VENV_ENV])
        if base.EXTERNAL_VENV_ENV in os.environ
        else None
    )
    txt2bin = args.txt2bin or (
        Path(os.environ[base.TXT2BIN_ENV]) if base.TXT2BIN_ENV in os.environ else None
    )
    htsim_rnic = args.htsim_rnic or (
        Path(os.environ[base.HTSIM_ENV]) if base.HTSIM_ENV in os.environ else None
    )
    loggopsim = args.loggopsim or (
        Path(os.environ[LOGGOPSIM_ENV]) if LOGGOPSIM_ENV in os.environ else None
    )
    missing = [
        name
        for name, value in (
            ("--bulk-root or " + BULK_ROOT_ENV, bulk_root),
            ("--external-venv or " + base.EXTERNAL_VENV_ENV, external_venv),
            ("--txt2bin or " + base.TXT2BIN_ENV, txt2bin),
            ("--htsim-rnic or " + base.HTSIM_ENV, htsim_rnic),
            ("--loggopsim or " + LOGGOPSIM_ENV, loggopsim),
        )
        if value is None
    ]
    if missing:
        parser.error("missing " + ", ".join(missing))
    assert bulk_root is not None
    assert external_venv is not None
    assert txt2bin is not None
    assert htsim_rnic is not None
    assert loggopsim is not None
    record = _coordinator(
        bulk_root=bulk_root,
        external_venv=external_venv,
        txt2bin=txt2bin,
        htsim_rnic=htsim_rnic,
        loggopsim=loggopsim,
        write_tracked=args.write_tracked,
    )
    print(json.dumps(record["family_tallies"], sort_keys=True))


if __name__ == "__main__":
    main()
