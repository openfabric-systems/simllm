from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/matched_seam_frontier_v1"
RUNNER = STUDY / "run_study.py"
DEPLOY12_RUNNER = STUDY / "run_deploy12_arm.py"
PLOTTER = STUDY / "plot_study.py"
CONFIG = STUDY / "study_config.json"
EXPECTATIONS = STUDY / "expectations.md"
FIGURE_ADDENDUM = STUDY / "figure_addendum.md"
EXPECTATIONS_V2 = STUDY / "expectations_v2.md"
EXPECTATIONS_DEPLOY12 = STUDY / "expectations_deploy12.md"
ADJUSTMENTS = STUDY / "external_adjustments.json"
RECORD = STUDY / "record.json"
RESULTS_CSV = STUDY / "results.csv"
DEPLOY12_RECORD = STUDY / "deploy12_record.json"
DEPLOY12_RESULTS_CSV = STUDY / "deploy12_results.csv"
DEPLOY12_RESULTS = STUDY / "DEPLOY12_RESULTS.md"
PDF = STUDY / "figures/matched-seam-frontier.pdf"
PNG = STUDY / "figures/matched-seam-frontier.png"
EXTERNAL_VENV_ENV = "SIMLLM_EXTERNAL_AIC_VENV"
RECORD_SHA256 = "bddd7cb040a3c0f0ec8afd7ea836d873fa22cad2131f98ff36e38da5441b2d50"
RESULTS_CSV_SHA256 = "4113ab2413084b7da957de60002abc4a4f8530bbb89837a5a5f73b9852f4448d"
PDF_SHA256 = "4ecc3bf2822f916bfd53107b55d1344406efea01fd0b1ad7a417019391712dbb"
PNG_SHA256 = "852378a01d3c9e0aeab74423259afe86b456dca0b193e27c23e48256322069c4"
EXPECTATIONS_DEPLOY12_SHA256 = (
    "ed784f7514fe766c509b02ed591391370129b84c63cc51552e278f5fcee44812"
)
DEPLOY12_RECORD_SHA256 = (
    "c2b4fa9b8e8c2401d01a36731e9e1989ef27918b5bb170813b436c0e61ab630f"
)
DEPLOY12_RESULTS_CSV_SHA256 = (
    "4057d5f321ae60bd7e34bd8b3e9ca663694f189788632c34806b4bfe1b7bc4a8"
)
DEPLOY12_RESULTS_SHA256 = (
    "502c835fd33fd5bd0abee11ae2548eaf099e39653671d9a1a3c993a76530c6c3"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("matched_seam_frontier_run", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_plotter():
    spec = importlib.util.spec_from_file_location("matched_seam_frontier_plot", PLOTTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_deploy12_runner():
    spec = importlib.util.spec_from_file_location(
        "matched_seam_frontier_deploy12", DEPLOY12_RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_published_record_and_figures_are_locked() -> None:
    assert _sha256(RECORD) == RECORD_SHA256
    assert _sha256(RESULTS_CSV) == RESULTS_CSV_SHA256
    assert _sha256(PDF) == PDF_SHA256
    assert _sha256(PNG) == PNG_SHA256
    assert b"\r" not in RECORD.read_bytes()
    assert b"\r" not in RESULTS_CSV.read_bytes()

    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["schema"] == "simllm-matched-seam-frontier-record-v2"
    assert record["run_state"] == "nonvoid"
    assert record["voiding_guards"] == []
    assert record["attempt"] == "attempt-0001"
    assert record["run_commit"] == "998a6900c0991c79e533c22164c5ef4a6bb56d3b"
    assert all(record["fatal_guards"].values())
    assert record["first_published_run"] == {
        "attempt": "attempt-0002",
        "reason": (
            "the first freeze forbade roofline and fitted terms throughout a "
            "composition whose external resolver and empirical adjustments require them"
        ),
        "run_commit": "3e752d58c9e874f234110af69851384ea02873cd",
        "state": "void",
        "voiding_guard": "FG-1",
    }
    assert record["family_tallies"] == {
        "S": {"passed": 13, "denominator": 13},
        "R": {"passed": 10, "denominator": 10},
        "F": {"passed": 12, "denominator": 13},
        "M": {"passed": 2, "denominator": 2},
        "W": {"passed": 1, "denominator": 1},
    }
    misses = [
        row
        for row in record["rows"]
        if row["kind"] == "scored" and not row["passed"]
    ]
    assert [(row["id"], row["observed"]) for row in misses] == [
        ("F-2-09", "0.607495219355")
    ]
    assert record["families"]["M"][
        "maximum_packet_priced_to_unpriced_network_quotient"
    ]["decimal"] == pytest.approx(1.0427153998047758)
    assert record["families"]["M"]["unpriced_network_service_ps"] == 0
    assert record["families"]["M"]["third_loggopsim_priced_arm"] == {
        "ran": False,
        "scope": "not run; no isolated receiver-side serialization claim is made",
    }
    assert record["families"]["D"]["raw_prefill_pass_ms"] == pytest.approx(
        99.20380474486889
    )
    assert record["families"]["D"]["residual_from_raw_pass_ms"] == pytest.approx(
        97.21919525513111
    )
    assert len(record["families"]["candidate_grid"]["agg"]) == 25
    assert len(record["families"]["candidate_grid"]["disagg"]) == 10
    assert len(record["families"]["F"]["ideal_frontier"]) == 10
    assert len(record["families"]["F"]["packet_frontier"]) == 9
    boundary_rows = record["families"]["F"]["boundary_proximity_rows"]
    assert [row["row"] for row in boundary_rows] == [1, 2, 6, 7, 9]
    assert [row["selected_frontier_answer"] for row in boundary_rows] == [
        "external-disagg-row-02",
        "external-disagg-row-03",
        "external-disagg-row-07",
        "external-disagg-row-08",
        "external-disagg-row-10",
    ]
    assert record["determinism"] == {
        "comparison": "byte-for-byte complete scored evaluation JSON",
        "equal": True,
        "evaluation_sha256": [
            "85a37550456d753efa95d8260d291328626ef8da07b2938cd35e57deb4152f74",
            "85a37550456d753efa95d8260d291328626ef8da07b2938cd35e57deb4152f74",
        ],
        "excluded_by_name": ["elapsed_seconds", "W-1"],
        "fresh_processes": 2,
    }
    sensitivity = {
        row["adjustment_id"]: (
            row["minimum_quotient"]["decimal"],
            row["maximum_quotient"]["decimal"],
            row["baseline_reachable"],
        )
        for row in record["families"]["R"]["remove_one_sensitivity"]
    }
    assert sensitivity == {
        "prefill_latency_correction": (
            pytest.approx(0.9999466085336709),
            pytest.approx(1.0000763449740957),
            False,
        ),
        "decode_latency_correction": (
            pytest.approx(0.9258764893830284),
            pytest.approx(0.9259966157167553),
            True,
        ),
        "prefill_rate_matching_degradation": (
            pytest.approx(0.9999466085336709),
            pytest.approx(1.0000763449740957),
            False,
        ),
        "decode_rate_matching_degradation": (
            pytest.approx(0.9999466085336709),
            pytest.approx(1.0000763449740957),
            False,
        ),
        "autoscale_ttft_correction": (
            pytest.approx(0.9999466085336709),
            pytest.approx(1.0000763449740957),
            False,
        ),
        "memory_bandwidth_empirical_scale": (
            pytest.approx(0.9958990449307074),
            pytest.approx(0.9994128852086287),
            True,
        ),
        "memory_empirical_constant_latency": (
            pytest.approx(0.8736469070444556),
            pytest.approx(0.964963420119815),
            True,
        ),
        "context_attention_extra_latency_correction": (
            pytest.approx(0.9999466085336709),
            pytest.approx(1.0000763449740957),
            False,
        ),
    }
    serialized = RECORD.read_text(encoding="utf-8")
    assert "/data3/" not in serialized
    assert "/home/" not in serialized

    with RESULTS_CSV.open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(record["rows"])


def test_deploy12_third_arm_result_is_locked_and_nonvoid() -> None:
    assert _sha256(DEPLOY12_RECORD) == DEPLOY12_RECORD_SHA256
    assert _sha256(DEPLOY12_RESULTS_CSV) == DEPLOY12_RESULTS_CSV_SHA256
    assert _sha256(DEPLOY12_RESULTS) == DEPLOY12_RESULTS_SHA256
    assert b"\r" not in DEPLOY12_RECORD.read_bytes()
    assert b"\r" not in DEPLOY12_RESULTS_CSV.read_bytes()
    assert b"\r" not in DEPLOY12_RESULTS.read_bytes()
    assert "\N{EM DASH}" not in DEPLOY12_RESULTS.read_text(encoding="utf-8")

    record = json.loads(DEPLOY12_RECORD.read_text(encoding="utf-8"))
    assert record["schema"] == "simllm-matched-seam-deploy12-record-v1"
    assert record["run_state"] == "nonvoid"
    assert record["voiding_guards"] == []
    assert record["attempt"] == "attempt-0001"
    assert record["run_commit"] == "d736ec6bbbf7a246a032dbe88b74b6b3070df836"
    assert len(record["fatal_guards"]) == 15
    assert all(record["fatal_guards"].values())
    assert record["family_tallies"] == {
        "S": {"passed": 13, "denominator": 13},
        "R": {"passed": 10, "denominator": 10},
        "F": {"passed": 12, "denominator": 13},
        "M": {"passed": 2, "denominator": 2},
        "W": {"passed": 1, "denominator": 1},
    }
    assert record["determinism"] == {
        "comparison": "byte-for-byte complete scored evaluation JSON",
        "equal": True,
        "evaluation_sha256": [
            "18d29e03fb3fc7a48bdf160c8e75129b0fe72f7f4a994666eac2a23156c880bd",
            "18d29e03fb3fc7a48bdf160c8e75129b0fe72f7f4a994666eac2a23156c880bd",
        ],
        "excluded_by_name": ["elapsed_seconds", "W-1"],
        "fresh_processes": 2,
    }
    assert record["bypass_arm"] == {
        "mode": "bypass",
        "invocation_count": 0,
        "cells": [
            {
                "configuration_id": f"tp4-to-tp{decode_tp}",
                "decode_tp": decode_tp,
                "network_service_ps": 0,
            }
            for decode_tp in (2, 4, 8)
        ],
    }
    assert [
        cell["network_service_ps"] for cell in record["priced_arm"]["cells"]
    ] == [2_295_758_000, 2_295_756_000, 2_295_752_000]
    assert record["disposition"] == {
        "frontier_visible_residual_survives": True,
        "frontier_visible_rows": [1, 2, 3, 7, 8],
        "maximum_network_residual_ps": 2_365_525_200,
        "maximum_residual_penalty": {
            "numerator": 37_928_489_473,
            "denominator": 37_139_981_073,
            "decimal": pytest.approx(1.0212307162583136),
        },
    }
    rows = record["families"]["T"]["rows"]
    assert len(rows) == 10
    assert [row["row"] for row in rows if row["frontier_visible_residual"]] == [
        1,
        2,
        3,
        7,
        8,
    ]
    assert rows[0]["network_service_ps"] == {
        "unpriced": 0,
        "loggopsim_priced": 2_295_758_000,
        "packet": 4_661_283_200,
    }
    assert rows[0]["priced_penalty"]["decimal"] == pytest.approx(
        1.0210380310780114
    )
    assert rows[0]["residual_penalty"]["decimal"] == pytest.approx(
        1.0212307162583136
    )
    assert rows[0]["total_packet_penalty"]["decimal"] == pytest.approx(
        1.0427153998047758
    )
    assert all(row["multiplicative_identity_holds"] for row in rows)
    serialized = DEPLOY12_RECORD.read_text(encoding="utf-8")
    assert "/data3/" not in serialized
    assert "/home/" not in serialized

    with DEPLOY12_RESULTS_CSV.open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(record["rows"])


def test_freezes_and_external_adjustment_table_are_locked() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frozen = config["frozen_inputs"]
    assert _sha256(EXPECTATIONS) == frozen["expectations"]["sha256"]
    assert _sha256(EXPECTATIONS_V2) == frozen["expectations_v2"]["sha256"]
    assert _sha256(ADJUSTMENTS) == frozen["external_adjustments"]["sha256"]
    assert _sha256(FIGURE_ADDENDUM) == (
        "cc4dcb8c82bbcd5e542457b56d91ddf172af2cbe05e6bac5c865535dcc307762"
    )
    table = json.loads(ADJUSTMENTS.read_text(encoding="utf-8"))
    adjustments = {row["id"]: row for row in table["adjustments"]}
    assert set(adjustments) == {
        "prefill_latency_correction",
        "decode_latency_correction",
        "prefill_rate_matching_degradation",
        "decode_rate_matching_degradation",
        "autoscale_ttft_correction",
        "memory_bandwidth_empirical_scale",
        "memory_empirical_constant_latency",
        "context_attention_extra_latency_correction",
    }
    assert adjustments["prefill_rate_matching_degradation"]["value"] == "0.9"
    assert adjustments["decode_rate_matching_degradation"]["value"] == "0.92"
    assert "transposes those phase names" in table["phase_assignment_finding"]
    assert all(row["source"]["start_line"] > 0 for row in adjustments.values())
    assert all(row["documentation"]["start_line"] > 0 for row in adjustments.values())


def test_deploy12_freeze_and_prior_publications_are_locked() -> None:
    runner = _load_deploy12_runner()

    assert _sha256(EXPECTATIONS_DEPLOY12) == EXPECTATIONS_DEPLOY12_SHA256
    assert runner.EXPECTATIONS_SHA256 == EXPECTATIONS_DEPLOY12_SHA256
    assert runner._protected_hashes() == runner.PROTECTED_PRIOR_SHA256
    assert runner.PINNED_LOGGOPSIM_SHA256 == (
        "7e0f13ee3c87a20e9d2e94dbbd74c46075fd03df2f1b04d1ed9739c43ee0a2bf"
    )


def test_deploy12_loggopsim_parameters_match_the_freeze(tmp_path: Path) -> None:
    runner = _load_deploy12_runner()
    config = runner._loggopsim_config(tmp_path / "same-packet-goal.bin")

    assert config.latency_ns == 2_000
    assert config.overhead_ns == 0
    assert config.message_gap_ns == 0
    assert config.byte_gap_ns_string == "0.02"
    assert config.byte_overhead_ns == 0
    assert config.rendezvous_threshold_bytes == (1 << 63) - 1
    assert config.network_type == "LogGP"


def test_deploy12_explicit_bypass_starts_no_loggopsim(monkeypatch) -> None:
    runner = _load_deploy12_runner()
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    packet_cells = record["families"]["M"]["packet_cells"]

    def forbidden(*args, **kwargs):
        raise AssertionError("bypass reached LogGOPSim")

    monkeypatch.setattr(runner, "run_loggopsim", forbidden)
    arm = runner._network_arm(mode="bypass", packet_cells=packet_cells)

    assert arm["invocation_count"] == 0
    assert [cell["decode_tp"] for cell in arm["cells"]] == [2, 4, 8]
    assert all(cell["network_service_ps"] == 0 for cell in arm["cells"])


def test_deploy12_priced_arm_reuses_the_packet_goal_binary(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_deploy12_runner()
    cell_root = tmp_path / "tp4-to-tp2"
    cell_root.mkdir()
    goal = cell_root / "kv-redistribution.goal"
    goal.write_bytes(b"num_ranks 16\n")
    goal_binary = cell_root / "kv-redistribution.bin"
    goal_binary.write_bytes(b"same schedule")
    goal_sha256 = _sha256(goal)
    binary_sha256 = _sha256(goal_binary)

    class Result:
        max_finish_ps = runner.base.PCIE_SUBMISSION_PS + 2_295_760_000
        max_finish_host = 8
        rank_count = 16
        quiescent = True

    monkeypatch.setattr(runner, "run_loggopsim", lambda *args, **kwargs: Result())
    arm = runner._network_arm(
        mode="priced",
        packet_cells=[
            {
                "configuration_id": "tp4-to-tp2",
                "aggregate_kv_bytes": runner.base.KV_BYTES,
                "flow_count": 8,
                "artifacts": {
                    "goal": {
                        "path": "tp4-to-tp2/kv-redistribution.goal",
                        "sha256": goal_sha256,
                    },
                    "goal_binary": {
                        "path": "tp4-to-tp2/kv-redistribution.bin",
                        "sha256": binary_sha256,
                    },
                },
            }
        ],
        packet_root=tmp_path,
        loggopsim=tmp_path / "LogGOPSim",
    )

    assert arm["invocation_count"] == 1
    cell = arm["cells"][0]
    assert cell["network_service_ps"] == 2_295_760_000
    assert cell["goal_sha256"] == cell["packet_goal_sha256"] == goal_sha256
    assert (
        cell["goal_binary_sha256"]
        == cell["packet_goal_binary_sha256"]
        == binary_sha256
    )
    assert cell["argv"][0] == "LogGOPSim"
    assert cell["argv"][cell["argv"].index("-G") + 1] == "0.02"


def test_deploy12_bypass_reproduces_the_corrected_unpriced_arm() -> None:
    runner = _load_deploy12_runner()
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    ideal_points = record["families"]["F"]["ideal_points"]
    services = {2: 0, 4: 0, 8: 0}

    points, frontier = runner._project_arm(
        ideal_points, services, arm_name="explicit-bypass"
    )

    assert points == ideal_points
    assert frontier == record["families"]["F"]["ideal_frontier"]


def test_deploy12_three_arm_identity_is_exact_and_conditional() -> None:
    runner = _load_deploy12_runner()
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    packet_cells = record["families"]["M"]["packet_cells"]
    priced_cells = [
        {
            "decode_tp": int(cell["configuration_id"].rsplit("tp", 1)[1]),
            "network_service_ps": int(cell["packet_service_ps"]) - 1_000,
        }
        for cell in packet_cells
    ]
    bypass_cells = [
        {"decode_tp": decode_tp, "network_service_ps": 0}
        for decode_tp in (2, 4, 8)
    ]

    result = runner._decomposition(
        base_evaluation={"families": record["families"]},
        priced_arm={"cells": priced_cells},
        bypass_arm={"cells": bypass_cells},
    )

    assert len(result["rows"]) == 10
    assert result["frontier_visible_residual_survives"] is True
    for row in result["rows"]:
        priced = runner.base._fraction(row["priced_penalty"])
        residual = runner.base._fraction(row["residual_penalty"])
        total = runner.base._fraction(row["total_packet_penalty"])
        assert priced * residual == total
        assert row["multiplicative_identity_holds"] is True
        assert row["network_residual_ps"] == 1_000


def test_scored_value_trace_rejects_reachable_simllm_fitted_value() -> None:
    runner = _load_runner()
    trace = {
        "nodes": [
            {
                "id": "external",
                "kind": "external_composed_service",
                "origin": "external-aiconfigurator",
                "value": 10,
                "dependencies": [],
            },
            {
                "id": "local-fit",
                "kind": "fitted_constant",
                "origin": "simllm-study",
                "value": 2,
                "dependencies": [],
            },
            {
                "id": "scored",
                "kind": "scored_coordinate",
                "origin": "simllm-projection",
                "value": 12,
                "dependencies": ["external", "local-fit"],
            },
        ],
        "scored_roots": ["scored"],
    }

    assert runner._validate_scored_value_trace(trace) == [
        "forbidden SimLLM-authored fitted_constant reaches scored root through local-fit"
    ]


def test_runner_starts_directly_without_pythonpath(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    external_venv = tmp_path / "external-venv"
    external_venv.mkdir()
    txt2bin = tmp_path / "txt2bin"
    txt2bin.touch()
    htsim_rnic = tmp_path / "htsim_rnic"
    htsim_rnic.touch()
    completed = subprocess.run(
        [
            sys.executable,
            os.fspath(RUNNER),
            "--bulk-root",
            os.fspath(tmp_path / "bulk"),
            "--external-venv",
            os.fspath(external_venv),
            "--txt2bin",
            os.fspath(txt2bin),
            "--htsim-rnic",
            os.fspath(htsim_rnic),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "ModuleNotFoundError" not in completed.stderr
    assert "SIMLLM_EXTERNAL_AIC_VENV has no Python interpreter" in completed.stderr


def test_deploy12_runner_starts_directly_without_pythonpath(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    external_venv = tmp_path / "external-venv"
    external_venv.mkdir()
    txt2bin = tmp_path / "txt2bin"
    txt2bin.touch()
    htsim_rnic = tmp_path / "htsim_rnic"
    htsim_rnic.touch()
    loggopsim = tmp_path / "LogGOPSim"
    loggopsim.touch()
    completed = subprocess.run(
        [
            sys.executable,
            os.fspath(DEPLOY12_RUNNER),
            "--bulk-root",
            os.fspath(tmp_path / "bulk"),
            "--external-venv",
            os.fspath(external_venv),
            "--txt2bin",
            os.fspath(txt2bin),
            "--htsim-rnic",
            os.fspath(htsim_rnic),
            "--loggopsim",
            os.fspath(loggopsim),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "ModuleNotFoundError" not in completed.stderr
    assert "SIMLLM_EXTERNAL_AIC_VENV has no Python interpreter" in completed.stderr


def test_tracked_external_grid_constructs_all_declared_candidates() -> None:
    runner = _load_runner()
    disagg = runner._csv_rows(runner.DISAGG_PATH)
    agg = runner._csv_rows(runner.AGG_PATH)
    inventory = "a" * 64

    disagg_candidates = [
        runner._candidate(
            row,
            row_number=index,
            disaggregated=True,
            inventory_sha256=inventory,
        )
        for index, row in enumerate(disagg, start=1)
    ]
    agg_candidates = [
        runner._candidate(
            row,
            row_number=index,
            disaggregated=False,
            inventory_sha256=inventory,
        )
        for index, row in enumerate(agg, start=1)
    ]

    assert len(disagg_candidates) == 10
    assert len(agg_candidates) == 25
    assert {candidate.pools[1].tensor_parallel for candidate in disagg_candidates} == {
        2,
        4,
        8,
    }
    assert {candidate.pools[0].tensor_parallel for candidate in agg_candidates} == {
        4,
        8,
    }
    assert all(
        sum(pool.engines * pool.gpus_per_engine for pool in candidate.pools)
        == candidate.budget.max_gpus
        for candidate in (*disagg_candidates, *agg_candidates)
    )


def test_results_csv_writer_is_lf_only() -> None:
    runner = _load_runner()
    payload = runner._csv_bytes(
        [
            runner._scored_row(
                "S",
                "S-test",
                True,
                expected="exact",
                observed="exact",
            )
        ]
    )

    assert b"\r" not in payload
    assert len(list(csv.DictReader(payload.decode().splitlines()))) == 1


def test_directed_figure_is_a_record_only_projection() -> None:
    plotter = _load_plotter()
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    plot = plotter.prepare_plot_data(record)

    assert [series["id"] for series in plot["series"]] == [
        "external-agg",
        "external-disagg",
        "simllm-ideal",
        "simllm-packet",
    ]
    assert plot["agreement"] == {
        "rows": [
            {
                "row": row["row"],
                "quotient": pytest.approx(row["quotient"]["decimal"]),
            }
            for row in record["families"]["R"]["rows"]
        ],
        "frozen_band": [0.98, 1.02],
        "minimum": "0.999946608534",
        "maximum": "1.000076344974",
    }
    assert plot["mechanism"] == {
        "arrow_enabled": True,
        "candidate_rows": [1, 3],
        "selected_row": 3,
        "x": pytest.approx(84.00604166008027),
        "ideal_y": pytest.approx(644.3347078275151),
        "packet_y": pytest.approx(617.9391883424295),
        "quotient": "1.042715399805",
        "label": (
            "Their planner class prices no network cost.\n"
            "Our unpriced-network arm charges zero network service.\n"
            "This workload: packet-priced / unpriced-network\n"
            "= 1.042715399805.\n"
            "Unpriced: MEASURED-EXTERNAL.\n"
            "Packet-priced: MEASURED-EXTERNAL + SIM-DERIVED."
        ),
    }
    assert plot["f209"] == {"quotient": "0.607495219355", "passed": False}
    assert "different schedule regime and is not plotted" in plot["caption"]
    assert "does not isolate receiver-side serialization" in plot["caption"]


def test_live_sdk_reproduces_frozen_service_oracles() -> None:
    raw_venv = os.environ.get(EXTERNAL_VENV_ENV)
    if raw_venv is None:
        pytest.skip(
            f"live Family S check requires {EXTERNAL_VENV_ENV}; the locked record remains covered"
        )
    venv = Path(raw_venv)
    python = next(
        (
            path
            for path in (venv / "bin/python", venv / "Scripts/python.exe")
            if path.is_file()
        ),
        None,
    )
    if python is None:
        pytest.skip(
            f"live Family S check requires a Python interpreter in {EXTERNAL_VENV_ENV}"
        )
    completed = subprocess.run(
        [os.fspath(python), os.fspath(RUNNER), "--worker", "live-sdk"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    observed = {
        row["id"]: row["service_ms_hex"]
        for row in json.loads(completed.stdout)["services"]
    }
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = {
        oracle["id"]: oracle[
            "expected_step_ms_hex" if phase == "decode" else "expected_service_ms_hex"
        ]
        for phase in ("decode", "prefill")
        for oracle in config["oracles"][phase]
    }

    assert observed == expected
