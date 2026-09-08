import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from examples.data_recovery_v1 import run_study as study
from simllm.backends.htsim_rnic import FlowCompletion

TOPOLOGY = """Nodes 64
Tiers 2
Tier 0
Radix_Down 8
Radix_Up 8
Downlink_speed_Gbps 400
Downlink_Latency_ns 1000
Tier 1
Radix_Down 8
Downlink_speed_Gbps 400
Downlink_Latency_ns 1000
"""
HEADER = ("profile,flow_id,source,destination,tag,payload_bytes,"
          "start_time_ps,completion_time_ps,fct_ps\n")


def fixture_cell(tmp_path, *, formerly_failed=False):
    reference = tmp_path / "collective-all-to-all-w8-400g-rnic-cn"
    ideal = tmp_path / "collective-all-to-all-w8-400g-rnic-nn"
    reference.mkdir()
    ideal.mkdir()
    (reference / "collective.goal").write_text(
        "num_ranks 64\nrank 0 {\na: send 100b to 8 tag 1\n}\n")
    (reference / "collective.bin").write_bytes(b"frozen-binary")
    topology = tmp_path / "clos.topo"
    topology.write_text(TOPOLOGY)
    (reference / "completion.csv").write_text(HEADER + "rnic-cn,1,0,8,1,100,0,5000000,5000000\n")
    (ideal / "completion.csv").write_text(HEADER + "rnic-nn,1,0,8,1,100,0,4002000,4002000\n")
    cell = study.Cell("collective", reference, 8, 400, "all-to-all", topology,
                      "collective", "1", formerly_failed)
    bounds = study.pre_run_bounds(cell)
    bounds["ideal_phase_ps"] = study.ideal_reference(cell)
    bounds["engineering_budget_over_ideal"] = (
        bounds["engineering_budget_ps"] / bounds["ideal_phase_ps"])
    return cell, bounds


def manifest(arm, bounds):
    selected = study.selections(arm, bounds)
    data = (f"rnic_cn_data_recovery={selected['data_recovery']} rnic_cn_retry_probe_windows=4 "
            f"rnic_cn_initial_window={'bounded' if selected['initial_window_bytes'] is not None else 'none'} "
            f"rnic_cn_initial_window_bytes={selected['initial_window_bytes'] or 0} "
            f"rnic_cn_initial_window_fan_in={selected['initial_window_fan_in'] or 0} "
            "rnic_cn_initial_buffer_bytes=1048576 rnic_cn_initial_sizing=F-times-U-at-most-B "
            "rnic_cn_probe_epoch=physical-retry-serialization-end "
            "rnic_cn_recovery_release=actual-arrival-tick rnic_cn_tail_probes=0 "
            "rnic_cn_tail_probe_wire_bytes=0 rnic_cn_deterministic_retransmissions=0 "
            "rnic_cn_deterministic_retransmission_wire_bytes=0 "
            "rnic_cn_late_retry_admissions=0 rnic_cn_initial_window_holds=0 "
            "rnic_cn_initial_grants_dispatched=0")
    return (f"[RNIC manifest] rnic_cn_control_recovery={selected['control_recovery']} "
            "rnic_cn_control_headroom_admissions=0 rnic_cn_control_headroom_remaining_bytes=0 "
            "rnic_cn_control_headroom_peak_egress_bytes=0\n"
            f"[RNIC manifest] {data}\n"
            "[RNIC manifest] maximum_retransmissions=8 control_deadline_ps=10000000 "
            "shared_buffer_bytes=1048576 max_wire_packet_bytes=4160\n"
            "[RNIC manifest] physical_quiescence=verified\n"
            "Maximum finishing time at host 0: 5000\n")


def test_input_bound_counts_every_send_per_leaf_and_budget_terms(tmp_path):
    cell, bounds = fixture_cell(tmp_path)
    cell.goal.write_text("num_ranks 64\nrank 0 {\n"
                         "a: send 100b to 8 tag 1\nb: send 100b to 9 tag 1\n"
                         "c: send 100b to 8 tag 2\nd: send 100b to 16 tag 1\n}\n")
    bounds = study.pre_run_bounds(cell)
    assert bounds["leaf_input_send_counts"] == {"1": 3, "2": 1}
    assert bounds["declared_fan_in"] == 3
    assert bounds["initial_window_bytes"] == 1048576 // 3
    assert bounds["declared_fan_in"] * bounds["initial_window_bytes"] <= 1048576
    assert bounds["receiver_serialization_floor_ps"] == 4000
    assert bounds["phase_floor_ps"] == 4004000
    q = 3 * 1048576 * 20 + 8_000_000 + 4160 * 20
    assert bounds["budget_queue_allowance_ps"] == q
    assert bounds["engineering_budget_ps"] == 9 * 4000 + 8 * (40_000_000 + q)
    assert bounds["phase_ceiling_ps"] is None
    assert bounds["flow_ceiling_ps"] is None


def test_matrix_contains_44_identity_cases_and_all_mechanism_arms(tmp_path):
    cells = []
    for pattern in ("ring", "all-to-all"):
        for width in (8, 16, 32, 64):
            for rate in (200, 400):
                cells.append(study.Cell("collective", tmp_path / f"{pattern}-{width}-{rate}",
                                        width, rate, pattern, tmp_path, "collective", "1",
                                        pattern == "all-to-all" and width == 64))
    for attachment in ("rail", "node-local"):
        for spines in (2, 8):
            for pp in (2, 4, 8):
                for ep in (0, 8, 32):
                    cells.append(study.Cell("pipeline", tmp_path / f"{attachment}-{spines}-{pp}-{ep}",
                                            ep, 400, "pipeline", tmp_path, "step", "1",
                                            spines == 2 and ep == 32))
    jobs = study.planned_jobs(cells)
    assert len(jobs) == len({(cell.name, arm) for cell, arm in jobs}) == 96
    assert sum(arm == "legacy-none" and not cell.formerly_failed for cell, arm in jobs) == 44
    assert sum(arm == "combined" for cell, arm in jobs) == 14
    assert sum(arm == "recovery-only" for cell, arm in jobs) == 14


@pytest.mark.parametrize("arm,failed,status,fatal", [
    ("combined", True, "failed", True), ("recovery-only", True, "failed", True),
    ("window-only", False, "diagnostic-failure", False),
    ("legacy-none", True, "diagnostic-failure", False),
    ("legacy-none", False, "failed", True), ("disabled", True, "diagnostic-failure", False),
])
def test_failed_phase_is_never_reconstructed_from_completed_subset(tmp_path, arm, failed, status, fatal):
    cell, bounds = fixture_cell(tmp_path, formerly_failed=failed)
    row = study.observe(cell, cell.reference,
                        "htsim_rnic: deterministic retransmission exhausted", 2, arm, bounds)
    assert row["status"] == status
    assert bool(row["fatal_findings"]) == fatal
    assert row["phase_makespan_ps"] is row["cn_nn_phase_ratio"] is row["fct_p99_ps"] is None
    assert row["behavioral_relations"] == []


def test_unknown_window_failure_is_not_a_survivable_diagnostic(tmp_path):
    cell, bounds = fixture_cell(tmp_path)
    row = study.observe(cell, cell.reference, "invalid command line", 2, "window-only", bounds)
    assert row["fatal_findings"] == ["required cell failed to complete and quiesce"]


def test_success_exit_with_missing_flow_has_no_phase(tmp_path):
    cell, bounds = fixture_cell(tmp_path)
    (cell.reference / "completion.csv").write_text(HEADER)
    row = study.observe(cell, cell.reference, manifest("combined", bounds), 0, "combined", bounds)
    assert row["fatal_findings"] == ["flow identity loss or duplication"]
    assert row["phase_makespan_ps"] is None


def test_active_window_can_change_csv_without_changing_disabled_identity_contract(tmp_path):
    cell, bounds = fixture_cell(tmp_path)
    output = tmp_path / "candidate"
    output.mkdir()
    (output / "completion.csv").write_text(HEADER + "rnic-cn,1,0,8,1,100,0,6000000,6000000\n")
    row = study.observe(cell, output, manifest("combined", bounds), 0, "combined", bounds)
    study.apply_guards(cell, output, row)
    assert row["fatal_findings"] == []
    assert row["compatibility_oracle"] is None
    row = study.observe(cell, output, manifest("legacy-none", bounds), 0, "legacy-none", bounds)
    study.apply_guards(cell, output, row)
    assert row["fatal_findings"] == ["protected legacy completion CSV changed"]


def test_behavior_is_measured_before_guard_and_void_has_no_score(tmp_path):
    cell, bounds = fixture_cell(tmp_path)
    bounds["engineering_budget_ps"] = 4_900_000
    row = study.observe(cell, cell.reference, manifest("combined", bounds), 0, "combined", bounds)
    assert row["behavioral_relations"][0]["within_band"] is False
    assert row["fatal_findings"] == []
    bounds["phase_floor_ps"] = 5_100_000
    study.apply_guards(cell, cell.reference, row)
    result = study.summarize([row], {})
    assert result["verdict"] == "void"
    assert len(result["behavioral_findings"]) == 1
    assert "score" not in result
    assert result["exact_oracles"]["legacy_disabled_csv"] == []


def test_complete_outside_engineering_budget_is_behavioral_finding(tmp_path):
    cell, bounds = fixture_cell(tmp_path)
    bounds["engineering_budget_ps"] = 5_000_000
    row = study.observe(cell, cell.reference, manifest("combined", bounds), 0, "combined", bounds)
    study.apply_guards(cell, cell.reference, row)
    assert study.summarize([row], {})["verdict"] == "outside-budget"


@pytest.mark.parametrize("counter", ["tail_probes", "tail_probe_wire_bytes"])
def test_dormant_recovery_requires_exact_csv_and_inactive_counters(tmp_path, counter):
    cell, bounds = fixture_cell(tmp_path)
    rows = []
    for arm in ("disabled", "recovery-only"):
        row = study.observe(cell, cell.reference, manifest(arm, bounds), 0, arm, bounds)
        rows.append(study.apply_guards(cell, cell.reference, row))
    result = study.summarize(rows, {})
    assert result["verdict"] == "consumer-valid"
    assert result["exact_oracles"]["dormant_recovery_csv"][0]["byte_identical"]
    rows[1]["data_recovery"][counter] = 1
    assert study.summarize(rows, {})["verdict"] == "void"


@pytest.mark.parametrize("probes,probe_bytes,retries,retry_bytes,finding", [
    (1, 4160, 2, 8320, None),
    (1, 164, 2, 4324, None),
    (0, 4160, 2, 8320, "packet and wire-byte counts disagree"),
    (1, 0, 2, 8320, "packet and wire-byte counts disagree"),
    (2, 8321, 3, 12480, "packet and wire-byte counts disagree"),
    (3, 6000, 2, 8320, "probe counter exceeds all retransmissions: tail_probes"),
    (2, 8320, 3, 4324, "probe counter exceeds all retransmissions: tail_probe_wire_bytes"),
])
def test_probe_accounting_counts_physical_packets_and_subset_wire_bytes(
        tmp_path, probes, probe_bytes, retries, retry_bytes, finding):
    cell, bounds = fixture_cell(tmp_path)
    stdout = manifest("combined", bounds)
    for field, value in (("tail_probes", probes), ("tail_probe_wire_bytes", probe_bytes),
                         ("deterministic_retransmissions", retries),
                         ("deterministic_retransmission_wire_bytes", retry_bytes)):
        stdout = stdout.replace(f"rnic_cn_{field}=0", f"rnic_cn_{field}={value}")
    row = study.observe(cell, cell.reference, stdout, 0, "combined", bounds)
    findings = study.manifest_findings(row)
    if finding is None:
        assert findings == []
    else:
        assert any(finding in value for value in findings)


def test_completed_probe_bytes_required_and_disabled_bytes_remain_zero(tmp_path):
    cell, bounds = fixture_cell(tmp_path)
    stdout = manifest("disabled", bounds)
    row = study.observe(cell, cell.reference, stdout.replace("rnic_cn_tail_probe_wire_bytes=0 ", ""),
                        0, "disabled", bounds)
    assert "missing final data recovery counter: tail_probe_wire_bytes" in study.manifest_findings(row)
    row = study.observe(cell, cell.reference,
                        stdout.replace("rnic_cn_tail_probe_wire_bytes=0", "rnic_cn_tail_probe_wire_bytes=1"),
                        0, "disabled", bounds)
    assert "disabled recovery used probes or late retry admission" in study.manifest_findings(row)


def test_raw_outcomes_are_written_before_identity_guards_and_resume_rejects_changes(tmp_path, monkeypatch):
    cell, bounds = fixture_cell(tmp_path)
    root = tmp_path / "run"
    binary = tmp_path / "backend"
    binary.write_bytes(b"native-binary")
    provenance = {"expectations_sha256": ["frozen"], "binary_sha256": study.digest(binary),
                  "script_sha256": "runner", "helper_sha256": "helper", "wrapper_sha256": "wrapper"}

    def execute(command, **kwargs):
        output = Path(command[command.index("-completion_csv") + 1])
        output.write_text(HEADER + "rnic-cn,1,0,8,1,100,0,6000000,6000000\n")
        return CompletedProcess(command, 0, manifest("legacy-none", bounds), "")

    monkeypatch.setattr(study, "run_owned_process", execute)
    row = study.run_cell(cell, "legacy-none", binary, root, provenance, bounds)
    output = root / cell.name / "legacy-none"
    observed = json.loads((output / "observations.json").read_text())
    assert observed["fatal_findings"] == []
    assert row["fatal_findings"] == ["protected legacy completion CSV changed"]
    assert (output / "per_flow_normalization.csv").exists()
    assert study.run_cell(cell, "legacy-none", binary, root, provenance, bounds, resume=True) == row
    with pytest.raises(ValueError, match="resume input mismatch"):
        study.run_cell(cell, "legacy-none", binary, root,
                       {**provenance, "script_sha256": "different"}, bounds, resume=True)
    (output / "completion.csv").write_text(HEADER)
    with pytest.raises(ValueError, match="raw output digest changed"):
        study.run_cell(cell, "legacy-none", binary, root, provenance, bounds, resume=True)


def test_immutable_record_rejects_rescoring(tmp_path):
    path = tmp_path / "record.json"
    study.write_once(path, {"verdict": "void"})
    study.write_once(path, {"verdict": "void"})
    with pytest.raises(ValueError, match="immutable"):
        study.write_once(path, {"verdict": "consumer-valid"})


def test_missing_matrix_member_voids_even_when_individual_cell_passes(tmp_path):
    cell, bounds = fixture_cell(tmp_path)
    row = study.observe(cell, cell.reference, manifest("combined", bounds), 0, "combined", bounds)
    row = study.apply_guards(cell, cell.reference, row)
    result = study.summarize([row], {}, [(cell.name, "combined"), (cell.name, "recovery-only")])
    assert result["verdict"] == "void"
    assert any("matrix" in f["finding"] for f in result["fatal_findings"])


def test_receiver_prefix_floor_retains_physical_impossibility_check():
    flows = [FlowCompletion("rnic-cn", i, i, 8, 1, 100, 0, 4_003_000, 4_003_000)
             for i in (1, 2)]
    findings, ratio = study.receiver_prefix_findings(flows, 400)
    assert findings == ["receiver 8 prefix 2 beats byte floor"]
    assert ratio < 1
