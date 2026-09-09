"""Map frozen component contracts to executable fixtures without scoring tests."""

from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET

from .common import ROOT, read, receipts, write

CASES = {
    "visibility-before-exact-after": ["all_shards_join_without_early_visibility_or_fabricated_queue_fields", "reused_publication_and_late_publication_are_terminal"],
    "separate-shard-completions": ["all_shards_join_without_early_visibility_or_fabricated_queue_fields"],
    "same-time-distinct-native-callbacks": ["distinct_same_time_callbacks_are_all_drained_before_join_publication"],
    "engine-arrival-packet-tie": ["composed_engine_packet_arrival_and_prior_eligibility_tie"],
    "prior-injection-eligibility-tie": ["lookahead_yields_before_local_action_and_keeps_eligibility_boundary_open", "composed_engine_packet_arrival_and_prior_eligibility_tie"],
    "zero-submission-delay": ["zero_submission_delay_rejects_before_starting_another_child"],
    "duplicate-request": ["duplicate_request_rejects_before_more_shards_are_injected"],
    "foreign-request": ["bad_private_receipt_inventory_poison_and_reap[foreign-request]"],
    "endpoint-bijection-and-mutation": ["invalid_endpoint_bijection_rejects_before_native_launch[missing-node]",
        "invalid_endpoint_bijection_rejects_before_native_launch[duplicate-node]",
        "invalid_endpoint_bijection_rejects_before_native_launch[endpoint-count]", "mutable_input_manifests_cannot_reroute_bound_engines"],
    "shard-byte-change": ["bad_private_receipt_inventory_poison_and_reap[bytes]"],
    "wrong-owner-and-forged-receipt": ["bad_private_receipt_inventory_poison_and_reap[forged]"],
    "missing-and-duplicate-sequences": ["bad_private_receipt_inventory_poison_and_reap[missing]",
        "bad_private_receipt_inventory_poison_and_reap[duplicate]", "bad_private_receipt_inventory_poison_and_reap[sequence]"],
    "foreign-completion": ["native_foreign_row_is_rejected_by_complete_framed_admission"],
    "partial-join": ["all_shards_join_without_early_visibility_or_fabricated_queue_fields"],
    "early-late-reused-publication": ["reused_publication_and_late_publication_are_terminal", "reused_receipt_cannot_publish_twice"],
    "policy-replacement": ["shared_policy_replacement_rejects_before_admission_and_clock_change", "serialized_pending_policy_override_rejects_before_clock_or_identity_changes"],
    "partial-injection-failure": ["partial_injection_retains_first_accepted_shard_and_original_failure"],
    "pending-close": ["pending_close_does_not_drain_or_advance", "shared_session_pending_engine_work_remains_uncancellable"],
    "cleanup-preserves-first-exception": ["cleanup_abort_keeps_the_original_exception",
        "interrupted_shared_teardown_finishes_every_owner_and_preserves_first_failure[direct]",
        "interrupted_shared_teardown_finishes_every_owner_and_preserves_first_failure[body-failure]"],
    "same-owner-across-batches": ["repeated_batches_preserve_native_owner_history_and_prior_evidence"],
}


def capture(path):
    path.mkdir()
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_shared_kv_handoff.py",
               "tests/test_shared_handoff_study.py", "tests/test_shared_handoff_capture.py",
               "tests/test_shared_handoff_admission.py", "tests/test_child_process.py", "tests/test_flow_session.py",
               "--junitxml=" + str(path / "software.xml")]
    write(path / "command.json", command)
    try:
        with (path / "software.log").open("wb") as log:
            result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, TMPDIR=str(path)),
                                    stdout=log, stderr=subprocess.STDOUT, timeout=180, check=False)
        write(path / "process.json", {"exit_code": result.returncode})
    finally:
        primary = sys.exc_info()[1]
        try:
            write(path.parent / "components-first-receipts.json", receipts(path))
        except BaseException as error:
            if primary is None:
                raise
            primary.__dict__["receipt_failure"] = {"type": type(error).__name__, "message": str(error)}


def admit(path, frozen, aliases, evidence):
    initial = read(path.parent / "components-first-receipts.json")
    evidence.equal("component:first-receipts", receipts(path), initial)
    evidence.equal("component:software-exit", read(path / "process.json")["exit_code"], 0)
    cases = list(ET.parse(path / "software.xml").getroot().iter("testcase"))
    identities = [(row.attrib["classname"], row.attrib["name"]) for row in cases]
    evidence.equal("component:unique-identities", len(set(identities)), len(identities))
    evidence.check("component:no-failures", all(not row.findall("failure") and not row.findall("error") for row in cases))
    names = [row.attrib["name"] for row in cases if row.attrib["classname"].endswith("test_shared_kv_handoff") and not row.findall("skipped")]
    evidence.equal("component:frozen-domain", list(CASES), frozen["component_controls"])
    catalog = []
    for family, patterns in CASES.items():
        matches = []
        for index, pattern in enumerate(patterns):
            prefix = "test_" + pattern
            selected = [name for name in names if name == prefix]
            evidence.equal("component:" + family + ":fixture:" + str(index), len(selected), 1)
            matches.extend(selected)
        catalog.append({"contract": family, "fixtures": sorted(set(matches)), "evidence_class": "unscored-component"})
    alias_cases = [row.attrib["name"] for row in cases if row.attrib["classname"].endswith("test_shared_handoff_study") and not row.findall("skipped")]
    for case in aliases["fatal_controls"]:
        evidence.check("component:alias-control:" + case, "test_alias_fatal_controls_reject_before_comparison[" + case + "]" in alias_cases)
    return {"component_contracts": catalog, "software_cases": len(cases),
        "software_skips": sum(bool(row.findall("skipped")) for row in cases), "alias_fatal_controls": aliases["fatal_controls"],
        "behavioral_instances": 0, "behavioral_score": None}
