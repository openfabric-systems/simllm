"""Build the expectations-only TRAF-70 freeze from the stable TRAF-65 catalog."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

STUDY_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_ROOT.parents[1]
PREVIOUS_ROOT = REPOSITORY_ROOT / "examples" / "a100_nvlink_packet_v1"
PREVIOUS_EXPECTATIONS_SHA256 = (
    "212a7a26f54e444c9b18f1e528bd0d00b5a28e4f9e005b0dc137f477ad642571"
)
PROTECTED_CANDIDATE_SHA256 = (
    "899712c4734f7a6b410d80231291663a404511528d46aab7497b73831e0e354f"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8", newline="") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _producer_classes(ordinal: int, inherited: list[str]) -> list[str]:
    """Keep copy-engine rows only where one batched operation applies the sweep."""

    sm_write = "persistent_sm_peer_write"
    sm_read = "dependent_sm_peer_read"
    if ordinal in (6, 7, 8, 9, 22, 23, 37, 38, 39, 40):
        return [sm_write, sm_read]
    if 50 <= ordinal <= 54:
        return [sm_write]
    if ordinal in (49, 55, 74):
        return [sm_read]
    if ordinal == 56:
        return ["persistent_sm_peer_atomic"]
    if 57 <= ordinal <= 73 or 77 <= ordinal <= 80:
        return [sm_write, sm_read]
    return list(inherited)


def _band(ordinal: int) -> dict[str, object]:
    common = {
        "fatal_guard_precondition": "all applicable row and cell fatal guards pass",
        "counter_quantum_bytes": 1024,
    }
    if ordinal <= 16:
        return {
            **common,
            "metric": "observed_data_bytes_over_expected_logical_bytes",
            "low": 0.98,
            "high": 1.02,
            "unit": "ratio after one-quantum allowance",
            "secondary_metric": "candidate_blind_packet_fit_holdout_residual",
            "secondary_low": -1,
            "secondary_high": 1,
            "secondary_unit": "counter quantum",
            "note": "The fit searches the frozen parameter grid before comparing any candidate value.",
        }
    if ordinal == 17:
        return {
            **common,
            "metric": "copy_engine_ordered_pair_payload_rate_gbps",
            "low": 94.0,
            "high": 94.07,
            "unit": "GB/s",
            "provenance": "published A100 hardware envelope, not a TRAF-70 observation",
        }
    if ordinal == 18:
        return {
            **common,
            "metric": "minimum_active_link_raw_rate_over_maximum_active_link_raw_rate",
            "low": 0.9,
            "high": 1.0,
            "unit": "ratio",
            "required_active_link_count": 4,
        }
    if ordinal == 24:
        return {
            **common,
            "metric": "copy_engine_three_way_fanout_payload_rate_gbps",
            "low": 275.0,
            "high": 285.0,
            "unit": "GB/s",
            "published_anchor_gbps": 281.65,
        }
    if 17 <= ordinal <= 32:
        return {
            **common,
            "metric": "observed_raw_rate_over_applicable_nameplate_ceiling",
            "low": 0.0,
            "high": 1.01,
            "unit": "ratio",
            "repeatability_ratio_low": 0.9,
            "repeatability_ratio_high": 1.1,
        }
    if 33 <= ordinal <= 48:
        return {
            **common,
            "metric": "destination_observed_raw_ingress_rate_gbps",
            "low": 0.0,
            "high": 303.0,
            "unit": "GB/s",
            "conservation_ratio_low": 0.98,
            "conservation_ratio_high": 1.02,
            "note": "No lower-rate candidate band is frozen for incast.",
        }
    if ordinal == 49:
        return {
            **common,
            "metric": "dependent_latency_repeat_ratio",
            "low": 0.9,
            "high": 1.1,
            "unit": "ratio",
        }
    if 50 <= ordinal <= 64:
        return {
            **common,
            "metric": "observed_data_bytes_over_expected_logical_bytes",
            "low": 0.98,
            "high": 1.02,
            "unit": "ratio after one-quantum allowance",
            "knee_repeatability": "first 95-percent knee within one frozen sweep point",
            "recovery_repeatability": "full-rate return within one frozen gap point",
        }
    if ordinal == 77:
        metric = "latency_over_dispersed_region_control"
        low = 0.9
        high = 1.1
    elif ordinal == 79:
        metric = "drain_time_over_serialization_floor"
        low = 1.0
        high = 2.0
    else:
        metric = "latency_flow_completion_over_isolated_control"
        low = 0.25
        high = 4.0
    return {
        **common,
        "metric": metric,
        "low": low,
        "high": high,
        "unit": "ratio",
        "conservation_ratio_low": 0.98,
        "conservation_ratio_high": 1.02,
    }


def _rule_ids(ordinal: int) -> list[str]:
    if ordinal <= 16:
        return [
            "TX_PACKET_PAYLOAD_AND_HEADER",
            "TX_REQUEST_RESPONSE_DIRECTION" if ordinal in (13, 14, 15) else "RX_DELIVERY",
            "RX_DELIVERY",
            "SWITCH_DIRECT_MESH_IDENTITY",
        ]
    if ordinal <= 32:
        rules = ["TX_LINK_COUNT_RATE_AND_BOND", "SWITCH_DIRECT_MESH_IDENTITY"]
        if ordinal in (24, 26, 27, 28, 29, 30):
            rules.append("TX_ENDPOINT_EGRESS_RATE")
        if ordinal == 25:
            rules.append("RX_INGRESS_RATE")
        return rules
    if ordinal <= 48:
        return ["RX_INGRESS_RATE", "RX_EFFECTIVE_BUFFER", "RX_DELIVERY"]
    if ordinal <= 64:
        return [
            "TX_EFFECTIVE_CREDITS",
            "RX_CREDIT_RETURN_LATENCY",
            "RX_EFFECTIVE_BUFFER",
            "TX_REQUEST_RESPONSE_DIRECTION",
        ]
    return ["TX_RX_QUEUE_SCOPE", "RX_DELIVERY"]


def _required_observables(ordinal: int) -> list[str]:
    values = [
        "applied_controls",
        "observed_counter_deltas.per_gpu_per_link_per_direction",
        "observed_data_bytes",
        "observed_raw_bytes",
        "replay_recovery_crc_ecc_deltas",
        "destination_checksum",
        "ordering_ledger",
        "throttle_verdict",
    ]
    if ordinal <= 16:
        values.append("candidate_blind_fit_membership")
    if 17 <= ordinal <= 48:
        values.extend(["flow_rate_ledger", "active_link_ledger"])
    if 49 <= ordinal <= 64:
        values.extend(["offered_inflight_bytes", "completion_and_drain_time"])
    if ordinal >= 65:
        values.extend(["latency_flow_ledger", "bulk_flow_ledger", "drain_time"])
    return values


def _catalog(previous: dict[str, Any]) -> list[dict[str, Any]]:
    catalog = copy.deepcopy(previous["catalog"])
    for case in catalog:
        ordinal = int(case["ordinal"])
        case["producer_classes"] = _producer_classes(
            ordinal, list(case["producer_classes"])
        )
        case["expected_band"] = _band(ordinal)
        case["identification_rule_ids"] = _rule_ids(ordinal)
        case["required_observables"] = _required_observables(ordinal)
        case["candidate_field_separation"] = (
            "No candidate packet count, raw-byte value, parameter value, or expected "
            "counter delta is an observation."
        )
    return catalog


def _decision_rules() -> dict[str, object]:
    return {
        "TX_PACKET_PAYLOAD_AND_HEADER": {
            "module": "tx",
            "parameters": ["max_payload_bytes", "header_bytes"],
            "identifying_cases": list(range(1, 17)),
            "observations": [
                "source-side per-link raw_tx and data_tx deltas",
                "payload_bytes and message_count stimulus fields",
                "producer class and applied alignment, access-width, lane-mask, stride and reuse controls",
                "case 16 blind holdout residuals",
            ],
            "candidate_blind_search": {
                "max_payload_bytes": {"minimum": 16, "maximum": 4096, "step": 16},
                "header_bytes": {"minimum": 0, "maximum": 128, "step": 1},
                "training_cases": list(range(1, 16)),
                "held_out_case": 16,
                "tolerance": "one 1024-byte counter quantum per active link and direction",
            },
            "identify_when": (
                "One parameter pair is the unique minimum-residual fit for both SM producers, "
                "has no producer interaction above one quantum, and predicts every blind holdout "
                "within one quantum. Copy-engine rows must agree but cannot rescue an SM failure."
            ),
            "refute_when": (
                "A unique non-candidate pair passes, or only copy-engine rows fit the candidate. "
                "Publish the passing fitted values or the copy-engine-only refutation literally."
            ),
            "inconclusive_when": "No unique pair passes, counters quantize the fit, or a fatal guard fires.",
            "identified_evidence_class": "measured_effective_nvml_counter_fit",
        },
        "TX_LINK_COUNT_RATE_AND_BOND": {
            "module": "tx",
            "parameters": [
                "links_per_peer",
                "per_link_rate_bytes_per_second",
                "bond_policy",
            ],
            "identifying_cases": list(range(17, 33)),
            "observations": [
                "remote-GPU-mapped per-link raw_tx deltas",
                "per-link elapsed rate at ordered-pair saturation",
                "balance under payload, stream, burst, direction and pair sweeps",
            ],
            "identify_when": (
                "Every ordered pair has one stable active-link count, each active link remains at "
                "or below 25 GB/s, saturated minimum-over-maximum balance is at least 0.90, and "
                "the median of each link's three highest unthrottled plateaus is repeatable within "
                "10 percent. This identifies the link count and measured effective per-link rate."
            ),
            "bond_policy_limit": (
                "Balanced aggregate counters identify four-link balanced striping only. They do "
                "not identify earliest-available scheduling. Keep that exact policy declared "
                "unless a packet-order observable distinguishes it."
            ),
            "refute_when": (
                "A stable count differs from the candidate, a link exceeds the physical ceiling, "
                "or saturated traffic remains imbalanced below 0.90 after control sweeps."
            ),
            "inconclusive_when": "No saturation plateau exists or any applicable fatal guard fires.",
            "identified_evidence_class": "measured_effective_link_counter_plateau",
        },
        "TX_ENDPOINT_EGRESS_RATE": {
            "module": "tx",
            "parameters": ["endpoint_egress_rate_bytes_per_second"],
            "identifying_cases": [24, 26, 27, 28, 29, 30],
            "observations": [
                "sum of one source GPU's per-link raw_tx deltas",
                "three-destination fanout elapsed time",
                "throttle verdict and all per-link ceiling checks",
            ],
            "identify_when": (
                "The median of the three highest unthrottled source raw-egress plateaus is within "
                "10 percent across repeats and no constituent link exceeds 25 GB/s."
            ),
            "refute_when": "A repeatable unthrottled plateau differs from the candidate by more than 10 percent.",
            "inconclusive_when": "No plateau, throttling, counter unavailability, or another fatal guard.",
            "identified_evidence_class": "measured_effective_endpoint_counter_plateau",
        },
        "TX_REQUEST_RESPONSE_DIRECTION": {
            "module": "tx",
            "parameters": ["request_response_direction"],
            "identifying_cases": [13, 14, 15, 49, 55, 56, 64, 73, 74, 75, 76],
            "observations": [
                "write source raw_tx/data_tx and destination raw_rx/data_rx",
                "read issuer raw_tx/data_tx and raw_rx/data_rx",
                "read target raw_rx/data_rx and raw_tx/data_tx",
            ],
            "identify_when": (
                "Writes carry observed data source-to-destination; reads carry nonzero raw-only "
                "request traffic issuer-to-target and observed data target-to-issuer, consistently "
                "for all direction controls."
            ),
            "refute_when": "A guarded direction ledger consistently differs from that mapping.",
            "inconclusive_when": "Raw-only request traffic is below counter resolution or a guard fires.",
            "identified_evidence_class": "measured_directional_counter_conformance",
        },
        "TX_EFFECTIVE_CREDITS": {
            "module": "tx",
            "parameters": ["credit_unit_bytes", "credits_per_destination"],
            "identifying_cases": list(range(49, 65)),
            "observations": [
                "applied outstanding window and offered in-flight bytes",
                "per-direction raw bytes and completion rate",
                "first 95-percent knee across payload, burst and gap controls",
            ],
            "identify_when": (
                "The first 95-percent knee repeats within one sweep point for every payload arm, "
                "scales with the independently fitted wire-byte unit, is destination-scoped in "
                "cases 62 to 64, and returns after the frozen recovery gap. The fitted quantities "
                "are effective, never register claims."
            ),
            "refute_when": "A reproducible guarded knee identifies different effective unit or count values.",
            "inconclusive_when": "No common knee, an occupancy-only knee, counter quantization, or a guard.",
            "identified_evidence_class": "measured_effective_credit_knee_fit",
        },
        "RX_INGRESS_RATE": {
            "module": "rx",
            "parameters": ["ingress_rate_bytes_per_second"],
            "identifying_cases": [25, *list(range(33, 49))],
            "observations": [
                "destination GPU per-link raw_rx deltas",
                "destination aggregate raw ingress rate",
                "source-count, skew, burst and region controls",
            ],
            "identify_when": (
                "The median of the three highest unthrottled destination raw-ingress plateaus is "
                "repeatable within 10 percent, checksum complete, and at or below 300 GB/s."
            ),
            "refute_when": "A repeatable guarded plateau differs from the candidate by more than 10 percent.",
            "inconclusive_when": "No plateau or any applicable fatal guard.",
            "identified_evidence_class": "measured_effective_ingress_counter_plateau",
        },
        "RX_EFFECTIVE_BUFFER": {
            "module": "rx",
            "parameters": ["buffer_capacity_bytes"],
            "identifying_cases": [*list(range(33, 49)), *list(range(49, 65))],
            "observations": [
                "offered in-flight bytes at the first destination knee",
                "destination raw ingress, completion and post-burst drain time",
                "checksum loss, duplication and order ledger",
            ],
            "identify_when": (
                "A loss-free first knee repeats within one sweep point across incast and outstanding "
                "arms, localizes to one destination, and the same effective capacity predicts drain."
            ),
            "refute_when": "A different guarded effective capacity uniquely predicts both knee and drain.",
            "inconclusive_when": "No localized knee, no common capacity, or a fatal guard.",
            "identified_evidence_class": "measured_effective_buffer_knee_fit",
        },
        "RX_CREDIT_RETURN_LATENCY": {
            "module": "rx",
            "parameters": ["credit_return_latency_ps"],
            "identifying_cases": list(range(49, 65)),
            "observations": [
                "completion and drain time",
                "first inter-burst gap that restores at least 95 percent of plateau",
                "payload and effective-credit cross checks",
            ],
            "identify_when": (
                "The recovery threshold repeats within one frozen gap point and predicts dependent "
                "round-trip and post-burst recovery within 10 percent or 1 microsecond."
            ),
            "refute_when": "A different guarded latency is the unique cross-case fit.",
            "inconclusive_when": "No recovery threshold, timer resolution, or a fatal guard.",
            "identified_evidence_class": "measured_effective_recovery_gap_fit",
        },
        "RX_DELIVERY": {
            "module": "rx",
            "parameters": ["reassembly_policy", "delivery_order"],
            "identifying_cases": [*list(range(1, 17)), *list(range(33, 49)), *list(range(65, 81))],
            "observations": [
                "expected destination byte SHA-256",
                "observed destination byte SHA-256",
                "extent sequence digest, missing, duplicate and out-of-order counts",
            ],
            "identify_when": (
                "Every applicable isolated and ordered-frame row has identical expected and observed "
                "destination bytes, zero missing, duplicate and out-of-order extents, and a matching "
                "sequence digest. This measures behavioral conformance, not hidden implementation."
            ),
            "refute_when": "A guarded repeatable mismatch identifies a different delivered behavior.",
            "inconclusive_when": "Any checksum or order field is unavailable or a fatal guard fires.",
            "identified_evidence_class": "measured_behavioral_delivery_conformance",
        },
        "TX_RX_QUEUE_SCOPE": {
            "module": "tx_rx",
            "parameters": ["egress_queue_scope", "ingress_queue_scope"],
            "identifying_cases": list(range(65, 81)),
            "observations": [
                "latency-flow and bulk-flow completion ledger",
                "same-pair, other-peer, remote-incast, direction and region controls",
                "per-link and per-direction raw/data deltas",
            ],
            "identify_when": (
                "Other-peer interference from one source identifies TX scope; remote incast into one "
                "destination identifies RX scope; same-pair and region controls must preserve the "
                "localization. A shared-cache-line-only effect is memory acceptance."
            ),
            "refute_when": "Guarded localization consistently contradicts the candidate module ownership.",
            "inconclusive_when": "No stable localized asymmetry or any fatal guard.",
            "identified_evidence_class": "measured_effective_queue_scope",
        },
        "SWITCH_DIRECT_MESH_IDENTITY": {
            "module": "switch",
            "parameters": ["mode"],
            "identifying_cases": list(range(1, 81)),
            "observations": [
                "four-GPU NV4 topology and remote-link mapping",
                "source TX versus destination RX counter conservation",
                "absence of an enumerated NVSwitch hop",
            ],
            "identify_when": (
                "The topology remains a direct GPU mesh and endpoint counters conserve bytes. This "
                "retains pass-through as a structural invariant only; it never promotes a physical "
                "switch timing, FIFO, arbitration or buffer parameter to measured."
            ),
            "refute_when": "An enumerated intermediate switch or guarded endpoint byte mismatch exists.",
            "inconclusive_when": "Topology or endpoint counters are unavailable.",
            "identified_evidence_class": "structural_direct_mesh_invariant_not_measurement",
        },
    }


def _fatal_guards() -> list[dict[str, object]]:
    return [
        {
            "id": "FG01_DESTINATION_INTEGRITY_AND_ORDER",
            "scope": "row",
            "observables": [
                "destination_checksum.expected_sha256",
                "destination_checksum.observed_sha256",
                "ordering_ledger.expected_sequence_sha256",
                "ordering_ledger.observed_sequence_sha256",
                "ordering_ledger.missing",
                "ordering_ledger.duplicate",
                "ordering_ledger.out_of_order",
            ],
            "pass_when": "checksums and sequence digests match and all three counts are zero",
            "fatal_when": "any mismatch or nonzero count",
            "decidable_when": "all named fields are present for every hardware row",
        },
        {
            "id": "FG02_QUALIFIED_NV4_PATH",
            "scope": "cell",
            "observables": [
                "four A100-SXM4-80GB identities before and after",
                "NV4 topology before and after",
                "per-link remote GPU mapping",
                "CUDA peer-access result for every ordered pair",
            ],
            "pass_when": "all four observations name the same direct NV4 mesh with peer access",
            "fatal_when": "non-NV4, an intermediate path, missing peer access, or topology change",
            "decidable_when": "before and after guard records plus row link maps are complete",
        },
        {
            "id": "FG03_COUNTER_AVAILABILITY_AND_MONOTONICITY",
            "scope": "row",
            "observables": [
                "data_tx_kib before and after",
                "data_rx_kib before and after",
                "raw_tx_kib before and after",
                "raw_rx_kib before and after",
                "counter API return status and timestamp",
            ],
            "pass_when": "all statuses are success and every modulo-safe delta is nonnegative",
            "fatal_when": "unsupported, missing, reset, ambiguous wrap, or non-monotone counter",
            "decidable_when": "every active GPU link has both snapshots and statuses",
        },
        {
            "id": "FG04_RAW_DATA_CONSISTENCY",
            "scope": "row",
            "observables": ["per-link raw TX/RX deltas", "per-link data TX/RX deltas"],
            "pass_when": "raw is at least data in each link direction after one-quantum allowance",
            "fatal_when": "raw is below data beyond one 1024-byte quantum",
            "decidable_when": "FG03 passes",
        },
        {
            "id": "FG05_REPLAY_RECOVERY_AND_LINK_ERRORS",
            "scope": "row",
            "observables": [
                "DL replay delta",
                "DL recovery delta",
                "CRC flit delta",
                "CRC data delta",
                "ECC data delta",
            ],
            "pass_when": "every nominal delta is zero",
            "fatal_when": "any unexplained nominal delta is nonzero",
            "decidable_when": "all five before and after counters and statuses exist per active link",
        },
        {
            "id": "FG06_PHYSICAL_RATE_CEILINGS",
            "scope": "row",
            "observables": [
                "per-link raw directional delta and elapsed time",
                "ordered-pair raw directional sum and elapsed time",
                "per-GPU raw directional sum and elapsed time",
            ],
            "pass_when": "rates are at most 25.25, 101 and 303 GB/s respectively",
            "fatal_when": "any ceiling including its frozen 1 percent timer allowance is exceeded",
            "decidable_when": "FG03 passes and elapsed time is positive",
        },
        {
            "id": "FG07_THROTTLE_AND_EXCLUSIVITY",
            "scope": "row_and_cell",
            "observables": [
                "before and after clock-event reason masks",
                "SM and memory clocks, power and temperature",
                "explicit row throttle_verdict",
                "before and after competing-process lists",
                "exclusive a100-hourly allocation metadata",
            ],
            "pass_when": "verdict is CLEAR, no fatal clock-event bit is set, and no competing process exists",
            "fatal_when": "thermal or hardware slowdown, power-brake, unknown throttle, competitor, or nonexclusive allocation",
            "decidable_when": "all row telemetry and both cell process snapshots exist",
        },
        {
            "id": "FG08_APPLIED_SWEEP_CONTROLS",
            "scope": "row",
            "observables": [
                "point input controls",
                "applied_controls with effect ledger",
                "applied_control_sha256",
                "copy_engine_batch_mode, host enqueue count and logical message count",
            ],
            "pass_when": (
                "every named input is present in the effect ledger, the canonical digest matches, "
                "and copy-engine rows use one batched or graph enqueue rather than one host enqueue per message"
            ),
            "fatal_when": "any parsed-only control, digest mismatch, or one copy-engine host enqueue per message",
            "decidable_when": "the complete input and effect ledgers are present",
        },
        {
            "id": "FG09_OBSERVATION_HYPOTHESIS_SEPARATION",
            "scope": "row_and_score",
            "observables": ["result JSON field names", "candidate-blind fit trace"],
            "pass_when": (
                "rows contain no candidate_packet_count, candidate_raw_bytes, predicted counter, "
                "or candidate-valued observation field and the scorer enumerates its full search grid"
            ),
            "fatal_when": "a candidate-derived field is labeled or consumed as an observation",
            "decidable_when": "raw rows and scorer fit trace are available",
        },
        {
            "id": "FG10_COMPLETION_AND_DRAIN",
            "scope": "row",
            "observables": [
                "positive batch elapsed time",
                "completion event time",
                "post-work drain event time",
                "terminal extent count",
            ],
            "pass_when": "times are ordered and every expected extent has exactly one terminal",
            "fatal_when": "missing completion, negative ordering, timeout, or terminal mismatch",
            "decidable_when": "all named timing and terminal fields are present",
        },
    ]


def build_expectations() -> dict[str, object]:
    previous_path = PREVIOUS_ROOT / "expectations.json"
    candidate_path = PREVIOUS_ROOT / "candidate-profile.json"
    if _sha256(previous_path) != PREVIOUS_EXPECTATIONS_SHA256:
        raise RuntimeError("TRAF-65 expectations changed before the TRAF-70 freeze")
    if _sha256(candidate_path) != PROTECTED_CANDIDATE_SHA256:
        raise RuntimeError("the protected A100 candidate changed before the TRAF-70 freeze")
    previous = _read_json(previous_path)
    return {
        "schema": "simllm-a100-nvlink-packet-expectations-v2",
        "study_id": "a100_nvlink_packet_v2",
        "task_id": "TRAF-70",
        "status": "expectations_only_frozen_before_harness",
        "frozen_on": "2026-08-27",
        "chronology": {
            "rule": "Commit this record before the corrected harness and before any timed TRAF-70 cell.",
            "prohibited_writeback": (
                "No TRAF-70 observation may change expectations.json or expectations.md. "
                "A changed rule requires a new study version and a new unscored freeze."
            ),
            "scored_cell_definition": (
                "A cell becomes scored when a hardware attempt with this freeze digest writes its immutable plan.json."
            ),
            "prior_void_is_not_measurement": "TRAF-65 job 198968 is procedure-defect evidence only.",
        },
        "protected_inputs_until_score_publication": {
            "traf65_expectations": {
                "path": "examples/a100_nvlink_packet_v1/expectations.json",
                "sha256": PREVIOUS_EXPECTATIONS_SHA256,
            },
            "a100_candidate_profile": {
                "path": "examples/a100_nvlink_packet_v1/candidate-profile.json",
                "sha256": PROTECTED_CANDIDATE_SHA256,
            },
            "rule": (
                "Both files remain byte-identical until the TRAF-70 score is published. "
                "Afterward only parameters literally identified by these rules may change."
            ),
        },
        "target": {
            "cluster": "Merlin",
            "partition": "a100-hourly",
            "node_shape": "one exclusive qualified four-A100-SXM4-80GB NV4 node",
            "array_pacing": "short cells, at most one four-GPU cell active with %1",
            "bulk_root_local": "/" + "data3/yifeng/simllm-dev/wave-runs/traf70",
            "bulk_root_merlin": "~" + "/simllm-data/traf70",
            "resumption": "skip only digest-complete immutable attempts",
            "ssh_loss": "staging is atomic and scheduler signals write a clean resumable stop record",
        },
        "published_measurement_anchors": previous["published_measurement_anchors"],
        "candidate_reference_not_observation": previous["candidate_hypothesis"],
        "observation_contract": {
            "schema": "simllm-a100-nvlink-packet-observation-v2",
            "candidate_separation": (
                "Candidate-derived packet counts, raw bytes and expected parameter values are forbidden from observation fields."
            ),
            "required_per_row": _required_observables(80),
            "raw_data_units": "NVML throughput-counter KiB, converted to bytes only as observed_kib * 1024",
            "direction_vocabulary": "GPU-local TX and RX, with remote GPU and physical link IDs recorded",
            "copy_engine_contract": (
                "One batched 2D transfer or one CUDA graph launch per flow/stream; never one host enqueue per logical message."
            ),
            "checksum_contract": (
                "SHA-256 over bytes copied back from the actual delivery destination, compared to independently generated expected bytes."
            ),
            "order_contract": (
                "Expected and observed extent-sequence digests plus missing, duplicate and out-of-order counts."
            ),
        },
        "global_acceptance": {
            "cell_count": 86,
            "isolated_case_count": 80,
            "ordered_corner_frame_count": 5,
            "all_corners_frame_count": 1,
            "fatal_guard_requirement": "every guard is decidable and passes for each applicable scope",
            "counter_quantum_bytes": 1024,
            "held_out_error": "at most 10 percent or 1 microsecond, whichever is larger",
            "first_saturation_knee": "within one frozen sweep point",
            "parameter_publication": (
                "Only a parameter with an IDENTIFIED or REFUTED result from its named rule may change value or evidence class."
            ),
            "flow_dynamics_gate": (
                "Open only if all 86 cells complete, every fatal guard is decidable, and the score publishes a non-void verdict for link bonding, effective credits, RX ingress/buffer, and queue-scope observability."
            ),
        },
        "fatal_guards": _fatal_guards(),
        "decision_rules": _decision_rules(),
        "catalog": _catalog(previous),
    }


def render_markdown(expectations: dict[str, object]) -> str:
    rules = expectations["decision_rules"]
    guards = expectations["fatal_guards"]
    catalog = expectations["catalog"]
    if not isinstance(rules, dict) or not isinstance(guards, list) or not isinstance(catalog, list):
        raise TypeError("expectations sections have invalid types")
    lines = [
        "# A100 NVLink packet v2 expectations",
        "",
        "## Freeze boundary",
        "",
        "This is the expectations-only TRAF-70 record. It is committed before the",
        "corrected harness and before any timed TRAF-70 cell. Neither this file nor",
        "`expectations.json` may be amended after a scored cell. A new hypothesis",
        "requires a new version and a new unscored freeze.",
        "",
        "TRAF-65 job `198968` is evidence that the earlier capture procedure was void.",
        "It is not a packet measurement. TRAF-65 expectations and the existing A100",
        "candidate profile remain byte-identical until this score is published.",
        "",
        "The new study records NVML data and raw KiB counter deltas per GPU, physical",
        "link and local TX/RX direction. It also records replay, recovery, CRC and ECC",
        "deltas, actual destination bytes, an extent-order ledger, applied-control",
        "effects and an explicit throttle verdict. Candidate packet counts and",
        "candidate raw bytes are forbidden from observation fields.",
        "",
        "## Parameter decision rules",
        "",
        "A value or evidence class changes only when its named rule returns",
        "`IDENTIFIED` or `REFUTED`. A completed but inconclusive rule changes nothing.",
        "",
        "| Rule | Module | Parameters | Identifying cases | Evidence on identification |",
        "|---|---|---|---|---|",
    ]
    for rule_id, value in rules.items():
        if not isinstance(value, dict):
            raise TypeError(f"decision rule {rule_id} is invalid")
        case_text = _compact_ordinals(list(value["identifying_cases"]))
        lines.append(
            "| `{}` | {} | {} | {} | `{}` |".format(
                rule_id,
                value["module"],
                ", ".join(f"`{item}`" for item in value["parameters"]),
                case_text,
                value["identified_evidence_class"],
            )
        )
    for rule_id, value in rules.items():
        lines.extend(
            [
                "",
                f"### `{rule_id}`",
                "",
                f"Observations: {'; '.join(value['observations'])}.",
                "",
                f"Identification: {value['identify_when']}",
                "",
                f"Refutation: {value['refute_when']}",
                "",
                f"Inconclusive: {value['inconclusive_when']}",
            ]
        )
        if "bond_policy_limit" in value:
            lines.extend(["", f"Policy limit: {value['bond_policy_limit']}"])
    lines.extend(
        [
            "",
            "## Frozen fatal guards",
            "",
            "Every guard has named observables and therefore must be scored as pass or",
            "fatal. Missing observables are themselves fatal, never undecidable.",
            "",
            "| Guard | Scope | Pass condition | Decidable when |",
            "|---|---|---|---|",
        ]
    )
    for guard in guards:
        lines.append(
            f"| `{guard['id']}` | {guard['scope']} | {guard['pass_when']} | {guard['decidable_when']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen 80-case catalog",
            "",
            "Each case runs isolated and in its ordered corner frame. The complete",
            "catalog also runs once in `all_corners_frame`, for exactly 86 resumable",
            "cells. Per-case bands are observation bands, not candidate-generated raw",
            "byte expectations.",
            "",
            "| Case | Sweep | Producers | Frozen primary band | Rules |",
            "|---|---|---|---|---|",
        ]
    )
    for case in catalog:
        band = case["expected_band"]
        band_text = (
            f"`{band['metric']}` [{band['low']}, {band['high']}] {band['unit']}"
        )
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                case["stable_name"],
                case["sweep"],
                ", ".join(case["producer_classes"]),
                band_text,
                ", ".join(f"`{item}`" for item in case["identification_rule_ids"]),
            )
        )
    lines.extend(
        [
            "",
            "## Flow-dynamics gate",
            "",
            "The maintainer's `prompts/nvlinkflows-DRAFT.md` study opens only after all",
            "86 cells complete, every fatal guard is decidable, and the score publishes",
            "a non-void verdict for link bonding, effective credits, RX ingress/buffer",
            "and queue-scope observability. The final report must state `OPEN` or",
            "`CLOSED` explicitly and name any failed prerequisite.",
            "",
        ]
    )
    return "\n".join(lines)


def _compact_ordinals(ordinals: list[int]) -> str:
    if not ordinals:
        return "none"
    values = sorted(int(value) for value in ordinals)
    ranges: list[str] = []
    start = values[0]
    previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = value
        previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def main() -> int:
    expectations = build_expectations()
    _write_json(STUDY_ROOT / "expectations.json", expectations)
    _write_text(STUDY_ROOT / "expectations.md", render_markdown(expectations))
    print(_sha256(STUDY_ROOT / "expectations.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
