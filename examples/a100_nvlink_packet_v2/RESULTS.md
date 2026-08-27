# TRAF-70 corrected A100 NVLink packet score

## Per-module identification

| Module | Parameter | Status | Identified value | Candidate relation | Evidence class |
|---|---|---|---|---|---|
| tx | `max_payload_bytes` | INCONCLUSIVE | none | UNCHANGED | `declared_candidate_not_hardware_measurement` |
| tx | `header_bytes` | INCONCLUSIVE | none | UNCHANGED | `declared_candidate_not_hardware_measurement` |
| tx | `links_per_peer` | INCONCLUSIVE | none | UNCHANGED | `declared_candidate_not_hardware_measurement` |
| tx | `per_link_rate_bytes_per_second` | INCONCLUSIVE | none | UNCHANGED | `declared_candidate_not_hardware_measurement` |
| tx | `bond_policy` | INCONCLUSIVE | none | UNCHANGED | `declared_candidate_not_hardware_measurement` |
| tx | `endpoint_egress_rate_bytes_per_second` | IDENTIFIED | 160795737454 | REFUTED_AND_REPLACED | `measured_effective_endpoint_counter_plateau` |
| tx | `request_response_direction` | IDENTIFIED | write payload travels as request; read control travels as request and read payload travels as response | CONFIRMED | `measured_directional_counter_conformance` |
| tx | `credit_unit_bytes` | INCONCLUSIVE | none | UNCHANGED | `declared_candidate_not_hardware_measurement` |
| tx | `credits_per_destination` | INCONCLUSIVE | none | UNCHANGED | `declared_candidate_not_hardware_measurement` |
| switch | `mode` | STRUCTURAL | pass_through | RETAINED_STRUCTURAL | `structural_direct_mesh_invariant_not_measurement` |
| rx | `ingress_rate_bytes_per_second` | IDENTIFIED | 207101921876 | REFUTED_AND_REPLACED | `measured_effective_ingress_counter_plateau` |
| rx | `buffer_capacity_bytes` | INCONCLUSIVE | none | UNCHANGED | `declared_candidate_not_hardware_measurement` |
| rx | `credit_return_latency_ps` | INCONCLUSIVE | none | UNCHANGED | `declared_candidate_not_hardware_measurement` |
| rx | `reassembly_policy` | IDENTIFIED | extent_sequence | CONFIRMED | `measured_behavioral_delivery_conformance` |
| rx | `delivery_order` | IDENTIFIED | per_extent | CONFIRMED | `measured_behavioral_delivery_conformance` |
| tx_rx | `queue_scope` | INCONCLUSIVE | none | UNCHANGED | `declared_candidate_not_hardware_measurement` |

## Fatal-guard verdicts

| Guard | Verdict | Decidable | Observations | Failures | Missing |
|---|---|---|---:|---:|---:|
| `FG01` | PASS | yes | 11542 | 0 | 0 |
| `FG02` | PASS | yes | 11628 | 0 | 0 |
| `FG03` | PASS | yes | 11542 | 0 | 0 |
| `FG04` | PASS | yes | 11542 | 0 | 0 |
| `FG05` | PASS | yes | 11542 | 0 | 0 |
| `FG06` | PASS | yes | 11542 | 0 | 0 |
| `FG07` | PASS | yes | 11628 | 0 | 0 |
| `FG08` | PASS | yes | 11542 | 0 | 0 |
| `FG09` | PASS | yes | 11543 | 0 | 0 |
| `FG10` | PASS | yes | 11542 | 0 | 0 |

## Flow-dynamics gate

Gate verdict: `OPEN`. all frozen prerequisites have a decidable non-void outcome

## Execution coverage

- Status: `COMPLETE_VALID_86_OF_86`.
- Measurement validity: `VALID_FOR_FROZEN_RULES`.
- Scheduler job: `199957,199960`.
- Expectations SHA-256: `f0ab026e054873a56614af63ab3a7ae3219dc0b045423808cb41522910fa6da6`.
- Digest-complete cells: 86 of 86.
- Hardware rows: 11,542.
- Exact pending array: `none`.

## Score-authorized profile changes

| Module | Parameter | Value | Candidate relation | Evidence class | Rule |
|---|---|---|---|---|---|
| tx | `endpoint_egress_rate_bytes_per_second` | 160795737454 | REFUTED_AND_REPLACED | `measured_effective_endpoint_counter_plateau` | `TX_ENDPOINT_EGRESS_RATE` |
| tx | `request_response_direction` | write payload travels as request; read control travels as request and read payload travels as response | CONFIRMED | `measured_directional_counter_conformance` | `TX_REQUEST_RESPONSE_DIRECTION` |
| rx | `ingress_rate_bytes_per_second` | 207101921876 | REFUTED_AND_REPLACED | `measured_effective_ingress_counter_plateau` | `RX_INGRESS_RATE` |
| rx | `reassembly_policy` | extent_sequence | CONFIRMED | `measured_behavioral_delivery_conformance` | `RX_DELIVERY` |
| rx | `delivery_order` | per_extent | CONFIRMED | `measured_behavioral_delivery_conformance` | `RX_DELIVERY` |

The scorer did not amend the frozen expectations. Candidate-derived packet
fields were not consumed as observations.
