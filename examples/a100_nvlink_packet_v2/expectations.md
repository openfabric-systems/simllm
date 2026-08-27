# A100 NVLink packet v2 expectations

## Freeze boundary

This is the expectations-only TRAF-70 record. It is committed before the
corrected harness and before any timed TRAF-70 cell. Neither this file nor
`expectations.json` may be amended after a scored cell. A new hypothesis
requires a new version and a new unscored freeze.

TRAF-65 job `198968` is evidence that the earlier capture procedure was void.
It is not a packet measurement. TRAF-65 expectations and the existing A100
candidate profile remain byte-identical until this score is published.

The new study records NVML data and raw KiB counter deltas per GPU, physical
link and local TX/RX direction. It also records replay, recovery, CRC and ECC
deltas, actual destination bytes, an extent-order ledger, applied-control
effects and an explicit throttle verdict. Candidate packet counts and
candidate raw bytes are forbidden from observation fields.

## Parameter decision rules

A value or evidence class changes only when its named rule returns
`IDENTIFIED` or `REFUTED`. A completed but inconclusive rule changes nothing.

| Rule | Module | Parameters | Identifying cases | Evidence on identification |
|---|---|---|---|---|
| `TX_PACKET_PAYLOAD_AND_HEADER` | tx | `max_payload_bytes`, `header_bytes` | 1-16 | `measured_effective_nvml_counter_fit` |
| `TX_LINK_COUNT_RATE_AND_BOND` | tx | `links_per_peer`, `per_link_rate_bytes_per_second`, `bond_policy` | 17-32 | `measured_effective_link_counter_plateau` |
| `TX_ENDPOINT_EGRESS_RATE` | tx | `endpoint_egress_rate_bytes_per_second` | 24, 26-30 | `measured_effective_endpoint_counter_plateau` |
| `TX_REQUEST_RESPONSE_DIRECTION` | tx | `request_response_direction` | 13-15, 49, 55-56, 64, 73-76 | `measured_directional_counter_conformance` |
| `TX_EFFECTIVE_CREDITS` | tx | `credit_unit_bytes`, `credits_per_destination` | 49-64 | `measured_effective_credit_knee_fit` |
| `RX_INGRESS_RATE` | rx | `ingress_rate_bytes_per_second` | 25, 33-48 | `measured_effective_ingress_counter_plateau` |
| `RX_EFFECTIVE_BUFFER` | rx | `buffer_capacity_bytes` | 33-64 | `measured_effective_buffer_knee_fit` |
| `RX_CREDIT_RETURN_LATENCY` | rx | `credit_return_latency_ps` | 49-64 | `measured_effective_recovery_gap_fit` |
| `RX_DELIVERY` | rx | `reassembly_policy`, `delivery_order` | 1-16, 33-48, 65-80 | `measured_behavioral_delivery_conformance` |
| `TX_RX_QUEUE_SCOPE` | tx_rx | `egress_queue_scope`, `ingress_queue_scope` | 65-80 | `measured_effective_queue_scope` |
| `SWITCH_DIRECT_MESH_IDENTITY` | switch | `mode` | 1-80 | `structural_direct_mesh_invariant_not_measurement` |

### `TX_PACKET_PAYLOAD_AND_HEADER`

Observations: source-side per-link raw_tx and data_tx deltas; payload_bytes and message_count stimulus fields; producer class and applied alignment, access-width, lane-mask, stride and reuse controls; case 16 blind holdout residuals.

Identification: One parameter pair is the unique minimum-residual fit for both SM producers, has no producer interaction above one quantum, and predicts every blind holdout within one quantum. Copy-engine rows must agree but cannot rescue an SM failure.

Refutation: A unique non-candidate pair passes, or only copy-engine rows fit the candidate. Publish the passing fitted values or the copy-engine-only refutation literally.

Inconclusive: No unique pair passes, counters quantize the fit, or a fatal guard fires.

### `TX_LINK_COUNT_RATE_AND_BOND`

Observations: remote-GPU-mapped per-link raw_tx deltas; per-link elapsed rate at ordered-pair saturation; balance under payload, stream, burst, direction and pair sweeps.

Identification: Every ordered pair has one stable active-link count, each active link remains at or below 25 GB/s, saturated minimum-over-maximum balance is at least 0.90, and the median of each link's three highest unthrottled plateaus is repeatable within 10 percent. This identifies the link count and measured effective per-link rate.

Refutation: A stable count differs from the candidate, a link exceeds the physical ceiling, or saturated traffic remains imbalanced below 0.90 after control sweeps.

Inconclusive: No saturation plateau exists or any applicable fatal guard fires.

Policy limit: Balanced aggregate counters identify four-link balanced striping only. They do not identify earliest-available scheduling. Keep that exact policy declared unless a packet-order observable distinguishes it.

### `TX_ENDPOINT_EGRESS_RATE`

Observations: sum of one source GPU's per-link raw_tx deltas; three-destination fanout elapsed time; throttle verdict and all per-link ceiling checks.

Identification: The median of the three highest unthrottled source raw-egress plateaus is within 10 percent across repeats and no constituent link exceeds 25 GB/s.

Refutation: A repeatable unthrottled plateau differs from the candidate by more than 10 percent.

Inconclusive: No plateau, throttling, counter unavailability, or another fatal guard.

### `TX_REQUEST_RESPONSE_DIRECTION`

Observations: write source raw_tx/data_tx and destination raw_rx/data_rx; read issuer raw_tx/data_tx and raw_rx/data_rx; read target raw_rx/data_rx and raw_tx/data_tx.

Identification: Writes carry observed data source-to-destination; reads carry nonzero raw-only request traffic issuer-to-target and observed data target-to-issuer, consistently for all direction controls.

Refutation: A guarded direction ledger consistently differs from that mapping.

Inconclusive: Raw-only request traffic is below counter resolution or a guard fires.

### `TX_EFFECTIVE_CREDITS`

Observations: applied outstanding window and offered in-flight bytes; per-direction raw bytes and completion rate; first 95-percent knee across payload, burst and gap controls.

Identification: The first 95-percent knee repeats within one sweep point for every payload arm, scales with the independently fitted wire-byte unit, is destination-scoped in cases 62 to 64, and returns after the frozen recovery gap. The fitted quantities are effective, never register claims.

Refutation: A reproducible guarded knee identifies different effective unit or count values.

Inconclusive: No common knee, an occupancy-only knee, counter quantization, or a guard.

### `RX_INGRESS_RATE`

Observations: destination GPU per-link raw_rx deltas; destination aggregate raw ingress rate; source-count, skew, burst and region controls.

Identification: The median of the three highest unthrottled destination raw-ingress plateaus is repeatable within 10 percent, checksum complete, and at or below 300 GB/s.

Refutation: A repeatable guarded plateau differs from the candidate by more than 10 percent.

Inconclusive: No plateau or any applicable fatal guard.

### `RX_EFFECTIVE_BUFFER`

Observations: offered in-flight bytes at the first destination knee; destination raw ingress, completion and post-burst drain time; checksum loss, duplication and order ledger.

Identification: A loss-free first knee repeats within one sweep point across incast and outstanding arms, localizes to one destination, and the same effective capacity predicts drain.

Refutation: A different guarded effective capacity uniquely predicts both knee and drain.

Inconclusive: No localized knee, no common capacity, or a fatal guard.

### `RX_CREDIT_RETURN_LATENCY`

Observations: completion and drain time; first inter-burst gap that restores at least 95 percent of plateau; payload and effective-credit cross checks.

Identification: The recovery threshold repeats within one frozen gap point and predicts dependent round-trip and post-burst recovery within 10 percent or 1 microsecond.

Refutation: A different guarded latency is the unique cross-case fit.

Inconclusive: No recovery threshold, timer resolution, or a fatal guard.

### `RX_DELIVERY`

Observations: expected destination byte SHA-256; observed destination byte SHA-256; extent sequence digest, missing, duplicate and out-of-order counts.

Identification: Every applicable isolated and ordered-frame row has identical expected and observed destination bytes, zero missing, duplicate and out-of-order extents, and a matching sequence digest. This measures behavioral conformance, not hidden implementation.

Refutation: A guarded repeatable mismatch identifies a different delivered behavior.

Inconclusive: Any checksum or order field is unavailable or a fatal guard fires.

### `TX_RX_QUEUE_SCOPE`

Observations: latency-flow and bulk-flow completion ledger; same-pair, other-peer, remote-incast, direction and region controls; per-link and per-direction raw/data deltas.

Identification: Other-peer interference from one source identifies TX scope; remote incast into one destination identifies RX scope; same-pair and region controls must preserve the localization. A shared-cache-line-only effect is memory acceptance.

Refutation: Guarded localization consistently contradicts the candidate module ownership.

Inconclusive: No stable localized asymmetry or any fatal guard.

### `SWITCH_DIRECT_MESH_IDENTITY`

Observations: four-GPU NV4 topology and remote-link mapping; source TX versus destination RX counter conservation; absence of an enumerated NVSwitch hop.

Identification: The topology remains a direct GPU mesh and endpoint counters conserve bytes. This retains pass-through as a structural invariant only; it never promotes a physical switch timing, FIFO, arbitration or buffer parameter to measured.

Refutation: An enumerated intermediate switch or guarded endpoint byte mismatch exists.

Inconclusive: Topology or endpoint counters are unavailable.

## Frozen fatal guards

Every guard has named observables and therefore must be scored as pass or
fatal. Missing observables are themselves fatal, never undecidable.

| Guard | Scope | Pass condition | Decidable when |
|---|---|---|---|
| `FG01_DESTINATION_INTEGRITY_AND_ORDER` | row | checksums and sequence digests match and all three counts are zero | all named fields are present for every hardware row |
| `FG02_QUALIFIED_NV4_PATH` | cell | all four observations name the same direct NV4 mesh with peer access | before and after guard records plus row link maps are complete |
| `FG03_COUNTER_AVAILABILITY_AND_MONOTONICITY` | row | all statuses are success and every modulo-safe delta is nonnegative | every active GPU link has both snapshots and statuses |
| `FG04_RAW_DATA_CONSISTENCY` | row | raw is at least data in each link direction after one-quantum allowance | FG03 passes |
| `FG05_REPLAY_RECOVERY_AND_LINK_ERRORS` | row | every nominal delta is zero | all five before and after counters and statuses exist per active link |
| `FG06_PHYSICAL_RATE_CEILINGS` | row | rates are at most 25.25, 101 and 303 GB/s respectively | FG03 passes and elapsed time is positive |
| `FG07_THROTTLE_AND_EXCLUSIVITY` | row_and_cell | verdict is CLEAR, no fatal clock-event bit is set, and no competing process exists | all row telemetry and both cell process snapshots exist |
| `FG08_APPLIED_SWEEP_CONTROLS` | row | every named input is present in the effect ledger, the canonical digest matches, and copy-engine rows use one batched or graph enqueue rather than one host enqueue per message | the complete input and effect ledgers are present |
| `FG09_OBSERVATION_HYPOTHESIS_SEPARATION` | row_and_score | rows contain no candidate_packet_count, candidate_raw_bytes, predicted counter, or candidate-valued observation field and the scorer enumerates its full search grid | raw rows and scorer fit trace are available |
| `FG10_COMPLETION_AND_DRAIN` | row | times are ordered and every expected extent has exactly one terminal | all named timing and terminal fields are present |

## Frozen 80-case catalog

Each case runs isolated and in its ordered corner frame. The complete
catalog also runs once in `all_corners_frame`, for exactly 86 resumable
cells. Per-case bands are observation bands, not candidate-generated raw
byte expectations.

| Case | Sweep | Producers | Frozen primary band | Rules |
|---|---|---|---|---|
| `CORNER_NVPKT_001_payload_bytes` | payload bytes 1 through 512 | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_PACKET_PAYLOAD_AND_HEADER`, `RX_DELIVERY`, `RX_DELIVERY`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVPKT_002_candidate_boundaries` | candidate-boundary neighbours through 4096 bytes | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_PACKET_PAYLOAD_AND_HEADER`, `RX_DELIVERY`, `RX_DELIVERY`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVPKT_003_256b_residuals` | 256*k+r residual matrix | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_PACKET_PAYLOAD_AND_HEADER`, `RX_DELIVERY`, `RX_DELIVERY`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVPKT_004_destination_alignment` | destination alignment | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_PACKET_PAYLOAD_AND_HEADER`, `RX_DELIVERY`, `RX_DELIVERY`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVPKT_005_source_alignment` | source alignment | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_PACKET_PAYLOAD_AND_HEADER`, `RX_DELIVERY`, `RX_DELIVERY`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVPKT_006_access_width` | access width | persistent_sm_peer_write, dependent_sm_peer_read | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_PACKET_PAYLOAD_AND_HEADER`, `RX_DELIVERY`, `RX_DELIVERY`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVPKT_007_active_warp_lanes` | active warp lanes | persistent_sm_peer_write, dependent_sm_peer_read | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_PACKET_PAYLOAD_AND_HEADER`, `RX_DELIVERY`, `RX_DELIVERY`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVPKT_008_lane_mask_shape` | lane-mask shape | persistent_sm_peer_write, dependent_sm_peer_read | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_PACKET_PAYLOAD_AND_HEADER`, `RX_DELIVERY`, `RX_DELIVERY`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVPKT_009_address_stride` | address stride | persistent_sm_peer_write, dependent_sm_peer_read | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_PACKET_PAYLOAD_AND_HEADER`, `RX_DELIVERY`, `RX_DELIVERY`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVPKT_010_fixed_total_bytes` | fixed total bytes with varied message size | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_PACKET_PAYLOAD_AND_HEADER`, `RX_DELIVERY`, `RX_DELIVERY`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVPKT_011_fixed_message_size` | fixed message size with varied count | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_PACKET_PAYLOAD_AND_HEADER`, `RX_DELIVERY`, `RX_DELIVERY`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVPKT_012_address_reuse` | address reuse versus separation | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_PACKET_PAYLOAD_AND_HEADER`, `RX_DELIVERY`, `RX_DELIVERY`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVPKT_013_peer_write` | peer-write length and alignment | persistent_sm_peer_write | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_PACKET_PAYLOAD_AND_HEADER`, `TX_REQUEST_RESPONSE_DIRECTION`, `RX_DELIVERY`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVPKT_014_peer_read` | peer-read length and alignment | dependent_sm_peer_read | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_PACKET_PAYLOAD_AND_HEADER`, `TX_REQUEST_RESPONSE_DIRECTION`, `RX_DELIVERY`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVPKT_015_producer_comparison` | copy-engine versus SM producers | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_PACKET_PAYLOAD_AND_HEADER`, `TX_REQUEST_RESPONSE_DIRECTION`, `RX_DELIVERY`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVPKT_016_blind_holdout` | seeded blind lengths, masks, and alignments | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_PACKET_PAYLOAD_AND_HEADER`, `RX_DELIVERY`, `RX_DELIVERY`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVBOND_017_ordered_pair_matrix` | all ordered GPU pairs | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `copy_engine_ordered_pair_payload_rate_gbps` [94.0, 94.07] GB/s | `TX_LINK_COUNT_RATE_AND_BOND`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVBOND_018_per_link_balance` | per-link counter balance | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `minimum_active_link_raw_rate_over_maximum_active_link_raw_rate` [0.9, 1.0] ratio | `TX_LINK_COUNT_RATE_AND_BOND`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVBOND_019_stream_count` | stream count | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_raw_rate_over_applicable_nameplate_ceiling` [0.0, 1.01] ratio | `TX_LINK_COUNT_RATE_AND_BOND`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVBOND_020_producer_concurrency` | producer concurrency | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_raw_rate_over_applicable_nameplate_ceiling` [0.0, 1.01] ratio | `TX_LINK_COUNT_RATE_AND_BOND`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVBOND_021_bandwidth_ramp` | bandwidth ramp | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_raw_rate_over_applicable_nameplate_ceiling` [0.0, 1.01] ratio | `TX_LINK_COUNT_RATE_AND_BOND`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVBOND_022_offered_rate` | offered rate | persistent_sm_peer_write, dependent_sm_peer_read | `observed_raw_rate_over_applicable_nameplate_ceiling` [0.0, 1.01] ratio | `TX_LINK_COUNT_RATE_AND_BOND`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVBOND_023_burst_length` | burst length | persistent_sm_peer_write, dependent_sm_peer_read | `observed_raw_rate_over_applicable_nameplate_ceiling` [0.0, 1.01] ratio | `TX_LINK_COUNT_RATE_AND_BOND`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVBOND_024_source_fanout` | source fan-out | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `copy_engine_three_way_fanout_payload_rate_gbps` [275.0, 285.0] GB/s | `TX_LINK_COUNT_RATE_AND_BOND`, `SWITCH_DIRECT_MESH_IDENTITY`, `TX_ENDPOINT_EGRESS_RATE` |
| `CORNER_NVBOND_025_destination_fanin` | destination fan-in | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_raw_rate_over_applicable_nameplate_ceiling` [0.0, 1.01] ratio | `TX_LINK_COUNT_RATE_AND_BOND`, `SWITCH_DIRECT_MESH_IDENTITY`, `RX_INGRESS_RATE` |
| `CORNER_NVBOND_026_symmetric_bidirectional` | symmetric bidirectional traffic | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_raw_rate_over_applicable_nameplate_ceiling` [0.0, 1.01] ratio | `TX_LINK_COUNT_RATE_AND_BOND`, `SWITCH_DIRECT_MESH_IDENTITY`, `TX_ENDPOINT_EGRESS_RATE` |
| `CORNER_NVBOND_027_asymmetric_bidirectional` | asymmetric bidirectional traffic | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_raw_rate_over_applicable_nameplate_ceiling` [0.0, 1.01] ratio | `TX_LINK_COUNT_RATE_AND_BOND`, `SWITCH_DIRECT_MESH_IDENTITY`, `TX_ENDPOINT_EGRESS_RATE` |
| `CORNER_NVBOND_028_disjoint_unidirectional` | disjoint unidirectional pairs | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_raw_rate_over_applicable_nameplate_ceiling` [0.0, 1.01] ratio | `TX_LINK_COUNT_RATE_AND_BOND`, `SWITCH_DIRECT_MESH_IDENTITY`, `TX_ENDPOINT_EGRESS_RATE` |
| `CORNER_NVBOND_029_disjoint_bidirectional` | disjoint bidirectional pairs | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_raw_rate_over_applicable_nameplate_ceiling` [0.0, 1.01] ratio | `TX_LINK_COUNT_RATE_AND_BOND`, `SWITCH_DIRECT_MESH_IDENTITY`, `TX_ENDPOINT_EGRESS_RATE` |
| `CORNER_NVBOND_030_full_mesh` | all ordered mesh flows | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_raw_rate_over_applicable_nameplate_ceiling` [0.0, 1.01] ratio | `TX_LINK_COUNT_RATE_AND_BOND`, `SWITCH_DIRECT_MESH_IDENTITY`, `TX_ENDPOINT_EGRESS_RATE` |
| `CORNER_NVBOND_031_startup_state` | cold versus warm startup | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_raw_rate_over_applicable_nameplate_ceiling` [0.0, 1.01] ratio | `TX_LINK_COUNT_RATE_AND_BOND`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVBOND_032_node_time_repeatability` | node and time repeatability | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `observed_raw_rate_over_applicable_nameplate_ceiling` [0.0, 1.01] ratio | `TX_LINK_COUNT_RATE_AND_BOND`, `SWITCH_DIRECT_MESH_IDENTITY` |
| `CORNER_NVINC_033_one_source` | one-source baseline | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `destination_observed_raw_ingress_rate_gbps` [0.0, 303.0] GB/s | `RX_INGRESS_RATE`, `RX_EFFECTIVE_BUFFER`, `RX_DELIVERY` |
| `CORNER_NVINC_034_two_source` | two-source simultaneous incast | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `destination_observed_raw_ingress_rate_gbps` [0.0, 303.0] GB/s | `RX_INGRESS_RATE`, `RX_EFFECTIVE_BUFFER`, `RX_DELIVERY` |
| `CORNER_NVINC_035_three_source` | three-source simultaneous incast | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `destination_observed_raw_ingress_rate_gbps` [0.0, 303.0] GB/s | `RX_INGRESS_RATE`, `RX_EFFECTIVE_BUFFER`, `RX_DELIVERY` |
| `CORNER_NVINC_036_fixed_aggregate_rate` | fixed aggregate rate across fan-in | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `destination_observed_raw_ingress_rate_gbps` [0.0, 303.0] GB/s | `RX_INGRESS_RATE`, `RX_EFFECTIVE_BUFFER`, `RX_DELIVERY` |
| `CORNER_NVINC_037_per_source_rate` | per-source offered rate | persistent_sm_peer_write, dependent_sm_peer_read | `destination_observed_raw_ingress_rate_gbps` [0.0, 303.0] GB/s | `RX_INGRESS_RATE`, `RX_EFFECTIVE_BUFFER`, `RX_DELIVERY` |
| `CORNER_NVINC_038_start_skew` | start skew | persistent_sm_peer_write, dependent_sm_peer_read | `destination_observed_raw_ingress_rate_gbps` [0.0, 303.0] GB/s | `RX_INGRESS_RATE`, `RX_EFFECTIVE_BUFFER`, `RX_DELIVERY` |
| `CORNER_NVINC_039_join_leave` | step-wise join and leave | persistent_sm_peer_write, dependent_sm_peer_read | `destination_observed_raw_ingress_rate_gbps` [0.0, 303.0] GB/s | `RX_INGRESS_RATE`, `RX_EFFECTIVE_BUFFER`, `RX_DELIVERY` |
| `CORNER_NVINC_040_burst_depth` | burst depth | persistent_sm_peer_write, dependent_sm_peer_read | `destination_observed_raw_ingress_rate_gbps` [0.0, 303.0] GB/s | `RX_INGRESS_RATE`, `RX_EFFECTIVE_BUFFER`, `RX_DELIVERY` |
| `CORNER_NVINC_041_equal_message_size` | equal message sizes | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `destination_observed_raw_ingress_rate_gbps` [0.0, 303.0] GB/s | `RX_INGRESS_RATE`, `RX_EFFECTIVE_BUFFER`, `RX_DELIVERY` |
| `CORNER_NVINC_042_unequal_message_size` | unequal message sizes | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `destination_observed_raw_ingress_rate_gbps` [0.0, 303.0] GB/s | `RX_INGRESS_RATE`, `RX_EFFECTIVE_BUFFER`, `RX_DELIVERY` |
| `CORNER_NVINC_043_one_elephant_two_mice` | one elephant with two mice | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `destination_observed_raw_ingress_rate_gbps` [0.0, 303.0] GB/s | `RX_INGRESS_RATE`, `RX_EFFECTIVE_BUFFER`, `RX_DELIVERY` |
| `CORNER_NVINC_044_two_elephants_one_mouse` | two elephants with one mouse | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `destination_observed_raw_ingress_rate_gbps` [0.0, 303.0] GB/s | `RX_INGRESS_RATE`, `RX_EFFECTIVE_BUFFER`, `RX_DELIVERY` |
| `CORNER_NVINC_045_push_distinct_buffers` | push incast into distinct buffers | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `destination_observed_raw_ingress_rate_gbps` [0.0, 303.0] GB/s | `RX_INGRESS_RATE`, `RX_EFFECTIVE_BUFFER`, `RX_DELIVERY` |
| `CORNER_NVINC_046_pull_gather` | pull gather from three buffers | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `destination_observed_raw_ingress_rate_gbps` [0.0, 303.0] GB/s | `RX_INGRESS_RATE`, `RX_EFFECTIVE_BUFFER`, `RX_DELIVERY` |
| `CORNER_NVINC_047_hot_destination` | hot destination region versus dispersed control | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `destination_observed_raw_ingress_rate_gbps` [0.0, 303.0] GB/s | `RX_INGRESS_RATE`, `RX_EFFECTIVE_BUFFER`, `RX_DELIVERY` |
| `CORNER_NVINC_048_long_soak` | long fairness, tail, and drain soak | persistent_sm_peer_write, dependent_sm_peer_read, copy_engine_reference | `destination_observed_raw_ingress_rate_gbps` [0.0, 303.0] GB/s | `RX_INGRESS_RATE`, `RX_EFFECTIVE_BUFFER`, `RX_DELIVERY` |
| `CORNER_NVCRD_049_dependent_round_trip` | dependent round trip | dependent_sm_peer_read | `dependent_latency_repeat_ratio` [0.9, 1.1] ratio | `TX_EFFECTIVE_CREDITS`, `RX_CREDIT_RETURN_LATENCY`, `RX_EFFECTIVE_BUFFER`, `TX_REQUEST_RESPONSE_DIRECTION` |
| `CORNER_NVCRD_050_outstanding_write_16b` | outstanding 16-byte writes | persistent_sm_peer_write | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_EFFECTIVE_CREDITS`, `RX_CREDIT_RETURN_LATENCY`, `RX_EFFECTIVE_BUFFER`, `TX_REQUEST_RESPONSE_DIRECTION` |
| `CORNER_NVCRD_051_outstanding_write_32b` | outstanding 32-byte writes | persistent_sm_peer_write | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_EFFECTIVE_CREDITS`, `RX_CREDIT_RETURN_LATENCY`, `RX_EFFECTIVE_BUFFER`, `TX_REQUEST_RESPONSE_DIRECTION` |
| `CORNER_NVCRD_052_outstanding_write_64b` | outstanding 64-byte writes | persistent_sm_peer_write | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_EFFECTIVE_CREDITS`, `RX_CREDIT_RETURN_LATENCY`, `RX_EFFECTIVE_BUFFER`, `TX_REQUEST_RESPONSE_DIRECTION` |
| `CORNER_NVCRD_053_outstanding_write_128b` | outstanding 128-byte writes | persistent_sm_peer_write | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_EFFECTIVE_CREDITS`, `RX_CREDIT_RETURN_LATENCY`, `RX_EFFECTIVE_BUFFER`, `TX_REQUEST_RESPONSE_DIRECTION` |
| `CORNER_NVCRD_054_outstanding_write_256b` | outstanding 256-byte writes | persistent_sm_peer_write | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_EFFECTIVE_CREDITS`, `RX_CREDIT_RETURN_LATENCY`, `RX_EFFECTIVE_BUFFER`, `TX_REQUEST_RESPONSE_DIRECTION` |
| `CORNER_NVCRD_055_outstanding_read` | outstanding reads | dependent_sm_peer_read | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_EFFECTIVE_CREDITS`, `RX_CREDIT_RETURN_LATENCY`, `RX_EFFECTIVE_BUFFER`, `TX_REQUEST_RESPONSE_DIRECTION` |
| `CORNER_NVCRD_056_outstanding_atomic` | outstanding atomics | persistent_sm_peer_atomic | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_EFFECTIVE_CREDITS`, `RX_CREDIT_RETURN_LATENCY`, `RX_EFFECTIVE_BUFFER`, `TX_REQUEST_RESPONSE_DIRECTION` |
| `CORNER_NVCRD_057_burst_length` | burst length at fixed window | persistent_sm_peer_write, dependent_sm_peer_read | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_EFFECTIVE_CREDITS`, `RX_CREDIT_RETURN_LATENCY`, `RX_EFFECTIVE_BUFFER`, `TX_REQUEST_RESPONSE_DIRECTION` |
| `CORNER_NVCRD_058_inter_message_gap` | inter-message gap | persistent_sm_peer_write, dependent_sm_peer_read | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_EFFECTIVE_CREDITS`, `RX_CREDIT_RETURN_LATENCY`, `RX_EFFECTIVE_BUFFER`, `TX_REQUEST_RESPONSE_DIRECTION` |
| `CORNER_NVCRD_059_inter_burst_gap` | inter-burst recovery gap | persistent_sm_peer_write, dependent_sm_peer_read | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_EFFECTIVE_CREDITS`, `RX_CREDIT_RETURN_LATENCY`, `RX_EFFECTIVE_BUFFER`, `TX_REQUEST_RESPONSE_DIRECTION` |
| `CORNER_NVCRD_060_duty_cycle` | duty cycle | persistent_sm_peer_write, dependent_sm_peer_read | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_EFFECTIVE_CREDITS`, `RX_CREDIT_RETURN_LATENCY`, `RX_EFFECTIVE_BUFFER`, `TX_REQUEST_RESPONSE_DIRECTION` |
| `CORNER_NVCRD_061_adaptive_knee_zoom` | adaptive zoom around first 95-percent knee | persistent_sm_peer_write, dependent_sm_peer_read | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_EFFECTIVE_CREDITS`, `RX_CREDIT_RETURN_LATENCY`, `RX_EFFECTIVE_BUFFER`, `TX_REQUEST_RESPONSE_DIRECTION` |
| `CORNER_NVCRD_062_two_streams_one_pair` | two streams on one pair | persistent_sm_peer_write, dependent_sm_peer_read | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_EFFECTIVE_CREDITS`, `RX_CREDIT_RETURN_LATENCY`, `RX_EFFECTIVE_BUFFER`, `TX_REQUEST_RESPONSE_DIRECTION` |
| `CORNER_NVCRD_063_one_source_two_peers` | one source to two peers | persistent_sm_peer_write, dependent_sm_peer_read | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_EFFECTIVE_CREDITS`, `RX_CREDIT_RETURN_LATENCY`, `RX_EFFECTIVE_BUFFER`, `TX_REQUEST_RESPONSE_DIRECTION` |
| `CORNER_NVCRD_064_opposite_directions` | opposite directions on one pair | persistent_sm_peer_write, dependent_sm_peer_read | `observed_data_bytes_over_expected_logical_bytes` [0.98, 1.02] ratio after one-quantum allowance | `TX_EFFECTIVE_CREDITS`, `RX_CREDIT_RETURN_LATENCY`, `RX_EFFECTIVE_BUFFER`, `TX_REQUEST_RESPONSE_DIRECTION` |
| `CORNER_NVHOL_065_small_behind_large` | small behind large | persistent_sm_peer_write, dependent_sm_peer_read | `latency_flow_completion_over_isolated_control` [0.25, 4.0] ratio | `TX_RX_QUEUE_SCOPE`, `RX_DELIVERY` |
| `CORNER_NVHOL_066_large_behind_small` | large behind small | persistent_sm_peer_write, dependent_sm_peer_read | `latency_flow_completion_over_isolated_control` [0.25, 4.0] ratio | `TX_RX_QUEUE_SCOPE`, `RX_DELIVERY` |
| `CORNER_NVHOL_067_separate_streams` | separate streams on one pair | persistent_sm_peer_write, dependent_sm_peer_read | `latency_flow_completion_over_isolated_control` [0.25, 4.0] ratio | `TX_RX_QUEUE_SCOPE`, `RX_DELIVERY` |
| `CORNER_NVHOL_068_alternating_sizes` | alternating sizes | persistent_sm_peer_write, dependent_sm_peer_read | `latency_flow_completion_over_isolated_control` [0.25, 4.0] ratio | `TX_RX_QUEUE_SCOPE`, `RX_DELIVERY` |
| `CORNER_NVHOL_069_bimodal_mix` | seeded bimodal mix | persistent_sm_peer_write, dependent_sm_peer_read | `latency_flow_completion_over_isolated_control` [0.25, 4.0] ratio | `TX_RX_QUEUE_SCOPE`, `RX_DELIVERY` |
| `CORNER_NVHOL_070_same_pair_bulk` | latency flow under same-pair bulk | persistent_sm_peer_write, dependent_sm_peer_read | `latency_flow_completion_over_isolated_control` [0.25, 4.0] ratio | `TX_RX_QUEUE_SCOPE`, `RX_DELIVERY` |
| `CORNER_NVHOL_071_other_peer_bulk` | latency flow under other-peer bulk from same source | persistent_sm_peer_write, dependent_sm_peer_read | `latency_flow_completion_over_isolated_control` [0.25, 4.0] ratio | `TX_RX_QUEUE_SCOPE`, `RX_DELIVERY` |
| `CORNER_NVHOL_072_remote_incast` | latency flow under remote incast | persistent_sm_peer_write, dependent_sm_peer_read | `latency_flow_completion_over_isolated_control` [0.25, 4.0] ratio | `TX_RX_QUEUE_SCOPE`, `RX_DELIVERY` |
| `CORNER_NVHOL_073_write_write` | write with write | persistent_sm_peer_write, dependent_sm_peer_read | `latency_flow_completion_over_isolated_control` [0.25, 4.0] ratio | `TX_RX_QUEUE_SCOPE`, `RX_DELIVERY` |
| `CORNER_NVHOL_074_read_read` | read with read | dependent_sm_peer_read | `latency_flow_completion_over_isolated_control` [0.25, 4.0] ratio | `TX_RX_QUEUE_SCOPE`, `RX_DELIVERY` |
| `CORNER_NVHOL_075_same_direction_read_write` | read and write with same payload direction | persistent_sm_peer_write, dependent_sm_peer_read | `latency_flow_completion_over_isolated_control` [0.25, 4.0] ratio | `TX_RX_QUEUE_SCOPE`, `RX_DELIVERY` |
| `CORNER_NVHOL_076_opposite_direction_read_write` | read and write with opposite payload directions | persistent_sm_peer_write, dependent_sm_peer_read | `latency_flow_completion_over_isolated_control` [0.25, 4.0] ratio | `TX_RX_QUEUE_SCOPE`, `RX_DELIVERY` |
| `CORNER_NVHOL_077_distinct_regions` | distinct memory regions | persistent_sm_peer_write, dependent_sm_peer_read | `latency_over_dispersed_region_control` [0.9, 1.1] ratio | `TX_RX_QUEUE_SCOPE`, `RX_DELIVERY` |
| `CORNER_NVHOL_078_shared_cache_line` | shared-cache-line hotspot versus dispersed control | persistent_sm_peer_write, dependent_sm_peer_read | `latency_flow_completion_over_isolated_control` [0.25, 4.0] ratio | `TX_RX_QUEUE_SCOPE`, `RX_DELIVERY` |
| `CORNER_NVHOL_079_post_burst_drain` | post-burst drain | persistent_sm_peer_write, dependent_sm_peer_read | `drain_time_over_serialization_floor` [1.0, 2.0] ratio | `TX_RX_QUEUE_SCOPE`, `RX_DELIVERY` |
| `CORNER_NVHOL_080_blind_mixed_soak` | seeded blind mixed soak | persistent_sm_peer_write, dependent_sm_peer_read | `latency_flow_completion_over_isolated_control` [0.25, 4.0] ratio | `TX_RX_QUEUE_SCOPE`, `RX_DELIVERY` |

## Flow-dynamics gate

The maintainer's `prompts/nvlinkflows-DRAFT.md` study opens only after all
86 cells complete, every fatal guard is decidable, and the score publishes
a non-void verdict for link bonding, effective credits, RX ingress/buffer
and queue-scope observability. The final report must state `OPEN` or
`CLOSED` explicitly and name any failed prerequisite.
