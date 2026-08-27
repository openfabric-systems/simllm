# TRAF-65 A100 NVLink packet hardware score

Status: `COMPLETE_VOID_86_OF_86`.

## Per-corner verdicts

| Corner | Verdict | Pass | Refuted | Unscorable | Pending |
|---|---|---:|---:|---:|---:|
| packetization | UNSCORABLE_RUN_VOID | 0 | 0 | 16 | 0 |
| bond_and_wire | MEASURED_BAND_REFUTED_BUT_RUN_VOID | 0 | 3 | 13 | 0 |
| incast_and_destination_fifo | MEASURED_BAND_REFUTED_BUT_RUN_VOID | 0 | 16 | 0 | 0 |
| credit_depletion_and_return | UNSCORABLE_RUN_VOID | 0 | 0 | 16 | 0 |
| fifo_partition_and_head_of_line | UNSCORABLE_RUN_VOID | 0 | 0 | 16 | 0 |

## Module-parameter identification

| Module | Parameters | Status | Reason |
|---|---|---|---|
| TX | maximum payload; header; packet granularity | UNIDENTIFIABLE_RUN_VOID | no per-row observed raw/data counter deltas |
| TX | links per peer; per-link rate; bond policy | UNIDENTIFIABLE_RUN_VOID | no per-row per-link counter deltas |
| TX | request/response direction | UNIDENTIFIABLE_RUN_VOID | no direction-specific request/data counter ledger |
| TX | effective credit unit and destination window | UNIDENTIFIABLE_RUN_VOID | no recorded applied window and no valid first-knee evidence |
| Switch | pass-through mode; zero bytes and time | STANDS_STRUCTURALLY_NOT_MEASURED | mandatory direct-mesh model invariant; hardware rows expose no switch visit |
| RX | ingress rate and buffer capacity | UNIDENTIFIABLE_RUN_VOID | no valid destination counter, occupancy, overflow, or drain ledger |
| RX | credit return latency | UNIDENTIFIABLE_RUN_VOID | no credit-return event or applied recovery-gap observation |
| RX | reassembly and delivery order | UNIDENTIFIABLE_RUN_VOID | checksum_ok is reported without measured destination bytes or an order ledger |

## Candidate-profile decision

The decision is `RETAIN_DECLARED_CANDIDATE_NO_HARDWARE_PROMOTION`. No completed timing can satisfy the frozen measurement-validity contract while the raw/data, replay, recovery, corruption, and throttle observables are absent.

The switch result is `PASS_THROUGH_STANDS_STRUCTURALLY_NOT_MEASURED`. The packet-overhead result is `UNIDENTIFIABLE_FROM_CAPTURED_ROW_SCHEMA`, and the copy-engine coalescing result is `UNIDENTIFIABLE_FROM_CAPTURED_ROW_SCHEMA`.

## Execution coverage and exact remainder

- Scheduler job: `198968`.
- Frozen execution head: `2ab092f9255d77c00c547446b65534a3b273ec82`.
- Frozen expectations SHA-256: `212a7a26f54e444c9b18f1e528bd0d00b5a28e4f9e005b0dc137f477ad642571`.
- Digest-complete cells: 86 of 86.
- Hardware rows: 14,035, including 4 protocol-validation rows.
- Consecutive completed prefix: indices 0 through 85.
- Exact pending array: `none`.
- Batch binary audit: `BUILD_REPRODUCIBILITY_MISMATCH`; observed `992eaa12d5953806a1f21d12fce612d72f721a141d425a666404ffb26770c3e1` against compile-check `96b4c544de54457d1fbed8e56b0a1cbe61344bcdab02d6445c07a0ab637277a4`.

## Freeze chronology and guard ruling

The written resume record held submissions until `2026-08-28T06:30` for maintenance reservation `SD26082026`. On 2026-08-27 the integrator verified that the reservation lifted early and the A100 partitions were again visible in mixed and allocated states. That verified node state superseded only the submission date, not the freeze or occupancy rules.

The freeze says a fired guard voids a run. The capture does not record enough information to decide five guards, so completed timings cannot be promoted to measurement evidence.

The capture contract is `REFUTED_AS_IDENTIFICATION_CAPTURE`: candidate packet and raw-byte counts are derived fields rather than observations; access width, lane mask, stream count, outstanding window, burst length, gap, and offered rate are parsed but not applied by the hardware path; and the copy-engine loop enqueues one peer copy per message. The emitted checksum is a point-id hash, not a measured destination checksum.

The scorer made no expectations amendment and changed no candidate parameter value.

## Per-case frozen-band score

| Case | Coverage | Metric status | Measurement verdict |
|---|---|---|---|
| `CORNER_NVPKT_001_payload_bytes` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVPKT_002_candidate_boundaries` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVPKT_003_256b_residuals` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVPKT_004_destination_alignment` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVPKT_005_source_alignment` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVPKT_006_access_width` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVPKT_007_active_warp_lanes` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVPKT_008_lane_mask_shape` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVPKT_009_address_stride` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVPKT_010_fixed_total_bytes` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVPKT_011_fixed_message_size` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVPKT_012_address_reuse` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVPKT_013_peer_write` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVPKT_014_peer_read` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVPKT_015_producer_comparison` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVPKT_016_blind_holdout` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVBOND_017_ordered_pair_matrix` | COMPLETE | REFUTED | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVBOND_018_per_link_balance` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVBOND_019_stream_count` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVBOND_020_producer_concurrency` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVBOND_021_bandwidth_ramp` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVBOND_022_offered_rate` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVBOND_023_burst_length` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVBOND_024_source_fanout` | COMPLETE | REFUTED | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVBOND_025_destination_fanin` | COMPLETE | REFUTED | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVBOND_026_symmetric_bidirectional` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVBOND_027_asymmetric_bidirectional` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVBOND_028_disjoint_unidirectional` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVBOND_029_disjoint_bidirectional` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVBOND_030_full_mesh` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVBOND_031_startup_state` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVBOND_032_node_time_repeatability` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVINC_033_one_source` | COMPLETE | REFUTED | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVINC_034_two_source` | COMPLETE | REFUTED | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVINC_035_three_source` | COMPLETE | REFUTED | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVINC_036_fixed_aggregate_rate` | COMPLETE | REFUTED | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVINC_037_per_source_rate` | COMPLETE | REFUTED | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVINC_038_start_skew` | COMPLETE | REFUTED | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVINC_039_join_leave` | COMPLETE | REFUTED | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVINC_040_burst_depth` | COMPLETE | REFUTED | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVINC_041_equal_message_size` | COMPLETE | REFUTED | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVINC_042_unequal_message_size` | COMPLETE | REFUTED | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVINC_043_one_elephant_two_mice` | COMPLETE | REFUTED | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVINC_044_two_elephants_one_mouse` | COMPLETE | REFUTED | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVINC_045_push_distinct_buffers` | COMPLETE | REFUTED | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVINC_046_pull_gather` | COMPLETE | REFUTED | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVINC_047_hot_destination` | COMPLETE | REFUTED | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVINC_048_long_soak` | COMPLETE | REFUTED | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVCRD_049_dependent_round_trip` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVCRD_050_outstanding_write_16b` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVCRD_051_outstanding_write_32b` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVCRD_052_outstanding_write_64b` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVCRD_053_outstanding_write_128b` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVCRD_054_outstanding_write_256b` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVCRD_055_outstanding_read` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVCRD_056_outstanding_atomic` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVCRD_057_burst_length` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVCRD_058_inter_message_gap` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVCRD_059_inter_burst_gap` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVCRD_060_duty_cycle` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVCRD_061_adaptive_knee_zoom` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVCRD_062_two_streams_one_pair` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVCRD_063_one_source_two_peers` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVCRD_064_opposite_directions` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVHOL_065_small_behind_large` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVHOL_066_large_behind_small` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVHOL_067_separate_streams` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVHOL_068_alternating_sizes` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVHOL_069_bimodal_mix` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVHOL_070_same_pair_bulk` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVHOL_071_other_peer_bulk` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVHOL_072_remote_incast` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVHOL_073_write_write` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVHOL_074_read_read` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVHOL_075_same_direction_read_write` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVHOL_076_opposite_direction_read_write` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVHOL_077_distinct_regions` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVHOL_078_shared_cache_line` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVHOL_079_post_burst_drain` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
| `CORNER_NVHOL_080_blind_mixed_soak` | COMPLETE | UNSCORABLE | VOID_FATAL_GUARD_COVERAGE_INCOMPLETE |
