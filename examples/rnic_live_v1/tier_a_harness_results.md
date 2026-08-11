# HTSIM-9 Tier A preparation results

Current status, 2026-08-11: this file records the earlier preparation
checkpoint. The later qualifying Tier C run is recorded in
[RESULTS.md](RESULTS.md#tier-c-abi-v2-packet-chain-chronology-and-closure), and
HTSIM-9 is now closed.

The deterministic fake-port preparation gate passes. This is component
evidence for the SimLLM side of HTSIM-9, not a composed htsim result and not a
TTFT or TPOT claim. The successor remains HTSIM-9, which must compile the
unchanged scenario runner with the htsim port-factory translation unit and run
the same checker.

## Chronology and expectation integrity

The broader composition gate was originally frozen at
`65b56097d0409488e274b83eb2d0d2e6cb34a2f9` and last amended before its run at
`d5d98a29c0bb5a3e1f61abb5eacdf33f27258f61`. The post-specified fixture audit
records the exact chronology and does not edit that frozen file.

The executable preparation behavior, matrix, signed relations, exact bands,
negative control, raw schema and evidence classes were frozen in
`35c2ee4a2f9feb159ac4a884c90bd7dcc6237a6b` before implementation began. Both
registered commands passed `--check-only` before that commit and produced no
observation or summary files.

The first nonfinal producer smoke exposed a Python machinery defect in the
one-token-per-WQE check: `Counter(cell_wqes)` treated raw WQE dictionaries as
counts. Post-specified correction
`21f9a4c` changed only those two operands to
`Counter(cell_wqes.keys())`. It changed no expectation, relation, band,
schema, invariant or evidence count. This correction occurred after the
nonfinal smoke, so it is not represented as a pre-registration. The complete
implementation is commit `f8eeb34`; the registered study below ran after that
commit.

## Method and external evidence

The historical run used the same executable basenames, scripts, options and
pinned inputs; resolved machine-local paths are intentionally omitted. The
following is a portable post-run rendering, not a verbatim transcript. Source
the local configuration first:

```bash
cmake -S examples/rnic_live_v1/native \
  -B "${SIMLLM_TIER_A_RUN_ROOT:?configure SIMLLM_TIER_A_RUN_ROOT}/fake/build" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build \
  "${SIMLLM_TIER_A_RUN_ROOT:?configure SIMLLM_TIER_A_RUN_ROOT}/fake/build" \
  --config Release --parallel
ctest --test-dir \
  "${SIMLLM_TIER_A_RUN_ROOT:?configure SIMLLM_TIER_A_RUN_ROOT}/fake/build" \
  -C Release --output-on-failure
.venv/bin/python examples/rnic_live_v1/tier_a_acceptance.py \
  --factory fake \
  --producer "${SIMLLM_TIER_A_RUN_ROOT:?configure SIMLLM_TIER_A_RUN_ROOT}/fake/build/simllm_rnic_tier_a" \
  --run-dir "${SIMLLM_TIER_A_RUN_ROOT}/fake"
```

Bulk evidence remains outside Git:

- `raw_observations.json`: 38,069 bytes, SHA-256
  `5fb58e513f6313ebe23fc751ce05bafc07a51ef0a4892d9035c24cdff20fafbb`
- `summary.json`: 503 bytes, SHA-256
  `6825d3ae34f079ec5cc5e3d91faa59948dd31ee724b89c400306a5fac5b869fb`

The native target built with warnings as errors. Its CTest entry passed 1 of
1. This executable count is component evidence and is not added to any
behavioral denominator.

## Exact single-WQE rows

All eight exact-oracle rows pass. Times are picoseconds.

| Payload bytes | Rate Gbit/s | D | Eligible | Port TX | Terminal | CQE visible | Poll | JCT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 200 | 0 | 0 | 0 | 163,840 | 163,840 | 163,840 | 163,840 |
| 4,096 | 200 | 1,000 | 1,000 | 1,000 | 164,840 | 164,840 | 164,840 | 164,840 |
| 4,096 | 400 | 0 | 0 | 0 | 81,920 | 81,920 | 81,920 | 81,920 |
| 4,096 | 400 | 1,000 | 1,000 | 1,000 | 82,920 | 82,920 | 82,920 | 82,920 |
| 1,048,576 | 200 | 0 | 0 | 0 | 41,943,040 | 41,943,040 | 41,943,040 | 41,943,040 |
| 1,048,576 | 200 | 1,000 | 1,000 | 1,000 | 41,944,040 | 41,944,040 | 41,944,040 | 41,944,040 |
| 1,048,576 | 400 | 0 | 0 | 0 | 20,971,520 | 20,971,520 | 20,971,520 | 20,971,520 |
| 1,048,576 | 400 | 1,000 | 1,000 | 1,000 | 20,972,520 | 20,972,520 | 20,972,520 | 20,972,520 |

These eight rows are exact-oracle evidence. They are reported separately from
the behavioral families below.

## Scored behavioral families

Each family retains its own denominator.

| Family | Passed | Frozen quantitative relation | Observation |
|---|---:|---|---|
| D-additivity | 4 of 4 | Every absolute boundary and JCT shifts by exactly +1,000 ps; service is unchanged. | Every `(payload, rate)` pair shifted by exactly 1,000 ps in all six checked fields. |
| Inverse-rate serialization | 4 of 4 | Service at 200 Gbit/s is exactly 2 times service at 400 Gbit/s. | Both payloads and both D values matched the exact factor 2. |
| Two-WQE FIFO | 4 of 4 | W0 is `D` to `D+L`; W1 is `D+L` to `D+2L`; W1 wait is L; JCT is `D+2L`. | Every `(rate, D)` row matched all timing equations. |

The FIFO observations were:

| Rate Gbit/s | D | W0 TX | W0 terminal | W1 eligible | W1 TX | W1 terminal | JCT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 0 | 0 | 163,840 | 0 | 163,840 | 327,680 | 327,680 |
| 200 | 1,000 | 1,000 | 164,840 | 1,000 | 164,840 | 328,680 | 328,680 |
| 400 | 0 | 0 | 81,920 | 0 | 81,920 | 163,840 | 163,840 |
| 400 | 1,000 | 1,000 | 82,920 | 1,000 | 82,920 | 164,840 | 164,840 |

## Fatal unscored evidence

All seven fatal families pass, but they do not increase a behavioral count:

- Authority exclusivity: the structural case records one native session and
  one native post with no legacy activity; bypass records only one legacy
  ledger/post/mutation; dual construction throws before its all-zero snapshot
  changes.
- Token conservation: each frozen v1 WQE issues one nonzero token and records
  one matching terminal; every session has zero live tokens at quiescence.
- Quiescence: every device ends nonfatal with empty SQ, CQ and unpublished
  sets and no pending physical work.
- Terminal atomicity: duplicate, unknown and cross-WQE calls each throw the
  frozen `std::invalid_argument`; caller time remains 81,920 ps before and
  after, and a 150,000 ps progress probe returns zero changes with no
  exception.
- Controlled drop: one unsignaled 4 KiB SEND accepts at 0 ps, drops in the
  fabric at 81,920 ps, produces exactly one `transport_error` CQE, produces no
  Success CQE, records one injected network-drop fact and leaves no live work.
- FIFO ordering: all four FIFO rows complete in W0, W1 order, grounded in the
  landed ordered-retirement implementation.
- Wrapper-bypass sensitivity: the requested D=1,000 mutant constructs the
  device with D=0. All D-sensitive deltas are 0 ps, its D=0 observation equals
  the accepted baseline, and the same D-additivity predicate rejects it.

## Genuine-risk fraction

This fraction estimates exposure before considering the observed pass. It is
not a failure count, and common-mode defects can affect several instances at
once.

| Scored family | Plausibly at-risk instances | Why a competent implementation could fail |
|---|---:|---|
| D-additivity | 4 of 4, 100% | The wrapper could omit D, charge it twice, put it after network service or pass it through the factory. Each payload-rate pair exercises that ownership seam. |
| Inverse-rate serialization | 4 of 4, 100% | A bit/byte, Gbit/ps or integer-conversion error, or an accidental header/propagation term, could break the exact factor at either payload or D value. |
| Two-WQE FIFO | 4 of 4, 100% | Busy retry time, same-time terminal-before-progress order, or SQ-head retention could shift or reorder W1 at either rate and either D value. |

## Deliberate omissions and residual work

- HTSIM-9 (Completeness; P1; L) remains open. No backend repository was
  changed, no htsim factory was linked and no composed binary result is
  claimed.
- BACK-24 (Precision; P0; S) remains open for transactional validation at the
  direct `RnicDevice::onNetworkEvent` boundary. Wrapper prevalidation is not
  that repair.
- BACK-25 (Completeness; P1; L) remains open for versioned packet-attempt,
  TX-start, TX-finish and native-RX vocabulary plus session token identity.
- BACK-26 (Completeness; P1; L) remains open for ECN/CNP and rate updates, PFC
  pause/resume and link-state events.
- BACK-8 (Completeness; P1; L) remains open for the production run record,
  structural projection, hardware hash, bypass equivalence and live metric
  chain. No `CompletionEvent`, `ExecutionResult`, `StepResult`, TTFT or TPOT
  result is produced here.
- The frozen [`expectations.md`](expectations.md) was not edited. Its proposed
  clarification remains a draft in the post-specified fixture audit pending
  maintainer approval.

Postscript, 2026-08-11: BACK-25 and BACK-26 later closed at the vocabulary and
relay boundary; see [the ABI-v2 packet study](../rnic_packet_v2/RESULTS.md) and
its registered producer residuals.
