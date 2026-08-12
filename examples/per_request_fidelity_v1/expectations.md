# Per-request replay fidelity v1 expectations

Tasks: PLAY-11 and CORE-28.

## Decision and claim boundary

The routed supply is the sole authority for the scheduled request, forwarded
token and captured expert assignment. The implementation may project that
authority into an aggregate physical pair table, an immutable per-request byte
partition and a rendered GOAL message record. It must not derive a second
routing decision or let the per-request partition alter the physical GOAL.

The decision-relevant relation is discrimination under an
aggregate-preserving request permutation. If a rendered request partition can
be permuted between two co-scheduled requests while the fidelity gate still
passes, the proposed read-only partition is not an adequate interface. The
design must then move the ownership key closer to the routed-token projection
or use separately correlated physical operations. If the gate rejects the
permutation while the aggregate physical GOAL stays identical, one aggregate
operation with a checked request partition is sufficient and avoids changing
packetization.

This task claims exact captured MoE dispatch and combine byte attribution for
each scheduled request, layer and directed pair under the selected placement.
It does not claim expert compute fidelity, gate-weight fidelity, TP collective
attribution, KV-cache fidelity, packet-level calibration or per-request
latency. `StepResult.step_latency_ps` remains one whole-step makespan. No rule
divides that makespan among co-scheduled requests.

## Pre-freeze source audit

The audited repository state is commit
`15a617859d9b52ad1a241434da67bd04525f2fcb`.

- `simllm/preplay/routing.py:52-95` retains each routed request's stable
  identity and offers identity lookup. Lines 175-210 validate the exact
  prompt and nonterminal decode-token rows belonging to that request.
- `simllm/traffic/step_comm.py:109-166` selects each scheduled request's
  routed token slice, then returns one identity-free token tuple. Lines
  220-248 aggregate every selected token directly into one pair table. This
  is the loss point addressed by PLAY-11.
- `simllm/core/execution.py:148-169` carries only aggregate sparse pair
  payloads. `simllm/traffic/patterns.py:103-127` renders only those aggregate
  sizes, and `simllm/goal/emitter.py:25-92` retains no structured message or
  request attribution. This is the projection gap addressed by CORE-28.
- `examples/routed_supply_v1/expectations.md:242-250` freezes the already
  audited fluid rule used for the sanity grid: 2,000,000 ps propagation and
  exactly 40 ps/byte at 200 Gbit/s or 20 ps/byte at 400 Gbit/s for one flow
  per two-rank port. Lines 328-336 define the serial phase makespan relation.

The external Granite source is supplied under `SIMLLM_MOE_E2E_ROOT`; no site
path is part of this study. The audit before this freeze found:

- `capture/granite-greedy.jsonl` is 120 LF-terminated rows with SHA-256
  `5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6`.
  Line 2 names `r0`, lines 3-24 are its 22 prefill forwards; line 48 names
  `r1`, lines 49-60 are its 12 prefill forwards; line 68 names `r2`, lines
  69-88 are its 20 prefill forwards. Line 120 reports three requests, 54
  prefill forwards and 61 decode forwards.
- `replay-400g/steps.jsonl` has SHA-256
  `824cd9557293328bb42b593ac893b6a067302e545b087c9219195ccb8031d755`.
  Line 1 schedules all 22, 12 and 20 prefill tokens under request identities
  `r0`, `r1` and `r2`.
- `replay-400g/routed-experts.json` has SHA-256
  `24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f`.
  It is the supported routed-supply projection consumed by the study.
- `replay-400g/htsim/step-000000.goal` is 334,432 bytes with SHA-256
  `08a0403af66ff8a9d6b18f93afd15ae0bc925cc85555acf8a0593438a3d7bc92`.
  Its 2,688 sends total 207,499,264 bytes. A standard-library audit of the
  capture with expert owner `expert_id % 8` reproduced every tag and pair of
  that aggregate GOAL exactly before deriving the per-request values below.

The external artifacts are observed inputs, not live pin literals. The study
records their observed hashes and the binaries used for a run; it assumes no
equality between any frozen commit literal and a current submodule gitlink.

## Synthetic sweep

The exact fixture has two MoE layers, four experts, top-k two, two EP ranks
and eight bytes per hidden vector. Each request contributes one prefill input
token:

| Request | Layer 0 experts | Layer 1 experts |
|---|---|---|
| `alpha` | 0, 2 | 0, 1 |
| `beta` | 0, 1 | 2, 3 |
| `gamma` | 2, 3 | 1, 3 |

Placement epoch 0 assigns experts 0 and 1 to rank 0 and experts 2 and 3 to
rank 1 at both layers. Epoch 1 assigns layer-0 experts 0 and 2 to rank 0 and
layer-1 experts 0 and 3 to rank 0; the other experts belong to rank 1.

The request-count sweep schedules the ordered prefixes `alpha`,
`alpha,beta` and `alpha,beta,gamma`. Author-defined request order, token IDs
and placement order are configuration and therefore unscored.

### Exact request byte oracle

Each entry below is one dispatch table. `0>1:8,1>0:8` means eight bytes on
each named directed pair; an omitted pair is zero. Combine is the exact
transpose of dispatch.

| Epoch | Request | Layer 0 dispatch | Layer 1 dispatch | Positive rows across both phases | Bytes across both phases |
|---:|---|---|---|---:|---:|
| 0 | `alpha` | `0>1:8,1>0:8` | `1>0:8` | 6 | 48 |
| 0 | `beta` | `1>0:8` | `0>1:8` | 4 | 32 |
| 0 | `gamma` | `0>1:8` | `0>1:8,1>0:8` | 6 | 48 |
| 1 | `alpha` | `1>0:8` | `0>1:8,1>0:8` | 6 | 48 |
| 1 | `beta` | `0>1:8,1>0:8` | `0>1:8,1>0:8` | 8 | 64 |
| 1 | `gamma` | `0>1:8,1>0:8` | `0>1:8,1>0:8` | 8 | 64 |

These tables are fatal exact oracles, evaluated after the scored raw
relations. They do not increase a behavioral denominator.

### PLAY-B1: raw rendered identity

For all six `(epoch, request_count)` cells, compare the routed operation's
per-request rows with the raw structured messages produced by direct GOAL
rendering before any exact hash or aggregate-conservation assertion runs.
Every included request, layer, phase and pair must match with zero missing,
extra or changed rows. These are six scored instances. A renderer that drops
identity, associates by request position or attaches one request's partition
to another aggregate message can fail this relation.

### CORE-B1: graph round-trip identity

For the same six cells, strict execution-graph JSON round trip followed by
graph-only GOAL rendering must retain the same raw per-request rows as direct
rendering. Evaluate the raw comparison before checking the exact wire shape or
aggregate sums. These are six scored instances. A strict reader that omits the
new optional field, a lowerer that keeps only aggregate pairs or a graph
renderer that loses the partition can fail this relation.

### PLAY-B2: aggregate-preserving permutation control

For request counts two and three at both placements, swap `alpha` and `beta`
only in the rendered request partition. Do not change any physical message
size, peer, tag, dependency or GOAL text. Evaluate the aggregate-only and
per-request comparisons directly on these raw permuted observations before
running any fatal exact oracle.

| Epoch | Request counts | Aggregate mismatched rows | Per-request mismatched rows | L1 attribution error bytes | Signed `alpha` byte error |
|---:|---|---:|---:|---:|---:|
| 0 | 2, 3 | 0 in each cell | 12 in each cell | 96 in each cell | -16 in each cell |
| 1 | 2, 3 | 0 in each cell | 4 in each cell | 32 in each cell | +16 in each cell |

The aggregate-only comparison must pass all four cells and the per-request
gate must reject all four. The signed error is observed minus expected for
`alpha`; `beta` has the opposite sign. These are four scored instances and
the primary decision-relevant family.

### Physical GOAL identity, fatal and unscored

Adding the read-only request partition must not change physical GOAL text.
The pre-implementation direct-render baselines are:

| Epoch | Requests | GOAL bytes | SHA-256 | Sends | Total send bytes |
|---:|---:|---:|---|---:|---:|
| 0 | 1 | 744 | `1eb2bbff8a981523b5f6733420aa9d5d3509aa473ed991409b8d455e619e5864` | 6 | 48 |
| 0 | 2 | 952 | `78a8e80589b156374b965634dd82251931219398c1e2cf2454b06cbe3629916c` | 8 | 80 |
| 0 | 3 | 964 | `8e38bf44631b9f3d7020452886552502fa567ec44559d05b5401a5dbbc825ab6` | 8 | 128 |
| 1 | 1 | 744 | `8c1738dbd01f320b0f5f005b9ea6acd19145c77db67af89eaac4a78219d494de` | 6 | 48 |
| 1 | 2 | 960 | `3023c39e472980ed6c689410a21fa626db3a73cf8a3d83bde425d8d41cfd4361` | 8 | 112 |
| 1 | 3 | 964 | `60cb32ca80a57d03b627de51d01fd292c0e87da3ec1482760faa8d304b075440` | 8 | 176 |

This identity is configuration-forced and therefore fatal-unscored.

### Whole-step fluid sanity, separate and unscored

The rendered program has one 1 ns calc per layer and four serial all-to-allv
phases. If `M(e,n,p)` is the larger aggregate pair in phase `p`, then

```text
JCT(e,n,B) = 2,000 + sum_p (2,000,000 + M(e,n,p) * 8 * 10^12 / B)
```

The result-producing study runs both rates and requires 0 ps residual:

| Epoch | Requests | JCT at 200 Gbit/s, ps | JCT at 400 Gbit/s, ps |
|---:|---:|---:|---:|
| 0 | 1 | 8,003,280 | 8,002,640 |
| 0 | 2 | 8,003,920 | 8,002,960 |
| 0 | 3 | 8,004,560 | 8,003,280 |
| 1 | 1 | 8,003,280 | 8,002,640 |
| 1 | 2 | 8,004,560 | 8,003,280 |
| 1 | 3 | 8,005,840 | 8,003,920 |

This proves that the gated renderer reaches the existing live step-makespan
path, but it is not scored as per-request fidelity evidence. The timing is
fully determined by the unchanged aggregate table and says nothing about how
latency should be divided among requests.

## Real Granite prefill relation

For step 0, use eight EP ranks with expert owner `expert_id % 8`, 24 layers
and 2,048 bytes per hidden vector. Canonical request rows are compact JSON
arrays `[request_id,layer,phase,source,destination,bytes]`, ordered by request
ID, layer, dispatch before combine, source and destination, encoded without
spaces plus one LF.

| Request | Positive rows | Total bytes | Canonical bytes | SHA-256 |
|---|---:|---:|---:|---|
| `r0` | 2,688 | 84,439,040 | 80,824 | `d2d5564c0507ae8e9946e377dfd9df0fca3eab20910d150faba03b1576e5e75a` |
| `r1` | 2,688 | 46,190,592 | 80,516 | `5f7603ec085e76e86b022b688404c428c90344115ac675ef40b59e609b90f568` |
| `r2` | 2,688 | 76,869,632 | 80,810 | `c441be8e81936ef0d32d32d59dfaf20f08bf496d588836edfee84058dbe0c89f` |
| all | 8,064 | 207,499,264 | 242,146 | `bcb21232c6f433e64ca0efb9bbfdaab4c008b087249f5d4b849dfb9bc646c077` |

These hashes are fatal exact external oracles and remain separate from the
scored relation.

### PLAY-B3: real aggregate-preserving permutation

Swap `r0` and `r1` in the raw rendered request partition. The aggregate
comparison must retain 0 mismatched rows and the same 2,688 physical sends and
207,499,264 bytes. The per-request comparison must report exactly 5,348
mismatched rows and 76,496,896 bytes of L1 error, then the fail-closed gate
must reject it. The signed errors are -38,248,448 bytes for `r0`, +38,248,448
bytes for `r1` and 0 for `r2`. This one external-capture instance is scored.

## Structural and wire guards, fatal and unscored

The implementation must reject a blank or unknown request identity; duplicate
`(request, source, destination)` entries; nonpositive sizes; self pairs;
ranks outside the collective; noncanonical ordering; request IDs absent from
the operation correlation; a request partition on any operation other than a
sparse pairwise all-to-allv; a partition whose per-pair sums differ from the
aggregate table; a rendered message whose partition differs from its physical
size; missing or extra attributed messages; and any fidelity mismatch before
a backend run begins.

The optional field must be omitted when empty. Existing uniform and sparse
graphs without attribution retain their exact v1 bytes, and all existing
physical GOAL hashes remain unchanged. Aggregate conservation, transpose,
canonical ordering, configuration echoes, frozen hashes and author-defined
sequences are fatal-unscored.

## Entailment and genuine-risk plan

The study records PLAY-B1, CORE-B1, PLAY-B2 and PLAY-B3 from raw observed
message and graph projections before it evaluates any frozen hash, exact-table
or aggregate-conservation guard. Therefore no earlier fatal oracle entails a
scored result. The later exact checks can fail the run but cannot add passes to
the scored denominator.

All four scored families are expected to be genuinely at risk. A competent
implementation can aggregate before saving request identity, serialize the
aggregate but omit the optional graph field, deserialize the field but drop it
in graph-only rendering, or compare aggregate pairs while ignoring ownership.
The result report must give both family-level and instance-level fractions and
must revise this estimate if execution shows an instance was unreachable or
entailed.

## Registered command and pre-freeze dry run

Configure the four paths in the gitignored local environment, then run:

```text
.venv/bin/python examples/per_request_fidelity_v1/run_study.py --out "$SIMLLM_PER_REQUEST_RUN_ROOT" --source-root "$SIMLLM_MOE_E2E_ROOT" --htsim-rnic "$SIMLLM_HTSIM_RNIC" --txt2bin "$SIMLLM_TXT2BIN"
```

The pre-freeze dry run is the same command with `--check-only`. It parses the
complete CLI and validates only the frozen registry above. It prints a
confirmation by design, does not inspect any supplied path, does not import
the implementation under study and produces no artifacts.
