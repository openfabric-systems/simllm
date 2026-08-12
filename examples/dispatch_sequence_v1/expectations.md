# Dispatch message sequence expectations

Date: 2026-08-12

This file freezes the TRAF-21 study before the sequence-faithful MoE traffic
generator is implemented or any result-producing run is made. The new traffic
precision level is named `captured-message-sequence`. It is an explicit
renderer entry point, not a repository-wide selector. CORE-36 remains the sole
owner of the future validated fidelity selection and provenance surface.

## Defect and authority audit

The accepted aggregate renderer is byte-correct but cannot represent message
arrival sequence or granularity:

- `simllm/traffic/step_comm.py:510-550` walks source ranks, converts each
  token's expert owners to a set, accumulates request and ordered-pair bytes,
  then sorts the aggregate table. The set removes top-k destination order and
  the sorted table replaces capture order with request, source and destination
  order.
- `simllm/traffic/patterns.py:183-239` emits exactly one positive send per
  ordered pair in nested source-rank and destination-rank order.
- `simllm/preplay/framework_runner.py:1141-1162` reads vLLM's returned
  `routed_experts` tensor and flattens it without sorting.
- `simllm/preplay/framework_runner.py:794-810` consumes each contiguous top-k
  slice in returned order and places that tuple into `ObservedLayerDispatch`.
- `simllm/preplay/framework_trace.py:306-323` writes expert IDs in tuple order,
  while `simllm/preplay/framework_trace.py:328-368` reads the JSON array in
  that same order.
- `simllm/preplay/framework_schema.py:256-297` requires the source to be
  observed dispatch, validates top-k cardinality and uniqueness, and does not
  sort the expert tuple.

The v2 trace therefore determines request order, phase-local token order,
layer order and the framework-returned top-k tuple order. It does not observe
the order in which a fused kernel, NCCL implementation or RNIC posted bytes to
the wire. The implementation must not rename framework-return order as kernel
wire order.

The packet arithmetic is audited against the repository's accepted RNIC
model. `examples/rnic_packet_v2/back34_expectations.json:6-36` fixes a 4,096
byte maximum wire packet and a 64 byte data header, including the exact short
final packet. `examples/rnic_packet_v2/expectations.md:108-132` fixes the
work-conserving packet serializer and inverse-rate arithmetic.

## Frozen generator contract

The generator consumes a scheduled routed step and an explicit expert
placement epoch. For each layer it derives one ordered base stream in this
lexicographic traversal:

1. `StepRecord.scheduled` request order;
2. selected phase-local token order;
3. framework-returned top-k position;
4. EP source-rank order only to project the same captured route onto each
   source.

Within one token, several experts on the same destination rank require one
hidden-vector transfer. The first top-k position naming that destination owns
the transfer's position. A local destination emits no fabric message. No
ordering is asserted between different source ranks. Each source's sends are
issued in the order obtained by filtering the base stream for that source.

Dispatch sends the hidden vector from the EP source to the selected expert
owner. Combine transposes each message while retaining its captured routing
ordinal. This is a traffic replay convention, not evidence of expert-kernel
completion order.

The two declared grouping rules are:

- `per-token`: one hidden-vector message for every unique remote destination
  of every token and source;
- `per-expert-group`: coalesce all vectors for one
  `(request, source, destination, layer, phase)` into one message, ordered by
  the first contributing token and top-k position. This rule represents
  whole-layer buffering before issue.

Every message retains request identity and all contributing
`(token_index, top_k_index)` routing ordinals. Sequence is data, not a
dictionary iteration side effect. The aggregate ordered-pair and per-request
tables remain read-only projections of the sequence.

The existing `step_moe_alltoalls`, `pairwise_all_to_allv` and
`render_step_goal` entry points remain the default aggregate compatibility
level. Their signatures, tuple values, GOAL text and accepted artifacts must
remain byte-identical. The new level uses separate traffic APIs so this change
does not add a second runtime selector ahead of CORE-36.

## Frozen observed fixture

The deterministic decision fixture is a strict
`simllm-preplay-trace-v2` value with one request, one prefill layer, four
tokens, top-k two, four experts and four EP ranks. Expert `e` is owned by rank
`e`. The hidden vector is 2,048 bytes. The framework-returned top-k tuples are,
in token order:

```text
(3, 1)
(2, 1)
(3, 2)
(1, 3)
```

The dispatch sequence filtered by source must be:

```text
source 0: 3, 1, 2, 1, 3, 2, 1, 3
source 1: 3, 2, 3, 2, 3
source 2: 3, 1, 1, 3, 1, 3
source 3: 1, 2, 1, 2, 1
```

Per-token rendering has 24 dispatch and 24 combine messages. Expert-group
rendering has 9 dispatch and 9 combine messages. Both carry exactly 49,152
bytes per phase and 98,304 bytes for the complete dispatch-plus-combine step.
There are nine positive dispatch pairs and the combine table is their exact
transpose. Every byte belongs to the fixture's one request.

These counts, the declared sequence, pair conservation, request conservation
and combine transpose are fatal exact or structural oracles. They are not
scored behavioral evidence.

## Frozen backend matrix

The fixture is rendered at zero compute cost. The matrix is the Cartesian
product of:

- renderer: `aggregate`, `per-expert-group`, `per-token`;
- profile: packetized `rnic-nn`, fluid `rnic-nn-fluid`;
- endpoint rate: 200 and 400 Gbit/s.

The packetized rows set the maximum wire packet to 4,096 bytes, the data
header to 64 bytes and null-network propagation to zero. The fluid rows use
the same payloads and zero propagation. All runs require verified physical
quiescence and record raw per-flow FCT plus whole-program completion.

### Packet signed relation

A 6,144 byte expert group needs two packets, while its three separate 2,048
byte messages need three packets. A 4,096 byte group and its two separate
messages both need two packets because the 64 byte header leaves 4,032 payload
bytes in the first packet. The fixture has six remote three-vector groups in
dispatch and six in combine. Per-token rendering therefore adds exactly 12
data headers, or 768 wire bytes, relative to expert-group rendering.

The narrow incident-resource floor is 192 extra header bytes in dispatch plus
192 in combine. The conservative work ceiling counts all 768 extra bytes at
both endpoint serializers. For rate `R` in bits per second:

```text
floor(R)   = 384 * 8 * 10^12 / R ps
ceiling(R) = 1536 * 8 * 10^12 / R ps
```

The registered raw packet-completion deltas are therefore positive and in
these closed bands:

| Comparison | 200 Gbit/s | 400 Gbit/s |
|---|---:|---:|
| per-token minus per-expert-group | [15,360, 61,440] ps | [7,680, 30,720] ps |
| per-token minus aggregate | [15,360, 61,440] ps | [7,680, 30,720] ps |

Both comparisons are evaluated from raw backend observations before any
byte-conservation, exact-sequence or packet-ledger oracle. They form one
decision-relevant family with four parameterized instances. The two
comparators share the same per-token subject, so the result reports both
instances but does not pretend they are two independent families.

The 200 Gbit/s delta must be twice the corresponding 400 Gbit/s delta within
2,000 ps of whole-nanosecond schedule quantization. This is a second family
with two instances. It is also evaluated before exact packet checks.

### Fluid diagnostic relation

Fluid service has no packet header term. For each rate, report both sequenced
minus aggregate completion deltas even if they are zero. The pre-run
expectation is an absolute difference no larger than 1,000 ps for each of the
four grouping-rate cells. These are two grouping families with two rate
instances each. A nonzero value inside the band is reported, not rounded away.

### Physical sanity before digits

The peak incident payload is 18,432 bytes in each phase. Before reading any
backend result, the runner records these first-principles bounds:

| Rate | Payload floor for two serial phases | Conservative packet ceiling |
|---|---:|---:|
| 200 Gbit/s | 1,474,560 ps | 9,000,000 ps |
| 400 Gbit/s | 737,280 ps | 4,500,000 ps |

The floor is two times 18,432 bytes over rate. The ceiling serializes all
101,376 per-token wire bytes at both endpoints and leaves margin for
whole-nanosecond GOAL boundaries. A measured value outside its rate's range
is fatal. Halving rate should approximately double every serialization-only
completion and exactly doubles the registered packet delta band.

## Granite scale and cost record

The provided Granite scale input is the existing v1 routed projection, not a
v2 observed-dispatch trace. It is used only to measure size and cost. No result
may call its reconstructed tuple order framework-observed.

The authored-against input identities are:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `routed-experts.json` | recorded by the run | `24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f` |
| `steps.jsonl` | recorded by the run | `824cd9557293328bb42b593ac893b6a067302e545b087c9219195ccb8031d755` |
| accepted aggregate step-0 GOAL | 334,432 | `08a0403af66ff8a9d6b18f93afd15ae0bc925cc85555acf8a0593438a3d7bc92` |

The result separately records the input commits and hashes it actually
observed. It does not require a live submodule pin to equal a frozen literal.

For aggregate, expert-group and per-token rendering of Granite prefill step
zero, measure message count, Python plan time, GOAL render time, peak traced
Python memory, GOAL bytes, binary compile time and packet and fluid backend
wall time when the registered 600 second timeout permits completion. A level
is labelled practical for a large sweep only if one step renders and compiles
in at most 30 seconds, uses at most 1 GiB peak traced Python memory, produces
at most 64 MiB of GOAL text and completes each requested backend run in at
most 60 seconds. Crossing a threshold is a measured finding, not a study
failure.

## Evidence accounting and entailment

The scored evidence classes are kept separate:

- packet signed-delta family: four instances;
- packet inverse-rate family: two instances;
- fluid grouping families: four instances.

The expected genuine-risk headline is three families and ten parameterized
instances. Each can fail from raw completion observations before an entailing
oracle runs. Correct pair totals can still produce the wrong completion if
the renderer aggregates, reorders, serializes too strongly or fails to reach
the selected backend. Exact sequence, byte conservation, request
conservation, default identity, physical bounds, quiescence and input hashes
are fatal unscored evidence. Grouping counts and the expected top-k sequence
are author-defined structural checks and never enter the behavioral
denominator. Unit tests and repository gates are separate executable
evidence.

## Closure scope and residual

TRAF-21 closes only if the result demonstrates all of these clauses:

1. `captured-message-sequence` preserves the v2 framework-returned request,
   token and top-k order under both declared grouping rules;
2. every aggregate ordered-pair and per-request byte total matches the
   accepted aggregate authority exactly;
3. the default aggregate APIs and accepted artifacts remain byte-identical;
4. the registered packet and fluid matrix reaches a native backend and
   reports the frozen signed relations, raw FCT and completion;
5. the Granite scale record reports cost and practicality without describing
   v1 reconstructed order as observed;
6. no repository-wide fidelity selector is added ahead of CORE-36.

The kernel-to-wire ordering gap is intentionally outside TRAF-21. PLAY-14 is
reserved for a residual that requires an observed per-message post sequence
from the framework collective or kernel boundary through NCCL and RNIC WQE
submission, joined to token, layer, top-k position, source and destination.
That evidence, or an equivalent hardware trace with the same identities, is
what can replace the framework-return order surrogate.

## Registered command and pre-freeze dry run

The complete registered invocation is:

```bash
.venv/bin/python examples/dispatch_sequence_v1/run_study.py \
  --out "$SIMLLM_WAVE6_RUN_ROOT/dispatch_sequence_v1" \
  --granite-root "$SIMLLM_GRANITE_REPLAY_ROOT" \
  --htsim-rnic "$SIMLLM_HTSIM_RNIC" \
  --txt2bin "$SIMLLM_TXT2BIN"
```

Before the expectations commit, this exact command is run with
`--check-only`. Check-only parses the complete CLI and validates only the
frozen registries, arithmetic, paths as strings and evidence counts. It does
not import SimLLM, read the Granite artifacts, create the output directory,
invoke a native tool or write an artifact.
