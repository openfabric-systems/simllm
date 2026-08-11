# HTSIM-18 persistent session and CORE-24 result codec expectations

The original expectations freeze preceded either implementation and every
result-producing run. A later expectations-only amendment corrected only the
wall-clock calibration after a dry run exposed that the historical BRIDGE-1
workload was not the lightweight flow replay registered here. The amendment
precedes the HTSIM implementation commit and every result-producing run. The
codec contract and all simulated-time relations remain exactly as first
frozen. This file covers the opt-in HTSIM flow session and the full
`StepResult` wire codec. The one-GOAL CLI remains the exact default off path.
BRIDGE-2 remains open because it owns the graph-level client above these two
deliverables.

## External-source audit and design decisions

The audit used SimLLM base commit
`90ada43070adb3b1e624b6819aff34d8620e8571` and HTSIM base commit
`4885c647eecdfdf81479d1df052223c016ad086b` before this freeze.

- `docs/modules/backends.md:866-891` specifies the flow-session framing,
  verbs, sequence rejection, retained authorities, drain proof, and CLI
  identity acceptance.
- `docs/modules/core.md:584-611` specifies the full result codec and assigns
  the later graph, event, ledger-cursor, and result publication client to
  BRIDGE-2.
- HTSIM `htsim/sim/datacenter/main_rnic.cpp:152-217` parses exactly one CLI
  configuration, creates one event list and runtime assembly, executes one
  GOAL, validates quiescence, optionally writes one CSV, and exits.
- HTSIM `htsim/sim/eventlist.h:29-48` owns one monotonic picosecond event
  clock and exposes pending events at exact timestamps. HTSIM
  `htsim/sim/simllm_atlahs_flow_runtime.cpp:188-234` constructs the endpoint
  devices, binds one network runtime, and records one native authority;
  lines 237-303 post each accepted flow into that authority at event-list
  time.
- HTSIM `htsim/sim/rnic_packetized_manifold_runtime.cpp:349-489` reserves
  source serialization slots from retained flow and calendar state and emits
  the real packet timing observations. Resetting that object resets the source
  calendar, so overlapping same-source flows discriminate a persistent
  session from per-step process reset.
- `simllm/core/step.py:1-28` contains the old
  `atlahs-closed-loop-result-v1` name and prose sketch. Repository history
  shows that no payload, reader, writer, fixture, or accepted legacy bytes
  ever accompanied it. It is therefore a legacy name, not an accepted legacy
  wire form. The new reader must reject that name explicitly instead of
  inventing fields or fabricating CORE-5 attribution.
- `simllm/core/step.py:122-275` defines the conserved
  `LatencyAttribution`, separately typed `AdditiveVisitTotals`, exact
  `Fraction` TPOT, `RequestMetric`, and full `StepResult` that the new codec
  must preserve.
- `examples/bridge_persistent_v1/RESULTS.md:14-18` records the measured
  pre-existing baseline: `txt2bin` took 0.011178458 seconds and one isolated
  simulator invocation took 7.252140791 seconds. That result used complete
  recorded TP-8 steps with many flows and computation events. It motivates
  removal of the serial process boundary but cannot calibrate the lightweight
  one-flow invocations in this study.

### Wall calibration amendment

Before the HTSIM implementation commit, the unchanged base binary at commit
`4885c647eecdfdf81479d1df052223c016ad086b` ran the exact isolated replay
side of this study five times. Two-node replay elapsed seconds were
`[0.010377614, 0.005665112, 0.004582836, 0.004237526, 0.004083072]`.
Four-node replay elapsed seconds were
`[0.008126185, 0.008239501, 0.008102946, 0.009297478, 0.011039933]`.
These diagnostic-only observations are calibration inputs, not scored study
evidence. No session mode exists in that binary. They replace the
workload-inapplicable BRIDGE-1 values for the broad bands below. The session
bands are deliberately broad because no preimplementation session timing can
exist; the signed within-run ratio remains the decision-relevant wall-clock
check.

### Causal boundary

`advance` uses an inclusive virtual-time horizon named `through_ps`. This is
the concrete boundary type. It matches the simulator's sole picosecond event
clock and authorizes every already accepted injection whose
`eligible_at_ps <= through_ps`, plus every causally generated event at or
before that time. It does not authorize a later event. A later `advance` must
not lower the horizon, and an injection accepted after an advance must have
`eligible_at_ps > through_ps`, because inserting work into an already
authorized interval would rewrite causal history.

The alternatives are weaker here. A sequence-only boundary cannot stop a
long-lived transport timer from running beyond the caller's intended time,
while an operation-set boundary would duplicate the runtime's event
authority. The virtual-time horizon controls the existing authority without
creating a second scheduler.

### Discriminating retention observable

The retention observable is the second same-source flow's completion
timestamp and the source SQ high-water mark. Two equal-size flows are
eligible at 0 ps in one session. The first consumes source serialization
slots while the second remains live behind it. A fresh per-step reset gives
the second flow an empty source calendar and SQ. The persistent result must
therefore have both:

- `persistent_second_fct_ps > reset_second_fct_ps`; and
- persistent source SQ high-water mark `2`, greater than reset value `1`.

This differs from a stateless implementation. Merely completing both steps,
or retaining an absolute clock after full quiescence, is not acceptance.

### Necessary open field refinement

The registered sketch names topology identity but not endpoint cardinality.
Identity is an audit label and cannot safely size endpoint devices or a
generated Clos. `open` therefore also carries positive `node_count`. The
server verifies that `topology_identity` is exactly
`<profile>:nodes=<node_count>` for topology-free profiles and includes the
accepted identity in its response. This refines the construction input
without widening the authority seam.

## Frozen protocol

Every request and response is one canonical JSON object prefixed by its
unsigned 32-bit big-endian byte length. Canonical JSON uses UTF-8, ascending
object keys, no insignificant whitespace, minimal string escapes, integers
only for numeric fields, and no duplicate keys. The maximum frame body is
1 MiB. A noncanonical, truncated, oversized, nonobject, or unknown-field
frame is rejected before dispatch.

Every request carries `schema: "simllm-htsim-flow-session-v1"` and one of
these verbs:

- `open`: `session_id`, `profile`, `topology_identity`, `node_count`,
  `link_rate_bps`, `seed`, `effective_hardware_sha256`, and
  `wqe_authority`. The only accepted authority is
  `simllm-native-rnic-session`. This is the first frame and returns the exact
  accepted configuration with `sequence: 0`.
- `inject`: contiguous positive `sequence`, nonblank `execution_id`,
  `operation_id`, and `flow_id`, distinct in-range `source` and
  `destination`, uint32 `tag`, positive `payload_bytes`,
  `eligible_at_ps`, and `policy_context_token`. The policy token must equal
  the effective native device context. Acceptance schedules, but does not
  independently advance, the existing event authority.
- `advance`: `through_sequence` equal to the last accepted injection and the
  inclusive `through_ps` horizon. Its response carries newly visible
  `accepted`, `queued`, `started`, and `completed` projections ordered by
  `(timestamp_ps, sequence, phase)`, with the original identities and native
  WQE, SQ, CQ, and transport aliases.
- `drain`: `through_sequence` equal to the last accepted injection. It is
  legal only after every accepted injection has run and the physical runtime
  is quiescent. It returns all completion rows, exclusive authority counters,
  per-source SQ high-water marks, the last accepted sequence, and
  `quiescent: true`.
- `close`: `through_sequence` equal to the drained sequence. It is legal only
  after drain, emits a terminal success response, and leaves subsequent
  complete frames as explicit post-terminal errors.

Success responses echo the request verb and use `status: "ok"`. Error
responses use `status: "error"`, a stable error code, the observed exclusive
authority counters, and `terminal: true`; the server processes no later
authority action. Duplicate, skipped, stale, cursor-disagreeing, and
post-terminal requests are checked before native post counts can change. EOF
between a frame prefix and its declared body is a truncated-frame error, and
the partial body is never parsed or committed.

The supported session profiles in this slice are the composed structural
profiles `rnic-nn` and `rnic-cn`. The already unsupported `rnic-ss` remains
rejected. `rnic-nn-fluid` remains the explicit nonstructural diagnostic
profile and is not falsely advertised as a native-session authority.

## Frozen StepResult wire form

The new schema is `simllm-step-result-v2`. Its exact JSON object contains:

- `schema`, `step_index`, `step_latency_ps`, and `completed_at_ps`;
- `request_metrics`, preserving order and every request ID, phase, token
  index, completion, interval latency, TTFT, attribution field, and additive
  visit total;
- TPOT as either `null` or an object with signed integer `numerator` and
  positive integer `denominator`, reduced by `Fraction`; and
- graph-wide `additive_visit_totals` as either `null` or the same separately
  typed visit-total object.

The attribution object has exactly `queue_ps`, `kv_ps`, `kernel_ps`,
`dma_ps`, `collective_ps`, `nic_ps`, and `control_ps`. The additive object has
exactly `queue_wait_ps`, `service_ps`, `visibility_ps`, and `visit_count`.
Readers reject missing or unknown fields, booleans in integer fields,
negative times, duplicate request IDs, zero or negative denominators,
unreduced or sign-noncanonical fractions, nonconserved attribution, request
completion after the step, and every unsupported schema. The old schema name
gets a specific unsupported-legacy error because the audit found no accepted
legacy payload to upgrade.

## Fixed study inputs

All transport cells use `rnic-nn`, 400 Gbit/s links, seed 0, default
4160-byte maximum wire packets with 64-byte headers, zero native device
service, and the exact hardware hash reported by the matching one-GOAL CLI.
The stateless-equivalent replays are generated from these frozen rows:

| replay | nodes | steps | `(source,destination,payload_bytes)` |
|---|---:|---:|---|
| two-node | 2 | 2 | `(0,1,4096)`, `(1,0,8192)` |
| four-node | 4 | 4 | `(0,1,4096)`, `(1,2,8192)`, `(2,3,4096)`, `(3,0,8192)` |

Successive persistent injections in these replays are eligible 10,000,000 ps
apart, beyond the preceding isolated completion. Each CLI comparison is one
fresh one-flow GOAL with eligibility 0. The complete ordered FCT integer list
is serialized with `json.dumps(..., sort_keys=True, separators=(",", ":"))`
and UTF-8. Those bytes are the latency identity surface.

The state-retention cells use two same-source, same-destination flows both
eligible at 0 ps, with payload in `{4096, 8192}`. Payload is the varied
parameter. Replay step count and endpoint count are independently varied by
the two stateless and wall-time cells.

The codec cases are empty, prefill-only, decode-only, and mixed prefill plus
decode. The decode and mixed cases include `Fraction(1, 3)`, whose decimal
expansion does not terminate.

## F1: off-path and simulated identity, fatal and unscored

Invoking `htsim_rnic` without `--flow-session` must retain the pre-change
help bytes, one-GOAL stdout, stderr, completion CSV, exit status, and simulated
timestamps exactly for the frozen one-flow inputs. In each stateless-equivalent
replay, the persistent ordered FCT byte stream must equal the concatenated
isolated CLI stream byte for byte. Completion cardinality, identities,
payloads, native aliases, exclusive authority counters, canonical event
ordering, and physical quiescence must conserve exactly.

Protocol rejection and atomicity checks are also fatal and unscored. They
cover duplicate and skipped injections, stale horizons, cursor disagreement,
post-drain injection, post-close frames, malformed canonical JSON, oversize
and truncated frames, and disconnect during a declared body. These are
guards, not behavioral relation instances.

All four codec cases must satisfy in-memory to JSON-ready object to real JSON
bytes to JSON-ready object to in-memory identity. The nonterminating TPOT must
retain numerator 1 and denominator 3 without float conversion. Strict-reader
negative cases and attribution conservation are fatal and unscored.

## R1: retained-state relation

For payload `P` in `{4096, 8192}`, let `S_P` be the raw second-flow FCT from
the overlapping persistent session and `I_P` the raw FCT from a fresh
one-flow CLI invocation. Let `H_P` and `h_P` be their source SQ high-water
marks. Each instance passes only when:

`S_P > I_P`, `H_P = 2`, and `h_P = 1`.

The signed direction is greater persistent second-flow latency and greater
persistent queue occupancy. There is deliberately no fatal exact oracle for
`S_P` in the stateful cells. The study records and evaluates this relation
directly from raw completion rows before any explanatory packet arithmetic.

R1 has two live-runtime instances. Both are genuine risk: a server that
recreates the runtime per injection, advances past the wrong horizon, drains
too early, loses FIFO state, or reports a projection from a parallel ledger
can reach the check and fail it. Planned genuine-risk fraction: `2/2`.

### Entailment analysis

F1 does not entail R1. F1 uses quiescent injections separated by 10,000,000 ps
and compares them to isolated runs. R1 uses overlapping injections at 0 ps
and compares the second completion against a different reset execution. No
earlier exact check pins `S_P`, `I_P`, `H_P`, or `h_P` to the R1 predicate.
The harness computes R1 from raw observations before applying any packet
serialization explanation. R1 can therefore fail after the run reaches it.

## R2: live wall-clock relation

Elapsed time for isolated mode begins before rendering the first GOAL and ends
after the final completion CSV is parsed. It includes every GOAL write,
`txt2bin`, child startup, simulator run, and CSV parse. Session time begins
before starting the one persistent process and ends after its close response
and process reap. It includes framing, open, every inject and advance, drain,
close, simulation, and response parsing. Matrix construction, binary hashing,
and evidence comparison are outside the timed boundaries.

| replay | isolated band, s | session band, s | minimum speedup |
|---|---:|---:|---:|
| two-node | `[0.002, 0.5]` | `[0.0005, 0.25]` | `1.2x` |
| four-node | `[0.004, 1.0]` | `[0.0005, 0.25]` | `1.5x` |

The signed expectation is lower wall time in session mode. The bands derive
from the exact base-binary calibration above, with broad allowance for machine
load and new process setup. R2 has two live-runtime instances. Both can fail
through framing, startup, simulation, process scheduling, or accidental
per-step child reuse, so the planned genuine-risk fraction is `2/2`. F1 does
not entail R2 because byte-identical simulated time places no bound on host
elapsed time.

## Registered command and pre-freeze dry run

Bulk outputs remain outside Git. The literal registered command is:

```bash
SIMLLM_HTSIM_RNIC="${SIMLLM_HTSIM_RNIC:?configure the paired binary}" \
SIMLLM_TXT2BIN="${SIMLLM_TXT2BIN:?configure the matching converter}" \
.venv/bin/python examples/persistent_session_v1/run_study.py \
  --out "${SIMLLM_DATA_ROOT:?configure the data root}/persistent_session_v1"
```

Before the original freeze and again before the wall-only amendment, the same
command must be executed with `--check-only`.
Check-only validates both executable paths, the fixed replay and state
matrices, all wall bands, the four codec cases, the frame limit, and output
placement. It prints the complete plan and creates no artifacts. The harness
existed untracked during the original freeze and contained only the literals
frozen in this file. At the amendment it is tracked and has only the corrected
CLI spelling, canonical UTF-8 rendering, and the amended wall literals as
unstaged changes.
