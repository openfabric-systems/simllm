# NVLink locality v1 expectations

This is the expectations-only record for TRAF-10. The study freezes one
captured-routing MoE collective while varying physical node span and hidden
vector size. No locality implementation or result-producing run existed when
this file was committed.

## Decision and source audit

The audited SimLLM source state is repository commit `6973bd0`. Before this
freeze:

- `simllm/placement/manifest.py:46-61` records each global rank's hostname and
  local GPU rank. `simllm/placement/mapper.py:45-47` already answers
  `is_intra_node` from that manifest. The placement manifest therefore remains
  the single locality authority; this change must not copy a locality flag into
  `ExecutionGraph`.
- `simllm/traffic/step_comm.py:220-269` expands captured router assignments
  into sparse directed dispatch pairs and their exact combine transpose.
  `simllm/traffic/step_comm.py:433-451` hands every positive pair to
  `pairwise_all_to_allv` without a physical-locality join.
- `simllm/traffic/patterns.py:103-127` renders every positive directed pair as
  one GOAL send and one matching receive. It omits only zero or absent pairs.
- `simllm/backends/step_sink.py:266-370` renders that all-remote GOAL, runs
  htsim, and returns its makespan through `StepResult`. Its configuration has
  no placement input at the audited state.
- `examples/m5/expectations.md:87-113` freezes the fluid manifold's integer
  max-min allocation: each active peer receives a whole-bit/s water level,
  serialization rounds up to whole picoseconds, and propagation is
  2,000,000 ps. This study reuses that accepted backend law rather than fitting
  a new network form.

The routed input is the tracked real Granite capture
`examples/preplay_trace_v1/granite_length_cap.jsonl`, whose accepted SHA-256
is `36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341`.
It contains 22 captured prefill tokens over 24 MoE layers, top-k 8 and 32
experts. The study holds those router assignments fixed. Four EP ranks own
contiguous expert ranges of eight experts each.

The first-cut local rate is the maintainer-directed declared constant
`450,000,000,000 bytes/s` per source GPU. It is not a measurement. It is a
conservative H100-class one-direction surrogate obtained by halving NVIDIA's
published 900 GB/s bidirectional H100 NVLink figure. It is not the B100
same-generation value: NVIDIA documents 1.8 TB/s bidirectional per Blackwell
GPU. TRAF-11 must calibrate the surrogate against same-generation captures.
Sources:

- <https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html>
- <https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html>

## Fixed sweep

The EP ranks are `(0, 1, 2, 3)`. The two payload points change only one hidden
vector's byte count, `V in {1,024, 2,048}`. Model dtype is two bytes and hidden
size is `V / 2`. The step has a fixed 24,000 ps compute estimate, represented
as one 1 ns calc on each of 24 layers.

Physical placement is the other swept parameter:

| Placement | Rank host pattern | Meaning |
|---|---|---|
| `AAAA` | one shared hostname | all directed pairs are intra-node |
| `AABB` | ranks 0/1 and 2/3 share hosts | one fixed two-node span |
| `ABCD` | four distinct hostnames | every directed pair is remote |

`ABCD` is also the identity-off cell. Explicit `ABCD` locality and omitted
locality must use the accepted all-remote renderer and preserve its GOAL bytes
exactly.

## Registered analytic form

Each dispatch or combine operation is one serial communication phase. For a
phase `p` and source rank `s`:

```text
local_egress_bytes(s, p)
    = sum(payload(s, d) for same-host destinations d)

source_service_ns(s, p)
    = ceil(local_egress_bytes(s, p) * 10^9 / 450,000,000,000)

nvlink_phase_ps(p)
    = 1,000 * max_s(source_service_ns(s, p))
```

The whole-nanosecond ceiling is part of this first cut because GOAL calc units
are nanoseconds. There is no added propagation term. A participant's local
visibility waits for the relevant source serializer, and a source waits for
its own egress serializer. Local and remote service start from the same phase
frontier. The phase completes at `max(local NVLink, remote htsim)`, never at
their sum. Phases and layer compute remain serial under the pre-existing
TRAF-7 off path.

The additive `nvlink_service_ps` report is the sum of `nvlink_phase_ps` over
serial phases. It is local work accounting. In a mixed phase it need not be a
separate additive critical-path contribution because remote service may hide
it.

## Frozen independent arithmetic

An independent standard-library audit of the accepted captured pair table
produced 48 phases, 576 positive directed pairs, and these raw totals:

| Vector bytes | Placement | Total bytes | Fabric bytes | NVLink bytes | NVLink service ps |
|---:|---|---:|---:|---:|---:|
| 1,024 | `AAAA` | 11,870,208 | 0 | 11,870,208 | 7,097,000 |
| 1,024 | `AABB` | 11,870,208 | 7,913,472 | 3,956,736 | 2,442,000 |
| 1,024 | `ABCD` | 11,870,208 | 11,870,208 | 0 | 0 |
| 2,048 | `AAAA` | 23,740,416 | 0 | 23,740,416 | 14,156,000 |
| 2,048 | `AABB` | 23,740,416 | 15,826,944 | 7,913,472 | 4,838,000 |
| 2,048 | `ABCD` | 23,740,416 | 23,740,416 | 0 | 0 |

The accepted all-remote renderer's frozen GOAL oracles, each with a 1 ns calc
on all 24 layers, are:

| Vector bytes | GOAL bytes | SHA-256 |
|---:|---:|---|
| 1,024 | 72,819 | `0417832c8788a0477d48b414cf2d8456b87215abd1d0193ba46fb8db46185d8a` |
| 2,048 | 72,819 | `bcd72e63546d03efaddd48c16e160457d1e28f19795036d1f871788d78cf5a02` |

At 400 Gbit/s on `rnic-nn-fluid`, the registered live `StepResult` makespans
are:

| Vector bytes | `AAAA` JCT ps | `AABB` JCT ps | `ABCD` JCT band ps |
|---:|---:|---:|---:|
| 1,024 | 7,121,000 | 139,195,840 | [160,781,760, 160,781,808] |
| 2,048 | 14,180,000 | 182,367,680 | [225,539,520, 225,539,568] |

The one-node values are compute plus the registered whole-nanosecond NVLink
form. In every two-node phase, remote service dominates local service; the
two-node point is compute plus the exact two-peer fluid port-work form. The
all-remote band permits at most one whole-picosecond max-min quantization
increment per phase after `floor(400e9 / 3)`.

## Scored behavioral families

All scored checks consume raw observations before any exact table, digest or
conservation oracle is evaluated.

### TRAF-B1: raw locality response

For each vector size, observed fabric bytes must increase strictly and
observed NVLink bytes must decrease strictly across `AAAA`, `AABB`, `ABCD`.
The six raw cells are inspected before conservation. This is one scored family
over two payload instances.

### TRAF-B2: analytic local-service response

The four nonzero `AAAA` and `AABB` local-service observations must match the
registered whole-nanosecond closed form exactly. Doubling `V` must increase
the raw service, but that direction is not counted again because the exact
cells entail it. This is one scored family over four placement/payload
instances.

### TRAF-B3: live metric response

All six placement/payload cells must reach `HtsimStepSink -> StepResult` and
meet the point or band above. Consequently the raw JCT is strictly ordered
`AAAA < AABB < ABCD` at both payloads. The exact cells and bands are evaluated
directly from `StepResult.step_latency_ps` before byte/digest oracles; the
entailed order is descriptive, not an extra score.

For signature-metric visibility, each cell is replayed three times with the
same frozen captured collective. TTFT is the first `StepResult` latency and
TPOT is the mean of the next two virtual-clock deltas, following the M4
virtual-mode construction. This controlled fixed-step replay tests metric
reachability; it is not a claim about decode routing or workload fidelity.
TTFT and TPOT equal the cell JCT by construction and therefore do not add
another scored family or instance.

The scored headline is 3/3 families over 8 parameterized instances: two B1,
four B2, and two B3 payload grids. The six B3 cells are grouped into two
payload instances because placement order is the decision relation.

## Fatal and unscored evidence

These checks are required but never increase the behavioral denominator:

- every positive directed pair appears exactly once on either NVLink or the
  fabric, and `fabric_bytes + nvlink_bytes == legacy_total_bytes`;
- `AAAA` emits zero fabric flows, while `ABCD` emits no local segment;
- explicit `ABCD` and omitted locality reproduce the frozen GOAL length and
  SHA-256 for both payloads, plus identical htsim flow bytes and timestamps;
- dispatch and combine remain exact transposes, phase/tag order is stable,
  and the backend reports physical quiescence;
- single-node TP widths 1 through 8 produce zero fabric bytes; width 1 has no
  collective work, and widths 2 through 8 use only the analytic NVLink path;
- unknown ranks, incomplete manifests, invalid bandwidths and inconsistent
  rank groups fail before output mutation;
- existing absent-placement byte locks in the M4, M5, routed-supply and step
  sink tests remain unchanged.

The zero-fabric cell, identity hashes, conservation, author-defined placement
sequence, fixed phase count and transpose are exact or by-construction guards.
They are fatal-unscored by design.

## Entailment analysis

B1, B2 and B3 can each fail in a run that reaches it. B1 reads the raw
classification counters before conservation, so the later exact split cannot
entail its result. B2 reads the raw analytic duration before the later byte
table is checked; bit/byte confusion, global rather than per-source
serialization, or wrong rounding can fail B2 even with conserved bytes. B3
reads live `StepResult` values before any digest or exact-byte oracle; correct
looking counters can still fail to reach the sink or can compose local and
remote service incorrectly. The later fatal checks do not enter the scored
denominator.

## Registered command and pre-freeze dry run

The result-producing command is:

```text
.venv/bin/python examples/nvlink_locality_v1/run_study.py --out "$SIMLLM_NVLINK_LOCALITY_RUN_ROOT"
```

Before this expectations commit, the exact command above was run with
`--check-only`. Check-only parses the complete production CLI, validates only
the frozen literal shapes and arithmetic, imports no target SimLLM module,
invokes no native tool, creates no output directory or artifact, and prints a
confirmation by design.

The result report must cite the final expectations-only commit and record the
SimLLM and htsim revisions observed by the run separately from the revision
against which this evidence was authored. It must not require either observed
revision to equal a live submodule pin.
