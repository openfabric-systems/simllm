# NVLink locality v1 results

TRAF-10's placement split worked at the byte and analytic-service boundary,
and its decision-relevant signed placement order reached `StepResult`. The
registered exact live-metric family did not fully pass. The final behavioral
score is **2/3 genuine-risk families and 6/8 parameterized instances**.

The miss is informative. The frozen all-remote JCT bands assumed a global
barrier between the 48 collective phases. The accepted byte-identical legacy
GOAL actually advances each rank from its own completion frontier, so adjacent
phases overlap. This report preserves that failed expectation and assigns the
active causal-semantics gap to TRAF-12.

The byte, service and JCT values below are the historical pre-TRAF-25
source-multiplied observations. The corrected single-engine table is in
[the token ownership results](../token_ownership_v1/RESULTS.md#nvlink_locality_v1).
Its corrected single-node values are baseline observations pending CORE-41's
destination-ingress correction, as recorded in
[the dependency authority refreeze](../dependency_authority_v1/RESULTS.md#corrected-six-cell-sweep).

## Chronology and provenance

The expectations-only commit is
`dd1eefebc84091c547d8cad10225b21ab85a7706`. Its exact production command was
first exercised with `--check-only`; the target implementation was not
imported, no native binary ran, and the requested output path remained absent.
The behavior then landed in
`3fa568e8ce684e32382cea67132d5b9f3c45a174`.

One result-producing run was executed. It wrote
`$SIMLLM_NVLINK_LOCALITY_RUN_ROOT/summary.json` and then exited nonzero because
TRAF-B3 missed its frozen all-remote bands. No second outcome-producing run
was used to replace it. The phase-overlap audit below is explicitly
post-specified and reads the failed run's existing CSV timestamps.

| Provenance field | Revision or value |
|---|---|
| Evidence authored against | `6973bd0e3ed6091c403c7055ee01c2d8ae0ae970` |
| Expectations commit | `dd1eefebc84091c547d8cad10225b21ab85a7706` |
| SimLLM revision observed by the run | `3fa568e8ce684e32382cea67132d5b9f3c45a174` |
| htsim gitlink observed by the run | `fc4400e4ca619223481536632074045cb6af2756` |
| htsim build input | the separately recorded observed gitlink above |
| Captured trace SHA-256 | `36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341` |
| Runtime | Python 3.12.12 on Linux x86-64 |

These revisions are observations, not an assertion that any frozen literal
must equal a future live submodule pin.

## Declared model and sweep

The local term is analytic and uncalibrated. Its declared constant is
450,000,000,000 bytes/s per source GPU, with zero propagation:

```text
source_service_ns(s, p)
    = ceil(local_egress_bytes(s, p) * 1e9 / 450,000,000,000)

nvlink_phase_ps(p)
    = 1,000 * max_s(source_service_ns(s, p))
```

The value is a conservative H100-class one-direction surrogate derived from
half of NVIDIA's published 900 GB/s bidirectional H100 figure. It is not a
B100 measurement or same-generation calibration. TRAF-11 owns replacement of
the declared constant with same-generation measurements.

The study held one captured 22-token, 24-layer Granite routing step fixed. It
swept vector bytes 1,024 and 2,048 across one-node `AAAA`, two-node `AABB`, and
all-remote `ABCD` placements. Compute stayed fixed at 24,000 ps. Each cell was
replayed three times so the same live step latency was visible as controlled
TTFT and TPOT.

## Raw observations

| Vector bytes | Placement | Fabric bytes | NVLink bytes | NVLink service ps | StepResult JCT ps |
|---:|---|---:|---:|---:|---:|
| 1,024 | `AAAA` | 0 | 11,870,208 | 7,097,000 | 7,121,000 |
| 1,024 | `AABB` | 7,913,472 | 3,956,736 | 2,442,000 | 139,195,840 |
| 1,024 | `ABCD` | 11,870,208 | 0 | 0 | 156,569,755 |
| 2,048 | `AAAA` | 0 | 23,740,416 | 14,156,000 | 14,180,000 |
| 2,048 | `AABB` | 15,826,944 | 7,913,472 | 4,838,000 | 182,367,680 |
| 2,048 | `ABCD` | 23,740,416 | 0 | 0 | 217,222,486 |

Fabric bytes increased and NVLink bytes decreased strictly across
`AAAA < AABB < ABCD` at both payloads. JCT had that same signed order at both
payloads. Controlled TTFT and TPOT equal the JCT shown because all three
replays used the same fixed step; this demonstrates metric reachability, not
decode-workload fidelity.

## Scored behavioral evidence

Evidence classes are not combined. The headline below includes only
genuine-risk behavioral families and their registered parameter instances.

| Family | Instances | Result | Raw relation |
|---|---:|---|---|
| TRAF-B1 locality response | 2 | 2/2 pass | Fabric bytes strictly rise and local bytes strictly fall over node span at both payloads. |
| TRAF-B2 local service | 4 | 4/4 pass | Each nonzero local service equals the registered per-source whole-nanosecond form. |
| TRAF-B3 live metric | 2 | 0/2 pass | All cells reached `StepResult` and both signed placement orders passed, but each payload instance failed because its all-remote exact band missed. |

The genuine-risk headline is **2/3 families, 6/8 instances**. The two B3
placement-order relations are reported as raw observations but are not added
to the score because the frozen family groups each payload's six exact cells
and band into one instance.

### Entailment check

The runner evaluated TRAF-B1 from raw fabric and local byte counters before
the later exact split and conservation checks. It evaluated TRAF-B2 from raw
analytic service before the later exact cell table. It evaluated TRAF-B3 from
live `StepResult.step_latency_ps` values before GOAL digests, CSV identity,
zero-resource assertions, transpose checks, or conservation. Therefore no
earlier fatal oracle pinned a scored result.

Payload-doubling direction, controlled TTFT/TPOT equality, exact split totals,
all-intra zero fabric, all-remote zero local bytes, and identity digests are
entailed, fixed-configuration, or by-construction facts. They are fatal or
descriptive and never enter the behavioral denominator.

## Registered miss and post-specified diagnosis

The 1,024-byte all-remote JCT was 156,569,755 ps, 4,212,005 ps below the lower
registered band. The 2,048-byte all-remote JCT was 217,222,486 ps, 8,317,034 ps
below its lower band. The one-node and two-node points matched exactly.

The frozen arithmetic treated all 48 remote phases as global serial phases.
The accepted all-remote identity path deliberately retained the monolithic
legacy GOAL, whose dependencies are rank-local. A post-specified audit of the
raw completion CSV found:

| Vector bytes | Adjacent transitions | Transitions with an early next-phase start | First overlap ps | Largest overlap ps |
|---:|---:|---:|---:|---:|
| 1,024 | 47 | 46 | 368,640 | 1,413,120 |
| 2,048 | 47 | 46 | 737,280 | 3,675,091 |

For the 1,024-byte cell, tag 1001 begins at 2,984,041 ps while tag 1000's last
flow completes at 3,352,681 ps. Thus the registered phase-additive band was
not an oracle for the byte-identical legacy renderer. Rewriting the frozen
band after seeing this result would manufacture a pass, so it remains failed.

The localized path uses separate phase executions because a single rank-local
GOAL cannot express the required source eligibility and global phase barrier.
This yields internally sound `max(local, remote)` composition, but it exposes
a timing-semantics mismatch with the legacy identity path. TRAF-12 is a P0
precision residual for establishing one causal authority while retaining the
all-remote GOAL-byte lock.

## Fatal and unscored evidence

All fatal-unscored guards passed:

- all six exact byte splits and local-service cells matched;
- the two explicit all-remote GOALs matched omitted placement and their
  preimplementation length and SHA-256 locks;
- explicit and omitted all-remote flow CSV bytes, timestamps, and legacy
  `StepNetworkOutcome` records matched;
- all-intra cells had zero fabric bytes, zero fabric segments, zero backend
  runs, zero flows, and zero GOAL files;
- all-remote cells had zero local bytes and used the compatibility fast path;
- both payloads retained 48 phases, 576 positive ordered pairs, exact
  dispatch/combine transpose, exact partition and stable tags;
- single-node TP widths 1 through 8 had zero fabric bytes;
- all live backend runs reported physical quiescence;
- malformed placement and bandwidth inputs failed before output mutation in
  the native test suite.

These checks are required for acceptance but contribute zero to the scored
fraction.

## Existing accepted artifacts

Locality is explicit opt-in. M4, M5, `examples/breakdown`, routed supply, and
other callers that omit physical placement retain the accepted monolithic
all-remote renderer. The full test suite preserved the existing GOAL and
serialization locks. No frozen historical expectations or result artifact was
edited. The committed breakdown fabric-TP columns remain byte-identical and
should now be read as the all-remote, cross-node what-if rather than a
single-node deployment.

## TRAF-10 closure map

Each registered clause is quoted and mapped below. Any fidelity not
demonstrated here has a new task ID.

| Registered clause | Evidence and disposition |
|---|---|
| "Intra-node segments of a collective should not ride the fabric" | Exact split cells passed; both `AAAA` cells had zero fabric work and no backend invocation. |
| "model them as a point-to-point NVLink-class network" | Directed segments are grouped by source into an analytic per-source serializer. This is an uncalibrated first cut; TRAF-11 owns measured fidelity. |
| "a flat same-generation NVLink per-GPU bandwidth, analytic, no packet simulation" | The analytic mechanism and per-GPU source authority are implemented. The declared 450 GB/s source is H100-class rather than B100 calibration, stated plainly above; TRAF-11 owns the same-generation replacement. |
| "send only inter-node segments to htsim" | The ordered phase partition and fabric render tests passed; local-only phases invoked no native backend. |
| "single-node TP (any width up to 8 on the 8-GPU reference node) has no fabric component at all" | The fatal width sweep passed for widths 1 through 8. |
| "the fabric story applies to cross-node placements" | `AABB` sent exactly its cross-node pairs to htsim; `ABCD` retained all accepted fabric bytes. |
| "Needs the locality knowledge of the placement manifest (`is_intra_node`)" | `RankMapper` snapshots and validates the manifest, then classifies semantic ranks through `is_intra_node`. No graph locality copy was added. |
| "composes with the `unique-nic` GOAL-rank mapping (PLACE-2)" | Classification occurs before `goal_rank`; the fabric renderer permits multiple semantic ranks per endpoint and rejects only a collapsed cross-node pair. PLACE-2 itself remains deliberately unimplemented. |
| "the committed examples/breakdown fabric-TP columns become the cross-node what-if under this model" | The absent-placement identity path and full-suite byte locks passed; the interpretation is recorded above without rewriting historical artifacts. |
| "the same collective must produce a registered signed change when the participants move from all-intra-node to spanning two nodes" | Both payloads produced strict raw byte and live JCT changes from `AAAA` to `AABB`. |
| "the intra-node case has zero bytes on the fabric and an NVLink term given by the registered closed form" | Both zero-fabric guards and both exact `AAAA` analytic-service cells passed. |
| "the spanning case puts exactly the cross-node segments on the fabric and the rest on NVLink" | Both `AABB` exact split cells and conservation partitions passed. |
| "the all-remote case reproduces today's accepted bytes exactly" | Both explicit `ABCD` GOALs match the frozen preimplementation hashes and omitted-placement bytes. |
| "Sweep at least the node span and one payload size" | The study swept three node spans and two payload sizes. |
| "The all-remote cell is your identity off path" | GOAL, flow CSV, timestamps, and legacy outcome equality passed. The unexpected legacy phase overlap remains visible rather than being hidden. |

TRAF-10 closes for the implemented locality split. TRAF-11 owns hardware
calibration. TRAF-12 owns the exact causal-timing inconsistency exposed by the
failed B3 family. PLACE-4, PLACE-5 and CORE-25 were not consumed because the
existing placement manifest, mapper join, and live `StepResult` seam suffice.

## Contradiction sweep

After closure, the required sweep found integrator-owned text that should be
reconciled without editing it in this branch:

- `README.md:256` describes one compute-owned flat NVLink egress serializer
  but does not distinguish it from the new traffic-owned analytic authority.
- `docs/README_PRO.md:260-280` and `335-340` describe intra-node service as a
  full NCCL-kernel path through the live stack. TRAF-10's landed sink path is a
  separate analytic traffic term and does not demonstrate that full stack.
- `docs/README_PRO.md:418` omits the implemented locality split and its TRAF-12
  timing residual from the traffic status row.
- `docs/architecture.md:116-120` reads as if `unique-nic` mapping is available,
  while PLACE-2 still rejects it.
- `docs/architecture.md:527-529` says intra-node service stays inside the
  compute model, while this live sink currently has a separate traffic-owned
  analytic authority. The two authorities are not enabled together in this
  study, but the ownership wording is stale.

`README.md:244-245`, which says intra-node traffic stays off the fabric, is
consistent with this result. These hits are reported only, as required by the
wave contract.
