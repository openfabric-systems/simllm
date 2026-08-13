# NVLink locality v1 requalification expectations (CORE-42)

This expectations-only record freezes the CORE-42 requalification of
`nvlink_locality_v1` before the study is rerun. It registers the corrected
all-local `AAAA` service and JCT values, restates the unchanged `AABB` and
`ABCD` rows as independently derived rather than merely inherited, and settles
in advance how the refrozen `AAAA` instances are to be classified.

The original TRAF-10 freeze in [expectations.md](expectations.md) and the
post-TRAF-25 ownership refreeze recorded in
[the token ownership results](../token_ownership_v1/RESULTS.md#nvlink_locality_v1)
remain historical chronology. Neither is rewritten here.

## Why a requalification is needed

CORE-41 replaced the analytic intra-node service. The superseded charge was the
maximum source egress over a phase; the accepted charge is the maximum endpoint
load, where one endpoint's load is `max(egress_bytes, ingress_bytes)` because
the modeled NVLink port is full duplex. The runner still carries the superseded
all-local literals 4,538,000 ps and 9,047,000 ps, so it now rejects its own
fixture.

Those two cells are not baseline observations. They are scored TRAF-B2
instances, so the corrected values are registered here, in their own commit,
before any rerun.

## Provenance and chronology

| Field | Value |
|---|---|
| Evidence authored against | `aeb40ac95cdd8163942297335948c94df0376e04` |
| Original TRAF-10 expectations commit | `dd1eefebc84091c547d8cad10225b21ab85a7706` |
| Fixture | `examples/preplay_trace_v1/granite_length_cap.jsonl` |
| Fixture SHA-256 | `36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341` |

The commit that lands this file also updates the runner's frozen acceptance
literals, because the runner is this study's checker rather than its
implementation. It contains no behavior change, no generated result and no
measured value: every literal below comes from the independent derivation in
the next section. The behavior under test, the CORE-41 endpoint charge, landed
earlier in `78d8c14f52e7bfa3a45adcf61b24b3e77038ae0e`.

The result report will record the SimLLM revision and htsim gitlink that the
rerun observes as separate provenance fields. No frozen literal here is
asserted equal to a live submodule pin.

## Independent derivation of every registered literal

The derivation reads the tracked capture with the standard library only. It
imports no SimLLM module and runs no native tool. It applies the declared
payload rule (one home rank owns every routed token, and a token crosses to a
destination rank once per phase however many of its top-k experts that rank
owns), the contiguous eight-expert ownership of the four EP ranks, and the
CORE-41 endpoint charge

```text
endpoint_load(r, p)   = max(egress_bytes(r, p), ingress_bytes(r, p))
endpoint_service_ns   = ceil(endpoint_load * 1e9 / 450,000,000,000)
nvlink_phase_ps(p)    = 1,000 * max_r(endpoint_service_ns(r, p))
```

Its control is that it must reproduce the four literals CORE-41 did not change:
the `AABB` local and fabric byte counts and both `AABB` services. It does.

| Vector bytes | Placement | Total bytes | Fabric bytes | NVLink bytes | Service ps |
|---:|---|---:|---:|---:|---:|
| 1,024 | `AAAA` | 2,983,936 | 0 | 2,983,936 | **6,652,000** |
| 1,024 | `AABB` | 2,983,936 | 2,011,136 | 972,800 | 2,194,000 |
| 1,024 | `ABCD` | 2,983,936 | 2,983,936 | 0 | 0 |
| 2,048 | `AAAA` | 5,967,872 | 0 | 5,967,872 | **13,286,000** |
| 2,048 | `AABB` | 5,967,872 | 4,022,272 | 1,945,600 | 4,358,000 |
| 2,048 | `ABCD` | 5,967,872 | 5,967,872 | 0 | 0 |

The same derivation reproduces the superseded egress-only services, 4,538,000 ps
and 9,047,000 ps, from the same pair table, which is the check that the two
charges are being compared on one fixture rather than on two readings.

The registered live JCT points and bands are:

| Vector bytes | `AAAA` JCT ps | `AABB` JCT ps | `ABCD` JCT band ps |
|---:|---:|---:|---:|
| 1,024 | **6,676,000** | 136,246,720 | [155,702,720, 155,702,864] |
| 2,048 | **13,310,000** | 176,469,440 | [215,381,440, 215,381,584] |

The all-local JCT is the analytic service plus the fixed 24,000 ps compute
estimate, with no fabric term because those cells emit no flow. The two-node and
all-remote points are unchanged and are derived below rather than inherited.

## Physical sanity before the exact comparison

Floors and ceilings first, in one line each, from first principles.

**Floor for an all-local cell.** No local phase can beat its own peak endpoint
bytes over the declared one-direction rate, so the step cannot beat the sum of
those peaks over 48 serial phases. On this fixture rank 0 is the star hub of
every phase: it sources every dispatch byte and sinks every combine byte, so a
phase's peak endpoint load is exactly that phase's local byte count and the sum
of the 48 peaks is exactly the cell's local byte total. The floor is therefore
`total_local_bytes * 1e12 / 450e9` ps.

**Ceiling for an all-local cell.** GOAL calc units are whole nanoseconds, so
each of the 48 serial phases rounds up by strictly less than 1 ns. The ceiling
is the floor plus 48,000 ps.

| Vector bytes | Placement | Floor ps | Ceiling ps | Registered ps | Position |
|---:|---|---:|---:|---:|---|
| 1,024 | `AAAA` | 6,630,969 | 6,678,969 | 6,652,000 | floor + 21,031 |
| 2,048 | `AAAA` | 13,261,938 | 13,309,938 | 13,286,000 | floor + 24,062 |
| 1,024 | `AABB` | 2,161,778 | 2,209,778 | 2,194,000 | floor + 32,222 |
| 2,048 | `AABB` | 4,323,556 | 4,371,556 | 4,358,000 | floor + 32,444 |

**Floor and point for the fabric cells.** The accepted fluid manifold charges
2,000,000 ps of propagation per serial phase and serializes the phase's star
bottleneck at 400 Gbit/s, i.e. 20 ps per byte. With the fixed 24,000 ps compute
term that gives `96,000,000 + 20 * fabric_bytes + 24,000` ps:

| Vector bytes | Placement | Fabric bytes | Derived ps | Registered ps |
|---:|---|---:|---:|---|
| 1,024 | `AABB` | 2,011,136 | 136,246,720 | 136,246,720 exact |
| 2,048 | `AABB` | 4,022,272 | 176,469,440 | 176,469,440 exact |
| 1,024 | `ABCD` | 2,983,936 | 155,702,720 | band lower edge |
| 2,048 | `ABCD` | 5,967,872 | 215,381,440 | band lower edge |

The all-remote band adds at most one whole picosecond of max-min quantization
per positive flow, i.e. 144 ps. The two-node cells are exact points because the
local term is about 45 ns per phase while the remote term is above 2 us per
phase, so the registered `max(local, remote)` composition is remote everywhere
and the local term is fully hidden.

**Sanity against the system being imitated.** 6,652,000 ps of NVLink service to
move 2,983,936 bytes across one node is 449 GB/s of realized hub throughput,
which is the declared 450 GB/s rate minus quantization, as it must be for a
star whose hub is saturated in every phase. It is a conservative H100-class
one-direction surrogate, not a B100 measurement; TRAF-11 still owns
calibration.

**The number that must scale with it.** Doubling the vector doubles every
payload, so the endpoint charge must double the service up to quantization
only:

```text
2 * 6,652,000 - 13,286,000 = 18,000 ps
```

which lies in the registered `[0, 48,000]` ps window, because the doubled cell
rounds up 48 phases once instead of twice. A ratio that is exactly 2.000000, or
that misses 2 by more than 48 ns, refutes the whole-nanosecond phase model
rather than confirming it. The corrected-over-superseded ratio is registered as
1.4658 at 1,024 bytes and 1.4686 at 2,048 bytes, matching the 1.466 undercharge
CORE-41 reported at this EP width.

## What must not move

Every `AABB` and `ABCD` row is registered unchanged: byte splits, services,
both exact two-node JCT points and both all-remote bands, the frozen direct
GOAL length of 20,392 bytes with its two SHA-256 values, the 48 phases and 144
positive directed pairs, the zero-fabric and zero-local guards, the tag
sequence, the transpose, the TP width sweep and physical quiescence. A change
in any of them is not a requalification result; it is a separate defect and
voids this run.

## Scored families and their instances

The registered families are unchanged from TRAF-10: TRAF-B1 raw locality
response over 2 payload instances, TRAF-B2 analytic local service over 4
placement/payload instances, and TRAF-B3 live metric response over 2 payload
instances. The headline remains 3 families and 8 instances. Raw observations are
read before any exact byte, conservation, digest or transpose oracle, so no
earlier fatal guard pins a scored instance.

## Registered classification of the refrozen AAAA instances

CORE-42 asks whether the refrozen `AAAA` instances still carry genuine risk or
have become exact-oracle evidence. The answer is decided here, before the run,
because it follows from the arithmetic rather than from the outcome.

They remain genuine risk, and they are weaker than they were. Both halves of
that are registered:

1. **They can still fail.** The alternative the requalification exists to
   detect, the superseded egress-only charge, produces 4,538,000 ps, which is
   2,114,000 ps outside the registered window. TRAF-B2 reads the raw analytic
   service before any exact-cell or conservation oracle, so nothing pins it
   first.
2. **Their magnitude is nearly entailed.** Because rank 0 is the hub of every
   phase, the sum of the per-phase peak endpoint loads is identically the local
   byte total, which the fatal exact-cell and conservation guards already pin.
   Given those guards, the endpoint charge admits only the 48,000 ps
   quantization window above. The frozen value is a point inside a window that
   is 0.72 percent wide at 1,024 bytes and 0.36 percent wide at 2,048 bytes, so
   the surviving discrimination is over the per-phase byte split and the
   rounding rule, not over the magnitude.
3. **This fixture cannot falsify the duplex choice.** The hub is pure egress in
   a dispatch phase and pure ingress in a combine phase, so a half-duplex
   `egress + ingress` port yields exactly the same load. The full-duplex ruling
   in `simllm/traffic/locality.py` is therefore untested by these cells and must
   not be reported as demonstrated by them.

The `AABB` instances are weaker still: their local group is a single directed
pair per phase, where the endpoint charge and the superseded egress charge are
algebraically identical. They are registered as unchanged controls that
discriminate the rate, the locality split and the rounding, and not the charge.

## Registered acceptance clauses

1. The tracked fixture still matches its frozen SHA-256, and the study reruns
   end to end from the registered command.
2. Both `AAAA` cells report 6,652,000 ps and 13,286,000 ps of analytic service
   and 6,676,000 ps and 13,310,000 ps of live JCT, each inside its registered
   floor and ceiling.
3. Every `AABB` and `ABCD` row is exactly the registered value, including both
   independently derived fabric JCT points and both all-remote bands.
4. The doubling relation lands inside the registered `[0, 48,000]` ps
   quantization window.
5. All fatal-unscored guards hold. A violation makes the run void rather than
   scored, and fatal guards are never reported as a fraction.
6. The result report states the classification above against the observed run
   and reports the contradiction sweep.

## Registered command and check-only dry run

```text
.venv/bin/python examples/nvlink_locality_v1/run_study.py --out "$SIMLLM_NVLINK_LOCALITY_RUN_ROOT"
```

Before this commit the same command was run with `--check-only`. That path
parses the full production CLI, validates only the frozen literal shapes and
arithmetic, imports no SimLLM module, invokes no native tool and creates no
output directory or artifact.
