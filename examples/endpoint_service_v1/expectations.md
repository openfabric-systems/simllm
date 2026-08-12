# Endpoint service v1 expectations

This is the expectations-only record for CORE-41. It freezes the endpoint
accounting decision, arithmetic sweep, live JCT relation, dependency-authority
refreeze and physical bounds before implementation or any result-producing
run.

## Decision and model boundary

The analytic intra-node model will charge one full-duplex endpoint per rank.
For each already serial communication phase and endpoint rank `r`, define:

```text
egress(r) = sum(payload_bytes for local segments whose source is r)
ingress(r) = sum(payload_bytes for local segments whose destination is r)
endpoint_load(r) = max(egress(r), ingress(r))
phase_service_ns = max_r ceil(endpoint_load(r) * 1e9 / B)
```

`B` is the declared one-direction endpoint bandwidth in bytes per second. The
production classifier must construct an explicit, sorted ledger containing
both byte directions for every local endpoint. It must not recover ingress
from symmetry or transpose assumptions. Endpoint ledgers are per phase because
the graph already declares the phases serial. Aggregating bytes across phase
boundaries would create a different schedule.

The rejected alternative is a shared half-duplex port charged as
`egress(r) + ingress(r)`. That rule would double a symmetric phase even though
the hardware has independent direction capacity. The flat 450 GB/s rate and
zero propagation remain uncalibrated first-cut parameters; TRAF-11 still owns
their calibration. CORE-41 changes directional accounting, not the routed
renderer, collective phase boundaries or byte population.

If symmetric service does not remain exact, the full-duplex maximum or ledger
projection is wrong and this design will not land. If combine service does not
increase by the exact registered amount, the directional aggregation is wrong.
If the same increase does not reach live JCT, endpoint accounting is not on the
supported execution path and CORE-41 remains open.

## Pre-freeze source audit

The evidence was authored against SimLLM commit
`76223875557a552deb5aa2c2c529a07f000135ba`. The observed htsim gitlink during
the audit was `fc4400e4ca619223481536632074045cb6af2756`. These are separate
provenance facts. Neither is an equality requirement on the revision or live
gitlink observed by a later run.

The repository audit at that SimLLM revision found:

- `simllm/traffic/locality.py:259-303` partitions local segments, accumulates
  only `source_egress_bytes`, and charges the largest source duration;
- `simllm/traffic/step_comm.py:927-1001` builds one dispatch star from the
  engine rank and an exact combine transpose for uniform MoE traffic;
- `simllm/traffic/step_comm.py:1682-1720` expands graph-owned pair payloads
  into the same communication phases;
- `simllm/backends/step_sink.py:599-628` places each analytic phase service on
  the checked execution artifact;
- `simllm/backends/step_sink.py:730-800` composes analytic and fabric service,
  sums ordered artifacts and obtains the step makespan;
- `simllm/backends/step_sink.py:838-887` projects that makespan into the
  scheduler-visible `StepNetworkOutcome` and `StepResult`.

Two NVIDIA hardware sources determine the duplex choice:

- NVIDIA, *DGX GB Rack Scale Systems User Guide*, section 1.3, "NVLink
  Switch Trays", PDF page 11, describes fifth-generation NVLink switch
  bandwidth as full duplex:
  <https://docs.nvidia.com/dgx/dgxgb200-user-guide/dgxgb200-user-guide.pdf>.
- NVIDIA, *NVIDIA NVSwitch Technical Overview*, April 2018, PDF page 3,
  states that each NVSwitch port supports bandwidth in each direction and
  that the crossbar is nonblocking:
  <https://images.nvidia.com/content/pdf/nvswitch-technical-overview.pdf>.

These sources establish independent transmit and receive directions. They do
not calibrate the repository's 450 GB/s aggregate endpoint parameter.

## Frozen fixture sweep

Sweep payload `P` in `{1,024, 2,048}` bytes and EP width `W` in `{2, 4, 8}`
at `B = 450,000,000,000` bytes/s. Every fixture is one all-local phase:

- symmetric: every ordered pair carries `P` bytes;
- dispatch star: rank 0 sends `P` bytes to each of the other `W - 1` ranks;
- combine star: each of the other `W - 1` ranks sends `P` bytes to rank 0.

The symmetric fixture has `W(W - 1)P` total group bytes. Each star has only
`(W - 1)P` total group bytes. All three nevertheless have the same peak
directional endpoint load, `(W - 1)P`. Total group bytes and peak endpoint
load are different observables and must be reported in separate fields.

The corrected service is the same for all three shapes:

| Payload bytes | EP width 2 | EP width 4 | EP width 8 |
|---:|---:|---:|---:|
| 1,024 | 3,000 ps | 7,000 ps | 16,000 ps |
| 2,048 | 5,000 ps | 14,000 ps | 32,000 ps |

Symmetric and dispatch service already use the critical source load and must
remain bit exact. The old combine surrogate charges only one peer payload:
3,000 ps for 1,024 bytes and 5,000 ps for 2,048 bytes. Therefore the exact
signed combine changes are:

| Payload bytes | EP width 2 | EP width 4 | EP width 8 |
|---:|---:|---:|---:|
| 1,024 | 0 ps | +4,000 ps | +13,000 ps |
| 2,048 | 0 ps | +9,000 ps | +27,000 ps |

Width two is the degenerate transpose in which one source and one destination
have equal load. Its zero change is a fatal-unscored compatibility guard, not
a claimed positive-response instance.

### CORE-B1: symmetric preservation

Compare the six raw baseline and corrected symmetric services before applying
any exact ledger, conservation or service oracle. Every corrected-minus-
baseline value must equal zero. These are six scored genuine-risk instances.
A half-duplex implementation, or one that sums endpoint directions, fails all
six even if its byte ledger conserves exactly.

### CORE-B2: dispatch preservation

Compare the six raw baseline and corrected dispatch services before exact
checks. Every corrected-minus-baseline value must equal zero. These are six
scored genuine-risk instances. They protect the already correct one-to-many
path independently of the symmetric compatibility fixture.

### CORE-B3: combine response

For widths four and eight, compare raw corrected-minus-baseline combine
service before exact checks. The four values must be exactly `+4,000`,
`+13,000`, `+9,000` and `+27,000` ps in payload-major, width-minor order.
These are four scored genuine-risk instances. A source-only implementation,
an aggregate-group serializer and a half-duplex implementation all produce a
different response.

## Live JCT relation

For each payload and width, the supported live fixture is one uniform MoE
layer with one engine rank, a dispatch star, its combine transpose, fixed
compute and an all-local placement. It executes through `StepRecord`, the
serial graph lowerer, checked graph artifacts, `HtsimStepSink` and
`StepResult`. Three equal controlled steps expose JCT as `step_latency_ps`,
TTFT as the first JCT and TPOT as the mean of the later JCTs.

For widths four and eight, raw corrected JCT must exceed its recorded baseline
by exactly the corresponding CORE-B3 combine delta. These four singleton
signed bands form CORE-B4. The same implementation could correctly compute a
component ledger yet fail to place its service on the executed graph artifact,
so CORE-B4 is decision-relevant independently of CORE-B3.

Width two JCT, compute service, operation inventory and all byte fields must be
preserved exactly. Fixed equal steps entail `TTFT == TPOT == JCT`, so metric
equality establishes reachability but is fatal-unscored rather than a scored
family.

For the same six payload and width cells, an explicit all-remote placement and
omitted-placement compatibility mode execute through the native
`rnic-nn-fluid` path. Their rendered artifacts, flow rows, timestamps,
`StepResult`, TTFT and TPOT must match between baseline and corrected runs
exactly. These are compatibility and identity guards, not scored evidence.

## Physical sanity before exact comparison

For every corrected combine star, let `L = (W - 1)P`. The one-direction link
serialization floor is:

```text
floor_ps = L * 1e12 / 450e9
```

The analytic result is quantized upward to whole nanoseconds, so its strict
ceiling is `floor_ps + 1,000 ps`. The measured service must satisfy
`floor_ps <= service_ps < floor_ps + 1,000 ps` before its exact integer oracle
is read.

| Payload bytes | EP width | Peak endpoint bytes | Floor ps | Strict ceiling ps | Expected service ps |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 2 | 1,024 | 2,275.556 | 3,275.556 | 3,000 |
| 1,024 | 4 | 3,072 | 6,826.667 | 7,826.667 | 7,000 |
| 1,024 | 8 | 7,168 | 15,928.889 | 16,928.889 | 16,000 |
| 2,048 | 2 | 2,048 | 4,551.111 | 5,551.111 | 5,000 |
| 2,048 | 4 | 6,144 | 13,653.333 | 14,653.333 | 14,000 |
| 2,048 | 8 | 14,336 | 31,857.778 | 32,857.778 | 32,000 |

Doubling payload must move each ideal serialization floor and peak byte load
by exactly two. Increasing width from four to eight moves peak load by
`7/3`, not by the symmetric total-byte factor `56/12 = 14/3`. The payload,
width and group-versus-endpoint checks are independent physical-sanity views.

## Dependency-authority refreeze

Only the two all-local `AAAA` cells in `dependency_authority_v1` are expected
to move. The 24,000 ps represented compute term stays fixed:

| Vector bytes | Service old ps | Service new ps | Signed service change ps | JCT, TTFT and TPOT old ps | JCT, TTFT and TPOT new ps |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 4,538,000 | 6,652,000 | +2,114,000 | 4,562,000 | 6,676,000 |
| 2,048 | 9,047,000 | 13,286,000 | +4,239,000 | 9,071,000 | 13,310,000 |

The earlier TRAF-27 expectation record independently predicted the two new
service values before CORE-41. This study treats the rerun as exact
fatal-unscored consumer regression evidence. `AABB`, `ABCD`, all byte counts,
graph inventories, direct GOAL hashes, native flows, cross-check findings and
all-remote timestamps must remain exact. No dependency-authority scored family
or denominator changes.

## Seqgen consequence

The corrected full-duplex rule is the safe first-principles floor that the
historical dispatch-sequence run should have used. Its synthetic home endpoint
has 16,384 bytes in each direction, so `max(egress, ingress)` gives 655,360 ps
at 200 Gbit/s and 327,680 ps at 400 Gbit/s. The historical freeze summed both
directions and used 1,310,720 ps and 655,360 ps. Its fluid observations exceed
the corrected floors.

This finding makes a future refreeze recoverable, but it does not retroactively
unvoid or score the historical run. TRAF-22 retains requalification with a new
expectations-only commit, dependency-aware bounds and the missing scaling cell.

## Fatal guards, entailment and run outcome

After CORE-B1 through CORE-B4 are evaluated from raw baseline and corrected
observations, the runner checks:

- every explicit endpoint ledger exactly reproduces source egress and
  destination ingress from the local segments;
- ledger egress sum, ledger ingress sum and local directed bytes are equal;
- endpoint coverage, ordering, byte positivity and phase partitions are exact;
- renderer bytes and segment order do not change;
- the physical floor and strict quantization ceiling hold before exact service;
- exact service, width-two identity and live metric rows match the registry;
- all-remote artifacts, native flow timestamps, completions and quiescence are
  byte-identical across baseline and corrected runs;
- the dependency-authority refreeze matches every changed and preserved row;
- configuration, provenance and output-root guards pass.

These are fatal-unscored evidence. By-construction guards and fixed-replay
metric equalities never enter the behavioral denominator. Any fatal failure
makes the run void, with findings retained and no behavioral fraction
published. If fatal guards pass but a scored relation misses, the run is
failed and may report its fraction. Only four passed families and 20 passed
instances with every fatal guard passing is accepted.

The entailment order is deliberate. Raw service and JCT differences are
recorded and scored before exact corrected services, ledgers, conservation,
physical bounds, artifact identity or dependency rows are checked. No earlier
fatal oracle pins a scored instance.

## Registered acceptance clauses

1. Production service is derived from an explicit per-endpoint egress and
   ingress ledger and charges the full-duplex maximum, with the half-duplex
   alternative named and rejected against hardware evidence.
2. Payload and EP-width sweeps over symmetric, dispatch-star and combine-star
   fixtures conserve bytes, preserve symmetric and dispatch service exactly,
   and produce the frozen positive combine changes.
3. The corrected combine charge changes a supported live `StepResult` JCT by
   the exact signed amount, while fixed-replay TTFT and TPOT remain reachable.
4. Routed byte output, phase order, width-two behavior and all-remote
   artifacts, timestamps and metrics remain exact.
5. The two dependency-authority `AAAA` rows are refrozen with old and new
   values, every unaffected row remains exact, and the seqgen consequence is
   assigned to TRAF-22 without retroactively changing its void chronology.

## Registered commands and check-only dry runs

The pre-implementation baseline command is:

```bash
.venv/bin/python examples/endpoint_service_v1/run_study.py \
  --mode baseline \
  --out "$SIMLLM_WAVE6_RUN_ROOT/endpoint_service_v1-baseline" \
  --htsim-rnic "$SIMLLM_HTSIM_RNIC" \
  --txt2bin "$SIMLLM_TXT2BIN"
```

The corrected result command is:

```bash
.venv/bin/python examples/endpoint_service_v1/run_study.py \
  --mode corrected \
  --baseline-summary \
    "$SIMLLM_WAVE6_RUN_ROOT/endpoint_service_v1-baseline/summary.json" \
  --out "$SIMLLM_WAVE6_RUN_ROOT/endpoint_service_v1-corrected" \
  --htsim-rnic "$SIMLLM_HTSIM_RNIC" \
  --txt2bin "$SIMLLM_TXT2BIN"
```

Before this expectations-only commit, both complete commands are run with
`--check-only`. Check-only parses every production argument and validates only
the frozen registries and arithmetic. It imports no SimLLM implementation,
reads no input or baseline file, invokes no native executable, creates no
output directory and writes no artifact.
