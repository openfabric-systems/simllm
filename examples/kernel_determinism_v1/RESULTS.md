# Kernel determinism v1 results

**Verdict: nonvoid and accepted.** All 23 fatal guards held, all 3 controls
discriminated, and all 8 pre-registered scored instances passed with a zero
residual. The 5 derived rows and the 8 raw observations are reported and are
not counted.

The run establishes that the maintainer's kernel-time determinism ruling holds
as an observable property of the model on these fixtures, and that the
enforcement is discriminating rather than vacuous. It establishes nothing about
silicon, nothing about collectives, and nothing about the size of any real
latency tail.

## Chronology, stated no more strongly than the evidence supports

| Step | Commit | What it contained |
|---|---|---|
| Freeze | `1226d0d9577a89cc1e1bdc9e92c0b3742a3d324e` | `expectations.md` only, plus its end-of-line rule. No implementation, no harness, no generated row, no measured value. |
| Implementation | `62a9082c2b55777746e4c7d845cb0ab4a22b20ab` | The public contract in `docs/modules/compute.md`, the matching statement in the calibration doctrine, the UALink protocol, `tests/test_kernel_determinism.py`, and this study's harness. |
| Run | this commit | `results.json`, this report, and the byte lock. |

The freeze precedes both the implementation and the first run, so this is a
genuine pre-registration. Every predicted integer in `expectations.md` was
derived by hand from the closed forms stated there and cross-checked with a
calculator that does not import `simllm`; none was read off a run.

One prediction in the freeze concerns a source-level fact rather than a
measurement: that the two adapter geometry readers would spell the optional
dtype widths differently. That was predicted from reading the two readers, not
from running them, and it is recorded here as a prediction that held rather than
as a discovery.

## Fixtures

Fixture R is a dense two-layer geometry (hidden 128, intermediate 256, 4 heads,
2 KV heads, head size 32, vocab 1,024, 2-byte activations) priced by
`RooflineProvider(efficiency=0.5)` against a synthetic envelope whose nameplates
are exact powers of two, `peak_flops = 2**48` (281.5 TFLOP/s) and
`mem_bandwidth = 2**41` (2.199 TB/s). The host profile is
`HostInitiationModel.ideal()`, which contributes exactly zero. Fixture M is the
accepted `gpu_task_mix` synthetic 1 GHz architecture, loaded as a module so the
architecture object is that study's own, at a transaction size (192 bytes) that
appears in none of its published cells. Fixture A is one dense config per
adapter, shaped to fixture R.

No fixture makes a silicon claim. The power-of-two envelope exists so the
provider's IEEE-754 double arithmetic is exact and a frozen integer cannot be
missed by a one-unit truncation artifact.

## Physical bounds, checked before the digits

Every bound below was written in the freeze before the run.

| Bound | Stated floor or ceiling | Measured | Where it sits |
|---|---|---|---|
| Decode read floor, cell A1 | 917,504 bytes over the derated 2^40 bytes per second is 834,465 ps; the undercut full-nameplate figure is 417,232 ps | 834,465 ps | exactly at the derated floor, well above the nameplate figure |
| Decode pin margin | compute term 13,023.96 ps against memory term 834,465.03 ps, a factor of 64 | classified `memory` | not a knife edge, so the classification is a result and not a rounding accident |
| Prefill arithmetic floor, cell A3 | 436,469,760 FLOP over the derated 2^47 FLOP per second is 3,101,304 ps, against a memory term of 1,013,278 ps | 3,101,304 ps, classified `compute` | compute bound by 3.06 times |
| Prefill chunk scaling | the projection term doubles and the attention term quadruples, so the ratio is 2.6144, not 2.0 and not 4.0 | 2.6144 | exact |
| HBM cursor floor | 32 loads of 192 bytes is 6,144 bytes; at 64 bytes per cycle no ordering occupies the cursor for fewer than 96 cycles, and halving the bandwidth doubles that to 192 | 96 and 192 cycles of occupancy, plus the fixed 100-cycle return latency | exactly at the floor on both bandwidths |

The direction checks that matter against real serving stacks all hold: decode is
memory bound, prefill is compute bound, and prefill grows superlinearly in chunk
length. The absolute picoseconds of a two-layer toy carry no claim.

## Fatal guards: 23 held, none violated

Reported as a state, never as a fraction.

| Guard | What it asserts | Instances |
|---|---|---:|
| G1 | 64 repeated estimates of one cell through one provider instance serialize to exactly one byte string | 7 |
| G2 | a freshly constructed provider or `SmSchedulerModel` produces the same byte string | 7 |
| G3 | the ideal host profile leaves the duration unchanged and exposes zero picoseconds | 4 |
| G4 | no module under `simllm/compute` imports a random-number source, a wall clock or an environment reader | 1 |
| G5 | no pricing entry point accepts a rank, worker, adapter or device-index parameter, across 12 entry points | 1 |
| G6 | xGMI and UALink are rejected at configuration time naming COMP-35, and NVLink is unaffected | 3 |

G4's audit reads the module abstract syntax trees rather than grepping text, so
a mention inside a comment or a docstring cannot satisfy or trip it. The whole
`simllm/compute` package imports only deterministic standard-library modules:
`abc`, `collections`, `csv`, `dataclasses`, `decimal`, `enum`, `fractions`,
`hashlib`, `json`, `math`, `pathlib`, `re`, `statistics` and `typing`. There was
no RNG and no clock to remove. The guard exists so that adding one is loud, and
its own control in `tests/test_kernel_determinism.py` hands the audit a file
that really does import `random`, `numpy.random` and `time` and requires all
three to be reported.

G1 and G2 are guards and not scored rows, and the freeze says why: the
determinism of a deterministic function is entailed by its construction. These
are pure closed forms over frozen dataclasses. A variance would mean the
construction assumption is false, which voids the run, not that a prediction was
wrong. Counting them as passes would inflate a behavioral denominator with rows
that cannot lose.

Guard G7 of the freeze, byte identity of the accepted artifacts, is satisfied by
the existing locks rather than duplicated here: `tests/test_gpu_device_ports.py`
regenerates the `gpu_task_mix` artifacts, `gpu_service_model/results.csv` and the
`mixed_makespan_v1` record through the composed device and compares bytes, and
it carries its own mutation control. Those tests pass with the UALink protocol
landed. Adding a name to an enum moved nothing, which is the whole claim.

## Controls: 3 of 3 discriminated

| Control | Mutation | Result |
|---|---|---|
| C1 | price cell A1 through `RooflineProvider(efficiency=0.25)` | byte stream differs, so G1's equality is a measurement |
| C2 | read the SGLang geometry at `tp_size=2` | the price changes, so the adapter agreement is a measurement |
| C3 | halve the HBM cursor bandwidth | the duration changes, so the pin is a measurement |

The artifact byte lock carries two more controls in
`tests/test_kernel_determinism_study.py`: perturbing one frozen prediction
breaks the artifact and fails exactly row A1, and changing the mechanistic
transaction size breaks it and fails exactly rows B1, B2 and C1.

## Scored: 8 of 8, zero residual

### Family A, phase and token keying (4 of 4)

| Row | Cell | Predicted | Measured | Residual |
|---|---|---:|---:|---:|
| A1 | decode, 2 requests, context 64 | 834,465 ps, `memory` | 834,465 ps, `memory` | 0 |
| A2 | decode, 4 requests, context 64 | 894,069 ps, `memory` | 894,069 ps, `memory` | 0 |
| A3 | prefill, 512-token chunk | 3,101,304 ps, `compute` | 3,101,304 ps, `compute` | 0 |
| A4 | prefill, 1024-token chunk | 8,108,094 ps, `compute` | 8,108,094 ps, `compute` | 0 |

The phase changes both the constant and the selected roof. The token count moves
the constant within a phase. Neither row is entailed by another: a model that
ignored the KV read would return A1's value for A2, a model that ignored the
phase would keep the memory roof for A3, and a model linear in chunk length
would return 6,202,608 ps for A4.

### Families B and C, the memory-bound pin (3 of 3)

| Row | Cell | Predicted cycles | Measured cycles | Predicted ps | Measured ps |
|---|---|---:|---:|---:|---:|
| B1 | 192 bytes, 64 bytes per cycle, 1 SM | 196 | 196 | 196,000 | 196,000 |
| B2 | 192 bytes, 32 bytes per cycle, 1 SM | 292 | 292 | 292,000 | 292,000 |
| C1 | 192 bytes, 64 bytes per cycle, 2 SMs | 196 | 196 | 196,000 | 196,000 |

C1 is the statement that a memory-bound duration does not notice the SM count.
It is a replication of the accepted `gpu_task_mix` B2 family at a new cell, and
it is scored because a defect that let SM count leak into a memory-bound
duration would pass B1 and fail here.

The roofline half of the same pin is inside family A rather than a separate row:
`tests/test_kernel_determinism.py` asserts that A1 and A2 equal
`int(bytes_moved / (mem_bandwidth * efficiency) * 1e12)` exactly, with the
compute term contributing nothing.

### Family D, runner and adapter independence (1 of 1)

The vLLM and SGLang geometry readers, given equivalent dense configs, produce
the same integer geometry, the same resolved `weight_element_bytes` and
`kv_element_bytes`, the same derived `weight_bytes` and `lm_head_bytes`, and
neither defaults a field. Both then price cell A1 to 834,465 ps.

The picosecond equality is reported as derived rather than scored, and the
freeze said so before the run: once the geometry agrees, both adapters call the
same `estimate_step_latency_ps` on equal inputs, so equal output is entailed by
the function being a function. The genuine risk was in the two readers agreeing.

## Derived rows, reported and not counted

| Row | Relation | Value |
|---|---|---|
| DER-1 | A2 minus A1 is the 65,536-byte KV read delta at the derated roof | 59,604 ps, matched |
| DER-2 | A4 over A3 | 2.6144, matched |
| DER-3 | the cursor term scales inversely with bandwidth: B2 minus latency equals twice B1 minus latency | 192 cycles, matched |
| DER-4 | the two adapters price cell A1 identically | 834,465 ps both |
| DER-5 | the frozen A1 constant reaches an adapter end to end | 834,465 ps |

Each is arithmetic over rows already scored above. Adding them to the
denominator would count the same evidence twice.

## Findings

**F1. The contract is about the function, not about the shape, and the
distinction is load bearing.** The freeze recorded this before the run because
the repository already contains a case that looks like a violation and is not:
vLLM gives the low expert-parallel ranks the remainder of an uneven expert
split, so 30 experts over 8 ranks leaves ranks 0 to 5 with four resident experts
and ranks 6 and 7 with three. Those ranks stream different weight bytes and
their decode steps legitimately cost different amounts. A test that asserted
"every rank prices a step identically" would fail on a correct model. The
enforced invariant is instead that no provider accepts or reads a caller
identity, which G5 audits across 11 pricing entry points.

**F2. The two adapter geometry readers agree on every value the cost model reads
and disagree on how two of them are stored.** The vLLM reader resolves
`weight_dtype_bytes` and `kv_dtype_bytes` from the quantization and cache
configs and stores `2.0` and `2.0`; the SGLang reader stores `None` and lets
`ModelDims` fall back to the activation width. Both resolve to 2.0 through
`weight_element_bytes` and `kv_element_bytes`, so no picosecond moves and the
two adapters price the step identically. The dataclasses are nonetheless
unequal. This was predicted in the freeze from reading the two sources, is
pinned by `tests/test_kernel_determinism.py`, and is registered as COMP-42
because a consumer that ever compares or hashes `ModelDims` itself would see two
adapters disagree about one identical rank, which is the failure mode BACK-50
already records for the effective-hardware snapshot.

**F3. COMP-9's original scope is refuted rather than unfinished.** It promised a
measured or fitted per-kernel service-time distribution so that CORE-5 could
claim kernel-level p99 and p99.9 accuracy. Under the ruling there is no
kernel-level tail to fit. Worse, fitting one would double count: the same spread
would appear once in the kernel constant and again in the queueing that constant
feeds, and a reported p99 TTFT could then be reproduced by an arbitrary mix of
kernel noise and queue noise, which makes the attribution unfalsifiable at the
metric. COMP-9 keeps its ID and now owns locating and validating tail fidelity
in the network, batching and queueing chain. COMP-23, which asked for a
distribution provider carrying a seed, gained the matching scope constraint: the
fitted spread is calibration evidence and an uncertainty input, never a sampling
source, and no seed enters a service path.

**F4. UALink sharpens the taxonomy's own second rule rather than adding a new
one.** The UALink 200G 1.0 specification states a 200 GT/s per-lane data rate
carried at a 212.5 GT/s signalling rate, to cover forward error correction and
layer-1 encoding. Reading the headline as a payload ceiling is exactly the error
the GH200 freeze made with NVLink4's 26.5625 GB/s signalling rate against its
25.0 GB/s payload rate. The taxonomy row therefore carries the nameplate with
that caveat attached, and no code path can consume it: a UALink port is rejected
during configuration, naming COMP-35.

**F5. Naming a protocol changed nothing, which is the point.** Adding `UALINK`
to `GpuPortProtocol` and to the peer-link role table left the xGMI diagnostic
byte-identical (pinned by a test that asserts the full string), left NVLink port
construction unaffected, and left every accepted `gpu_task_mix`,
`gpu_service_model` and `mixed_makespan_v1` artifact reproducing byte for byte.

**F6. The artifact byte lock found a determinism defect in this study's own
harness, before the artifact was committed.** Guard G5 swept
`ComputeProvider.__subclasses__()` to discover pricing entry points and recorded
how many it had audited. That set grows as modules are imported, and more than a
dozen study harnesses under `examples/` define their own throwaway provider
subclass, so the artifact reproduced when the study ran alone and differed when
it ran inside the full test suite. The audited coverage of a determinism guard
was itself import-order dependent, which is exactly the class of defect this
study exists to catch. The fix imports the modules that define shipped providers
before sweeping and keeps only `simllm.` classes, so a study fixture cannot
change what the audit covers, and
`test_the_audited_entry_points_do_not_depend_on_import_order` pins it. The
correction changed no scored row: G5 measured an empty offender set before and
after, and the sweep now reports 12 shipped entry points, listed in observation
OBS-4. It is recorded here rather than quietly folded in, because the harness
was edited after a run had been observed.

## What this run does not establish

- Nothing about silicon. Both fixtures are synthetic and declared so.
- Nothing about collectives. They are the declared exception to the ruling and
  are owned by the traffic side; no collective is priced here.
- Nothing about the size of a real latency tail. The ruling relocates tails to
  the network, batching and queueing chain; validating them there is COMP-9,
  which is open.
- Nothing about absolute accuracy of any provider. The scored rows check that
  the implementation computes the closed form it claims to compute, on fixtures
  chosen so the closed form is checkable by hand.

## Reproducing

```bash
python examples/kernel_determinism_v1/run_study.py --check
```

Exit code 0 means the committed `results.json` regenerates byte for byte and no
guard, control or scored row failed.
