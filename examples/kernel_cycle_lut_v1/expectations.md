# Kernel-cycle lookup retained-fixture freeze

This freeze precedes the lookup implementation and every scored fixture run.
The tracked worktree was clean at commit
`7e5b1ce26104d5252f94ee94350947d874f844c9`; the required local sizing note is
ignored by Git. No GPU runs in this study.

## Physical story and bounds

The chip executes a sequence of kernels. For each kernel, arithmetic consumes
streaming-multiprocessor cycles while data movement consumes memory service.
Those two parts may overlap, so their service time is the larger part. A fixed
front-end or latency remainder follows it. Kernel service is therefore
`max(compute, memory) + fixed`, and step service is the sum over ordered
kernels.

Before reading the retained values, each observed kernel must take more than
zero time. No individual kernel may take longer than the enclosing Granite
batch-1 decode wall step, whose retained upper bound is 2,323,678,000 ps. Both
the observed streaming-multiprocessor clock and memory clock must be positive.
These are defect bounds, not evidence that the decomposition is correct.

## Frozen lookup contract

The schema is `simllm-kernel-cycle-lut-v1`. It uses the existing calibration
canonical byte grammar and keeps the SHA-256 identity outside the record. The
logical key order is framework identity, model identity, pool, launch mode,
parallelism, and then the pool shape. Decode uses batch size plus one KV length
for every request. Prefill uses computed new tokens plus existing context.

Dense families forbid a routing member. Routed mixture-of-experts families
require a routing member with per-expert loads and the digest of the captured
routing evidence. Token values do not otherwise enter the key.

Each ordered kernel carries measured time and cycles, compute cycles, memory
bytes and achieved bandwidth where known, fixed overhead in picoseconds, both
observed clocks, a distribution verdict, and nullable PTX and SASS code-object
digests with a closed implementation class. Unknown memory measurements stay
explicitly unknown. They are not converted to zero-byte evidence.

The retained graph fixture has fewer than the required 256 replays, so its
distribution verdict must be `insufficient-replays`. That verdict is evidence
about the fixture, not a stochastic service source.

## Campaign protocol frozen for the dry workflow

The rendered campaign covers decode and prefill, CUDA graph and eager launch,
and tensor, pipeline, data and expert parallel configuration. Decode uses 16 KV
points at 1, 2, 4, 6, 8, 12, 16, 24, 32, 40, 50, 60, 70, 80, 90 and 100 percent
of supported context. It crosses fresh contiguous and deliberately fragmented
page placement.

Graph cells request at least 256 replays and eager cells at least 64. Every
window records elapsed time, elapsed streaming-multiprocessor cycles, the
streaming-multiprocessor clock and the memory clock. The component pass requests
DRAM read and write bytes, achieved DRAM throughput and achieved compute
throughput with `--clock-control none`. Program-counter sampling is attempted
where permission allows it, and the record states granted, denied or
unavailable.

Code objects are classified as Triton just-in-time, wheel-precompiled CUDA,
closed library or unknown. The workflow harvests per-kernel PTX and SASS
digests twice from clean state and accepts the harvest only when the canonical
manifests are byte-identical.

## Evidence classes and expectations

The scored behavioral denominator is one relation family with five
parameterized kernel instances. Conservation, key rejection, byte identity,
round-trip compilation and physical bounds are fatal guards. They never add to
the score.

**R1-cross-instrument-elapsed-agreement.** For each of the five mapped kernel
families seen by both instruments, the Nsight Compute elapsed time divided by
the Nsight Systems median must lie in [0.5, 2.0]. The instruments perturb and
select launches differently, so equality is not expected. A factor larger than
two rejects the candidate fixture join.

**G1.** Every fixture excerpt carries its retained-source SHA-256 and the
analyzer rejects a digest mismatch.

**G2.** Every emitted lookup record is strict canonical calibration JSON and
its external identity is the SHA-256 of those exact bytes.

**G3.** Every kernel reconstructs measured elapsed time within one picosecond
as `max(compute service, memory service) + fixed overhead`, and the entry total
is their sum.

**G4.** A dense decode key missing per-request KV lengths rejects. A routed
family missing routing evidence also rejects.

**G5.** Two analyses of identical fixture bytes emit byte-identical records
with one identical content identity.

**G6.** The scalar projection round-trips through
`simllm-profile-table-v1`. The mechanistic projection round-trips through the
existing device-model service-entry readers. Neither changes the lookup
duration.

**G7.** Every retained per-kernel elapsed time is positive and below the
enclosing wall-step ceiling, and both clocks are positive.

## Closure boundary

This fixture study closes nothing and keeps COMP-64 open. It makes no GPU or
numerical calibration claim. Static compile-graph inference and
program-counter cycle attribution are expected residuals for COMP-65 and
COMP-66, but those IDs are registered only with the implementation change that
establishes their exact remaining scope.
