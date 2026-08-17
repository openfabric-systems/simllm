# RNIC GPU fabric-endpoint v1 results

## Chronology and provenance

Expectations-only commit `51d1ddfffd416b69d2e58cedcac72f770e7b7694` precedes
the second-device mechanism, every new native test, the study command registry
and every result-producing run. At that commit the only change in the working
tree was [expectations.md](expectations.md); no header, source file, test or
artifact of the mechanism existed. This is a local pre-run freeze, not a claim
of public pre-registration.

Implementation landed next, then the study command registry and its pytest
locks, then the single result-producing run. The frozen sweep, row schema,
relation families, entailment answers, artifact digests, staging literals and
the decision not to close BACK-46 were all written before the run and are
unchanged. No attempt was stopped and no outcome-dependent edit was made.

The registered command is:

```bash
.venv/bin/python examples/rnic_gpu_endpoint_v1/run_study.py \
  --out "$SIMLLM_WAVE17_RUN_ROOT/w17b/back46"
```

## What landed

The gap BACK-46 owns is composition, not placement. `GpuMemory` was already a
legal data-region endpoint and the accepted BACK-20 artifact already carries
`data_endpoint` as `gpu_memory`, but the region was owned by the reading NIC.
This change lands the second device:

- `PcieFabric` carries endpoint identities. `PcieFabricConfig::host_endpoint_id`
  names the fabric's host endpoint, a device claims its own identity the way it
  already claims ordering domains, and a per-endpoint ledger sits beside the
  per-service-class ledger. An endpoint pair names the two ends of one link
  traversal, so a `HostStore`, which moves zero link bytes in this model, is
  rejected if it names endpoints at all.
- `VirtualHostMemory` records each claimed device owner's fabric endpoint,
  device kind and the set of peers it grants read access to its data regions.
- `GpuDevice` attaches to a shared fabric with its own endpoint identity and
  ordering domain, owns its regions in a shared registry, grants named peers,
  and issues its own fabric transfers. It has no service model of its own: the
  fabric charges the transfer, and the GPU's copy engines and peer ports stay
  with COMP-31 and COMP-34.
- A WQE data descriptor may name a peer-owned region through
  `WorkRequestDataMemory::peer_device_owner_id`, and the access is legal only
  when the owner granted the reader. Both halves are required.

Every new field is additive and inert at zero, so no ABI, config, result or
record version constant moved and `defaultPcieFabricConfig()` is unchanged.

## Run configurations

Eight cells: two arms crossed with payloads 4096 and 16384 bytes and link
widths 8 and 16 lanes, on one shared fabric whose config is
`defaultPcieFabricConfig()` plus one GPU-memory path, `host_endpoint_id` 4000
and the swept lane count. The peer data region is identical across arms in
identity, MKey, virtual address, length and page geometry; only its endpoint
kind and path differ. Measured rows are [results.csv](results.csv).

| arm | payload | lanes | payload completer | staged ps | WQE CQE visible ps |
|---|---|---|---|---|---|
| host_bounce | 4096 | 8 | 4000 host | 142,188 | 308,249 |
| host_bounce | 4096 | 16 | 4000 host | 71,094 | 154,128 |
| host_bounce | 16384 | 8 | 4000 host | 568,750 | 1,155,279 |
| host_bounce | 16384 | 16 | 4000 host | 284,375 | 577,643 |
| gpu_direct | 4096 | 8 | 4002 gpu | 0 | 166,061 |
| gpu_direct | 4096 | 16 | 4002 gpu | 0 | 83,034 |
| gpu_direct | 16384 | 8 | 4002 gpu | 0 | 586,529 |
| gpu_direct | 16384 | 16 | 4002 gpu | 0 | 293,268 |

## Physical sanity before precision

Generation 5 is 32 GT/s per lane under 128b/130b encoding, so one byte costs
15.869140625 ps at 16 lanes and twice that at 8, i.e. 63.0 and 31.5 GB/s.
Bounds were written down before any measured value was read.

**Floor.** No transfer of `P` bytes can beat `P` over the link rate: 65,000 ps
for 4096 bytes at 16 lanes, 130,000 at 8 lanes, 260,000 for 16384 at 16 lanes,
520,000 at 8 lanes. Every staged transfer sits above its floor, and the excess
is exactly the 24-byte posted-write header per 256-byte TLP, a factor 1.09375
for these maximum-payload-aligned payloads.

**Ceiling.** `ceil(link_bytes * ps_per_byte)` is 71,094, 142,188, 284,375 and
568,750 ps. All four measured staged completions equal their ceiling exactly,
with zero credit wait and zero link-queue wait, so no fabric resource stalled
and nothing was absorbed silently.

**Second angle, end-to-end plausibility.** The 4096-byte GPU-direct cell
completes in 83,034 ps at 16 lanes. Of that, 70,078 ps is the payload read's
completion stream (4096 payload bytes plus sixteen 20-byte completion headers),
leaving 12,956 ps for the WQE fetch, the 256-byte QPC read, the MPT and MTT
reads and the CQE write. Those are the right magnitudes for 64-, 256- and
8-byte transfers on this link, so no term is unexplained. These are
serialization-only figures: every path base latency, service latency and
analytical penalty in this configuration is zero, so the numbers are not a
calibrated ConnectX-7 device latency and must not be read as one.

**Third angle, the value that should scale with it.** The payload-dependent
part of a completion is the read's completion stream, so growing the payload
from 4096 to 16384 bytes must add exactly
`((16384 + 20*64) - (4096 + 20*16))` bytes of h2d serialization. That is
210,234.375 ps at 16 lanes and 420,468.75 ps at 8 lanes. Measured increments
are 210,234 and 420,468 ps, agreeing to within the sub-picosecond reporting
ceiling. Halving the width doubles the completion: twice the 16-lane
GPU-direct 4096 cell is 166,068 ps against a measured 166,061 ps at 8 lanes,
7 ps apart because each of the roughly ten transactions reports its own
ceiling. Nothing double-charges the payload and nothing is width independent
that should not be.

**Ledger conservation by hand.** The NIC is charged 4568 useful bytes as
requester in the 4096-byte cells: 64 WQE plus 8 queue-page-list plus 256 QPC
plus 64 MPT plus 8 MTT plus 4096 payload plus 64 CQE plus 8 queue-page-list.
The host completer sees 8664 in the bounce arm (that 4568 plus the 4096-byte
staging write) and 472 in the direct arm (that 4568 minus the payload the GPU
completes). The 16384-byte cells shift by exactly the payload difference in
every column. Every figure reproduces by hand.

## Scored relation families

Sixteen scored instances, 16 of 16 pass. Families are reported separately and
are never summed with any other evidence class.

| Family | Instances | Result |
|---|---|---|
| Arm ordering: the host bounce completes later | 4 | 4/4 |
| The bounce penalty equals the staged serialization, exact | 4 | 4/4 |
| The staged transfer matches the closed form, exact | 4 | 4/4 |
| The GPU endpoint is charged as the direct read's completer, exact | 4 | 4/4 |

The exact-difference family is the substantive one. In every cell,
`wqe_cqe_visible_ps(host_bounce) - wqe_cqe_visible_ps(gpu_direct)` equals the
staged completion to the picosecond: 142,188, 71,094, 568,750 and 284,375 ps.
The two arms share one fabric object, so this could have failed through a
link-queue wait at the post time, a credit wait, an ordering horizon leaking
out of the GPU's domain, or a parameter difference between the host and GPU
paths. None occurred.

The GPU-completer family is the acceptance measurement: the direct arm charges
the GPU endpoint one completing transaction of exactly `payload_bytes`, and the
bounce arm charges it none, so the difference equals the payload in all four
cells. Inverse lane scaling of the staged transfer was deliberately left
unscored in the freeze because it follows arithmetically from the closed-form
family and is therefore unlosable.

## Fatal guards: void, not scored

No guard was violated, so the scored numbers above mean what they claim. None
of these is reported as a fraction.

- **G1** `examples/rnic_pcie_v1/results.csv` regenerated by
  `run_rnic_pcie_v1.py --check`: 35 of 35 exact-oracle rows, 10 of 10 relation
  families and 18 of 18 predicate instances reproduced, and the tracked bytes
  matched the measured rows. HOLDS.
- **G2** `examples/rnic_hostmem_v1/results.csv` regenerated from the native
  `host_memory_test --study-csv`: byte identical. HOLDS.
- **G3** `examples/rnic_submission_v1/results.csv` regenerated from the native
  `submission_test --study-csv`: byte identical, including the rows whose
  `data_endpoint` is `gpu_memory`. HOLDS.
- **G4** all seven tracked accepted artifacts hashed to their frozen digests
  before and after the run. HOLDS.
- **G5** the whole native CTest suite is green, 7 of 7 under
  `-DSIMLLM_RNIC_WARNINGS_AS_ERRORS=ON`, with every pre-existing test
  unchanged. HOLDS.
- **G6** idle-second-device equivalence: attaching a `GpuDevice` and issuing no
  transfer reproduces every RNIC WQE timestamp and every per-service-class
  accounting field of the run without it. HOLDS.
- **G7** identity-attribution equivalence: giving the NIC an endpoint identity
  over its own host-pinned region reproduces every timestamp and class charge
  exactly, and adds only endpoint-ledger rows: the eight link-crossing
  transactions, with the two host stores correctly left unattributed. Naming a
  fabric host endpoint with no attributed device changes nothing at all.
  HOLDS.
- **G8** cross-device rejection is transactional. Nine rejections were
  exercised: an ungranted peer region, a WQE naming its own device as peer, a
  named peer disagreeing with the region owner, a GPU transfer into an
  ungranted peer region, a GPU transfer naming its own owner as peer, a device
  charging another device's requester endpoint, an attributed host store, a
  half-named endpoint pair, and a duplicate host-memory owner claim. After all
  nine the fabric generation, per-class accounting, registry generation, live
  allocation count, SQ occupancy and work-queue counters were unchanged and the
  endpoint ledger was still empty. Endpoint and ordering-domain collisions and
  a GPU attempting to register a queue object are rejected the same way.
  HOLDS.
- **G9** `validateInvariants()` passes on the fabric, registry, RNIC device and
  GPU device in every cell, and the endpoint ledger conserves: requester and
  completer charge counts are equal, and equal to the attributed transaction
  count, in all eight rows. HOLDS.
- **G10** the four Python gates pass. HOLDS.
- **G11** no version constant moved and `defaultPcieFabricConfig()` keeps its
  fields. HOLDS.

G1 through G3 are the mutation-sensitive off-path locks: they rebuild the
native library and re-derive the accepted bytes from source, so a C++ change
that perturbs the off path fails them. G4 is the tracked-artifact lock, carried
in `tests/test_rnic_gpu_endpoint.py` with a negative control that mirrors the
artifact tree, flips one byte of each artifact in turn and requires the same
guard to reject it. Neither lock is claimed to do the other's job.

## Structural evidence, fatal and unscored

Reported separately: the eight run configuration rows; the payload-read
completer labels (4002 GPU in the direct arm, 4000 host in the bounce arm, with
the matching device kinds); the by-construction zeros (no GPU completer charge
and no staged transfer in the bounce and direct arms respectively); the
requester-side mirror of the completer family, where the bounce arm charges the
GPU endpoint one requesting transaction of exactly the payload; the eight
link-crossing NIC requester charges per cell; and the six native test
executables plus the probe rejection check that make up the CTest suite.

## Scope limits, stated plainly

- **This is not a TTFT or TPOT claim.** The metric here is a native WQE
  completion timestamp. No projected end-to-end metric was produced, and none
  should be inferred.
- **The bounce penalty measured here is unoverlapped.** The harness posts the
  WQE after the staging write completes, because the read cannot precede the
  data it depends on. A real pipeline can stage chunk `n+1` while the NIC reads
  chunk `n`, which hides part or all of that cost. The exact relation is a
  statement about how the shared fabric charges a single chunk, not a claim
  about steady-state deployment throughput.
- **The GPU-direct arm claims no GPUDirect penalty.** Every analytical path
  profile in this configuration is disabled, including `gpu_direct`. Whether
  and when that penalty occurs is BACK-16 precision scope, and this study does
  not touch it.
- **The endpoint ledger covers link traversals only.** Host stores, including
  the UAR doorbell and the doorbell record, carry no endpoint pair by
  construction. A GPU-owned UAR mapping fails closed: a device with an endpoint
  identity rejects that submission shape rather than charging the mapping to
  the host.
- **The session record does not yet describe endpoint identity.** Two devices
  differing only in fabric endpoint identity currently produce the same
  `hardware_config_sha256`. This is registered as BACK-50 rather than fixed
  here, because the effective-hardware schema carries its own frozen mutation
  corpora.

## Why BACK-46 is not closed

Acceptance clauses 1 through 3 are met and evidenced above. Clause 4 asks for
an end-to-end metric moved in a registered direction and names the
`rnic_live_v1` Tier-B-class machinery as the vehicle. That vehicle is out of
reach in this wave for two independent reasons, both established before the
freeze:

1. The wire leg runs the htsim binaries and reads a topology file out of the
   `third_party/htsim` submodule, which is not checked out in this working tree,
   and the wave forbids fetching.
2. The structural-only Tier-B path needs no htsim, but it ingests observations
   through `ComposedRnicCell`, whose validator requires
   `eligible_at_ps == doorbell_service_ps` for every WQE, where `eligible_at_ps`
   is the producer's projection of `admitted_at_ps`. That equality is a property
   of the scalar-service fixture. A DMA-mode device cannot satisfy it: DMA mode
   rejects a nonzero scalar doorbell service, and admission is derived from PCIe
   transactions instead. This study's own rows show the gap concretely, with
   `wqe_admitted_ps` at 80,811 ps in the 4096-byte 16-lane direct cell against a
   required zero. Ingesting a DMA-mode cell needs a second composed-observation
   schema family, which would put the accepted Tier A and Tier B artifacts at
   risk in the same change.

Per the instruction registered for exactly this case, the mechanism ships with
native-level completion-time relations, BACK-46 stays open, and the live-chain
residual is registered as BACK-49. Nothing here is stretched into a metric
claim.

## Native and study gates

- `cmake` configure and build of `simllm/backends/rnic` in Release with
  `-DSIMLLM_RNIC_WARNINGS_AS_ERRORS=ON`: clean.
- `ctest`: 7 of 7 passed, including the new `simllm_rnic_gpu_device_test`.
- `examples/rnic_wq_v1/run_rnic_wq_v1.py --check`: passed, tracked results
  match 11 measured rows and 11 of 11 checks.
- `examples/rnic_pcie_v1/run_rnic_pcie_v1.py --check`: passed, tracked results
  match 35 measured rows.
- `.venv/bin/ruff check .`, `.venv/bin/python -m pytest -q`,
  `scripts/check_docs_format.py`, `scripts/task_progress.py --check`: all pass.

## Genuine-risk fraction and boundary

The scored fraction is 16 of 16 across four families, all of them losable given
the registered guards: the ordering and difference families fail if the second
device's transfer is not really charged on the shared fabric or if the two arms
couple through it, the closed-form family fails if the new caller adds or drops
serialization, and the completer family fails if the completer identity resolves
to the wrong end. Byte identity, invariant validation, rejection atomicity,
ledger conservation and the by-construction zeros are fatal and unscored, and
their counts are never added to the scored total.

The boundary this study does not cross: occurrence, timing and calibration of
the enabled leg (BACK-16), the GPU's own port objects and service model
(COMP-31, COMP-34), the packetized intra-node leg (TRAF-45) and the live metric
projection (BACK-49).
