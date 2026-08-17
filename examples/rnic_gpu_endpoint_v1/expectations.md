# RNIC GPU fabric-endpoint v1 expectations

## Freeze status and scope

This is the expectations-only record for BACK-46. It precedes the second-device
mechanism, every new native test, the study command registry and every
result-producing run. Nothing in this file was written after observing a
measured number, and no part of the mechanism it describes exists at this
commit.

The gap this slice closes is composition, not placement. The GPUDirect
placement is already expressible and already exercised: `GpuMemory` is a legal
endpoint for a `DataRegion`, the accepted BACK-20 artifact carries
`data_endpoint` as `gpu_memory`, and the payload read really is issued as a
`PayloadRead` non-posted read. What is missing is the second device. This
study lands a separately modeled GPU that attaches to the same PCIe fabric as
the RNIC, owns its own regions, grants named peers access to them, and has its
transactions charged under its own endpoint identity.

Timing, occurrence and calibration of the enabled leg are BACK-16 precision
scope and are deliberately out of scope here. The GPU's internal service model
(copy engines, NVLink egress, typed ports) is COMP-31 and COMP-34 scope; this
slice gives the GPU no service model of its own, only a fabric identity, region
ownership and the ability to issue a fabric transfer.

## Decision recorded before the run: how far the metric claim goes

The registered acceptance clause 4 asks for an end-to-end metric moved in a
registered direction, and names the `rnic_live_v1` Tier-B-class machinery as
the strongest honest vehicle. That vehicle is not reachable in this wave, for
two independent reasons established before this freeze:

1. The wire leg of the Tier-B chain runs the htsim binaries and reads a
   topology file out of the `third_party/htsim` submodule. That submodule is
   not checked out in this worktree and the wave forbids fetching.
2. The structural-only Tier-B path needs no htsim, but it ingests observations
   through `ComposedRnicCell`, whose validator requires
   `eligible_at_ps == doorbell_service_ps` for every WQE. That invariant is a
   property of the scalar-service fixture. A DMA-mode device cannot satisfy it,
   because DMA mode rejects a nonzero scalar doorbell service and derives WQE
   eligibility from PCIe transactions instead. Ingesting a DMA-mode cell would
   require a second composed-observation schema family, which would put the
   accepted Tier A and Tier B artifacts at risk in the same change.

Therefore, per the registered instruction for that case, this study delivers
the fabric mechanism with native-level completion-time relations, **does not
close BACK-46**, and registers the live-chain residual. No projected TTFT or
TPOT claim is made anywhere in this study, and the results report must not
imply one.

## The mechanism under test

- `PcieFabric` gains endpoint identities. `PcieFabricConfig::host_endpoint_id`
  names the fabric's host endpoint; a device claims its own endpoint identity
  the way it already claims ordering domains. `PcieEndpointAttribution` names
  the requester and completer of one transaction, and the fabric keeps a
  per-endpoint ledger separate from its per-service-class ledger.
- An endpoint pair names the two ends of one link traversal. `HostStore`
  operations move zero link bytes in this model, so they carry no endpoint
  pair, and the fabric rejects an attributed host store.
- `VirtualHostMemory` records, per claimed device owner, that owner's fabric
  endpoint identity, its device kind and the set of peer device owners it
  grants read access to its data regions.
- `GpuDevice` is the second device kind. It attaches to a shared fabric with
  its own endpoint identity and its own ordering domain, claims its own device
  owner identity in a shared registry, registers its own regions, grants named
  peers, and can issue one fabric transfer at a time under its own identity.
- A WQE data descriptor may name a peer-owned region through
  `WorkRequestDataMemory::peer_device_owner_id`. Zero keeps exactly today's
  rule: the region must be owned by the posting device.
- Every new field is additive and inert at zero. A zero value reproduces the
  previous semantics exactly, so no ABI, config, result or record version
  constant moves, and `defaultPcieFabricConfig()` is unchanged.

## Frozen sweep

Two arms crossed with two payload sizes and two link widths, eight run
configurations:

- `arm` in `{host_bounce, gpu_direct}`
- `payload_bytes` in `{4096, 16384}`
- `lane_count` in `{8, 16}`

Both arms share one PCIe fabric object whose config is
`defaultPcieFabricConfig()` plus one added path 3 with endpoint `GpuMemory`,
plus `host_endpoint_id = 4000` and the swept `lane_count`. Everything else is
the default: generation 5, MPS 256, MRRS 512, RCB 64, 24-byte posted-write and
read-request overheads, 20-byte completion overhead, all analytical penalty
profiles disabled, all service latencies zero, `analytical_seed` zero.

Fixture identities, frozen:

| Object | Identity |
|---|---|
| Fabric host endpoint | 4000 |
| RNIC fabric endpoint | 4001 |
| GPU fabric endpoint | 4002 |
| RNIC device owner | 920 |
| GPU device owner | 930 |
| RNIC ordering domains | 21 submission, 20 completion |
| GPU ordering domain | 31 |
| RNIC QP number | 19 |
| GPU peer read grant | `{920}` |
| Peer data region | allocation 40, MKey 511 |

The peer data region is identical across the two arms in allocation identity,
MKey, virtual address, length and page geometry. Only its endpoint kind and
path differ: host-pinned on path 2 in `host_bounce`, GPU memory on path 3 in
`gpu_direct`. Both paths carry identical parameters, so the two arms differ
only in which endpoint completes the payload read and whether a staging
transfer is needed at all.

Composition per cell:

- `host_bounce`: the GPU stages `payload_bytes` into its own host-pinned region
  as a `PayloadWrite` posted write at t = 0. The RNIC posts and doorbells its
  WQE at t = `staging_completed_ps`, because an intermediate result cannot
  complete before the data it depends on arrives. The payload read then
  completes out of host memory.
- `gpu_direct`: no fabric staging transfer exists, because the GPU writes its
  own local memory. The RNIC posts and doorbells at t = 0 and its payload read
  names the GPU-owned GPU-memory region directly.

## Physical sanity: floor and ceiling before any measured value

PCIe generation 5 is 32 GT/s per lane with 128b/130b encoding, so one byte
costs `8 * 130 * 10^6 / (32000 * lanes * 128)` picoseconds, i.e. 15.869140625
ps at 16 lanes and 31.73828125 ps at 8 lanes. A posted write of `P` bytes at
MPS 256 with `P` a multiple of 256 and offset zero carries `P / 256` TLPs, each
adding 24 overhead bytes, so the modeled link bytes are `P * 1.09375`.

| payload | lanes | link bytes | floor: payload over link rate | expected staging completion |
|---|---|---|---|---|
| 4096 | 16 | 4480 | 65,000 ps | 71,094 ps |
| 4096 | 8 | 4480 | 130,000 ps | 142,188 ps |
| 16384 | 16 | 17,920 | 260,000 ps | 284,375 ps |
| 16384 | 8 | 17,920 | 520,000 ps | 568,750 ps |

The expected completions are `ceil(link_bytes * ps_per_byte)`. No staged
transfer may complete before its floor. A measured value above the expected
completion means a credit, header-credit or link-queue stall was added and must
be named, not absorbed: the study row therefore also carries the transfer's
credit wait and modeled link bytes so a deviation is diagnosable rather than
guessed.

The WQE cannot become visible before its own payload has been serialized, so
`wqe_cqe_visible_ps` must exceed `payload_bytes * ps_per_byte` in the
`gpu_direct` arm and must exceed `staging_completed_ps + payload_bytes *
ps_per_byte` in the `host_bounce` arm.

## Scored relation families and their entailment answers

Sixteen scored instances in four families. The entailment question asked of
each is: given the fatal guards already registered below, can this relation
fail?

**F1, arm ordering, direction, 4 instances (one per payload and lane cell).**
`wqe_cqe_visible_ps(host_bounce) > wqe_cqe_visible_ps(gpu_direct)`.
Can fail. If the GPU's staging transfer is not actually charged on the shared
fabric, or if the second device only relabels transactions that the RNIC would
have issued anyway, both arms complete at the same time. Nothing in the guard
set forces an inequality.

**F2, the host bounce costs exactly its staging serialization, exact, 4
instances.**
`wqe_cqe_visible_ps(host_bounce) - wqe_cqe_visible_ps(gpu_direct) ==
staging_completed_ps(host_bounce)`.
Can fail. The two arms share one fabric object, so any residual coupling makes
the difference exceed the staging time: a link-queue wait if the posted
reservation calendar is not free at the post time, a credit wait if credits are
not returned by then, an ordering horizon if the GPU's transfer lands in the
RNIC's ordering domain, or a path-parameter difference between path 2 and path
3. Not entailed.

**F3, the staged transfer matches the closed-form serialization, exact, 4
instances.**
`staging_completed_ps` equals the literal expected completion in the table
above, computed independently in the study runner from exact rational PCIe rate
arithmetic rather than read back from the model.
Can fail. The second device is a new caller of the fabric; it can add a setup
cost, drop or duplicate a fragment, or take a different segmentation path.
Not entailed by the byte-identity guards, which cover other configurations.

**F4, the GPU endpoint is charged as the completer of the direct payload read,
exact, 4 instances.**
`gpu_completer_useful_bytes(gpu_direct) -
gpu_completer_useful_bytes(host_bounce) == payload_bytes`.
Can fail. The completer identity is resolved from the region's owner and its
endpoint kind; wiring it to the host endpoint for a GPU-memory region, or
swapping the requester and completer roles, or charging the read to the RNIC
because the region is reached through the RNIC's registry, all produce a wrong
difference. Not entailed.

An earlier candidate family, inverse lane scaling of the staged transfer, is
deliberately **not** scored: it is entailed arithmetically by F3 holding on all
four staging cells, so it is unlosable and belongs with the structural
evidence.

Scored denominator: 16. Families are reported separately and never summed with
any other evidence class.

## Fatal guards: void, not scored

A single violation voids the run for the purpose of closing anything. None of
these is reported as a fraction.

- **G1** `examples/rnic_pcie_v1/results.csv` regenerated by
  `run_rnic_pcie_v1.py --check` is byte-identical to the tracked file.
- **G2** `examples/rnic_hostmem_v1/results.csv` regenerated from the native
  `host_memory_test --study-csv` output is byte-identical to the tracked file.
- **G3** `examples/rnic_submission_v1/results.csv` regenerated from the native
  `submission_test --study-csv` output is byte-identical to the tracked file,
  including the rows whose `data_endpoint` is `gpu_memory`.
- **G4** the tracked accepted artifacts hash to their frozen digests before and
  after the run:

| Artifact | Frozen SHA-256 |
|---|---|
| `examples/rnic_wq_v1/results.csv` | `598f0e10ca4e5a83a9dfb8ed8289e25cdc4c80fc24f92f2f70db967724be5682` |
| `examples/rnic_pcie_v1/results.csv` | `464b92fd5327287db6b5e71a5449add5b893285bc3c4bcdf6a4950355339a5e2` |
| `examples/rnic_device_v1/results.csv` | `7a0b8423d0a99de9538047f307bb7fd2f20c8d19bd408ef90fe02199da868934` |
| `examples/rnic_device_v1/native_tests.csv` | `969963477314bfb723770556a02e4f038c7220820d522ae60dfa8c80744a202d` |
| `examples/rnic_session_records_v1/results.json` | `d83575d1c873d3375bc24819c4d6eca0b85ea3a414fe8578f30262268a39fdf6` |
| `examples/rnic_hostmem_v1/results.csv` | `1bc7bcc8e72b7aef9fda1ed7e6ca2078d60c48a00377cbf8dfded75ff4d2fa53` |
| `examples/rnic_submission_v1/results.csv` | `8f74c6fd92d012f2c70c1c2b09d6f49a4d99bcc35fd418a239f7b577777edbc7` |

  Six of the seven digests are copied from the frozen inventories the accepted
  BACK-19 and BACK-20 registries already carry. The seventh, the BACK-20
  artifact itself, is not in any of those inventories, so it was hashed from
  the tracked file at this commit before any mechanism existed and is frozen
  here.
- **G5** the whole native CTest suite is green, with every pre-existing test
  unchanged, under `-DSIMLLM_RNIC_WARNINGS_AS_ERRORS=ON`.
- **G6** idle-second-device equivalence: attaching a `GpuDevice` to the shared
  fabric and issuing no transfer leaves every RNIC WQE timestamp and every
  per-service-class accounting field exactly equal to the run without the
  second device.
- **G7** identity-attribution equivalence: enabling the endpoint identities on
  a device that keeps its own host-pinned data region leaves every WQE
  timestamp and every per-service-class accounting field exactly equal to the
  unattributed run. Attribution may only add endpoint-ledger rows.
- **G8** cross-device rejection is transactional. Each of the following throws
  the registered exception type and leaves the fabric generation, the fabric
  per-class accounting, the registry generation, the registry live allocation
  count, the SQ occupancy and the work-queue counters unchanged:
  1. a WQE naming a peer region whose owner granted no access;
  2. a WQE whose named peer owner disagrees with the region's actual owner;
  3. a WQE naming its own device owner as a peer;
  4. a GPU transfer into a region owned by another device without a grant;
  5. a second device claiming a device owner identity already claimed;
  6. a second device claiming an endpoint identity already claimed;
  7. a transaction naming an unattached endpoint identity;
  8. an attributed `HostStore`.
- **G9** `validateInvariants()` passes on the fabric, the registry, the RNIC
  device and the GPU device at the end of every cell, and the per-endpoint
  ledger conserves: the sum of requester transactions over endpoints equals the
  attributed requester count, and likewise for completers.
- **G10** the four common Python gates pass: `ruff check .`, `pytest -q`,
  `check_docs_format.py`, `task_progress.py --check`.
- **G11** no version constant moves and `defaultPcieFabricConfig()` keeps its
  fields: `kPcieFabricConfigVersion`, `kPcieTransactionAbiVersion`,
  `kPcieTransactionResultVersion` and `kPcieAnalyticalDelayProfileVersion` all
  stay 1.

G1 through G3 are the mutation-sensitive off-path locks: they rebuild the
native library from source and re-derive the accepted bytes, so any C++ change
that perturbs the off path fails them. G4 is the tracked-artifact lock, carried
in pytest with a negative control that proves the check rejects a single
flipped byte. Neither lock alone is claimed to do the other's job.

## Structural evidence, fatal and unscored

Reported separately, never added to the scored denominator: the eight run
configuration rows; the endpoint-kind labels of the payload-read completer in
each arm; the by-construction zeros (`gpu_completer_transactions` is zero in
`host_bounce`, `gpu_requester_transactions` is zero in `gpu_direct`,
`staging_transfers` is zero in `gpu_direct`); the requester-side mirror of F4;
the per-endpoint ledger conservation identities; and the native test
executables and their pass counts.

## Registered command

The local machine configuration must set `SIMLLM_WAVE17_RUN_ROOT` to the
external wave-17 run root. The result-producing command is:

```bash
.venv/bin/python examples/rnic_gpu_endpoint_v1/run_study.py \
  --out "$SIMLLM_WAVE17_RUN_ROOT/w17b/back46"
```

The same command with `--check-only` appended validates the frozen registry,
the eight-cell matrix, the artifact inventory and the external-output rule
without configuring CMake, creating the output directory or producing an
artifact.

## Registry outcome, decided before the run

BACK-46 stays open. The live-chain residual is registered as BACK-49 and the
effective-hardware projection gap as BACK-50: the session-record hardware
snapshot does not describe fabric endpoint identities, so two devices differing
only in endpoint identity share one `hardware_config_sha256`. Both residual
texts are written in the closing commit, not here, because their exact wording
depends on what the run shows.
