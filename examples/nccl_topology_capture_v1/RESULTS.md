# NCCL topology capture result

## Outcome

What ran: `examples/nccl_topology_capture_v1`, the frozen second PLACE-1
slice, read the NCCL topology dump captured on one NERSC Perlmutter GPU node
(`nid001056`, four A100-SXM4-40GB in an NV4 mesh, four Slingshot NICs) with
the new strict reader, joined it to the fabric and peer-topology schemas, and
drove `HtsimStepSink` on `rnic-nn-fluid` with the captured node and with a
literal twin written from the frozen tables. Implementation commit
`edb07066`, harness `d13ca579`, expectations-only commit `718e4566`.

What came out: the result is `PASS` with no finding. The deciding row is the
twin identity: the captured node and its literal twin produce equal
`StepResult` rows, outcomes, locality outcomes, phase services and fabric
documents on one prefill and two decode steps, with zero fabric bytes and no
backend invocation. All five structural cells, all nine rejection controls
and all eight compatibility digests held, and both frozen relations held.

What it changes for the project: PLACE-1's "sourcing intra-node structure
from NCCL topology dumps" clause becomes literal for the captured single-NIC-
per-NUMA-node shape, and its registry text narrows to what remains. The
architecture doc's NCCL-dump sentence is now a statement of a landed reader.
The affinity shapes the reader refuses and the rendering of the captured NIC
inventory into a physical fabric are registered as PLACE-9.

What it does not change: PLACE-1 stays open (general NIC selection in the
mapper is PLACE-2's freeze), no timing is calibrated (the link rate and
propagation delay are declared inputs, and the peer domain keeps its declared
evidence class), no eight-GPU switched board is covered (PLACE-6), and no
serialized schema gained a field.

## Captured node

The fixture and its digests are in
`tests/fixtures/nccl_topology/perlmutter_a100_nid001056/PROVENANCE.md`. The
reader reproduces every frozen literal: four NUMA nodes, eight PCIe Gen4 x16
devices, four GPUs (`0x10de:0x20b0`, `sm` 80) at bus ids `03`, `41`, `82`,
`c1` on NUMA nodes 3, 2, 1, 0, twelve NVLink rows of count four, and four
Cassini NICs (`0x17db:0x0501`, 200 Gbit/s) at bus ids `01`, `c2`, `81`, `42`
on NUMA nodes 3, 0, 1, 2. The join gives GPU ranks 0 through 3 the NICs
`cxi3`, `cxi2`, `cxi1`, `cxi0`. The direct mesh carries 24 links, 48 GPU
ports and 12 routes of four single-link paths each, and validates inside a
`source="extracted"` fabric manifest. The inventory cross-check agrees: the
`nvidia-smi topo -m` matrix is `NV4` on all twelve ordered pairs with the same
NUMA column, `lspci` names the same eight devices, and twelve links at 25 GB/s
per GPU equal three peers times four lanes.

## Evidence

| Evidence class | Result |
|---|---|
| Scored identity family (N5 twin) | held on step results, outcomes, locality outcomes, phase services and fabric documents |
| Scored relation instances (N5 floor, N5 doubling) | 2 of 2 held; the pair floor held on 36 phase instances with minimum slack 20,200 ps |
| Structural exact guards (N1, N2, N3, N4, N7) | 5 of 5 cells exact |
| Rejection controls (N6) | 9 of 9 refused before any schema object is built |
| Fatal compatibility digests | 8 of 8 reference artifacts byte identical |

Counts in different evidence classes are not added. The tracked
[results](results.json) hold every cell; bulk manifests stay under the
configured data root.

The live cell used the breakdown study's first three step records over a
dense one-layer geometry at hidden width 512, a fixed 37,000 ps compute
provider, 1,000 ps propagation and the live peer packet study's pass-through
direct profile at 25 GB/s per link. Its numbers, stated against physics
first: the prefill step's twelve tensor-parallel phases each move 524,288
bytes from the peak endpoint, whose floor over the four-lane 100 GB/s pair
bound is 5,242,880 ps; the measured phase service is 22,294,120 ps, which is
4.25 times the floor because the declared profile bounds egress by its
25 GB/s endpoint feed and receiver ingress, not by the four lanes. The
prefill step latency is 267,566,440 ps, exactly 37,000 plus twelve times
22,294,120. Each decode step moves 256 bytes per phase and completes in
310,120 ps with 22,760 ps of service against a 2,560 ps floor. Doubling the
declared per-link rate to 50 GB/s leaves every service and latency identical,
so the never-slower relation holds with equality; the binding term is the
endpoint feed, as the profile declares.

## Choices the freeze left open

The mesh names ports and links by GPU `dev` and takes rank ownership from
the caller's map; the two coincide on this fixture. The reader refuses a few
shapes beyond the frozen list (a NIC serving zero or several GPUs, a NIC
element with more than one `net`, a PCI element without exactly one GPU or
NIC, differing host hashes, an NVLink class not matching its target, and a
document type declaration), each fail closed. The domain id is `<node_id>:nv4`
for any lane count, as frozen; PLACE-9 carries the name for other lane
counts. The fixture files carry no end-of-line attribute, so the tests and
harness normalize line endings before hashing.

## Reproduction

```bash
source .env.local.sh
python examples/nccl_topology_capture_v1/run_study.py
python examples/nccl_topology_capture_v1/run_study.py --check
```

The live cell invokes no packet backend. `SIMLLM_DATA_ROOT` or
`--output-root` owns the bulk artifacts. The capture itself is
`capture_perlmutter_node.sh` in this directory, run on an allocated node with
the site's own scheduler and paths configured locally.
