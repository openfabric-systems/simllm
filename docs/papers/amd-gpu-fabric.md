# AMD GPU fabric: xGMI, RCCL and the ROCm RDMA path

Evidence dossier for the AMD half of the pluggable packet-device frame. The
claim under test is that "GPU as a packet device" is vendor-pluggable: PCIe
plus NVLink with NCCL on NVIDIA, PCIe plus xGMI with RCCL on AMD. This note
does not assert that correspondence, it sources it. Every number and every
structural claim below carries a source that was fetched and read on
2026-08-17, and anything that could not be sourced is called out as
unverified rather than smoothed over. The format follows
[msg-size-vs-bandwidth.md](msg-size-vs-bandwidth.md): sources first, verbatim
extraction second, derived quantities third, and provenance carried in a
column of every number table rather than in the prose around it.

## Sources

All URLs below were fetched and read on 2026-08-17. AMD PDF white papers were
retrieved over HTTP and read as extracted text; RCCL sources were read through
the GitHub raw and contents endpoints.

| Source | Kind | URL | What it contributes |
|---|---|---|---|
| AMD, "Introducing AMD CDNA 4 Architecture" (white paper) | Vendor primary | https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf | MI350X and MI355X link count, signaling rate, per-link and aggregate bandwidth, 8-GPU topology, endnote MI350-007 which also restates the MI300X numbers |
| AMD, "Introducing AMD CDNA 3 Architecture" (white paper) | Vendor primary | https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf | MI300X, MI325X and MI300A spec table, the 7 links plus 1 PCIe split, MI300A on-package coherence and its 4-APU node |
| AMD, "Introducing AMD CDNA 2 Architecture" (white paper) | Vendor primary | https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf | MI250X per-link and per-OAM numbers, the in-package GCD to GCD link, and the bandwidth formula AMD uses in its own endnotes |
| "AMD Instinct MI300 Series microarchitecture", ROCm Documentation 7.14.0 | Vendor primary | https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300.html | 7 Infinity Fabric links forming a fully connected 8-GPU system, PCIe Gen 5 x16 host attach, MI300A die mix |
| "AMD Instinct MI250 microarchitecture", ROCm Documentation 7.14.0 | Vendor primary | https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi250.html | 16-wide xGMI links at 25 GT/s, per-link 50 GB/s per direction, per-GCD PCIe Gen 4 x16 host attach |
| "RCCL Documentation, Release 2.30.7" (develop) | Vendor primary | https://rocm.docs.amd.com/projects/rccl/en/develop/ and the PDF build at https://rocm.docs.amd.com/_/downloads/rccl/en/develop/pdf/ | What RCCL is, the tuner link types, symmetric memory prerequisites, the debug subsystem list including the proxy thread and the NVLS exclusion |
| "Using the NCCL Net plugin API", RCCL 2.30.7 | Vendor primary | https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/using-nccl.html | The net plugin ABI as RCCL consumes it: library name, `NCCL_NET_PLUGIN`, `ncclNet_vX` symbol versioning, the optional `collNet` structure |
| "GPU-enabled Message Passing Interface", AMD Instinct GPU cluster networking documentation | Vendor primary | https://instinct.docs.amd.com/projects/gpu-cluster-networking/en/latest/how-to/gpu-enabled-mpi.html | The PeerDirect statement: the AMD kernel driver exposing RDMA so NICs read and write GPU memory directly |
| RCCL source, `ROCm/rocm-systems`, branch `develop` at commit `07fe594b2a9bba7b4f977a308f02612dddca1973` (2026-08-17T09:12:20Z) | Vendor primary, source | https://github.com/ROCm/rocm-systems | Transport array, plugin loader, proxy threads, GIN backends, changelog |
| "AMD AINIC Network Plugin (ANP)" | Vendor primary, source | https://github.com/ROCm/amd-anp | A shipping first-party RCCL net plugin and how it is selected |
| `ROCm/rccl` (retired repository README) | Vendor primary | https://github.com/ROCm/rccl | The collective list, and the retirement notice pointing at `ROCm/rocm-systems` as the live tree |
| "GPUDirect RDMA 13.3 documentation", NVIDIA | Vendor primary (NVIDIA side) | https://docs.nvidia.com/cuda/gpudirect-rdma/index.html | The NVIDIA definition the ROCm path is compared against |
| "Environment Variables", NCCL 2.31.2 documentation, NVIDIA | Vendor primary (NVIDIA side) | https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html | `NCCL_NET_GDR_LEVEL` and `NCCL_P2P_LEVEL` semantics, including the `NVL` level |

Two source classes are deliberately absent. No press coverage, analyst
articles or wiki-style aggregators are cited for any number. No claim below
rests on recollection; where recollection and the sources disagreed, the
sources won and the disagreement is recorded.

## Reading AMD's fabric numbers before using any of them

AMD states link bandwidth as a bidirectional sum, and it says so in its own
arithmetic. The CDNA 2 white paper endnote 8 gives the formula outright:

> Peak theoretical inter GDC to GDC data transport rate performance is
> calculated by Baud Rate * # lanes * # directions * # links / 8 = GB/s per
> card.

"GDC" there is the white paper's own spelling of GCD, quoted as printed.

The `# directions` factor is the trap. A serialization model needs bytes per
second in one direction; every headline AMD number below is twice that.
Applying the vendor's own formula to the vendor's own signaling rates
reproduces every published aggregate exactly, which is the cheapest available
check that the numbers in this dossier were read correctly.

| Claim | Formula input | Computed | Vendor-published value | Source of the published value |
|---|---|---|---|---|
| MI250X in-package GCD to GCD | 25 Gbps, 16 lanes, 2 directions, 4 links | 400 GB/s | 400 GB/s | CDNA 2 white paper, endnote 8 and body |
| MI250X external per OAM | 25 Gbps, 16 lanes, 2 directions, 8 links | 800 GB/s | 800 GB/s | CDNA 2 white paper, endnote 9 |
| MI300X peer-to-peer aggregate | 32 Gbps, 16 lanes, 2 directions, 7 links | 896 GB/s | 896 GB/s | CDNA 3 white paper spec table; restated in CDNA 4 endnote MI350-007 |
| MI355X peer-to-peer aggregate | 38.4 Gbps, 16 lanes, 2 directions, 7 links | 1075.2 GB/s | 1075.2 GB/s | CDNA 4 white paper spec table and endnote MI350-007 |
| MI355X single link | 38.4 Gbps, 16 lanes, 2 directions, 1 link | 76.8 GB/s each direction, 153.6 GB/s summed | "76.8GB/s in each direction"; "up to 153.6 GB/s ... per AMD Infinity Fabric link" | CDNA 4 white paper body and endnote MI350-007 |
| PCIe Gen 5 x16 host link | 32 GT/s, 16 lanes, 2 directions | 128 GB/s | 1024 minus 896 = 128; 1203.2 minus 1075.2 = 128 | CDNA 3 and CDNA 4 spec table rows "TOTAL PEAK AGGREGATE I/O BANDWIDTH" |

The last row is a consistency result rather than a quoted figure: both white
paper spec tables publish a peer-to-peer aggregate and a strictly larger total
aggregate, and the difference is exactly one PCIe Gen 5 x16 link at the
encoding-free rate in both generations.

There is one wording conflict worth carrying forward. Endnote MI350-007 says
each part includes "up to eight AMD Infinity Fabric links" providing 896 GB/s
(MI300X) or 1075.2 GB/s (MI355X), while the spec tables and the topology text
of both white papers describe seven x16 Infinity Fabric links plus one x16
PCIe Gen 5 link to the host. Seven is the number that reproduces both
aggregates exactly. The eighth interface is the multi-purpose link: the CDNA 3
white paper states that "One of the links is multi-purpose and can be
configured to act as a x16 PCIe Gen 5 for pure I/O functionality", and the
CDNA 4 white paper says each GPU has "one Infinity Fabric link configured for
PCIe Gen 5 to connect to I/O devices such as storage and networking". Model
seven peer links and one host link; treat the endnote's "eight" as counting
the physical interface rather than the peer-reachable link.

## xGMI and Infinity Fabric nameplate figures

Every value in this table is vendor-claimed peak theoretical bandwidth, not
measured. Bandwidth columns are bidirectional sums in AMD's convention unless
the cell says otherwise.

| Part | Architecture | Link signaling | Per-link bandwidth | Links carrying peer traffic | Per-device peer aggregate | Host attach | Provenance |
|---|---|---|---|---|---|---|---|
| MI250X, per OAM | CDNA 2 | 25 GT/s, 16 lanes | 100 GB/s (50 GB/s each direction) | 8 external | 800 GB/s | PCIe Gen 4 x16 per GCD, or a coherent Infinity Fabric link when paired with an optimized 3rd Gen EPYC | CDNA 2 white paper endnote 9 and body; ROCm MI250 microarchitecture page for the per-direction rate and the host attach |
| MI250X, per GCD (the ROCm-addressable device) | CDNA 2 | as above | 100 GB/s | 4 external | 400 GB/s (derived, 800 halved) | as above | Derived from the same endnote; the ROCm MI250 page states each of the two GCDs in an OAM "constitutes one GPU device in the system" |
| MI300X | CDNA 3 | 32 Gbps, 16 lanes | 128 GB/s (derived, 896 over 7) | 7 | 896 GB/s | PCIe Gen 5 x16 | CDNA 3 white paper spec table row "P2P RING PEAK AGGREGATE I/O BANDWIDTH: 896 GB/s (8 GPUs)"; per-link value derived, not published |
| MI300A | CDNA 3 | 32 Gbps, 16 lanes | 128 GB/s (derived) | see the note below | 384 GB/s in a 4-APU node | "4x16 PCIe Gen 5 or AMD Infinity Fabric Links"; the CPU cores are on package, so there is no external host link | CDNA 3 white paper spec table rows for MI300A |
| MI350X and MI355X | CDNA 4 | 38.4 Gbps, 16 lanes | 153.6 GB/s (76.8 GB/s each direction) | 7 | 1075.2 GB/s | PCIe Gen 5 x16 | CDNA 4 white paper body, spec table and endnote MI350-007 |

MI250X needs the per-GCD row because the OAM is not the addressable unit, and
the ROCm MI250 microarchitecture page settles that in one sentence: it
describes "an OAM package that consists of two GCDs, each of which constitutes
one GPU device in the system". The same page says "Each GCD maintains its own
PCIe x16 link to the host part of the system", notes that "some platforms may
offer an x8 interface to the GCDs, which reduces the available host-to-GPU
bandwidth", and describes the in-package path as "four AMD Infinity Fabric
links running at a theoretical peak rate of 25 GT/sec, giving 200 GB/sec peak
transfer bandwidth between the two GCDs of an OAM, or a bidirectional peak
transfer bandwidth of 400 GB/sec for the same". Any per-GPU figure taken from
an MI250X spec sheet is therefore twice the per-device figure a collective
library sees. This is the single largest unit trap in the AMD lineup and it
disappears from MI300X onward, where one OAM is one device.

The same page states the bidirectional convention for the external links too,
in the same breath as the per-direction rate: each 16-wide link "operates at
25 GT/sec, which corresponds to a theoretical peak transfer rate of 50 GB/sec
per link (or 100 GB/sec bidirectional peak transfer bandwidth)". That is the
convention documented on both the white-paper side and the ROCm docs side, so
the halving rule above is not an interpretation.

This is the same class of error the repository already hit on the NVIDIA side
and recorded in [docs/modules/traffic.md](../modules/traffic.md), where taking
`nvidia-smi nvlink -s` at face value overstated a Hopper link ceiling by 6.25
percent because the tool reports raw signalling rate rather than payload rate.
The AMD trap is larger and simpler: it is a factor of two, and it is stated in
the vendor's own formula rather than hidden in a tool's output.

MI300A carries an internal tension that this dossier does not resolve. The
CDNA 3 spec table gives MI300A "4x16 AMD Infinity Fabric Links" plus "4x16
PCIe Gen 5 or AMD Infinity Fabric Links" and a peer aggregate of "384 GB/S
(4 APUs)", which is exactly three links at 128 GB/s, i.e. one link per peer in
a four-socket fully connected node. The body text of the same white paper
instead says that in a four-processor node "each processor is fully connected
to its peers using two AMD Infinity Fabric links with 256GB/s of bandwidth",
which needs six peer links. Both statements are quoted here verbatim because
they cannot both describe the same configuration. Use the spec table row for
any modeled aggregate, and treat the MI300A peer link count as
configuration-dependent until a source resolves it.

### Node topology

The 8-GPU shape is stable across CDNA 3 and CDNA 4 and is stated in first
party documentation, not inferred from a diagram. The ROCm MI300 Series
microarchitecture page says "The GPUs are using seven high-bandwidth,
low-latency AMD Infinity Fabric links (red lines) to form a fully connected
8-GPU system" and "The MI300X OAMs attach to the host system via PCIe Gen 5
x16 links (yellow lines)". The CDNA 4 white paper says the MI350 Series system
architecture "is identical to the prior generation with a fully connected
8-GPU system. Each GPU uses one PCIe Gen 5 link to connect to the host
processors and I/O devices; this topology can flexibly handle all
communication patterns within the server node."

Two consequences matter for a simulator. First, an 8-GPU AMD node is a full
mesh with one hop between any pair, not a ring and not a switched fabric. No
switch appears in either white paper's node description or spec table, and
seven links reaching seven peers leaves no link for one, so the absence of an
xGMI analogue of NVSwitch in these platforms is a consequence of the cited
topology rather than a separate vendor statement. The CDNA 4 white paper's
phrasing ("this topology can flexibly handle all communication patterns") is
the vendor asserting the mesh is sufficient. Second, the per-pair
link budget is a single x16 link, so an all-to-all pattern on 8 GPUs uses 7
disjoint links per GPU concurrently and a ring uses 2 of them. The published
aggregate is the all-links-busy number and a ring collective cannot reach it.

MI250X is a different shape. The ROCm MI250 page describes an asymmetric
peer-to-peer graph where, for example, "GCD pairs 2 and 6 as well as GCDs 0
and 4 connect via two XGMI links", so the MI250X node is not a uniform mesh
and per-pair bandwidth is not uniform either.

### Host attach

| Part | Host interface | Coherent? | Provenance |
|---|---|---|---|
| MI250X | PCIe Gen 4 x16 per GCD; some platforms offer x8 | Optional: the 16-lane link behaves as a coherent Infinity Fabric interface with an optimized 3rd Gen EPYC and "falls back to behave like a standard PCIe interface with non-coherent communication" otherwise | ROCm MI250 microarchitecture page; CDNA 2 white paper |
| MI300X, MI325X | PCIe Gen 5 x16 | Not claimed | ROCm MI300 Series page; CDNA 3 white paper spec table |
| MI300A | On package. The CDNA 3 white paper says "the on-package AMD Infinity Fabric connects both the accelerator complex dies (XCDs), and the CPU complex dies (CCDs) directly into the shared Infinity Cache and 8-stack of HBM3 at chiplet latency and interface throughput" | Yes, by construction: CPU and GPU share one memory | CDNA 3 white paper; ROCm MI300 Series page gives the MI300A die mix as 6 XCDs and 3 CCDs |
| MI350X, MI355X | PCIe Gen 5 x16 | Not claimed | CDNA 4 white paper spec table |

MI300A is the case that breaks a host-attach model built on PCIe. There is no
host DMA hop to price for CPU to GPU data movement, and the coherent shared
HBM3 means a CPU-side buffer and a GPU-side buffer can be the same bytes. Any
model that charges a PCIe crossing for host to device traffic is wrong for
MI300A specifically, and only for MI300A among the parts covered here.

## RCCL relative to NCCL

### Lineage

RCCL is a source fork of NCCL that tracks upstream version for version, and
the source says so directly rather than by resemblance. `src/transport.cc`
opens with:

```
Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
Modifications Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
```

`src/proxy.cc` carries the same pair plus a third line for Microsoft
modifications.

The documentation build read this session is "RCCL Documentation, Release
2.30.7", matching NCCL 2.30.x, and the RCCL usage tips page reasons about
upstream behavior by version: "This matches the behavior introduced in
upstream NCCL 2.28.7, which removed the topology-distance check from the
symmetric-memory decision." The source tree still carries `nccl.h.in` and
`libnccl.map`, and the AMD-published plugin page still explains that plugins
decouple builds made "against a particular version of the GPU stack (such as
NVIDIA CUDA)". The practical consequence is that the NCCL mental model
transfers by default and the interesting question is always which pieces do
not transfer, not which do.

### Transports

`src/transport.cc` defines the transport table as:

```c
struct ncclTransport* ncclTransports[NTRANSPORTS + 1] = {
  &p2pTransport, &shmTransport, &netTransport, &collNetTransport,
  &profilerTransport // Not really used for transport, only to create proxy ops polling on profiler counters.
};
```

This is NCCL's list unchanged, in NCCL's order, and `selectTransport` walks it
in that order taking the first transport whose `canConnect` succeeds. The four
real transports are peer-to-peer, shared memory, network and collective
network; the fifth entry exists only to create proxy operations that poll
profiler counters, per its own comment.

The transport directory (`src/transport/`) contains `p2p.cc`, `shm.cc`,
`net.cc`, `coll_net.cc`, `net_socket.cc`, `net_ib/`, `net_ib_cast/`,
`generic.cc`, `profiler.cc` and `nvls.cc`. The presence of `nvls.cc` in an AMD
tree is a fork artifact, not a capability: the RCCL environment variable
documentation lists the debug subsystems and says of one of them, verbatim,
"NVLS: Not valid for AMD/RCCL". This is the clearest documented case of an
NCCL concept with no AMD counterpart, and it is worth stating because NVLS is
exactly the NVLink SHARP in-switch reduction path that has no xGMI analogue in
the platforms above.

On the link types RCCL itself distinguishes, the tuner plugin documentation is
explicit: a tuner selects an algorithm and protocol "based on an input
configuration specifying the message size, number of nodes and GPUs, and link
types (for instance, PCIe, XGMI, or NET)". Three link classes, matching the
three physical paths in the packet-device frame.

### The net plugin path: RCCL consumes the NCCL net plugin ABI

This is the load-bearing question for the pluggable frame and the answer is
yes, with one rename. From "Using the NCCL Net plugin API" (RCCL 2.30.7):

> NCCL network plugins are packaged as a shared library called
> `librccl-net.so`.

> The `NCCL_NET_PLUGIN` environment variable allows multiple plugins to
> coexist. If it's set, NCCL looks for a library named
> `librccl-net-${NCCL_NET_PLUGIN}.so`.

> After a library is found, NCCL looks for a symbol named `ncclNet_vX`, with
> `X` increasing over time. This versioning pattern ensures that the plugin
> and the NCCL core are compatible.

> In addition to the `ncclNet` structure, network plugins can provide a
> `collNet` structure which implements any supported in-network collective
> operations.

The loader in `src/plugin/net.cc` confirms both the name and the version
range. It sets `const char* defaultNetPlugin = "librccl-net.so";` and declares
external accessors `getNcclNet_v6` through `getNcclNet_v12`, tried newest
first:

```c
getNcclNet_t* getNcclNet[NCCL_NET_VERSION_COUNT] = {getNcclNet_v12, getNcclNet_v11, getNcclNet_v10, getNcclNet_v9,
                                                    getNcclNet_v8,  getNcclNet_v7,  getNcclNet_v6};
```

So the ABI is NCCL's `ncclNet_vX`, versions 6 through 12, with the struct
layout documented on the AMD side too (the published RCCL API page documents
`ncclNet_v6` in full, including `regMrDmaBuf` and the `NCCL_PTR_DMABUF`
pointer-support flag). A net plugin written against the NCCL net API is
source-compatible with RCCL; what changes is the file name it must be
installed under.

The rename is narrower than it looks. `src/plugin/plugin_open.cc` carries the
prefix table for every plugin class:

```c
static const char* pluginPrefix[NUM_LIBS] = {"librccl-net",   "libnccl-gin",      "libnccl-rma",
                                             "libnccl-tuner", "libnccl-profiler", "libnccl-env"};
```

Only the net plugin was renamed to the `librccl-` prefix. The GIN, RMA, tuner,
profiler and env plugin classes keep NCCL's own `libnccl-` names in an AMD
build. The same file accepts a fully specified library name verbatim when it
starts with `lib` and ends with `.so`, and otherwise composes
`<prefix>-<name>.so`.

Two built-in networks are always registered after any external plugin, from
`src/plugin/net.cc`: an InfiniBand-class net and a socket net, with an
AMD-specific selection between two IB implementations:

```c
    if ((envNet && strcasecmp(envNet, "IB-CAST") == 0 && !extNetPluginRequested) ||
        (!envNet && rcclUseAinic() && !extNetPluginRequested)) {
      netPluginLibs[pluginCounter].ncclNet = &netIbCast;
```

`NCCL_NET=ROCM-IB` is mapped to `IB-CAST`, and on a host where
`rcclUseAinic()` is true the AINIC-oriented `netIbCast` implementation is
chosen over the generic `ncclNetIb` even with no environment variable set.
`ncclNetSocket` is appended last as the fallback.

That the plugin path is a real product surface and not just an inherited
header is demonstrated by `ROCm/amd-anp`, the AMD AINIC Network Plugin, whose
README says it "extends AMD's RCCL library for networking capabilities",
builds `librccl-anp.so`, and is selected with
`-x NCCL_NET_PLUGIN=librccl-anp.so`, at which point, per the README, "RCCL
will load this specific AINIC plugin library instead of the default plugin
library `librccl-net.so`". Note the ANP name does not follow the
`librccl-net-${NCCL_NET_PLUGIN}.so` pattern the documentation recommends; it
works because the loader takes a full `lib*.so` name as given.

### Proxy threads

RCCL keeps NCCL's host proxy structure intact. `src/proxy.cc` starts three
host threads:

| Thread entry point | Started in | Role as read from the source |
|---|---|---|
| `ncclProxyService` | `ncclProxyCreate`, as `comm->proxyState->thread` | The per-communicator proxy service that handles proxy RPCs and connection setup |
| `ncclProxyServiceUDS` | alongside it, as `comm->proxyState->threadUDS` | A Unix domain socket service used for memory handle exchange |
| `ncclProxyProgress` | `ncclProxyProgressCreate`, created by the service thread | The progress loop that advances active proxy operations |

The progress thread is explicitly a child of the service thread and inherits
its CPU affinity: the source comments "This thread is created by proxyService,
therefore setting the affinity is not needed", and affinity is configurable
through `NCCL_PROXY_CPUSET`. The RCCL documentation exposes the same structure
to users through its debug subsystem list, which includes "PROXY: Prints logs
related to the proxy thread."

For the packet-device frame this matters more than the transport list. The
host proxy is the component that turns a device-side collective into posted
network work, and on AMD it is the same design, the same thread names and the
same progress loop as on NVIDIA. A model that prices a host proxy hop for
NCCL prices the same hop for RCCL.

## The GPUDirect RDMA equivalent on ROCm

NVIDIA's definition, for comparison, is that GPUDirect RDMA is "a technology
introduced in Kepler-class GPUs and CUDA 5.0 that enables a direct path for
data exchange between the GPU and a third-party peer device using standard
features of PCI Express", with the constraint that "the two devices must share
the same upstream PCI Express root complex".

ROCm has the same capability under two mechanisms, and current RCCL selects
between them at runtime. The naming is the part most often gotten wrong, so
both names are sourced.

**PeerDirect (the `peermem` path).** AMD's GPU cluster networking
documentation states it directly:

> The AMD kernel driver exposes remote direct memory access (RDMA) through
> _PeerDirect_ interfaces. This allows network interface cards (NICs) to
> directly read and write to RDMA-capable GPU device memory, resulting in
> high-speed direct memory access (DMA) transfers between GPU and NIC.

**dma-buf.** The NCCL net plugin ABI carries dma-buf support as a first-class
member: the documented `ncclNet_v6` struct includes
`regMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle)`,
and the RCCL API documentation states that if a plugin "has set the
`NCCL_PTR_DMABUF` property in `ptrSupport`, NCCL uses `regMrDmaBuf` instead of
`regMr`". On the ROCm side the file descriptor comes from the HSA runtime: the
RCCL changelog entry for release 2.30.4 (ROCm 7.14.0) records that RCCL now
directly links the HSA runtime and "binds `hsa_init`,
`hsa_system_get_info`, `hsa_status_string`, and
`hsa_amd_portable_export_dmabuf` to those symbols".

**Which one runs.** The RCCL changelog is explicit about the selection policy,
in one bullet under "RCCL 2.28.3 for ROCm 7.12":

> Changed GPU Direct RDMA mode selection logic to prefer peermem over DMAbuf
> by default. `NCCL_DMABUF_ENABLE` now defaults to 1 (previously 0). When both
> peermem and DMAbuf are available, RCCL will use peermem. If peermem is
> unavailable, RCCL will automatically fall back to DMAbuf (if available and
> enabled). Setting `RCCL_FORCE_ENABLE_DMABUF=1` forces DMAbuf usage
> exclusively, skipping peermem even if available, and disables GPU Direct
> RDMA if DMAbuf is unavailable.

So on a current ROCm stack both paths exist, `peermem` is preferred when
present, dma-buf is the automatic fallback, `NCCL_DMABUF_ENABLE` gates it and
now defaults to enabled, and `RCCL_FORCE_ENABLE_DMABUF=1` pins dma-buf while
turning GPU Direct RDMA off entirely if dma-buf is missing. The release 2.30.4
notes also record a defect in exactly this selection logic (proxy channel
staging buffers ignoring the new mode selection on older HIP builds, so
"peermem-equipped hosts on older HIP no longer fall through to
`hsa_amd_portable_export_dmabuf` when peermem was selected", with
`NCCL_DMABUF_ENABLE=0` as the workaround). The two-path selection is live code
rather than settled history, and a run that cares which path it took should
check rather than assume.

Correspondence summary for this section: NVIDIA GPUDirect RDMA maps to ROCm
PeerDirect plus dma-buf, not to a single named AMD feature. A model that needs
one switch should model "NIC reads and writes GPU memory without a host bounce
buffer" as the capability and treat `peermem` versus dma-buf as an
implementation detail that changes neither the data path nor its cost.

## GPU-initiated networking on AMD

The brief for this dossier asks for GPU-initiated networking only if it is
documented, and the honest answer is that the user documentation is thin while
the source tree is not.

**What AMD documents.** The RCCL usage tips page lists the prerequisites for
the symmetric-memory fast path and the fourth one is:

> Either GPU-Initiated Networking (GIN) is available, or the communicator is a
> single locality (one-LSA) team.

The same page says that when the symmetric path is unavailable RCCL logs which
prerequisite was missing, naming `globalGinSupport` as one of the reported
tokens. That is the entire first-party user-facing description of GIN on AMD
found this session: GIN appears as a named, runtime-checked precondition, and
nothing in the RCCL documentation set read here describes what it does, which
hardware provides it, or how to enable it.

**What the source tree shows.** `src/gin/` on `develop` contains
`gin_host.cc`, `gin_host_proxy.cc`, a `proxy_gpucontext/` directory, and two
backend implementations: `gin_plugin_rocshmem_gda.cc` with
`gin_rocshmem_gda_factory.cc`, and `gin_plugin_anvil_sdma.cc` with
`gin_anvil_sdma_oss7_device.cc` and `gin_anvil_ipc_table_host.cc`. These are
AMD-authored rather than inherited: `gin_plugin_rocshmem_gda.cc` carries
"Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.", is
guarded by `#ifdef ENABLE_ROCSHMEM_GIN`, and its header comment describes a
"Built-in GIN plugin for the rocshmem GDA (QueuePair) backend" that "Follows
upstream vtable pattern like GDAKI", creating queue pairs at connect time and
registering symmetric memory regions. `src/plugin/plugin_open.cc` additionally
carries `libnccl-gin` as a loadable plugin class, so GIN is pluggable on AMD
the same way the net transport is.

**What is therefore true.** RCCL has an AMD-authored GPU-initiated networking
implementation in tree with at least two backends, it is compile-gated, and it
is referenced in user documentation only as a precondition token. No AMD
document read this session states that GIN is a supported feature on any
specific AMD GPU or NIC, gives an enabling procedure, or quotes any
performance figure for it. Treat GIN on AMD as present and evolving, not as a
capability a model may assume.

## NVIDIA to AMD correspondence

One row per concept, with the provenance for the AMD side, since that is the
half this dossier exists to establish.

| NVIDIA | AMD counterpart | Correspondence strength | Provenance |
|---|---|---|---|
| NVLink, the scale-up GPU to GPU link | xGMI, i.e. AMD Infinity Fabric links | Direct. Same role, same place in the topology, same "bidirectional sum" quoting convention | CDNA 2, CDNA 3, CDNA 4 white papers; RCCL documentation, "It uses PCIe and xGMI high-speed interconnects" |
| NVSwitch-based scale-up fabric | None in the 8-GPU platforms covered here | No counterpart. The AMD platform is a direct fully connected mesh with 7 links per GPU | ROCm MI300 Series microarchitecture page; CDNA 4 white paper, "identical to the prior generation with a fully connected 8-GPU system" |
| NCCL | RCCL | Direct. Source fork tracking NCCL version for version | Dual NVIDIA and AMD copyright headers in `src/transport.cc` and `src/proxy.cc`; "RCCL Documentation, Release 2.30.7" |
| NCCL transports p2p, shm, net, collNet | Identical array, identical order, plus the same profiler pseudo-transport | Identical | `src/transport.cc` `ncclTransports` |
| `ncclNet` plugin ABI in `libnccl-net.so` | Same ABI, library renamed to `librccl-net.so`; `ncclNet_vX` for X in 6 to 12 | Direct, with a file-name change only | RCCL "Using the NCCL Net plugin API"; `src/plugin/net.cc` |
| Other NCCL plugin classes (tuner, profiler, env, RMA, GIN) | Same ABIs and the same `libnccl-` file names, not renamed | Direct, and the net rename does not generalize | `src/plugin/plugin_open.cc` `pluginPrefix` table |
| GPUDirect RDMA | PeerDirect (`peermem`) and dma-buf, selected at runtime | Same capability, two named mechanisms instead of one | AMD GPU cluster networking documentation; RCCL changelog entries for `NCCL_DMABUF_ENABLE` and `RCCL_FORCE_ENABLE_DMABUF` |
| NCCL host proxy threads | `ncclProxyService`, `ncclProxyServiceUDS`, `ncclProxyProgress`, same structure and affinity handling | Identical | `src/proxy.cc`; RCCL debug subsystem "PROXY: Prints logs related to the proxy thread" |
| NVLink SHARP (NVLS) in-switch reduction | None | Explicitly excluded, despite `nvls.cc` existing in the fork | RCCL environment variable documentation, "NVLS: Not valid for AMD/RCCL" |
| GPU-initiated networking (GIN) | In-tree AMD backends (rocshmem GDA, anvil SDMA), compile-gated, referenced in docs only as a precondition | Partial and unverified as a supported feature | `src/gin/` listing; `gin_plugin_rocshmem_gda.cc`; RCCL usage tips symmetric-memory prerequisites |
| `NCCL_P2P_LEVEL`, `NCCL_NET_GDR_LEVEL` | Same variable names, and RCCL reasons about `NCCL_P2P_LEVEL` in its own usage tips | Direct on the variable names; the AMD docs read here do not restate the level token list, so whether the NVIDIA-flavored `NVL` token is accepted on AMD is unverified | NCCL environment variable documentation for the level list; RCCL usage tips page for the AMD-side discussion, whose worked example uses `PHB` |

The overall verdict this dossier supports: the vendor-pluggable claim holds at
the level of transports, plugin ABI, host proxy structure and NIC to GPU DMA,
and it fails at exactly two points, in-network reduction (NVLS has no AMD
counterpart) and switch-based scale-up (the AMD platform is a direct mesh). It
is not established for GPU-initiated networking in either direction of
confidence.

## What this dossier does not establish

- No measured bandwidth, latency or collective efficiency for any AMD part.
  Every number above is vendor-claimed peak theoretical bandwidth. The
  cross-architecture result already recorded in
  [docs/modules/traffic.md](../modules/traffic.md) applies here with full
  force: measured ring efficiency against a GPU's own link ceiling was 71.0
  percent on Ampere and 74.9 percent on Hopper, 3.9 percentage points apart,
  while the ceiling itself moved by exactly 1.5 times. Efficiency transferred
  across a link generation and the rate did not, so a nameplate xGMI aggregate
  is an upper bound and nothing more.
- No RCCL algorithm or protocol thresholds, no channel counts, no comparison
  of RCCL tuning against NCCL tuning.
- No verification that the MI300A peer link count is three or six per the
  conflict recorded above.
- No first-party statement that GIN is supported on any AMD GPU or NIC.
- No MI355X per-part datasheet was read directly; the MI350 Series numbers
  come from the CDNA 4 architecture white paper, which covers MI350X and
  MI355X together and gives identical fabric rows for both. Repeated attempts
  to read the product pages and product briefs on the vendor web site returned
  timeouts this session.

## What this constrains in simllm

Stated as constraints rather than as work items; no registry entry is created
by this note.

- `simllm/compute/transformer.py` currently defines `GPU_ENVELOPES` for
  `gtx1660-ti-sm75`, `a100`, `h100`, `h200`, `b100` and `b200`, all NVIDIA. An
  AMD entry needs compute and HBM figures, which this dossier does not carry:
  its scope is fabric, not roofline.
- `docs/architecture.md` prices intra-node traffic against "an NVLink-class
  resource" and keeps it off the fabric backend. Nothing in the evidence above
  requires a second mechanism for AMD: a per-GPU egress cursor over a fully
  connected mesh is the same abstraction. What differs is the parameter (7
  links at 128 or 153.6 GB/s bidirectional, so 448 or 537.6 GB/s of egress
  per GPU after halving) and, for MI250X, that the addressable device owns
  half of the advertised OAM figure.
- MI300A is the one part that would break the host-attach assumption, because
  its CPU and GPU share HBM3 on package and there is no PCIe crossing to
  charge for host to device movement.
- On the software side, no new seam is implied. RCCL uses NCCL's transport
  set, NCCL's net plugin ABI and NCCL's proxy threads, so an AMD path is a
  parameterization of the existing NCCL model rather than a parallel stack.
  The two documented divergences, absent NVLS and absent switch-based
  scale-up, both remove mechanisms rather than adding them.
