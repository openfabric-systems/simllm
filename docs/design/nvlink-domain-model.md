# NVLink in an eight-GPU DGX system

A source GPU supplies packet bytes to an NVLink. A switch accepts those bytes
into finite input storage, waits for a free input/output connection and
receiver space, then forwards them to the destination GPU. The destination
releases receive space after its ingress service and preserves the operation's
required visibility order. Credits travel upstream when downstream space is
released. Several unrelated transfers can run at once; traffic sharing the
same physical output competes for that output.

## Public system topology

![DGX A100 wiring and model queues](../../resources/figures/nvlink-domain-model.png)

The drawing is an original schematic based on NVIDIA's
[DGX A100 system topology](https://docs.nvidia.com/dgx/dgxa100-user-guide/introduction-to-dgxa100.html#dgx-a100-system-topology).
The upper CPU, PCIe and network context is simplified. The colored groups are
independent point-to-point NVLinks, not electrical buses. Storage, management
connections and optional network cards are omitted. The preset models the
GPU peer fabric; those contextual host connections do not install a PCIe model.

HGX is the GPU baseboard/platform that server manufacturers integrate. DGX
is NVIDIA's complete system. Their corresponding eight-GPU baseboards use:

| Product | GPU NVLink generation | NVSwitch generation | Chips | Links per GPU | Active GPU-facing ports per chip | GPU one-way capacity |
|---|---:|---:|---:|---:|---|---:|
| A100 | 3 | 2 | 6 | 12 | 16 each | 300 GB/s |
| H100 | 4 | 3 | 4 | 18 | 32, 40, 40, 32 | 450 GB/s |

A100 attaches two links from every GPU to each switch. H100 attaches four to
switches 1 and 4, five to switches 2 and 3, following NVIDIA's published table
ordering. Every link has 25 GB/s nominal capacity in each direction. The
advertised 600 and 900 GB/s add send and receive, so neither is the bandwidth
available to one unidirectional transfer. The six or four chips are parallel
planes in one switching tier. A packet crosses one chip. The separate
[H100 schematic](../../resources/figures/dgx-h100-nvlink.svg) makes its wiring
explicit. Sources and their generation limits are recorded in the
[public-source ledger](../papers/nvswitch-public-record.md).

A group of two or four active GPUs on this board still uses the board's
switches. It is not the same physical system as a four-A100 direct mesh.
Merlin's A100 NV4 and GH200 NV6 meshes keep their separate identities. Two
four-GPU nodes cannot substitute for an eight-GPU NVSwitch allocation.

## Switching tiers beyond one board

Count switch chips along a packet path, rather than the number of trays or
parallel chips. The eight-GPU boards above have one switching tier. Calling
this a one-tier Clos is an informal analogy: there is no intervening
leaf-to-spine-to-leaf path, only GPU to one crossbar to GPU.

The Blackwell NVL72 rack also uses one in-rack switching tier. It has nine
switch trays containing eighteen 72-port chips. Each GPU connects to every
chip with one of its eighteen links, so a transfer crosses one chip even
when its GPUs occupy different compute trays. The rack scales the crossbar
radix and physical reach rather than adding an internal switching tier.
[NVIDIA's rack guide](https://docs.nvidia.com/dgx/dgxgb200-user-guide/hardware.html#nvlink-switch-trays)
records the tray/chip counts, and its
[GB300 topology description](https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/components.html#nvidia-nvlink-switch-tray)
explicitly records each GPU-to-chip attachment.

A separate Hopper-era design illustrates two switching tiers: NVIDIA's
2022 [NVLink Switch System description](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
adds external second-level switches to GPU nodes in a 2:1 tapered fat tree,
for up to 256 GPUs. That topology is different from both a standalone DGX
H100 and a Blackwell NVL72 rack. These comparisons describe public hardware;
`dgx_peer_fabric` accepts only the eight-A100 and eight-H100 presets above.

## Constructing a physical preset

The installed `simllm.placement.dgx_peer_fabric` function returns the existing
`PeerFabric` representation, which is carried by `FabricTopologyManifest`:

```python
from simllm.placement import dgx_peer_fabric

peer = dgx_peer_fabric(
    "a100",  # "h100" selects the four-switch board
    node_id="node-0",
    ranks=tuple(range(8)),
    propagation_delay_ps=1000,       # declared example, not measured
    switch_input_buffer_bytes=65536, # declared example, not measured
)
```

The constructor requires eight distinct ranks and explicit timing/storage
assumptions. A100 creates 96 full-duplex links; H100 creates 144. All 56 ordered
rank pairs use the same physical inventory. Port names are logical attachment
identities, not hardware register indices or `nvidia-smi` enumeration. The
current route policy matches lane ordinals within a switch, covering all 12
or 18 GPU attachments. That deterministic striping choice is declared; it is
not a reconstruction of a proprietary address hash or route-table setting.
The physical path validator rejects links that join ports on different chips.

## Queue ownership and native C model

The model composes the existing packet interfaces:

| Resource or boundary | Timing/state owner |
|---|---|
| Packetization and source staging | Existing GPU peer packet engine |
| Source feed, physical link serializers, acknowledgements | Existing causal calendar |
| Finite switch input storage, per-destination virtual queues, credits | Existing causal calendar |
| Switch input/output occupancy and grant order | Python allocator, or selected native C/C++ kernel |
| Finite GPU receive storage, receiver service and ordered visibility | Existing causal calendar |
| Extent completion, graph operation and request latency | Read-only projections through `CompletionEvent` and `StepResult` |

Virtual output queues, abbreviated VOQs, separate waiting packets by their
input, virtual channel and destination. A blocked destination therefore does
not automatically hide a ready packet for another output. Grant selection
first respects physical port conflicts and available receiver bytes. The
native kernel commits a legal input/output matching and integer-picosecond
service boundaries. Independent chips and disjoint ports proceed concurrently.
Its identity policy preserves baseline ordering; the optional rotating policy
reproduces the existing declared candidate policy. Neither is claimed to be
the arbiter deployed in A100 or H100.

The C interface lives in `simllm/backends/nvswitch/switch.h`. It accepts
read-only queue-head and receiver-capacity snapshots. One context owns the
crossbar port clocks and arbitration cursor; the Python port clocks remain
inert when native service is selected. The Python calendar applies the grant
to its owned queues and credits and schedules the resulting events. There is
no independently advancing native event calendar and no duplicated buffer
accounting. Invalid native calls reject atomically before grants or cursor
changes. A library mismatch or missing library fails explicitly.

Build the library with a C++17 compiler, placing outputs outside the checkout:

```bash
cmake -S simllm/backends/nvswitch -B "$SIMLLM_NVSWITCH_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$SIMLLM_NVSWITCH_BUILD_DIR" --config Release
```

Select its explicit library filename using `PeerPacketConfig`'s
`native_switch_library` argument. The selected observation records an
implementation ID and library digest, with no machine-local path. The ordinary
Python selection has no native build requirement and loads no native library.
Direct meshes reject NVSwitch selection. The outer absent `peer_packet`
selection preserves the analytic path and its accepted artifacts exactly.

## Timing assumptions and observable evidence

The first slice uses complete-packet forwarding: switch service begins after
that packet reaches switch ingress. It couples input, crossbar and output
serializer release. Source feed overlaps input-link transmission, while GPU
receive service follows output-link arrival. Buffer capacity includes reserved
in-flight bytes, so a sender must acquire downstream space before injection.
Credit processing delay is additional to physical return propagation.

These choices and all queue depths, packet format, service rates and credit
parameters remain declared unless a product-scoped hardware profile supplies
identified values. A published patent supports possible mechanisms, not a
claim that a specific implementation appears in A100. H100 switch multicast
and reduction offload require separate protocol and service qualification;
the unicast example does not model them as a faster A100 switch.

For an uncontended packet of W wire bytes, input/output link rate R, endpoint
feed F, crossbar rate X, receive rate D and per-link propagation P, the current
model's visibility time is:

```text
max(ceil_ps(W/F), ceil_ps(W/R)) + P
+ max(ceil_ps(W/X), ceil_ps(W/R)) + P + ceil_ps(W/D)
```

Every ceiling rounds upward only after conversion to picoseconds. Shared-port
serialization and finite-space waits add to that base. A resource's queue
wait is its start minus eligibility; summing simultaneous waits across ports
is work accounting, not an additive request-latency breakdown. BACK-73 owns
the remaining detailed critical-path attribution.

The [DGX model study](../../examples/dgx_nvlink_v1/RESULTS.md) checks physical
bounds, native/Python agreement and original-graph prefill plus decode steps.
It uses explicit toy compute times to expose transport effects, not to predict
real model throughput. Product calibration and actual eight-GPU allocation
qualification remain TRAF-92. Optional reference RTL is BACK-75.
Remaining native queue ownership is BACK-74;
collective protocol fidelity is TRAF-54, and captured port binding is PLACE-6.
The [earlier public protocol reconstruction](nvlink-mechanism-reverse-engineering.md)
retains the Pascal-generation format, acknowledgement/replay and ordering
source chain. Existing compatibility and component studies remain reproducible
through their original entry points and frozen profiles.

## Reproducing the diagrams and study

```bash
python scripts/plot_dgx_nvlink.py
python examples/dgx_nvlink_v1/run_study.py --library "$SIMLLM_NVSWITCH_LIBRARY" --output "$SIMLLM_NVSWITCH_EVIDENCE_DIR"
```

The Matplotlib generator writes PDF for vector export, SVG for web viewing and
PNG for the README. Its topology counts come from the same preset constants.
The inset depicts the declared model queues, not a vendor die schematic.
