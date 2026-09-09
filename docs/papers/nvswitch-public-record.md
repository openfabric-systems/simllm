# Public NVSwitch evidence for the DGX model

The product topology and the model's internal queue assumptions have different
evidence. Sources were retrieved on 2026-09-09. Local copies are in the ignored
`papers/` archive; `papers/manifest.json` records retrieval URLs, byte counts
and SHA-256 digests. Bulk papers are not redistributed in Git.

| Source | Supported use | Limit |
|---|---|---|
| [DGX A100 User Guide, System Topology](https://docs.nvidia.com/dgx/dgxa100-user-guide/introduction-to-dgxa100.html#dgx-a100-system-topology) | Eight A100 GPUs, six second-generation switches, host/PCIe context | No queue timing specification |
| [DGX H100/H200 User Guide](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html#dgx-h100-200-system-topology) | Eight-GPU system context and third-generation switches | Optional devices and storage are omitted in our schematic |
| [Fabric Manager, NVLink Topology tables 10 and 11](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/#nvlink-topology) | A100 two links per GPU per chip; H100 bundle order 4, 5, 5, 4 | Hardware enumeration requires a captured mapping |
| [NVSwitch Technical Overview, 2018](https://images.nvidia.com/content/pdf/nvswitch-technical-overview.pdf) | First-generation fully connected crossbar, concurrent nonconflicting ports, GPU memory transactions | Volta-era mechanism evidence, not an A100 numeric profile |
| [NVSwitch and DGX-2, Hot Chips 30](https://www.old.hotchips.org/hc30/2conf/2.01_Nvidia_NVswitch_HotChips2018_DGX2NVS_Final.pdf) | Port logic, SRAM buffering, parallel switch planes | First generation; no later-generation buffer-depth inference |
| [US20230070690A1, Virtual channel starvation-free arbitration for switches](https://patents.google.com/patent/US20230070690A1/en) | NVIDIA disclosure of input queues, destination/virtual-channel credit masks and coupled input/output arbitration | An embodiment, not proof of the A100 or H100 deployed arbiter |
| [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) | Hopper-generation collective acceleration differs from A100 | Reduction/multicast timing is not qualified by the unicast model |


The topology comparison also uses the
[DGX GB rack hardware guide](https://docs.nvidia.com/dgx/dgxgb200-user-guide/hardware.html)
and [GB300 NVL72 reference architecture](https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/components.html).
They establish nine trays, eighteen chips and one link from each GPU to each
chip within the rack. The Hopper article above describes the distinct
2022 second-level external switch design. None of these comparisons installs
an NVL72 preset or transfers an A100 timing profile to Blackwell.

The archived Fabric Manager A100 table contains an apparent out-of-range
switch-port value, `233`, for GPU module 8. The preset does not silently repair
that row or claim register-exact bindings. Its stable logical port IDs encode
the documented GPU/chip attachment counts. PLACE-6 retains actual hardware
port binding, with TRAF-92 owning allocation and timing qualification.

The native allocator implements legal port matching and finite shared receiver
capacity checks; its rotating policy matches the existing software reference.
It is not the patent's full arbitration algorithm. Complete-packet forwarding,
queue sizes, grant timing and lane-matched routing remain declared choices.
No downloaded document justifies treating those choices as measured silicon.
The [mechanism reconstruction](../design/nvlink-mechanism-reverse-engineering.md)
contains the wider protocol and patent source chain.
