# Rented HGX B200 node topology capture provenance

One eight-GPU NVIDIA B200 host rented for under ten minutes on the vast.ai
marketplace on 2026-09-13 (offer 48327339, machine 71029; the host kernel
names itself an AWS build, `7.0.0-1010-aws`, with two Intel Xeon Platinum
8559C sockets). The capture ran inside the provider's container under NVIDIA
Fabric Manager control. These files are input evidence for the GPU-side
binding of an eight-GPU switched preset; they carry no timing measurement
and, as stated below, no switch identity.

| File | Content | SHA-256 |
|---|---|---|
| `nccl_topo.xml` | NCCL `NCCL_TOPO_DUMP_FILE` system dump, version 1 | `52206219b633ab5abf9e100172205cc4274a03cafb31b23c0868abb7527d5ebf` |
| `nccl_graph.xml` | NCCL `NCCL_GRAPH_DUMP_FILE` search result | `d07117f7c0e098094932d45b10f4b4ffd33b1411ef41d950d0f722f90f4661fa` |
| `node_inventory.txt` | `nvidia-smi -L`, `nvidia-smi`, `nvidia-smi topo -m`, `nvidia-smi topo -p2p n`, `nvidia-smi nvlink -s`, `nvidia-smi nvlink -c`, `nvidia-smi nvlink -R`, `nvidia-smi nvlink -p`, selected `nvidia-smi -q` identity lines, the PCI device classes visible in the container, and `lscpu` NUMA lines | `c29f3e2893a79764b3a95d636e2147ced3dad665d79111585f8d122b85732ae0` |
| `nccl_env.txt` | the framework and NCCL versions of the dumping process | `fa598d30173c5cce7456b19d764c1d1c8139a8ca9d0ed50f1f94d6bbb1a80606` |
| `nccl_run_excerpt.log` | `NCCL_DEBUG=INFO` lines for versions, the network plugin, ring and tree channels and NVLS of the dumping all-reduce | `a6ec2199f87fe6e3134ac5cc4b18f6d66c3387fce4ce9ffea0ff72dfd431243f` |

Capture facts, read from these files:

- GPUs: eight NVIDIA B200 (PCI device `0x10de:0x2901`, `sm` 100), PCIe
  Gen5 x16, driver 595.91.07, board part number `692-2G525-0220-500` on all
  eight, eight distinct Board IDs (`0x5100`, `0x5200`, `0x6200`, `0x6300`,
  `0x7500`, `0x7600`, `0x8600`, `0x8700`). Device order 0 through 7 sits at
  bus ids `51`, `52`, `62`, `63`, `75`, `76`, `86`, `87`, two GPUs behind
  each of four PCIe switches (class `0x060400`, vendor `0x1000`) at `45`,
  `56`, `67`, `7a`; GPUs 0 through 3 on NUMA node 0 and 4 through 7 on NUMA
  node 1.
- NVLink: every GPU reports eighteen active links at 53.125 GB/s (the
  NVLink 5 signalling rate); `nvidia-smi topo -m` shows `NV18` for all 56
  ordered pairs; every link's remote device is the virtual fabric address
  `FFFFFFFF:FF:FF.0`, no PCI device of class `0x0680` is visible, and no
  `Module ID` is exposed. The NVSwitch side is therefore not observable
  from this container.
- NCCL: version 2.27.3 through PyTorch 2.8.0 (CUDA 12.8 build), socket
  network only (`eth0` at 10 Gbit/s, no GPU Direct RDMA), NVLS multicast
  available. The dump gives each GPU one `<nvlink target="fffffff:ff:ff.0"
  count="18" tclass="0x068000"/>` row, i.e. the switch fabric collapsed to one
  bridge-class target, and lists the socket NIC once under each `<cpu>`
  element. The graph search selected 16 ring and 16 tree channels of type
  NVL and 8 NVLS channels.

The capture script is `examples/nccl_topology_capture_v1/capture_container.sh`.
Two earlier rentals of hosts advertised as eight-GPU A100 SXM4 boards showed
every NVLink inactive and PCIe-only paths; they are retained outside Git as
negative evidence and are not fixtures. The bulk `NCCL_DEBUG` log stays
outside Git; the excerpt keeps the lines the consumers read. Provider
account, host address and rental identifiers are site local and are not
recorded beyond the offer and machine numbers above.
