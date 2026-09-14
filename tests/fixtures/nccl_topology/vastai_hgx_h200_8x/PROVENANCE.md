# Rented HGX H200 node topology capture provenance

One eight-GPU NVIDIA H200 host rented for under ten minutes on the vast.ai
marketplace on 2026-09-13 (offer 50936499, machine 149962, kernel
`5.15.0-185-generic`, two Intel Xeon Platinum 8592+ sockets). The capture
ran inside the provider's container. Unlike the B200 host, this container
passes the NVSwitch devices through, so the switch side of the board is
observable. These files are input evidence for the switch-side and GPU-side
binding of the H100-generation switched preset; they carry no timing
measurement.

| File | Content | SHA-256 |
|---|---|---|
| `nccl_topo.xml` | NCCL `NCCL_TOPO_DUMP_FILE` system dump, version 1 | `6fd8e122c54349102bf86320018d9cf616d776e1ddbb0223e0183801877e6c5c` |
| `nccl_graph.xml` | NCCL `NCCL_GRAPH_DUMP_FILE` search result | `c803984b331e800bdd77941c2430203d320111f7d0f78e17f2e72e2861b8cbf0` |
| `node_inventory.txt` | `nvidia-smi -L`, `nvidia-smi`, `nvidia-smi topo -m`, `nvidia-smi topo -p2p n`, `nvidia-smi nvlink -s`, `nvidia-smi nvlink -c`, `nvidia-smi nvlink -R`, `nvidia-smi nvlink -p`, selected `nvidia-smi -q` identity lines, the PCI device classes visible in the container, `lspci`, and `lscpu` NUMA lines | `b4f9f5b28d276ddfeb08f09afc664ac86f1b71622f70c4e9ba228163a8a36e7a` |
| `nccl_env.txt` | the framework and NCCL versions of the dumping process | `fa598d30173c5cce7456b19d764c1d1c8139a8ca9d0ed50f1f94d6bbb1a80606` |
| `nccl_run_excerpt.log` | `NCCL_DEBUG=INFO` lines for versions, the network plugin, ring and tree channels and NVLS of the dumping all-reduce | `f9b8e7a276589d6413c30c0c0bb73d069aaf84a347b962e8ac7223cd97e5835a` |

Capture facts, read from these files:

- GPUs: eight NVIDIA H200 (PCI device `0x10de:0x2335`, `sm` 90), PCIe Gen5
  x16, driver 580.173.02, board part number `695-2G520-0280-001` on all
  eight. Device order 0 through 7 sits at bus ids `83`, `8b`, `93`, `9b`,
  `a3`, `ab`, `b3`, `bb`, one GPU behind each of eight PCIe switches (class
  `0x060400`, vendor `0x104c`, device `0x8232`) at `81`, `89`, `91`, `99`,
  `a1`, `a9`, `b1`, `b9`; GPUs 0 through 3 on NUMA node 0 and 4 through 7 on
  NUMA node 1. `Module Id` for device order 0 through 7 is 2, 4, 1, 3, 7,
  5, 6, 8.
- NVSwitches: four PCI devices of class `0x068000` (`0x10de:0x22a3`) at bus
  ids `c3`, `c4`, `c5`, `c6`, all on NUMA node 1.
- NVLink: every GPU reports eighteen active links at 26.562 GB/s (the
  NVLink 4 signalling rate); `nvidia-smi topo -m` shows `NV18` for all 56
  ordered pairs; `nvidia-smi nvlink -R` names a real remote device and port
  for every one of the 144 links: each GPU has 4 links to switch `c3`, 5 to
  `c4`, 5 to `c5` and 4 to `c6`, and the 144 `(switch, port)` pairs are all
  distinct (32 ports on `c3`, 40 on `c4`, 40 on `c5`, 32 on `c6`).
- NICs: eight Mellanox ConnectX virtual functions, one under each GPU's
  PCIe switch (`PIX` to its GPU in `nvidia-smi topo -m`, `NIC0` through
  `NIC7`), two virtio Ethernet devices, and the socket NIC `eth0` at
  10 Gbit/s. NCCL's dump lists only `eth0`, once under each `<cpu>`, and the
  run used the socket network with no GPU Direct RDMA.
- NCCL: version 2.27.3 through PyTorch 2.8.0 (CUDA 12.8 build). The dump
  gives each GPU four `<nvlink>` rows of class `0x068000` naming the four
  switch bus ids with counts 4, 5, 5 and 4. The graph search selected 12
  ring and 12 tree channels of type NVL and 8 NVLS channels.

The capture script is `examples/nccl_topology_capture_v1/capture_container.sh`.
The bulk `NCCL_DEBUG` log stays outside Git; the excerpt keeps the lines the
consumers read. Provider account, host address and rental identifiers are
site local and are not recorded beyond the offer and machine numbers above.
