# Perlmutter GPU node topology capture provenance

One NERSC Perlmutter GPU compute node, captured on 2026-09-13 inside Slurm
job 58271200 (`-C gpu -q interactive`, one node, four GPUs). These files are
the input evidence for the NCCL topology reader and the captured-node fabric
join; they are not a calibration result and carry no timing measurement.

| File | Content | SHA-256 |
|---|---|---|
| `nccl_topo.xml` | NCCL `NCCL_TOPO_DUMP_FILE` system dump, version 1 | `af3996ac37ffedec0053c574da1cc59aa22df034fb20bc043d844dbf4a921c49` |
| `nccl_graph.xml` | NCCL `NCCL_GRAPH_DUMP_FILE` search result (ring and tree channels) | `939b46a3cb0da5e263cae0b5fd0c26132d10a565dafabbd45b52d046475a33d8` |
| `node_inventory.txt` | `nvidia-smi -L`, `nvidia-smi`, `nvidia-smi topo -m`, `nvidia-smi topo -p2p n`, `nvidia-smi nvlink -s`, `nvidia-smi nvlink -c`, `nvidia-smi -q -d PCI`, `lspci`, `lspci -tv`, network and Slingshot device listings, `lscpu` NUMA lines, and the PCI-to-NUMA map of every NVIDIA device | `a0e72c8c66eef9c9317f00ebdf683cb461f62e6d0951a41265f44752b3462f81` |
| `nccl_env.txt` | the framework and NCCL versions of the dumping process | `aa548d4bea8546b9a4ae42cca46311105a7decb7510d3b09662abd65444b67f3` |
| `nccl_run_excerpt.log` | `NCCL_DEBUG=INFO` lines for the network plugin, GPU Direct RDMA, ring and tree channels of the dumping all-reduce | `251dd065e98559a2d98dbd4ca939024af0c8e0ca0a54576dd945d9548bac9dd7` |

Capture facts, read from these files:

- Host: node `nid001056`, kernel `6.4.0-150600.23.125_15.0.28-cray_shasta_c`,
  one AMD EPYC 7763 (64 cores, 128 threads, four NUMA nodes).
- GPUs: four NVIDIA A100-SXM4-40GB (PCI device `0x10de:0x20b0`), PCIe
  Gen4 x16, driver 580.159.04, CUDA 13.2 reported by the driver. Device
  order `0, 1, 2, 3` sits at bus ids `03:00.0, 41:00.0, 82:00.0, c1:00.0` on
  NUMA nodes `3, 2, 1, 0` respectively.
- NVLink: every GPU reports twelve links at 25 GB/s; every ordered GPU pair
  is joined by four bonded links (`NV4` in `nvidia-smi topo -m`, `count="4"`
  in the dump). There is no NVSwitch on this node.
- NICs: four Cray Cassini 1 Slingshot 200 Gb NICs (PCI device
  `0x17db:0x0501`), one per NUMA node: `cxi3` at `01:00.0` (NUMA 3), `cxi2`
  at `42:00.0` (NUMA 2), `cxi1` at `81:00.0` (NUMA 1), `cxi0` at `c2:00.0`
  (NUMA 0). Each GPU therefore has exactly one NIC on its own NUMA node, and
  the NIC order is the reverse of the GPU order.
- NCCL: version 2.29.7 through PyTorch 2.13.0 (CUDA 13.0 build), network
  plugin "AWS Libfabric" (Slingshot OFI), GPU Direct RDMA enabled on all four
  NICs. The graph search selected twelve intra-node ring channels of type
  NVL at speed class 20 and twelve tree channels; no inter-node channel
  existed in the single-node communicator.

The capture script is `examples/nccl_topology_capture_v1/capture_perlmutter_node.sh`.
The bulk `NCCL_DEBUG` log (about 120 KB) stays outside Git; the excerpt keeps
the lines the fixture consumers read. Configuration of the login, the
allocation account and the scratch root is site local and is not recorded
here.
