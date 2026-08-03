"""Fabric topology manifest: the physical graph under the ranks.

Schema ``simllm-fabric-topology-v1`` describes what no serving framework
knows: nodes, GPUs, PCIe/NVLink links, NICs, GPU→NIC affinity, switches,
switch links, bandwidths, propagation delays and queue configuration.

Sources:

- **Intra-node** (GPU↔GPU NVLink/PCIe, GPU→NIC affinity): NCCL's detected
  topology (``NCCL_TOPO_DUMP_FILE``) is authoritative for what a real run
  would use; for what-if studies the graph is declared.
- **Inter-node** (NIC → ToR → fabric): a cluster inventory or the simulator
  topology config (htsim ``.topo`` files). NCCL's local discovery is *not* a
  complete description of the routed network — the switch-level graph always
  comes from here.

The concrete schema lands with milestone M4 together with the mapper's NIC
selection; this module currently pins the schema name.
"""

FABRIC_SCHEMA = "simllm-fabric-topology-v1"
