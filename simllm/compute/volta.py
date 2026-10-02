"""Explicit FP16 Volta nameplate envelope for opt-in analytical studies."""

from simllm.compute.provider import GpuSpec

# Dense FP16 tensor arithmetic and HBM ceilings from the Tesla V100 datasheet.
# This profile does not describe native BF16/FP8 or measured sustained service.
V100_PCIE_FP16 = GpuSpec(name="v100-pcie-fp16", peak_flops=112e12, mem_bandwidth=900e9)
