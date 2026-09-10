# Public A100 examples and their comparison limits

The public examples inspected on 2026-09-10 do not provide a matched substitute
for Merlin's two- and four-GPU float32 all-reduce curve. They do provide useful
hardware and software context. The closest NVIDIA-published two-GPU example
uses a DGX A100 and an NVSHMEM reduction wrapper. The local reference in
[RESULTS.md](RESULTS.md) instead runs the pinned NVIDIA nccl-tests executable
on the same Merlin machines as the original harness.

The assessment below uses each source's documented controls. Missing details
mean that a numerical comparison is unqualified; they do not establish that
the source's own experiment is invalid. None of these results calibrates the
model or expands its shaded band. No percentage agreement is scored against
an unmatched figure, and no uninspected plot is digitized.

## NVIDIA and vendor examples

| Public source | Documented experiment | Material comparison limit | Use here |
|---|---|---|---|
| [NVIDIA, NVSHMEM 2.0 team collectives, Figure 5](https://developer.nvidia.com/blog/accelerating-nvshmem-2-0-team-based-collectives-using-nccl/) | Two GPUs in a DGX A100 on Selene; integer sum reduction through `nvshmem_int_sum_reduce_on_stream`; NCCL enabled and disabled on the same platform. | Different wrapper, integer data type and switched board. The article does not provide the exact NCCL binary, launch/timing block, per-size repetitions or clock record needed to match this capture. | A real public two-GPU latency curve and a controlled implementation comparison, but not a Merlin calibration target. |
| [DDN/NVIDIA DGX POD reference architecture, section 4.3](https://images.nvidia.com/data-center/resources/DDN-A3I-WITH-DGX-A100-V4f.pdf) | One to eight DGX A100 systems; reported single-node peak slightly above 235 GB/s. | Peak bandwidth by system count, not latency by payload; the section does not establish the exact operation, payload or algorithm-bandwidth versus bus-bandwidth convention. DGX topology differs. | System-scale bandwidth context only. Do not invert this peak to predict a 1-MiB collective. |
| [NVIDIA, Network IO](https://developer.nvidia.com/blog/accelerating-io-in-the-modern-data-center-network-io/) | DGX A100 communication discussion and separate MPI network measurements. | The 3.4-microsecond result is inter-node MPI ping-pong with GDRCopy. The 192-GB/s network plot aggregates eight pairs. Neither is a two-GPU NVLink all-reduce curve. | Exclude those values from collective-latency overlays. |

For the two-GPU DGX example, the hardware difference matters before any fit.
NVIDIA's A100 has twelve NVLink links, with 600 GB/s aggregate bidirectional
bandwidth. The DGX/HGX eight-GPU design connects them through NVSwitch. Merlin's
four-GPU mesh divides its twelve links among three peers: NV4 gives four links
and a 100-GB/s one-direction ceiling to one peer. The corresponding twelve-link
one-direction ceiling is 300 GB/s. This threefold capacity difference is an
upper-limit comparison, not a prediction that a small collective becomes three
times faster. Software latency and the actual routing still matter. Sources:
[NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/archive/12.1.0/ampere-tuning-guide/index.html)
and [HGX configuration guide](https://docs.nvidia.com/datacenter/tesla/hgx-software-guide/index.html).

## Paper and public-log controls

| Source | Controls visible in the source | Remaining limitation and decision |
|---|---|---|
| [MSCCL++, version 4, section 7 and artifact appendix](https://arxiv.org/html/2504.09014v4) | Eight GPUs per node, explicit A100/H100 environments, CUDA 12.4, NCCL 2.26.2, per-size/environment tuning of channels, chunks, algorithm and topology, CUDA/HIP Graphs, and an archived implementation. | This is a tuned graph-launch comparison. It does not match Merlin's default automatic selection and ordinary launches. The appendix asks for the latest nccl-tests rather than pinning its exact binary; the paper text does not give the capture's warmup/repetition and clock records. Useful implementation evidence, excluded from our numerical accuracy score. |
| [Every Microsecond Matters, version 1, sections VI and VII](https://arxiv.org/html/2607.16100v1) | GB200 NVL72, named container and software versions, ten trials, means and standard deviations; separate cache and peer-transfer probes. | The hardware is Blackwell and the low-latency kernels use a different interface. The paper's cache-resident lower bound is not A100 all-reduce service. Useful mechanism context, excluded from A100 calibration. |
| [Public nccl-tests issue 331](https://github.com/NVIDIA/nccl-tests/issues/331) | User-supplied A100 send/receive log, NCCL 2.22.3, explicit five warmups and twenty iterations, graph mode off, correctness columns and transport controls. | This is peer send/receive, not all-reduce, and it is a community report hosted in NVIDIA's repository rather than a NVIDIA-authored benchmark. Its 71-73 GB/s result cannot set our all-reduce curve. |

The MSCCL++ paper's explicit tuning is a meaningful control for its intended
best-achievable comparison. It is also why importing its NCCL curve as the
untuned default would be wrong. The artifact offers a concrete reproducibility
route, while an exact rerun would still require freezing the benchmark revision
and execution settings. This assessment follows the controls, not the venue.

## Consequence for the A100 gap

Two questions remain distinct. On Merlin, the two timing harnesses differ by a
measured offset; the isolated controls do not identify one general cause, and
the new model exposes that uncertainty. Across public systems, topology,
participant count, operation, launch mode and tuning differ. These differences
are sufficient reasons to reject an exact overlay, but they do not by
themselves quantify the cause of any particular latency difference.

The review therefore retains the vendor figures as contextual references and
keeps the matched local benchmark as the accuracy comparator. TRAF-43 retains
precision work, COMP-44 retains host-cost identification, and TRAF-44 retains
architecture-scoped profile selection. No public reference closes those tasks
or turns the component result into end-to-end inference validation.
