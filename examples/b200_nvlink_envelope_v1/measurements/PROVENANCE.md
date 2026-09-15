# B200 NVLink envelope measurement provenance

Six rentals of a two-GPU NVIDIA B200 slice and one of an eight-GPU board, all
on the vast.ai marketplace on 2026-09-15, each destroyed by the job-local
rental script on exit. Every timed file here was produced inside the
provider's container from the image
`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` by
`examples/b200_nvlink_envelope_v1/capture_stage.sh`, which runs the inventory,
the RDMA evidence and the NVLink precheck before any timed lane. The result of
record is attempt 6; the earlier attempts are kept because each one changed the
harness, and the freeze requires the chronology to be legible. Provider account
and host addresses are site local and are not recorded here.

| File | Content | Bytes | SHA-256 |
|---|---|---|---|
| `stage1_result.json` | attempt 6, the stage 1 result of record: lanes P1, P2 and P3 by both timing methods | 181,705 | `dcbd929d5b404796d07bd5efbd6a0dd0dc9441c635baecbb82a2e3c452b794be` |
| `scored.json` | the scorer's evaluation of that result against the freeze and its two amendments | 23,468 | `0464577dac129a75dc20a80b38b4bd8173e772de2161b0b35d9dfd91acc43a0a` |
| `stage1_eager_attempt3_result.json` | attempt 3, eager timing only, retained by amendment a as the dispatch-floor evidence | 80,699 | `aa2cc3813f64f5cbdcf905f57b57097ca7f232fffb7d174ae9611a51a2f0eba0` |
| `stage2_attempt1_refusal/stage.log` | the stage 2 refusal: precheck line and the benchmark's peer-access abort | 416 | `f21570379b8807ac7425ee149f825d98895f5bf1d7121794b264797acb3756de` |
| `stage2_attempt1_refusal/listing.txt` | the stage script's own listing for that refusal | 992 | `1bc17c5adcd8cc535d75886a060d70127e3064489c3bcfc80392e7ab62a91df6` |
| `stage2_attempt1_refusal/nvlink_status.txt` | `nvidia-smi nvlink -s` of the refused board, including its one dead GPU | 2,938 | `2ea7c3abeaf4df251f565bd329bb34942c5a43c049337919df57928fb2e7a2c0` |
| `stage2_attempt1_refusal/topo_matrix.txt` | `nvidia-smi topo -m` of the refused board | 1,124 | `e179ad71aa97c5f68504a8f9634bce318993c4b6c45e8563489d193fab4611ff` |

## Rentals

| Attempt | Offer | Machine | Outcome |
|---|---|---|---|
| stage 1, 1 | 51156549 | 56359 | destroyed at launch by the rental watchdog, no measurement |
| stage 1, 2 | 51156549 | 56359 | the NCCL lanes hung on a rank to device mix-up; inventory and RDMA evidence captured, no timed row |
| stage 1, 3 | 51156549 | 56359 | every lane completed, eager timing only; the sub-megabyte dispatch floor this documents triggered amendment a |
| stage 1, 4 | 51062340 | 142255 | graph-replay rows in lanes P2 and P3; every lane P1 capture failed and fell back to eager |
| stage 1, 5 | 51062340 | 142255 | graph rows in all three lanes; the forward eager copy rows were contaminated by the capture preceding them, which triggered amendment b |
| stage 1, 6 | 51062340 | 142255 | the result of record: graph rows in every lane, both eager directions physical |
| stage 2, 1 | 51061978 | 137781 | refused before timing, one GPU with every NVLink inactive; the machine is excluded from further rentals |

One further stage 1 search between attempts 3 and 4 listed no eligible offer
and rented nothing. Rental cost, from the provider's credit ledger: 6.34, 4.21,
0.77, 1.34, 0.94 and 1.51 US dollars for the six stage 1 attempts, 15.11 in
total; 1.27 for the stage 2 refusal; and about 1 dollar of trailing storage
charges across the day's instances.

## Capture facts, read from these files

- Substrate of the record attempt: two NVIDIA B200 GPUs at PCI bus ids
  `00000000:5F:00.0` and `00000000:70:00.0`, `NV18` between them, eighteen
  active NVLinks per GPU at 53.125 GB/s signalling, driver 595.91.07, NCCL
  2.27.3 through PyTorch 2.8.0 on CUDA 12.8. Machine 56359, which served
  attempts 1 to 3, also signalled 53.125 GB/s; the refused eight-GPU board of
  machine 137781 signalled 50 GB/s on its seven live GPUs, so link rate is a
  property of the host and not of the generation.
- Timing: each row at or below 1 MiB is a CUDA graph replay of 200 captured
  iterations, and each row above it is 20 eager iterations, 10 above 64 MiB.
  Both methods are recorded at every payload; `method`, `of_record` and
  `timed_on` on each row say which is which.
- NCCL selected no NVLS at width 2 on this board: every communicator reports
  `0 nvls channels` against 32 collective and 32 point-to-point channels, and
  the transport is `P2P/CUMEM`. The internal tuner was used, since
  `libnccl-tuner.so` is not present in the image.
- The refused eight-GPU board reported 126 active links across eight GPUs
  because GPU 4 had none: `NVML: Unable to retrieve Nvlink information as all
  links are inActive`, and `SYS` rather than `NV18` on all fourteen of its
  ordered pairs. The benchmark's peer-access check refused it before any timed
  lane; the stage script's precheck has since been tightened to catch the same
  board itself.
- RDMA, for the PLACE-6 NIC clause: the record host lists two InfiniBand class
  devices in the kernel's InfiniBand class directory, `ibp115s0f0` and
  `ibp116s0f0`, while machine 56359 listed fourteen `mlx5` devices. On both
  `ibdev2netdev` is absent from the image and `ibv_devinfo` reports no device,
  and NCCL logs `NET/IB : No device found` before falling back to the socket
  network over `eth0`. No GPU Direct RDMA path was available to this study on
  either host.
- The container sees no NVSwitch: no PCI device of class `0x0680` is visible,
  which is the same limit the eight-GPU topology fixture records.
