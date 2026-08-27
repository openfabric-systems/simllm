# CORE-54 fourth scored flagship expectations

Status: **EXPECTATIONS_ONLY**. This freeze follows the committed field reader,
source allowlist and pre-arithmetic doctrine. It precedes the run-4 scorer,
the one permitted MTP held-out read, the score, the result and both figures.

## MTP per-layer arithmetic

The successor evidence projection supplies a measured four-layer step service
of 2,033,951,000 ps for the exact per-GPU batch-16, KV-4000 MTP decode shape.
The disclosure's batch 128 per node maps to 16 requests on each of eight GPUs.
Pinned SGLang source makes `accept_lens` include the bonus token while correct
draft count subtracts one. The recovered `generation_16(32)` boundary is the
same accounting: one ordinary token plus one simulated speculative token per
request, or 256 emitted tokens per node per step.

The inherited linear depth doctrine gives

`2,033,951,000 ps * 61 / 4 = 31,017,752,750 ps`.

Therefore every applicable layer is frozen at

`256 * 10^12 / 31,017,752,750`

`= 1,024,000,000,000 / 124,071,011`

`= 8,253.338082 tokens/s/node`.

The physics-only, physics-plus-boundary and
physics-plus-boundary-plus-attenuation bands are all the same zero-width exact
fraction. Decode has no inherited prefill overlap boundary. The successor
retains two independent observations, partially unlocking COMP-74, but no
validated MTP distribution is propagated in this run, so a wider component
band would be invented.

## Simulated-MTP definition and caveat

The source accounting is pinned to SGLang commit
`bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3`, specifically
`eagle_worker_common.py` lines 584-650 and `eagle_worker_v2.py` lines 899-920.
The run prices the disclosure's simulated one-speculative-token step. It does
not claim realized rejection, replay or production acceptance distributions.
The disclosure states that MTP integration with data-parallel attention is
incomplete; that caveat remains part of the verdict scope.

## Depth and attenuation rulings

The 61-over-4 service extrapolation is the same doctrine as the standard
decode cell. CORE-61's depth-linearity question remains open. No depth fit,
depth correction or attenuation is permitted.

The run-3 factor was derived for EP32 prefill communication incidence and is
not reused. The fresh EP72 architecture has 40 ranks with four logical experts
and 32 ranks with three logical experts. Its uniform-routing destination
incidence could be written as

`[40 * (1 - C(252,8)/C(256,8)) + 32 * (1 - C(253,8)/C(256,8))] / 8`.

That ratio is not admissible here. The successor measures total MTP step
service and carries only a disclosed component overlay; no independent
coefficient maps destination incidence to total elapsed service. Applying the
incidence ratio to the complete step would misstate disclosed attribution as
measured latency. The incomplete integration and depth questions are modeling
residuals, which policy rule five forbids attenuating. Admitted decode factor
count is therefore zero.

## Fit, access and scoring boundary

The run inherits the serialized run-3 calibration-only fit at
`78a798178234932325381aa7328ebd0dc816400e5a9caa3d6e5577edd0724883`.
There is no refit and no MTP parameter. The 2K and 4K PASS rows remain under
the run-3 authority byte for byte and are never rescored.

The field reader was committed before the successor evidence read. That read
is logged externally and returned only the declared MTP projection. The MTP
anchor numeric row has not been read. After this freeze is committed, attempt
1 may read that row exactly once, score the frozen point against the 5 percent
bar and publish the result without widening or adjustment. The initial broad
search chronology breach remains disclosed.

All 57 named prior artifacts hash byte-identically, including every first,
second and third scored-run record and figure, the successor campaign payloads
and the frozen deployment-frontier publication.
