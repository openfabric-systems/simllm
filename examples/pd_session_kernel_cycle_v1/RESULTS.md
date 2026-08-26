# Disaggregated session kernel-cycle binding result

## What ran

The frozen `pd_session_kernel_cycle_v1` study ran two independent record-absent
roofline sessions and one explicit candidate-record session through the pinned
vLLM 0.27.1 prefill and decode engines over the four prompt-length and handoff
cells.

## What came out

The run is **void**. The candidate mechanism selected the exact frozen decode
row twice and produced the exact predicted movement, but the fatal complete
request-result byte-identity guard failed in all four record-absent cells. A
fatal guard makes the behavioral score uninterpretable, so the matching
movement rows are retained as findings and are not reported as a pass
fraction.

The selected candidate row moved only the first client-visible decode step
after a 16-token prompt. Its represented service changed from 75,288,000 ps
under the roofline comparator to 2,047,488,000 ps, a signed change of
+1,972,200,000 ps. The exact end-to-end findings are:

| Prompt | Handoff (ps) | Roofline TTFT (ps) | Lookup TTFT (ps) | Signed TTFT (ps) | Roofline TPOT (ps) | Lookup TPOT (ps) | Signed TPOT (ps) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 100,000,000 | 273,376,000 | 273,376,000 | 0 | 77,952,000 | 77,952,000 | 0 |
| 8 | 200,000,000 | 373,376,000 | 373,376,000 | 0 | 77,952,000 | 77,952,000 | 0 |
| 16 | 100,000,000 | 292,912,000 | 2,265,112,000 | +1,972,200,000 | 77,976,000 | 77,976,000 | 0 |
| 16 | 200,000,000 | 392,912,000 | 2,365,112,000 | +1,972,200,000 | 77,976,000 | 77,976,000 | 0 |

The record provenance remained explicit. Its SHA-256 is
`e495f3ca5d0858cf371b19205ae6b7747d633695020d10f58645c5f245086070`,
its status is `candidate`, its device is A100 SM80 and its coverage is
`partial-kernel-subset`. The decode provider recorded two exact hits and 14
roofline misses; the prefill provider recorded zero hits and four roofline
misses. `calibration_claim` remained false.

## Off-path identity finding

Both record-absent observations reproduced all four accepted compact cells,
including KV bytes and every timestamp. The 8-token cells carried 393,216
bytes and the 16-token cells carried 786,432 bytes in both observations. The
lookup provenance member was absent on both arms.

The complete request-result serializations were not byte-identical. Across all
four cells, the only differing fields were `prefill_internal_request_id` and
`decode_internal_request_id`. vLLM creates a fresh random suffix for each
pool-local identifier. Every other field was equal. This exact difference set
was established after the void and is therefore a post-specified diagnostic,
not a repair to the frozen guard. The per-cell byte digests are retained in
the machine-readable result.

## Physical sanity

The bounds were stated before the run. The candidate partial kernel service
had to be positive and no greater than its enclosing measured decode step of
2,323,678,000 ps. The observed 2,047,488,000 ps is inside that interval and
276,190,000 ps below the ceiling. Live nonempty step service ranged from
77,952,000 ps to 2,050,176,000 ps, inside the frozen 1,000,000 ps through
100,000,000,000 ps interval.

The network check is independent of compute pricing. At 400 Gbit/s, 393,216
bytes require at least 7,864,320 ps of serialization and 786,432 bytes require
at least 15,728,640 ps. The declared 100,000,000 ps and 200,000,000 ps
handoffs exceed those floors, and doubling the handoff added exactly
100,000,000 ps to TTFT in both lookup prompt rows while changing TPOT by zero.

The end-to-end cadence remained 12,824.46 through 12,828.41 client-visible
tokens per second, inside the frozen 10 through 100,000 range. These checks
reject impossible values but do not calibrate the A100 partial record to the
B100 session.

## What it changes for the project

CORE-53 stays open. CORE-58 is registered to freeze and validate an exact
record-absent comparison boundary that retains every pricing-relevant and
client-visible byte and timestamp while handling vLLM-owned opaque pool-local
identifiers without retrospectively passing this run. COMP-73 is registered
to produce a key-compatible target record covering every prefill and decode
shape in the frozen session grid. The next CORE-53 acceptance run depends on
both tasks.

## What it does not change

The content-addressed lookup binding and exact comparator fallback are live,
but this void run closes no task and moves no milestone. It does not close
COMP-64, claim a B100 or Hopper calibration, validate the candidate's routing
distribution, or claim complete prefill or decode coverage. The accepted
`pd_session_v1` files and their compact values did not change.

## Reproduction and chronology

Expectations were frozen in commit
`fda6eed557aef037bf1794da1c1d8556a10a1ee0`, before implementation commit
`6817019376d153be2a4b6cdd972bbec36dfa23e6` and harness commit
`2e87c766981bf33f386343b06c99334874cc6399`. The retained bulk result has
SHA-256
`d9d62a64e3ba61e9909f9bfebdb06202eb09d17c8f50bfb9abd69d75430f5196`.

Run from the repository root with machine-local values supplied through
environment variables:

```bash
HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH="$PWD" \
"${SIMLLM_VLLM_PYTHON:?configure SIMLLM_VLLM_PYTHON}" \
examples/pd_session_kernel_cycle_v1/run_study.py \
  --run-dir "${SIMLLM_RUN_ROOT:?configure SIMLLM_RUN_ROOT}/c53bind/candidate-v1" \
  --vllm-source "${SIMLLM_VLLM_SOURCE:?configure SIMLLM_VLLM_SOURCE}" \
  --model-config "${SIMLLM_MODEL_CONFIG:?configure SIMLLM_MODEL_CONFIG}" \
  --vllm-python "${SIMLLM_VLLM_PYTHON}"
```
