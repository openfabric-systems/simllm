# VLLM-40 clean load-delay qualification

Status: clean repetition complete; frozen direction and held-out claims refuted.

## Access ledger

- Clean ledger: `vllm40_access_ledger.jsonl`, SHA-256
  `8a61a6e0b58a213259a19a593b8b3f4ec08cea6a7c854f5481ccbd7bc2dc5914`.
- Committed field reader Git blob:
  `a417e941cfd502d7dab28479b1ea93a57c166ec2`; it was reused without edits.
- Five accesses passed: acceptance status, campaign ID, device kind ID, and
  exactly the Granite CUDA-graph decode batch-1 and batch-8 entries.
- The reader stopped at byte 45,043; `whole_record_loaded` is false on every
  ledger row.
- No DeepSeek row and no Granite batch-32 row was decoded or captured.
- The clean projection is byte-addressed under the external VLLM-40 root and
  has SHA-256
  `08be3c72490e4a63cadb6227dd512ebd07fcca569c1350db241dbb391088ad0a`.
  Its provenance and both canonical key hashes match the frozen surface
  exactly.

| Batch | Service (ps) | CV (ppm) | Replays | Evidence | Key SHA-256 |
|---:|---:|---:|---:|---|---|
| 1 | 1,110,576,000 | 4,232 | 300 | MEASURED calibration | `d8fdebd7051f530bed3232cfeb2e3d5c87a1ab9a22fe68db9cca51436853a502` |
| 8 | 1,892,831,500 | 1,538 | 300 | MEASURED calibration | `38978457b27e56dbc0e9ffbe2385b7d53538a2515c826284be4d6d414520c830` |

## Claim verdict

- Frozen qualification status: **REFUTED**. Only 16 of 30 signed segment
  expectations matched, and only 1 of 24 held-out points was inside its frozen
  inclusive band.
- Monotonic-delay claim: **VALIDATED**. All 30 observed adjacent segments
  increased; none decreased or stayed flat.
- Observed mechanism: scheduler queue wait dominates every segment in the
  frozen 250 to 8,000 requests/s ladder.
- No band was widened and no signed expectation was changed after observation.

The clean repetition therefore refutes the frozen batch-amortization direction
pattern and its quantitative held-out model while validating monotonic increase
over the measured ladder.

## Exact segment directions

Each cell is `expected -> observed (verdict)`.

| Configuration | 250 to 500 | 500 to 1K | 1K to 2K | 2K to 4K | 4K to 8K |
|---|---|---|---|---|---|
| `(1,1,8)` | decrease -> increase (refuted) | decrease -> increase (refuted) | increase -> increase (held) | increase -> increase (held) | increase -> increase (held) |
| `(1,1,16)` | decrease -> increase (refuted) | decrease -> increase (refuted) | increase -> increase (held) | increase -> increase (held) | increase -> increase (held) |
| `(1,2,8)` | decrease -> increase (refuted) | decrease -> increase (refuted) | decrease -> increase (refuted) | increase -> increase (held) | increase -> increase (held) |
| `(1,2,16)` | decrease -> increase (refuted) | decrease -> increase (refuted) | decrease -> increase (refuted) | increase -> increase (held) | increase -> increase (held) |
| `(2,1,8)` | decrease -> increase (refuted) | decrease -> increase (refuted) | increase -> increase (held) | increase -> increase (held) | increase -> increase (held) |
| `(2,1,16)` | decrease -> increase (refuted) | decrease -> increase (refuted) | increase -> increase (held) | increase -> increase (held) | increase -> increase (held) |

Exact signed movement fractions in picoseconds are retained in
`vllm40_results.json` for all 30 rows.

## Held-out band comparison

All values below are exact terminating decimals in milliseconds. Inclusive
comparison used the frozen rational picosecond bounds without rounding.

| Configuration | Load | Frozen prediction | Frozen band | Observed | Verdict |
|---|---:|---:|---:|---:|---|
| `(1,1,16)` | 250 | 0.7170281675 | [0.555509658, 0.878546677] | 1.6363500625 | REFUTED |
| `(1,1,16)` | 500 | 0.44988802525 | [0.3284405370875, 0.5713355134125] | 2.14155334375 | REFUTED |
| `(1,1,16)` | 1,000 | 0.2903379375 | [0.1928229625, 0.3878529125] | 2.815674625 | REFUTED |
| `(1,1,16)` | 2,000 | 3.80586196875 | [2.8294659859375, 4.7822579515625] | 6.20072021875 | REFUTED |
| `(1,1,16)` | 4,000 | 5.77461196875 | [4.3060284859375, 7.2431954515625] | 7.948210375 | REFUTED |
| `(1,1,16)` | 8,000 | 6.75898696875 | [5.0443097359375, 8.4736642015625] | 8.831092625 | REFUTED |
| `(1,2,16)` | 250 | 1.16431 | [0.935699215625, 1.392920784375] | 2.710309375 | REFUTED |
| `(1,2,16)` | 500 | 0.7170281675 | [0.555509658, 0.878546677] | 3.60047828125 | REFUTED |
| `(1,2,16)` | 1,000 | 0.44988802525 | [0.3284405370875, 0.5713355134125] | 4.55817784375 | REFUTED |
| `(1,2,16)` | 2,000 | 0.2903379375 | [0.1928229625, 0.3878529125] | 7.41943084375 | REFUTED |
| `(1,2,16)` | 4,000 | 2.048099953125 | [1.51114447421875, 2.58505543203125] | 8.99011721875 | REFUTED |
| `(1,2,16)` | 8,000 | 3.032474953125 | [2.24942572421875, 3.81552418203125] | 9.96127634375 | REFUTED |
| `(2,1,8)` | 250 | 0.7121501675 | [0.551119458, 0.873180877] | 1.6224960625 | REFUTED |
| `(2,1,8)` | 500 | 0.44501002525 | [0.3240503370875, 0.5659697134125] | 2.1168705625 | REFUTED |
| `(2,1,8)` | 1,000 | 0.2854599375 | [0.1884327625, 0.3824871125] | 2.9526630625 | REFUTED |
| `(2,1,8)` | 2,000 | 3.80098396875 | [2.8250757859375, 4.7768921515625] | 6.07474140625 | REFUTED |
| `(2,1,8)` | 4,000 | 5.76973396875 | [4.3016382859375, 7.2378296515625] | 7.66773925 | REFUTED |
| `(2,1,8)` | 8,000 | 6.75410896875 | [5.0399195359375, 8.4682984015625] | 8.42115734375 | HELD |
| `(2,1,16)` | 250 | 0.7170281675 | [0.555509658, 0.878546677] | 1.6363500625 | REFUTED |
| `(2,1,16)` | 500 | 0.44988802525 | [0.3284405370875, 0.5713355134125] | 2.14155334375 | REFUTED |
| `(2,1,16)` | 1,000 | 0.2903379375 | [0.1928229625, 0.3878529125] | 2.99258078125 | REFUTED |
| `(2,1,16)` | 2,000 | 3.80586196875 | [2.8294659859375, 4.7822579515625] | 6.3108776875 | REFUTED |
| `(2,1,16)` | 4,000 | 5.77461196875 | [4.3060284859375, 7.2431954515625] | 7.96733275 | REFUTED |
| `(2,1,16)` | 8,000 | 6.75898696875 | [5.0443097359375, 8.4736642015625] | 8.72828853125 | REFUTED |

## Conservation and batching

- All 36 exact conservation rows held.
- Admissions, handoffs and terminals: 2,304 / 2,304 / 2,304.
- Terminal decode tokens: 9,216; maximum TTFT decomposition residual: 0 ps.
- CORE-51 one-request control held exactly, and all pool-local prefill and
  decode identities remained unique.
- Genuine batching was observed in both roles: maximum prefill batch 8 in 26
  cells, and maximum decode batch 8 in 36 cells.

## Frozen identities and preservation

- Freeze commit: `121345e950b12a36018404084c7dcf9bd507f962`.
- Expectations SHA-256: `28cee81deffe771836b5c38d7fe605185f4dc31a953087c80288ceb7a3a84e22`.
- Surface SHA-256: `26fc547d8b47ccec7108872e05fbedfe71ebb6229b88799ca254089d3f2b6e9d`.
- The unchanged sweep contained 36 cells, 64 requests per cell, four decode
  tokens per request, six loads, two prompt lengths and three pool ratios.

| Preservation class | Files | Bytes | Manifest SHA-256 |
|---|---:|---:|---|
| CORE-51 one-request control | 6 | 61,248 | `092d79c35c7632e87427804cda11bc6fe0890d2c98ceec23ff30af2e1143ad4d` |
| Deterministic concurrent comparator | 9 | 56,495 | `d09202846afeba9efc019d4f44f881fde6639457c0398f1ed4ae8da1e0c804c3` |
| Scored flagship artifacts | 17 | 1,198,680 | `7630ebdaf91a722ff5004184a03a38fac98bbf11f2adbbfd5e8e32838ff130d5` |

## Run evidence

- Scored HEAD: `dea94a2a3a6a26e2ce8975735741ecd68db52eed`.
- Raw result:
  `/data3/yifeng/simllm-dev/wave-runs/vllm40/qualified-sweep/result.json`,
  7,169,930 bytes, SHA-256
  `5cee648ebb0d401426f5e55884e3ba2e8b0ce0562719f09256d32fda2450f238`.
- Runner exit code: 0; worktree Python 3.10.18; vLLM 0.27.1.
- Tracked compact result SHA-256: `2e70e4e5c01427d97afd74412a519847e87e5fe3a88163a69998a407f790ce0e`.
- An initial infrastructure-only launch under the external `clean-sweep`
  directory stopped before any observation because PyTorch was unavailable to
  that interpreter. It was retained, not scored and not deleted. The qualified
  run used the worktree Python 3.10 interpreter with the installed Python 3.10
  vLLM environment on `PYTHONPATH`.
- The complete run was offline; no model weights or other network content were
  downloaded.

## Registry movement

- **VLLM-40 closed**: the fresh field-addressed repetition is structurally
  clean and the refutation is published literally.
- **VLLM-39 closed**: its required clean repetition completed without a band
  change or signed-expectation change.
- **VLLM-35 closed**: the clean run conserves every request and token and
  demonstrates genuine prefill and decode batching while preserving CORE-51
  and the deterministic comparator.
- **VLLM-41 unchanged and open**: the ladder begins at 250 requests/s, so this
  run contains no sub-250 evidence and makes no onset claim below 250.
- VLLM-42 remains reserved and untouched.
