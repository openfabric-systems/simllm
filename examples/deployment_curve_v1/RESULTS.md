# Deployment-curve scaffold dry-run result

## What ran

The post-specified granite dry-run regression at commit `968b04a` drove two
independent vLLM 0.27.1 one-prefill plus one-decode sessions, each with eight
simulated workers per node, through offered loads of 8,000, 16,000 and 32,000
requests per second. One configuration used an 8-token prompt and the other
used a 16-token prompt. Both used the `granite-roofline-bootstrap` model column,
bootstrap pricing and the declared 100,000,000 ps KV-handoff constant.

The result and figure are external bulk artifacts:

- `$SIMLLM_CORE54_RUN_ROOT/granite-one-plus-one-bootstrap-v2/result.json`
- `$SIMLLM_CORE54_RUN_ROOT/granite-one-plus-one-bootstrap-v2/flagship-dry-run.pdf`
- `$SIMLLM_CORE54_RUN_ROOT/granite-one-plus-one-bootstrap-v2/flagship-dry-run.png`

## What came out

The scaffold emitted two `simllm-deployment-curve-v1` records and rendered them
as separate legends with exact record fractions and propagated interval bands.
The scope-deciding number is zero: the run read zero held-out disclosure
anchors. Its constant fit is `NOT_RUN`, its held-out score is `NOT_SCORED`, and
the figure is visibly labeled `DRY RUN`.

All six curve points conserved 8 admissions, 8 terminals and 32 output tokens.
The 8-token curve rose from 19,665.007 to 32,270.817 output tokens per second as
load increased. The 16-token curve rose from 19,848.260 to 27,844.489. The
16-token prompt increased per-token request delay by 14,135,250, 36,456,000 and
39,396,750 ps at the three matched loads. Its configured KV geometry doubles
handoff bytes exactly from 393,216 to 786,432.

The inverse-delay point estimates span 4,236.606 to 5,093.984 tokens per second
per request. These are granite bootstrap outputs, not DeepSeek measurements or
predictions. The dry-run bands combine the 50 percent bootstrap record bound
with the KV-constant envelope. The reusable interval engine also accepts
distribution spreads; its synthetic test exercises nonzero record,
distribution and constant contributions together and checks exact endpoint
reversal when delay is inverted.

## Physical sanity

Compute and memory first: 400 million active bfloat16 parameters are about 800
million bytes. Perfect eight-way sharding gives 100 million bytes per rank. At
the declared 8 trillion bytes per second per-rank B100 bandwidth, a complete
weight read cannot take less than 12,500,000 ps, a memory-only ceiling of 80,000
tokens per second per request. The observed 196,310,000 to 236,038,000 ps
per-token delays are 15.7 to 18.9 times that floor, and the corresponding
4,236.606 to 5,093.984 token rates remain inside the frozen 10 to 100,000 broad
session bound.

Network next: the 393,216-byte and 786,432-byte handoffs cannot serialize over
eight 400 Gbit/s links in less than 983,040 and 1,966,080 ps. The selected
100,000,000 ps handoff is above both wire floors and inside the declared
20,983,040 to 101,457,280 ps scaffold envelope. This is a bounded declared
constant, not a calibrated transfer measurement.

End to end last: the disclosure context is approximately 100 ms per output
token, or about 10 tokens per second per request. Granite's roughly 0.2 ms dry
run is hundreds of times faster because it is a different, much smaller model
under bootstrap pricing. The logarithmic vertical axis makes that mismatch
visible rather than suggesting an equivalent-system comparison.

## What it changes for the project

The CORE-54 anchor freeze is literal at commit `629fc7b` with SHA-256
`b1a918ed02329a242d033943fb18b93fd9be8fdaa18093477e6abb8298540df5`.
The project now has the machine-checkable disclosure split, bounded calibration
and held-out-only scoring machinery, interval propagation, reusable multi-curve
records and publication-sized plotting needed to conduct the later flagship
campaign. No task closes, opens or becomes blocked. CORE-54 stays open on its
existing registered dependencies, and no residual task is added.

## What it does not change

This result does not run SGLang, the 96-GPU target, the 448-rank what-if, the
DeepSeek lookup column or packetized KV traffic. It fits no constant, scores no
held-out anchor, establishes no pricing-derived flagship band and makes no
claim about DeepSeek accuracy. It does not close CORE-54 or any dependency, and
it does not make the flagship milestone literal.

## Artifact identities

| Artifact | SHA-256 |
|---|---|
| anchor freeze JSON | `b1a918ed02329a242d033943fb18b93fd9be8fdaa18093477e6abb8298540df5` |
| dry-run result JSON | `1b9fb5ca1d565cc1f397ad00088fbac06758a67f86cebde6cff026c4d358e54b` |
| dry-run PDF | `046ce5407ad028f732d1ec4cd1ea04c4b33c364dee97f2222e9aac2f96619783` |
| dry-run PNG | `a97368bcfbf9e64efe532a36bdf6320daefe39b7336582acd5036df5bf344741` |
