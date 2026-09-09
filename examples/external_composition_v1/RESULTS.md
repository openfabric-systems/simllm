# External serving composition result

COMP-88 qualifies: the corrected installed composition preserves all 13
historical service values and pass totals bit for bit. Two fresh Python
processes ran the unchanged frozen experiment over memory parameters, serving
pool counts and rates, and the external first-token heuristic. Their complete
evaluation records are identical. All 112 behavioral instances across nine
families pass, with no fatal findings.

This closes COMP-88's immutable composition boundary. Eight externally
specified adjustments now belong to one source-joined record; service keys,
estimator stamps and serving projections carry that identity. A changed
factor requires a derived record. Public prefill and decode calls apply the
record's phase factor when omitted and reject an independently conflicting
factor. Cached models reject changed source, shape or factor state before
serving another value. The aggregate path applies its three operation factors
and preserves its existing accumulation order.

This does not establish new GPU calibration or large-model validity. The
source operation database already contains empirical terms and analytical
floors. The new record preserves those terms and their provenance. It does
not fit them, repair the historical frontier mismatch or turn an external
first-token heuristic into a simulated request latency. DEPLOY-13 retains
that comparison's residual, and COMP-54 retains K3 qualification work.

The compact [results record](results.json) carries identities, evidence classes
and the retained chronology. The prospective freeze is
`483b31a0d98ec1c4f5fd0668026055e84e363456`, which precedes implementation and
the first run. The accepted source is
`e968f9f0b484ab2eca645f1fddb822ba11685820`.
The promoted baseline record is
[`6f1dccd1444fdd947b978551218b2a46b13842436222fe11df0b79f0aae438f3`](../../offline/calibration/external-compositions/6f1dccd1444fdd947b978551218b2a46b13842436222fe11df0b79f0aae438f3.json).

The first attempt is **VOID by review**, with a null closure score. Its raw
harness reported PASS, but a missed frozen admission guard allowed public
pass calls to apply factor 2.0 while stamping a record whose phase factors
were 1.1 and 1.08. Omitted factors also used the legacy 1.0 default. A retained
post-specified direct-call audit reproduced the defect in both phases against
that source. All original files remain unchanged. The corrected successor
adds the missing direct-entry controls and repeats the same expectation
commit without changing any value or acceptance band. No score from the void
attempt contributes to closure.

Evidence classes remain separate:

| Evidence class | Corrected outcome |
|---|---|
| Historical service and pass-total oracles | 13 exact binary64 matches |
| Historical decode quotient oracles | 10 exact rational matches |
| Active remove-one regression oracles | 30 exact matches |
| Complete first-token decomposition oracle | One exact match, including signed publication residual |
| Prospective behavioral relations | 112 passing instances in nine families |
| Fatal admission, identity and structural guards | All held; 715 guards, unscored |
| Inactive remove-one disclosures | 50 retained identity rows, unscored |
| Process determinism | Two complete evaluations have identical bytes |

The numerical configuration grids contain 52 memory cells, 16 capacity cells
and six autoscale cells. The aggregate regression retains 25 baseline points,
125 explicit legacy interventions and three active operation controls.
Neither configuration counts nor software tests enter the behavioral total.
All 58 protected historical files remain byte-identical. GPU-runtime imports,
backend subprocesses and SimLLM roofline calls are all absent.

The physical checks precede the relation checks. Each rank must stream at
least its share of 25,165,824,000 bytes of feed-forward weights. At the declared
4.8 TB/s memory rate, tensor widths two, four and eight have conservative
memory floors of 2.62144, 1.31072 and 0.65536 ms. Multiplying each weight by the
active token count also gives an independent compute floor at 1.978
petaflop/s. These floors omit attention and other work. Every cell exceeds
the greater floor. The smallest margin is 3.665697602 ms in decode and
66.892998083 ms in prefill.

The specified intervention has a separate ceiling: no positive term can
exceed twice its accepted baseline because only the inverse memory-bandwidth
factor can double, and the fixed operation constant never exceeds baseline.
Every cell stays below that ceiling, with at least 4.929970242 ms margin in
decode and 74.951159698 ms in prefill. This is a bound on the declared
composition, not a universal upper bound on hardware latency.

Before execution, operation counts predict that adding 3 microseconds per
memory visit adds 0.62532 ms to decode and 3.42474 ms to prefill. The latter
includes the attention normalization, position and cache-write visits.
Halving effective memory bandwidth changes only the independently counted
byte term. For example, the batch-128, tensor-width-two decode increment is
0.36605952 ms. The largest residual over all floating-point behavioral checks
is 1.288e-14 ms, below the frozen 1e-9 ms allowance. Comparisons hold the other
factor fixed; simultaneously reducing a constant and reducing bandwidth need
not increase the total service.

Pool capacity provides another independent check. Doubling a pool's engine
count or rate factor doubles that pool's exact rational capacity. Served
capacity is the lesser of the prefill and decode capacities, so gains stop
when the other pool limits service. Both limiter states occur. The 22 active
clipping instances match exactly; inactive zero increments remain unscored.
Record identity reaches the existing estimator stamps and the final capacity
projection without creating another timing owner.

The original first-token decomposition remains explicit:

| Component | Milliseconds |
|---|---:|
| Raw prefill service | 99.20380474486889 |
| Phase-corrected prefill service | 109.1241852193558 |
| External autoscaled first-token heuristic | 196.42353339484043 |
| Published first-token value | 196.423 |
| Signed publication reconciliation | -0.000533394840431356 |

The reconciliation exceeds half of the last published decimal unit. Its
retention is an exact regression result, not evidence that nearest
three-decimal rounding alone explains the difference. Handoff latency stays
outside the external autoscale operator. The original 0.607495219355 frontier
quotient is also preserved and remains DEPLOY-13's separate issue.

To repeat the study, select a new output directory outside the repository:

```bash
python -m examples.external_composition_v1.run_study \
  --output-root "$SIMLLM_COMP88_EVIDENCE_ROOT"
```

The driver requires a clean committed source tree, verifies freeze ancestry,
retains both workers' outputs and process metadata, and rejects source changes
during execution. The corrected evaluation digest is
`dc48c2fec3670e32e398f40207eef710eac78f9f209601daa20d98d148e58893`;
the summary digest is
`4a08dcb0330877da3d9545625c0ad0be685a0aa2067fb6409b5c7afb3157b4bc`.
