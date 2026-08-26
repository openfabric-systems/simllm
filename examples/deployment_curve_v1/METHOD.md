# Deployment-curve scaffold method

## Configuration and curve records

The harness consumes one or more declared configurations. Each names its
framework, prefill and decode node counts, GPUs per node, role-local tensor,
expert, data and pipeline arrangement, model column, pricing mode and request
sweep. A supported run constructs one disaggregated session per configuration.
The framework scheduler remains the batching authority and the shared virtual
clock remains the timing authority.

Every configuration emits one `simllm-deployment-curve-v1` record, the schema
established by `pd_session_concurrent_v1`. Each point retains exact numerator
and denominator pairs. Study metadata extends the record without redefining
the axes. Multiple records are independent plot legends.

The scaffold supports the vLLM bootstrap path used by the granite dry run.
The scored SGLang, DeepSeek lookup-pricing and target-scale paths remain owned
by CORE-54's registered dependencies. An unsupported framework or pricing mode
is rejected before construction.

## Tuned constants

Each tunable constant has one selected value, a closed physical envelope and a
provenance record naming its source, locator and physical argument. A selected
or fitted value outside that envelope is refused. A fitted value is always
labeled as a fitted parameter, never as a measurement.

The fit API accepts independent linear projections of the form
`prediction = baseline + sensitivity * constant`. It builds its anchor map
only from the frozen calibration IDs. Relative residuals are reduced by least
squares when a constant has more than one calibration row. A fit row naming a
held-out or context-only anchor fails before any value is read.

The scoring API builds a separate map containing only the held-out IDs. It
requires predictions for exactly that set and refuses calibration or
context-only IDs. A point records its relative error and its propagated
interval. The frozen pass rule is intersection with the closed 5 percent band
around the disclosure.

The granite dry run does not fit or score. It exercises the session, records,
uncertainty and renderer without presenting bootstrap pricing as DeepSeek
evidence.

## Error propagation

The curve band uses deterministic interval arithmetic, not a confidence
interval. The method is `deterministic-additive-interval-v1`:

1. The curve record's lower and upper bounds establish the initial throughput
   and delay intervals.
2. Each distribution spread contributes a symmetric absolute interval formed
   from its declared relative half-width at the point estimate.
3. Each tuned-constant envelope is projected through declared throughput and
   delay sensitivities. Both envelope endpoints are evaluated, which also
   handles a negative sensitivity.
4. The independent contribution intervals are added by a Minkowski sum. This
   is conservative: it retains the simultaneous worst endpoint rather than
   assuming cancellation.
5. Inverse delay is transformed only after the delay interval is complete.
   Its lower endpoint is one divided by the delay upper endpoint, and its upper
   endpoint is one divided by the delay lower endpoint.

Every point records each source contribution and the final exact bounds. A
bound that reaches zero is refused because neither plotted axis admits a
nonpositive physical value. Synthetic tests exercise nonzero record bounds,
distribution spreads and constant-envelope effects together, including the
endpoint reversal under inversion.

## Figure contract

The renderer uses a 7 inch two-column canvas and produces PDF plus PNG from the
same machine-readable result. Simulated configurations use the established
blue, orange and green palette (`#2a78d6`, `#eb6834`, `#1baf7a`). Vertical
bands show inverse-delay uncertainty and horizontal whiskers show throughput
uncertainty.

The first legend identifies simulated configurations. A second legend
identifies disclosure context. The SGLang decode markers combine each exact
aggregated decode-throughput row with the disclosure's approximate 100
millisecond headline latency and say so explicitly. They are context markers,
not paired held-out score rows. DeepSeek's H800 decode average is a vertical
reference because the disclosure gives no matching per-token delay; the plot
does not invent one.

Both axes increase toward the upper-right optimal corner. A dry-run result
receives a visible `DRY RUN` label in the title and plot body.
The inverse-delay axis is logarithmic so the small-model granite dry run does
not collapse the approximately 100 millisecond disclosure context onto the
axis border. This changes presentation only; the record and scoring quantities
remain the exact untransformed values described above.
