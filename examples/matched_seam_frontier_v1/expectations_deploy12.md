# Matched-seam LogGOPSim third-arm expectations

This expectations-only freeze defines the DEPLOY-12 third arm before its
implementation or first execution. It does not amend `expectations.md` or
`expectations_v2.md`, widen any original band, or rescore either existing
publication. The first publication remains void under its own FG-1. The
corrected publication remains the nonvoid two-arm record whose maximum
packet-priced to unpriced-network capacity-step quotient is
1.042715399805 on rows 1 and 3.

Nobody knows the third-arm result at freeze time. This contract fixes the
schedule, parameters, arithmetic, identities, classifications, and reporting
rules. A positive residual and a zero or negative residual are both admissible
outcomes. The run publishes whichever the native executions produce.

## Protected prior evidence

The third-arm runner verifies this complete prior publication set before and
after each fresh-process evaluation. A mismatch is fatal and voids the new
run. These bytes are never rewritten by this wave.

| Path | SHA-256 |
|---|---|
| `RESULTS.md` | `fa1170277fa8f3b9f1a14df353add3dbd4e8e490aeb4847748dd2120d4434e62` |
| `expectations.md` | `fc5af307fee560fc7050011543e18e1cf77030d0aa6a13e6c5a014cb159a5726` |
| `expectations_v2.md` | `fe403500575d674a25c8b7c6c59eb41aec65fce6cc29024609fa995b29585f35` |
| `external_adjustments.json` | `c6778a81cdc6078ce74f06733e4bce9d99a92b4ab3eccba4a83d14e7d063a09e` |
| `figure_addendum.md` | `cc4dcb8c82bbcd5e542457b56d91ddf172af2cbe05e6bac5c865535dcc307762` |
| `plot_publication.py` | `a98514cb985a9980a679357285a11dbe52418e774a55d69a6c9f30ba9ddda53d` |
| `plot_study.py` | `d4fe430f1fede23bcbcbb21834d98a51d3563c4b4e4c21dc887c7b8c837a7e4f` |
| `record.json` | `bddd7cb040a3c0f0ec8afd7ea836d873fa22cad2131f98ff36e38da5441b2d50` |
| `results.csv` | `4113ab2413084b7da957de60002abc4a4f8530bbb89837a5a5f73b9852f4448d` |
| `run_study.py` | `242b5f1ae46ac18ac2cb474ad6fa24acc4dba21c4b8ff1d6683137163fec3182` |
| `study_config.json` | `64c8e16de53e194e98f5ca7c9b27d533d4c7f7ca32311841a62e3c6cece21f17` |
| `figures/matched-seam-frontier-publication.pdf` | `511a0fb869d3397a87664d28c6b0c1d5adc17738dd84543973f66c7fcfd764cb` |
| `figures/matched-seam-frontier-publication.png` | `d79b5099cbbfeed9e4272a64d7007512ed1889a08fc3438c9f2eef41a28986d1` |
| `figures/matched-seam-frontier.pdf` | `4ecc3bf2822f916bfd53107b55d1344406efea01fd0b1ad7a417019391712dbb` |
| `figures/matched-seam-frontier.png` | `852378a01d3c9e0aeab74423259afe86b456dca0b193e27c23e48256322069c4` |

## Frozen arm and parameter source

The arm executes the pinned goalsim binary directly through
`simllm.backends.run_loggopsim`. It does not use `LogGopsimStepSink`. The
direct subprocess path consumes the exact binary GOAL already rendered for
the matching `rnic-nn` packet cell. The step-sink path would lower a new
`StepRecord` and would therefore fail the exact same-schedule comparison that
DEPLOY-12 needs.

The goalsim executable is pinned to SHA-256
`7e0f13ee3c87a20e9d2e94dbbd74c46075fd03df2f1b04d1ed9739c43ee0a2bf`.
This is the audited binary in
`examples/loggopsim_ideal_v1/expectations.md`, whose SHA-256 is
`934ee355e4d5a376d1eecdb1d0e62f6e4f7acfd9ada93def5ba1bcf8fa8508ff`.
The parameter set is the P-2L envelope declared in
`examples/frontier_ladder_v1/expectations.md`, whose SHA-256 is
`e3e83264df6e72e83736a06dddcba11a501c75a25c8c1fb0a9c7b1e9c0caeea3`:

- `L = 2000` ns, the repository's declared 2.000 microsecond propagation
  reference.
- `o = 0` ns, `g = 0` ns, and `O = 0` ns/byte.
- `G = "0.02"` ns/byte, passed with that exact decimal spelling and derived
  from 400,000,000,000 bit/s.
- `S = 9223372036854775807` bytes, so every frozen payload is eager.
- Network type `LogGP`.

With payload `s` in bytes, the audited single-send arithmetic is:

```text
d_G = floor_binary64((s - 1) * G)
d_O = (s - 1) * O
single_send_ns = L + 2*o + g + d_G + 2*d_O
```

The floor occurs once per message. The receiver charges no per-byte gap.
That omission is the reason this arm can price sender serialization and
propagation without silently claiming to price shared receiver ingress.

## Frozen grid and same-schedule rule

The imported timing base, candidate construction, rate factors, original
families, and original bands come from the protected corrected runner. The
third arm adds no compute duration and no fitted factor.

The network grid contains the three existing TP4 key/value redistribution
cells with 458,752,000 aggregate bytes:

| Cell | Source ranks | Destination ranks | Flow count | Bytes per flow |
|---|---:|---:|---:|---:|
| TP4 to TP2 | 4 | 2 | 8 | 57,344,000 |
| TP4 to TP4 | 4 | 4 | 16 | 28,672,000 |
| TP4 to TP8 | 4 | 8 | 32 | 14,336,000 |

Every LogGOPSim cell and its packet counterpart must carry identical GOAL
text and binary SHA-256 values. Aggregate bytes, endpoints, tags, dependencies,
and flow counts must match. The ten disaggregated frontier rows reuse these
three network cells while independently varying decode tensor parallelism,
decode batch, worker counts, and capacity limiting. Thus the study varies both
network shape and serving composition.

## Physical sanity before precision

These bounds are written before reading a third-arm value.

- Sender floor: each of four sources must place one quarter of the aggregate
  payload on a 400 Gbit/s link. No complete cell may beat 2.293760 ms.
- Conservative serial ceiling: sending all 458,752,000 bytes through one
  400 Gbit/s link costs 9.175040 ms. Adding one 2 microsecond propagation
  allowance per message gives ceilings of 9.191040 ms at TP2, 9.207040 ms at
  TP4, and 9.239040 ms at TP8.
- Scaling cross-check: the per-flow payload halves as destination width doubles.
  Its isolated `G` term must therefore halve, apart from the one-byte floor,
  while each source's aggregate serialized bytes stay constant.
- End-to-end check: the priced arm may only add network service to the imported
  prefill boundary. Decode service, the x coordinate, every imported operation
  duration, and every external adjustment remain unchanged.

A native value outside its floor and ceiling is fatal. Agreement with these
bounds is necessary, not proof of correctness.

## Fatal guards

A failed guard makes the new run void. Fatal rows are never counted in a
behavioral denominator.

- FG-A prior publication immutability: every protected byte hash matches before
  and after both full evaluations.
- FG-B parameter and tool identity: goalsim, `htsim_rnic`, and `txt2bin` hashes
  are checked before execution. Every recorded LogGOPSim argv contains exactly
  `-L 2000 -o 0 -g 0 -G 0.02 -O 0 -S 9223372036854775807 -n LogGP`.
- FG-C same schedule: each priced and packet cell has identical GOAL text and
  binary hashes and conserves the frozen flow endpoints and bytes.
- FG-D inherited validity: every corrected fatal guard except its coordinator
  determinism guard holds in each fresh evaluation, and the original S, R, F,
  M, D, and W definitions and bands are not edited or reinterpreted.
- FG-E chronology: this freeze commit precedes the third-arm implementation and
  first execution.
- FG-F determinism: two complete scored evaluations run in fresh processes and
  their canonical JSON bytes are identical. Only `elapsed_seconds` and `W-1`
  are excluded, by those exact names. No native timing, arm value, row,
  classification, or provenance field is excluded.
- FG-G explicit bypass: selecting the third-arm bypass returns exactly zero
  network service for TP2, TP4, and TP8, starts no LogGOPSim process, and
  reproduces the protected corrected record's complete unpriced point and
  frontier projections byte for byte.
- FG-H physical bounds: every priced native cell lies inside its predeclared
  sender floor and conservative serial ceiling.

## Existing scored families

The fresh evaluations rerun the corrected record's S, R, F, M, and W families.
Their bands and meanings remain unchanged. In particular:

- Family R stays `[0.98, 1.02]` for each of ten decode rows.
- Family F2 stays `[0.75, 1.35]` for each of ten frontier comparisons.
- Family M remains packet-priced divided by unpriced-network. M1 stays at least
  1.000000 for every candidate and M2 stays at least 1.02 for at least one
  candidate. The third arm does not replace, rename, or rescore those rows.
- Family W keeps the 600 second ceiling for the complete coordinator run.

The third-arm decomposition is a new unscored evidence class. Structural
identities, physical guards, and bypass identities are fatal when violated but
do not inflate a behavioral pass denominator.

## Three-arm decomposition and conditional interpretation

For every disaggregated row, publish:

- unpriced network service, LogGOP-priced network service, and packet network
  service in integer picoseconds;
- the complete unpriced, LogGOP-priced, and packet prefill boundaries;
- all three exact rational `tokens/s/gpu` coordinates;
- `priced_penalty = unpriced_y / loggop_y`;
- `residual_penalty = loggop_y / packet_y`;
- `total_packet_penalty = unpriced_y / packet_y`;
- the exact identity
  `total_packet_penalty = priced_penalty * residual_penalty`;
- `network_residual_ps = packet_service_ps - loggop_service_ps`.

The interpretation is fixed before the numbers are known:

- A frontier-visible contention residual survives only where
  `residual_penalty > 1`. Report its maximum, every row where it appears, and
  the associated TP width. Wording may narrow only to a packet residual beyond
  the priced LogGOP latency and sender-serialization terms. Receiver-side
  wording requires the width pattern itself to support it and must state the
  measured scope.
- If no row has `residual_penalty > 1`, report that the LogGOP-priced arm
  swallows the complete frontier gap and that no contention residual is
  demonstrated at this scale.
- A positive network-leg residual that is hidden by a decode capacity limiter
  is published but is not called frontier-visible.
- A negative residual is published without relabelling or clipping. It means
  the pinned LogGOP terms exceed the packet service for that cell; it does not
  become evidence of negative contention.

No outcome changes the original 1.042715399805 packet-to-unpriced observation.

## Publication and closure

The new run writes separate append-only DEPLOY-12 result JSON, CSV, and Markdown
artifacts. It does not overwrite the protected record, ledger, report, or
figures. The result cites this final freeze commit.

DEPLOY-12 closes if all fatal guards hold, the two complete evaluations are
byte-identical under the named wall exclusions, the explicit bypass reproduces
the corrected unpriced arm, and all ten three-arm decompositions are published.
Closure does not depend on whether a positive residual survives. DEPLOY-13's
rounded-axis refutation, the original Family F miss, calibration scope, and
silicon-accuracy scope remain unchanged.
