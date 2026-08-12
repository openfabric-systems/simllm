# Dispatch sequence ownership refreeze expectations

Date: 2026-08-12

This expectations-only supplement freezes the dispatch-sequence rerun after
TRAF-25 established that one `StepRecord` contains one engine's tokens. It was
written before the sequenced renderer was corrected or any new result-producing
run was made. The original TRAF-21 expectations at commit `7efd71e` remain an
unaltered chronology record.

## Reason for the refreeze

The original synthetic and Granite fixtures projected one captured engine's
routing table onto every EP source. Main has since proved that projection
arithmetically impossible. `RoutedMoeSupply.engine_rank` is now the sole home
of the scheduled tokens. Peer ranks own experts and return combine messages,
but carry no scheduled tokens in this isolated step.

The aggregate renderer already follows that rule. The sequenced renderer must
consume the same ordered routed-contribution authority. A second ownership
rule, even one that happens to read the same field, is not acceptable.

TRAF-26 owns a real multi-engine population. It requires explicit peer
workloads and independently observed or sampled routing. Copying one routing
table onto every peer remains forbidden because it manufactures correlated
hot-expert incast.

## Corrected synthetic exact oracles

The strict v2 fixture, route tuples, message grouping definitions and
framework-returned order claim stay unchanged. Its declared engine rank is
rank 0. The only dispatch source is therefore rank 0, with destinations:

```text
source 0: 3, 1, 2, 1, 3, 2, 1, 3
source 1:
source 2:
source 3:
```

The corrected exact rows are:

| Quantity | Corrected value |
|---|---:|
| Aggregate messages, dispatch plus combine | 6 |
| Per-expert-group messages, dispatch plus combine | 6 |
| Per-token messages, dispatch plus combine | 16 |
| Dispatch bytes | 16,384 |
| Combine bytes | 16,384 |
| Total bytes | 32,768 |
| Remote vector hops | 16 |
| Independent hop ceiling, `4 * 2 * 1 * 2` | 16 |

Every sequenced grouping must match every aggregate ordered-pair and request
total. Every dispatch message must source rank 0 and every combine message
must return to rank 0. Combine remains the exact transpose with unchanged
routing ordinals.

These are fatal-unscored guards. They supersede only the source-multiplied
fatal literals in the original expectations. They do not replace a behavioral
band after seeing a new outcome.

## Behavioral relations remain frozen

The original twelve-cell backend matrix remains unchanged: three renderers,
packet and fluid profiles, and 200 and 400 Gbit/s endpoint rates. The original
scored relations and evidence accounting remain the acceptance test:

- packet per-token minus each comparator stays positive and inside
  `[15,360, 61,440]` ps at 200 Gbit/s and `[7,680, 30,720]` ps at
  400 Gbit/s;
- each 200 Gbit/s packet delta remains twice its 400 Gbit/s counterpart within
  2,000 ps;
- each sequenced minus aggregate fluid delta remains within 1,000 ps of zero;
- the genuine-risk registry remains three family classes and ten instances.

The corrected single-source fixture has fewer three-vector groups, so the old
byte-only packet bands may be unreachable. That possibility does not authorize
changing the bands. If the run misses them, the result records the miss and
TRAF-22 retains the residual.

Scored relations are evaluated from raw completions before exact sequence,
pair, request, hop, quiescence or physical-bound guards. None of those fatal
guards may contribute to a behavioral numerator or denominator.

## Synthetic physical sanity before digits

The peak incident payload is 16,384 bytes in each of two serial phases. The
payload floors are therefore 1,310,720 ps at 200 Gbit/s and 655,360 ps at
400 Gbit/s. The packet backend reserves a complete 4,096-byte calendar
envelope for each of the 16 one-packet per-token messages. Serializing all 16
envelopes at both endpoint serializers gives 5,242,880 ps and 2,621,440 ps.
The registered conservative ceilings, including quantization margin, are
5,500,000 ps and 2,750,000 ps.

Every backend cell must lie inside its rate's floor and ceiling. Halving the
rate should move serialization-dominated terms by about two. An out-of-range
cell is a fatal physical-sanity failure and voids the run.

## Corrected Granite scale registry

The Granite v1 projection remains cost evidence only. Its reconstructed tuple
order is not framework-observed and must not be described as kernel or wire
issue order.

The input hashes remain:

| Artifact | SHA-256 |
|---|---|
| `routed-experts.json` | `24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f` |
| `steps.jsonl` | `824cd9557293328bb42b593ac893b6a067302e545b087c9219195ccb8031d755` |

The archived 334,432-byte aggregate GOAL with SHA-256
`08a0403af66ff8a9d6b18f93afd15ae0bc925cc85555acf8a0593438a3d7bc92`
is retained only as observed pre-TRAF-25 provenance. The corrected aggregate
GOAL is 47,399 bytes, carries 336 sends, and has SHA-256
`6bb83366a3936bcf1e435cab008bb55c2966777528a9e8d885dd44d47f5a4943`.

Accepted TRAF-25 arithmetic supplies these pre-run scale oracles:

| Grouping | Messages | Directed bytes |
|---|---:|---:|
| Aggregate | 336 | 25,563,136 |
| Per-expert-group | 1,008 | 25,563,136 |
| Per-token | 12,482 | 25,563,136 |

The per-token count is 12,482 remote vector hops, below the independent
`54 * 8 * 24 * 2 = 20,736` ceiling. Corrected peak per-rank egress is
12,781,568 bytes. Total group bytes fell by 8.117 times from the defective
projection, while peak egress fell by only 2.117 times. Completion must track
the critical-rank shape, not the total-byte ratio.

The 24 layer compute floor is 99,336,000 ps. Peak egress over 400 Gbit/s adds
a 255,631,360 ps network floor before propagation or packet overhead. The
corrected cost record must report where each completed cell sits relative to
those bounds and must check the corresponding 200 Gbit/s scaling term.

The original practicality limits remain unchanged: render plus compile within
30 seconds, peak traced Python memory within 1 GiB, GOAL text within 64 MiB,
and each requested backend run within 60 seconds. The 54-token, 24-layer
Granite step is the required scale point. Crossing a limit is a measured
finding assigned to TRAF-22, not a reason to alter the limit.

## Fatal and failed outcomes

A violated fatal guard makes the entire run void. A void run reports findings
but publishes no behavioral pass fraction and closes no task. If all fatal
guards pass but one or more scored relations miss, the run is failed and may
report its scored fraction. A failed scored band is not converted into a fatal
guard and is not refrozen after observation.

The EP-width-eight regression must be capable of failing independently of the
aggregate renderer. It checks exact per-pair equality plus the hop ceiling.
The old sequenced source loop emits 101,318 Granite hops and 207,499,264 bytes,
while the corrected aggregate emits 12,482 hops and 25,563,136 bytes. The old
rule therefore violates both pair equality and the 20,736-hop ceiling.

## Claim boundaries and existing residuals

The v2 trace observes the order returned by framework dispatch. It does not
observe the order in which a fused kernel, NCCL, or an RNIC posts bytes to the
wire. PLAY-14 remains the honest residual for that missing identity-preserving
wire evidence. This rerun does not upgrade the framework-returned sequence
claim.

The study uses all-remote native network profiles. It does not exercise the
analytic intra-node locality service. CORE-41 already owns the missing ingress
term in that model. Any intra-node number mentioned for context remains pending
CORE-41 and is not an acceptance oracle here.

## Registered command and dry run

The result-producing invocation remains:

```bash
.venv/bin/python examples/dispatch_sequence_v1/run_study.py \
  --out "$SIMLLM_WAVE6_RUN_ROOT/dispatch_sequence_v1-ownership-refreeze" \
  --granite-root "$SIMLLM_GRANITE_REPLAY_ROOT" \
  --htsim-rnic "$SIMLLM_HTSIM_RNIC" \
  --txt2bin "$SIMLLM_TXT2BIN"
```

Before this expectations-only commit, the complete command is run with
`--check-only`. Check-only parses the CLI and validates only the frozen
registries and arithmetic. It does not import SimLLM, read external artifacts,
create the output directory, invoke a native tool, or write an artifact.
