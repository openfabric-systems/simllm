# COMP-78 campaign remainder result

## Depth-8 signed residual

The signed CORE-61 residual is **unavailable**, not zero, and depth linearity
has no verdict. Exact base job `200120` retained an Nsys capture and digest
ledger but failed the compact analyzer. Exact decode jobs `200123` and `200128`
both failed before the scored boundary when vLLM's registered 65,536-token
startup profile could not allocate 896 MiB. Consequently the measured decode
service, `measured - prediction` residual against 3,751,359,511 ps, and 5
percent comparison all remain absent. CORE-61 stays open.

The first decode attempt and its unchanged cache-warmed retry remain distinct
on Merlin with file-manifest SHA-256 values `b7d23c47...` and `3345fc82...`.
The base digest ledger is `e7127bd2...`. No failed or completed attempt was
overwritten, and the registered commands and expectations were not changed.

## Granite completion state

Granite remains **0 of 1,212 digest-complete cells**. The completed prefix is
still the empty byte sequence with SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The exact resume point remains
`sglang-decode-cuda-graph-te1-pi1-da1-ex1-b1-kv1311-deliberately-fragmented`.
No canonical cell attempt directory was created, so the empty prefix is
conserved byte for byte.

Pinned real targets were staged for vLLM `0.26.0+cu129` and SGLang
`0.0.0.dev1+g8f2a3ad6d`. The first-cell capability audit fails closed before a
GPU run because the landed `run-cell` contract has no public control that can
prove deliberately fragmented KV placement, no routed-expert sidecar output
path, and no two-clean-run code-object index output path. The same driver also
has no digest-complete cell writer. Starting the canonical first cell could
therefore only create an uncompletable failed attempt, so the occupancy-bound
campaign was parked at its exact original resume point.

## Content-addressed successor

The landed compiler reproduced its output byte for byte in two fresh external
directories and emitted candidate successor
`58d169865109a5eaca3e69978a48080c25a6bb48ee6607d32e82ed8487d17fdd`.
It retains the CORE-61 attempts, target capability audit, staged target
digests, and corrected 1,212-cell prefix audit as sources without changing any
frozen service point or distribution.

The immutable predecessor
`ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52`
and partial successor
`d868a4f35d633032daa238168d00f42c2ab47fc569db649b19b907008072e107`
remain byte-identical. The new record remains `candidate`; it does not promote
coverage or invent a measurement.

| Artifact | SHA-256 |
|---|---|
| candidate record | `58d169865109a5eaca3e69978a48080c25a6bb48ee6607d32e82ed8487d17fdd` |
| profile table | `090a1673283a1815e2b636b4606e4704022e52fcd2c0702c230d650c1b1e357f` |
| device-service entries | `54f57ef009e7efcd856906764b14e47b94c97a268c7a6977d0f2198b9322ea40` |
| result | `c84724b54ae04f6e24e29cab9f3dcef6d335be78a09338e16abf2398c7409028` |
| artifact manifest | `9bbf8a762513a1eba87183ec1716342cfba9c79f6765d4d891171d953ca45220` |

## Closure and exact remainder

COMP-78 and literal COMP-72 remain open. The exact remainder is:

1. obtain a digest-complete exact registered depth-8 decode output without
   changing its frozen command, then retain the measured service and signed
   residual;
2. land framework target contracts for verified fragmented and contiguous KV
   placement, routed-expert sidecars, two clean code-object harvests, and the
   digest-complete marker; and
3. resume the unchanged canonical Granite plan at the cell named above and
   complete all 1,212 cells.

`COMP-79` is already allocated in the registry to key-local DeepSeek repeat
distribution work. It was not overwritten or silently reassigned. The exact
campaign remainder therefore remains registered under the still-open COMP-78
entry pending integrator assignment of an available new ID.
