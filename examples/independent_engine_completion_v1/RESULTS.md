# Independent-engine completion

The latest native campaign is **VOID at the frozen 1,800-second host limit**.
Its three serialized controls qualify 58 requests. The first independent
process retains 18 complete request rows but does not finish its required
20-request workload. No independent process is admitted, so CORE-68 stays open.

## Latest run and result

The campaign runs the unchanged component, service, arrival, engine-count and
handoff grid from source `72a395677ec56e9814a85518473940f2f95313c3`.
The original prospective behavior freeze is
`73270f6699440aa4a17ec7b2af5155698ab643b2`. The representation amendment
`7d756ad7cefdd8270c2a6b28e2b74dbe84139e6e` follows the first failure;
`1d7f5a9c1f1908f55e065f5d8ca77339bc5d17cb` freezes complete process-exit receipt
retention after the second failure. Neither changes the workload or time cap.

The serialized one-, two- and four-engine-per-pool processes exit successfully
in 362.541, 278.331 and 253.289 seconds, respectively. The first independent
process is killed at 1,800.254 seconds, with sampled maximum current resident
memory of 1,608,264 KiB, below the 16-GiB limit. The wall limit decides the
verdict. These are process diagnostics, not a source-paired speedup claim.

The independent process retains five complete cells, 18 complete request rows
and 438 checkpoints. Its sixth cell is incomplete. The last checkpoint follows
submission of decode step 82 at 7,414,640,000 ps, with 83 input records and 82
completed results. A submitted step is not a completed request. The remaining
two independent processes never start.

Nine of eighteen required stages finish. The retained evidence includes
249,340 unscored guards, 21 exact oracle rows and eight component behavioral
instances in two families. The campaign's behavioral score is null. The empty
violated-guard array does not override the fatal process exception, and the
partial native observations do not qualify concurrency.

The [portable publication](results.json) retains 59 raw process-file receipts,
totaling 83,866,110 bytes, plus root evidence receipts. Every started process,
including the timed-out process, has an initial receipt taken after exit and
before admission. All four initial inventories match their final inventories
and the retained files exactly. Earlier missing initial receipts are not
reconstructed. The latest raw summary SHA-256 is
`7279b687ec33f8ee394ca37c174f69bdd16713f96249eaf8b7d08b3b386aa3b1`.

The separately qualified [publication reader study](../publication_snapshot_v1/RESULTS.md)
removes repeated snapshot encoding and preserves exact model records and
rejections. The native campaign still exceeds its cap with that repair and the
qualified dependency lookup changes. This result does not identify the
remaining hot path; further host-work investigation precedes another attempt.

## Earlier retained attempts

| Attempt | Executed source | Fatal finding | Retained publication |
|---|---|---|---|
| First | `594fdfb001dc056b93fca5bef0595d548e754034` | Float/integer profile comparison mismatch after 24 requests; no process admitted | [Original record](void-frozen-v1.json) |
| Second | `ce8bf357e7477291d2a3cfc55b8a6875dd789c06` | 1,800-second timeout after eight complete independent request rows; three serialized processes admitted | [Original record](void-frozen-v2.json) |

Both earlier publications and their raw receipts remain unchanged and
unrescored. The first failure concerns exact checker representation, not the
8-TB/s memory bound. The second process's eleven finished decode observations
include three from an incomplete cell and are not eleven complete request
results. It has a final raw receipt but no initial pre-admission lock; that
chronology remains explicit in its archived publication.

## Project effect and limits

CORE-68 needs a complete fresh native campaign within the existing limits.
The repair does not close independent-engine timing qualification, CORE-52's
large retained session, CORE-51 or CORE-54's deployment frontier, CORE-69's
cancellation path, or CORE-70's resource compositions. Shared cache-transfer
work may proceed independently, but its native qualification depends on
CORE-68. No GPU, model weights or packet backend runs in these attempts.
