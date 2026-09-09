# Independent-engine completion

The first native campaign is **VOID**. Its first serialized process completes
24 requests, but no process is admitted because the checker treats an integer
physical bound and the numerically identical floating-point profile value as
different evidence. CORE-68 remains open.

The campaign runs the frozen component cases followed by the first native
one-prefill, one-decode process, with eight simulated workers per engine.
The source is `594fdfb001dc056b93fca5bef0595d548e754034`; the prospective
behavior contract is `73270f6699440aa4a17ec7b2af5155698ab643b2`.

The native process exits successfully in 425.15 host seconds and reaches
1,214,396 KiB sampled resident memory. Admission stops at
`serialized-p1-d1:simllm-prefill-0:memory-bandwidth`. The captured source
configuration uses floating-point `8000000000000.0` bytes per second; the
physical bound uses integer `8000000000000`. Both describe the same declared
8-TB/s memory interface. Exact type-aware JSON comparison refutes this checker
assumption. It does not refute the bandwidth bound or establish the accuracy
of the modeled service.

Three of eighteen stages finish before the fatal guard. The retained check
records contain 2,066 unscored guards, two component exact-oracle rows and eight
component behavioral instances. These are separate evidence inventories,
not an admitted score. The behavioral score is null. The native output contains
24 requests, but the admitted request and process counts are both zero. The
remaining five native processes are never started.

The [machine-readable record](results.json) retains initial receipts for all
fourteen raw process files, totaling 33,973,445 bytes, and the root evidence
receipts. Raw evidence stays under the configured external output root. Its
summary SHA256 is
`3464b261b79964cec73a0739d07f4c64e7eaefc9556fb390736b15f1ff1aaa0b`.
The run stays immutable and unrescored.

This finding requires an explicit representation contract and checker repair
before a fresh campaign. Any corrected assertion is a post-specified regression
check; the original service, arrival, width and handoff sweep stays frozen.
CORE-68 and CORE-54 gain no native concurrency qualification. CORE-52 target
feasibility, CORE-69 cancellation and CORE-70 shared-resource composition also
remain open. No GPU, model weights or packet backend runs in this attempt,
and no deployment frontier or serving-throughput claim follows from it.
