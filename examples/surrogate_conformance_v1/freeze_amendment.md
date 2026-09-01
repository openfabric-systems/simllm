# Pre-run freeze amendment

The original expectations were committed before any conformance run and
none has executed, so this is a pre-run amendment, not a supersession of a
run of record. The original file stays byte-identical; where this
amendment and the original disagree, this amendment governs. Three
changes, each forced by executed evidence (the version-delta audit that
diffed the pinned 0.26.0 sources against the supported 0.27.1 release and
reproduced identical step compositions on an executed arrival workload):

1. Pinned engine. The conformance target is the supported adapter pin,
   vLLM 0.27.1 (release commit
   6e448d0ea9bf3d88d898b65449ca6dc2aec170ac), with scheduler source
   SHA-256
   `c67bda2886b52865ddafabaae7d797c359e930752f374421a33e537d94a5f45a`
   recorded and matching at run time. The original 0.26.0 hash bullet is
   replaced. A source-hash mismatch still voids the run.
2. Causal-tuple provenance wording. The partial-prefill count fields were
   removed in 0.27.1; the tuple recorded per cell drops any mention of
   them. The delta audit's one behavioral divergence inside the pinned
   families' reach, the cache-disabled free-queue reinsertion rule
   (hashless blocks appended for locality rather than prepended when
   caching is off), is the 0.27.1 semantics and binds both sides of every
   prefix-free family.
3. Family F7 comparison alphabet. The live-side normalization bridge can
   emit only operations the oracle sidecar witnesses; the sidecar
   observes allocation outcomes after the capacity decision, so a
   pre-decision RESERVE is not witnessable on the live side. F7 therefore
   compares the declared witnessed alphabet (BIND_PREFIX, TOUCH,
   ALLOCATE, RELEASE, FREE, EVICT, RECOMPUTE) exactly on both sides; the
   surrogate's native RESERVE operations are recorded and reported but
   excluded from the F7 equivalence by this pre-run declaration, and the
   live pre-decision semantics remain owned by the registered VLLM-44
   residual. Every other clause of the original file stands unchanged.
