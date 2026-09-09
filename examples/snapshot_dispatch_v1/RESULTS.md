# Primitive snapshot dispatch result

**VOID: the first source-paired attempt stops at a duplicate checker name.** `snapshot_dispatch_v1` runs the before-source process to completion, then process and data admission both attempt to register `before:runtime-before`. The evidence registry correctly rejects the duplicate. No source is admitted, no behavioral relation is scored, and no dispatch reduction is qualified by this run.

The expectations-only commit `f08da959e9aca1be75e56a21e4cc5db0fa8b15f9` precedes implementation and this first campaign. The run uses source `4b76fb254c8ee9652afb8b17939bc803736d9b11`. Its immutable [receipt archive](void-frozen-v1.json) retains 138 raw files totaling 18,334,165 bytes with matching first and final inventories. The raw summary SHA-256 is `2b93da9b08998acece89bff6b3296b073ca3a0ac22230b2b4fdfc02070bcc07b`.

Two of nine stages finish and 705 unscored guards precede the collision. There are zero exact oracles and zero behavioral instances. The process exits successfully in 3.036 seconds with sampled resident memory of 97,604 KiB; these are host diagnostics, not model completion metrics.

The repair gives process admission its own check-name namespace and adds a regression that composes process and data admission. It changes no production behavior, expectation, workload, count relation or process limit. The first run remains VOID and a fresh campaign is required. CORE-68 remains open. Native completion, CORE-52, shared resources and the deployment frontier do not close here.
