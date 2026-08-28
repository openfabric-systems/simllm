# Kernel calibration namespace

This namespace holds reviewed kernel-calibration requests, suite configuration
and compact candidate results. Raw profiler traces and sample blobs stay
outside Git under a caller-supplied run root. A reviewed result retains their
relative names, byte counts and SHA-256 identities.

The common local-shard contract is implemented by
`simllm.calibration.local_shard`. One request names the exact framework
dispatch signature, model revision, logical parallelism, physical shard,
device ISA, launch mode, phase and shape. The target command owns the actual
framework compilation and execution. The collector verifies the returned
identity, sample closure and architecture before accepting a candidate result.

Run one target with:

```bash
simllm-calibrate run \
  --request request.json \
  --target framework-target \
  --output-root "$SIMLLM_KERNEL_CALIBRATION_RUN_ROOT/cell"
```

Targets that are Python entry points can pass their script with a repeated
`--target-arg`. The launcher never invokes a shell. An absent target creates no
output, and a nonempty output root is never overwritten.

This boundary measures rank-local compute only. Distributed collectives and
network service are excluded, even when their launch dependencies are visible
to the framework. The final kernel-cycle lookup and compact device model remain
the established pricing authorities; a local-shard result is candidate
evidence and does not promote itself.
