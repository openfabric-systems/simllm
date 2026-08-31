# CORE-61 depth-8 registered execution result

## Signed residual and verdict

The signed held-out residual is **unavailable**, not zero. Both exact decode
attempts failed during vLLM startup before the registered batch-32,
remote-KV-2000 boundary ran, so there is no measured service to subtract from
the preregistered 3,751,359,511 ps prediction. Depth linearity therefore has no
verdict. CORE-61 remains open.

If a valid measurement becomes available, the frozen sign convention is
`measured service minus predicted service`, and the 5 percent acceptance uses
measured service as its denominator. TRAF-66's finite-overlap term remains
separate and was not recomputed.

## Registered attempts

The base job `200120` completed its Nsys capture and digest ledger, but the
compact analyzer failed closed. Decode job `200123` then failed while vLLM's
65,536-token startup profile tried to allocate 896 MiB. Exact retry `200128`
ran after the compile cache was warm and reproduced the same allocation
failure. Neither failed decode attempt contains a scored decode boundary or a
digest-complete registered decode output.

All three attempt trees remain on Merlin below the CORE-61 campaign root. The
base digest ledger has SHA-256 `e7127bd2...`; the two fail-closed file manifests
have SHA-256 `b7d23c47...` and `3345fc82...`. No attempt was overwritten, no
command or expectation changed, and COMP-76 was untouched.

## Registry movement

CORE-61 moves from time-gated to `OPEN_BLOCKED_EXACT_DECODE_STARTUP_OOM`.
COMP-72 and COMP-78 remain open on a digest-complete registered decode output
and its signed residual. The partial campaign successor is candidate record
`58d169865109a5eaca3e69978a48080c25a6bb48ee6607d32e82ed8487d17fdd`.
