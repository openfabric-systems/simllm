# Pre-capture addendum: the hierarchical-topology attribution warning

Committed after the harness (4bdf437) and before any capture. The
flagship integrator's fence verdict cleared this campaign and added one
technical warning, preserved verbatim in the campaign evidence root as
fd-fence-verdict.md and binding on the analysis:

At width 8 with four GPUs per node, NCCL's hierarchical topology
traverses intra-node NVLink as a stage of each collective, so the phase
decomposition must attribute intra-node stages to the NVLink domain
rather than the fabric, or the per-port Cassini concentration numbers
will carry an intra-node component.

Consequences, frozen now:

- The T2B analysis attributes stages to domains (intra-node stages to
  the NVLink domain, cross-node stages to the fabric) before any
  concentration number is interpreted, using the NCCL topology and
  selection logs the harness captures.
- If E1's large-payload one-port to four-port ratio refutes below 2.0
  at width 8, the first candidate mechanism is dilution by the common
  intra-node stage, and the analysis must quantify that stage before
  any other interpretation is offered.
- No expectation, band or scored family changes; this addendum names a
  mechanism in advance, it does not soften a prediction.
