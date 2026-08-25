# Concurrent session load amendment

The original expectations remain the historical freeze. Their first complete
run is retained with SHA-256
`7121ab1b99eeb4809de8e2546351fd03653cd7acf30cee99a0c50155d401d5c5` and is
VOID because the required multi-request batch was absent in all three pool
ratios. All 18 admission, handoff and terminal conservation rows held with a
maximum TTFT decomposition residual of 0 ps, so the finding isolates the load
grid rather than the concurrent identity or completion mechanism.

The unit error is explicit. The original highest load used a 31,250,000,000 ps
interarrival, while the observed prefill services are 95,424,000 and
114,936,000 ps. Even the longest service is more than 271 times shorter than
that interarrival. The offered requests therefore cannot overlap at a stock
scheduler, and a genuine batch is physically impossible in the frozen cells.

This amendment replaces only the offered loads with 8,000, 16,000 and 32,000
requests per second, corresponding exactly to 125,000,000, 62,500,000 and
31,250,000 ps. At the highest load, a two-prefill-engine assignment sees each
engine every 62,500,000 ps, below both observed prefill services, while a
two-decode-engine assignment sees each engine every 62,500,000 ps, below the
77,952,000 ps short-prompt decode cadence. This creates the overlap the
original acceptance guard asks the stock schedulers to resolve.

The pool ratios, prompts, request count, token count, handoff constant, curve
definitions, behavioral directions, conservation checks, physical bounds and
CORE-51 byte lock do not change. The amended run is a post-specified
regression because both the implementation and the first scored run precede
this correction. It is not reported as public preregistration, and the void
run is not discarded or rescored.
