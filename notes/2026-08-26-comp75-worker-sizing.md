# COMP-75 worker sizing

COMP-75 is sized as a large, single-worker task.

The work requires a preregistered source boundary, an expectations-only
checkpoint, an exposure ledger, a clean calibration comparison, independent
signing of the 1K movement, preservation checks, implementation and document
updates, and full per-commit verification. A single worker preserves the
exposure ordering and makes the preregistration, expectations, and visible
comparison sequence auditable.
