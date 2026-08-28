# CORE-63 clean repetition expectations

Status: **EXPECTATIONS ONLY**. This freeze precedes every protected record
access and contains no derived residency or calibration number.

## Frozen derivation

The assignment formula remains `256 * 8 * 4 / 288`. Relative to the
256-assignment capture, only routed expert work is scaled by exactly `1/9`.
The component classification rule remains fixed: a case-insensitive kernel
name containing `fused_moe_kernel` is routed expert work; every other retained
noncollective kernel stays at scale one; the independently retained fixed term
is kept once. No parameter may be amended or refit.

Before values are visible, the expected signed direction is fixed as:

- corrected step: decrease;
- standard-decode prediction: increase;
- signed residual: less negative before any possible crossing.

## Frozen sources and access

The machine-readable freeze binds the tracked inputs by Git blob identity and
the external kernel summary by its already published catalog size and digest.
Only the six selectors in `core63_clean_access_protocol.json` are permitted.
The reader must produce twelve ordered access events, two for each selector,
with every completed byte count strictly below its source size.

The held-out MTP numeric value is neither an input nor a comparison target.
The forbidden-access ledger is frozen as the empty array. Scoring remains the
exclusive responsibility of the fifth scored run.

## Preservation and registry intent

The inherited CORE-63 preservation manifest is frozen at 93 entries and at
SHA-256
`3586ddd96cfad09d8b7c8015eda6641ba5f4180a1f55c5b9f34e06cd29eb445d`.
All 93 prior artifacts must remain byte-identical.

CORE-63 will move only as far as its literal acceptance permits after the
clean undercorrection result is known. CORE-64's conditional registration may
become unconditional only if the clean reproduction stands. CORE-65 must be
verified free on the base main commit before registry publication.
