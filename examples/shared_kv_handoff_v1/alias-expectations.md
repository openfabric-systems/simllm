# Native request identity comparison amendment

This expectations-only amendment precedes the shared campaign and its
source-pair comparison implementation. Source inspection and already retained
independent-engine records show that the pinned input processor appends a
fresh random suffix to each native request identity. The same logical request
therefore has different process-local names in separate native processes.
No shared native outcome is used to choose this comparison rule.

The initial shared freeze excludes the two declared opaque identifiers at the
root of `to_comparison_json`, but leaves their occurrences in complete engine
records ambiguous. Literal equality there is impossible for fresh native
processes. Resolve this using an explicit identity projection, not field
removal or changes to the native input processor.

Retain every original identifier and every complete raw record. Within each
process, admit a bijection from its native identifier to the tuple of actual
engine identity, public request identity and prefill/decode role. Verify that
binding against native request/cache/output observations and every scheduled
or finished step membership. The comparison projection replaces only the
identifier values in `scheduled[].request_id`, `finished_request_ids[]` and
`preempted_request_ids[]` by those stable tuples. It preserves those fields,
all list positions, ordering and multiplicity. No field is discarded from a
step. Unknown, duplicated, foreign or incorrectly bound identifiers are fatal.

The complete request comparison retains its original two root exclusions.
Every step's other fields, complete results, all sink collections, events,
visits, clock advances, public batches and selected profiles compare exactly.
An opaque native identifier found at an undeclared location is a fatal reader
failure. Do not perform recursive string replacement, suppress mismatches or
normalize times, bytes, token shapes, errors or ownership records. A coherent
bijection between two fresh alias inventories is permitted; changing only one
projection of an identity must fail admission.

The pinned input-processor file is added to the explicit native source hash
inventory. This is an identity-interpretation amendment, with no change to
workload, packet laws, services, native source code, resource limits, exact
oracle count or behavioral denominator. Cite this amendment alongside the
original shared freeze and the separate deadline amendment. CORE-71 still
requires the complete forty-request native campaign; no result is rescored.
