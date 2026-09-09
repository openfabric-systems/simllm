# Native host execution diagnostic

The first paired diagnostic is **VOID**, with zero admitted arms. The native
uninstrumented process completed all 14 requests in 260.017 seconds, but the
checker rejected the step-record writer's existing JSON encoding. The
instrumented process was never started, so this attempt attributes no hotspot
and supports no host optimization claim. CORE-52 remains open.

## Frozen protocol and chronology

The final original expectations-only commit is
`8f06648d26bbfb2bfff4751d29a2e2bf232b440d`. Implementation and native execution
follow it; the executed source is
`51ed0d1ef5fb86a2fc643dfba3949f67627c3659`.

The complete retained summary has SHA-256
`ae25e0fdf6f75cf14a66bffc50ce080e15f276b2766a9538ee74e5cf2019779b`.
Its behavioral score is null. Only protocol and uninstrumented capture stages
finished. The process exited normally, with 70 native scheduler steps and 98
clock advances. Fourteen advances are the unchanged serial driver's explicit
equal-time request admissions. Sampled maximum current resident memory was
1,147,248 KiB. These are descriptive capture facts from a void attempt.

## Finding

`StepRecordStream.append` writes each native step using
`json.dumps(step_record_to_json(record))` followed by a newline. That writer
preserves schema insertion order and uses standard spaces. The diagnostic
incorrectly required the compact sorted encoding used by its own JSON files.
The raw writer bytes are intact; interpreting this mismatch as changed model
semantics would be wrong. The checker defect voids the attempt all the same.

The first file receipts, full native records, process observations and
traceback remain retained. The [compact record](results.json) identifies all
16 raw files, totaling 24,977,530 bytes. Intermediate exact checks do not turn
an incomplete arm into admitted evidence.

The correction requires a separate source-writer encoding contract before the
parser changes and a fresh execution. That encoding assertion is
post-specified after this finding. The original run is never rescored and no
public pre-registration claim is made for the corrected checker.

## Project effect and limits

CORE-52 retains target host feasibility. Its original 1,200-second target
campaign also remains VOID. CORE-68 owns independent-engine completion;
CORE-51 and CORE-54 retain deployment and frontier obligations. No GPU was
used, no device service was calibrated, and this smaller diagnostic qualifies
neither the 56-engine target nor distributed model throughput.

Before this execution, 152 focused software checks passed in 13.85 seconds
and Ruff passed. The complete software and continuous-integration gates remain
pending for publication.
