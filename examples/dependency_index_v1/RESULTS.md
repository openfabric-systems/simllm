# Call-local dependency lookup result

The source-paired study **PASS** removes repeated dependency searches without
changing any modeled outcome. In the largest direct-verifier case, conversion
calls fall from 8,976 to 272, exactly the frozen factor of 33. CORE-52 gains a
qualified host-work optimization and can freeze its next native target
campaign. Its 56-engine feasibility task remains open.

## What ran

Two fresh local processes execute the same committed worker against the old
and new installed package sources. Twelve graph configurations cross depth
`{1,2,4,8}` with width `{2,4,8}`. Each retains the full graph, communication
schedule and direct-call profile. Two valid key controls per source exercise
distinct dependency origins and participant ranks. Fourteen corruptions in
each graph produce 168 rejected projections per source.

Four supported `HtsimStepSink` jobs per source cross prompt lengths 8 and 16
with one and four serial requests. Each request has one prefill and four
decode steps, giving 50 steps per source. The selected Granite configuration
uses 24 layers, eight local ranks, the B100 architecture envelope, roofline
efficiency 0.7 and ideal host initiation. No native serving frontend, GPU or
packet backend executes in this study.

All eight required stages complete. Evidence classes remain separate:
3,208 unscored guards have no violation; 24 exact conversion-count oracle
rows match; 12 behavioral instances in one reduction family pass; and five
semantic corruption controls reject the intended damage. These classes are
not added into a combined score.

## Expected work and observed reduction

Let `E` count effective-edge occurrences, `B` boundary occurrences and `S`
serialized occurrences. The retained conversion sites require at least
`E+B+S` calls. The old search can inspect at most `E` candidates for each
serialized occurrence, giving the ceiling `E+B+S+E*S`. These bounds are
recorded before either worker starts.

For this frozen graph family, the exact old count is
`(2*w+1)*d*(2+w*d)` and the new count is `2*(2*w+1)*d`. Their ratio is
`1+w*d/2`. Every observed count lies inside its bounds and equals its frozen
formula. The width-eight slice shows both the direction and shape:

| Depth | Old calls | New calls | Old/new |
|---|---|---|---|
| 1 | 170 | 34 | 5 |
| 2 | 612 | 68 | 9 |
| 4 | 2,312 | 136 | 17 |
| 8 | 8,976 | 272 | 33 |

At fixed width, doubling depth doubles the new conversion count; the old
count follows the frozen quadratic-plus-linear relation. In the largest
case, the floor is 272 and the ceiling is 17,680. The new path reaches the
floor because it retains each first matching edge while building the existing
occurrence counter. It no longer converts discarded search candidates.
This proves a reduction in this operation count, not linear complexity of the
whole verifier or a 33-fold reduction in elapsed execution time.

The complete before and after processes take 200.258 and 174.231 seconds.
Their sampled maximum current resident memory is 535,132 and 566,764 KiB.
These single-run observations include profiling, evidence construction and
sink jobs. They are diagnostics without a scored speedup or memory claim.

## Completion and rejection compatibility

The existing resident-weight streaming surrogate gives a conditional floor
of 320,864,256 bytes divided by 8 TB/s, or 40,108,032 ps per step. The sum of
the declared services gives the serial-job ceiling and exact completion
oracle. The observed decode services, 77,952,000 and 77,976,000 ps, exceed
that floor. Both sources finish each job exactly at its declared ceiling:

| Prompt tokens | Serial requests | Steps | Completion, ps |
|---|---|---|---|
| 8 | 1 | 5 | 407,232,000 |
| 8 | 4 | 20 | 1,628,928,000 |
| 16 | 1 | 5 | 426,840,000 |
| 16 | 4 | 20 | 1,707,360,000 |

Four serial requests take exactly four times one request. All full step
inputs, results and ten sink publication collections remain identical; each
paired sink file has identical bytes. This floor belongs to the existing
surrogate. COMP-7 retains routed-expert compute precision, so these checks do
not establish a hardware decoding rate.

Full graph and projection values also remain identical before and after.
Every invalid projection retains the exact exception class and message,
including which error appears first when two fields are corrupted. Inputs
remain unchanged after both accepted and rejected calls. Multiplicity,
canonical schedule order and the five-field dependency key keep their prior
meaning.

## Chronology and retained evidence

The expectations-only freeze is
`581d41531c42759e1fd1f9b7af3a0f45844f51a2`, preceding implementation and the
first source-paired run. The old package source is
`67cb2dbdfc393400f1c760a0d24534ad3c494094`; the executed new source is
`8dcf9858a881f8c2b42207c1ab644fb0d471ea88`.
Only `simllm/traffic/execution_goal.py` changes among installed package files.
The worker, configuration helper, actual imports, complete package inventories
and profile function identities are source-bound.

The raw summary SHA-256 is
`afe3e18026443e0b92f2f45de781465a4152456847f13a8c5ff526a8b53a0821`.
The [portable publication](results.json) retains the source and raw receipts,
conversion matrix and paired job results. There are 86 process raw files
totaling 334,338,908 bytes, with identical initial and final receipt domains
and hashes. Full profiles and captures stay in external evidence storage.

## Project effect and limits

CORE-52 can use the indexed checker in a separately frozen native target
campaign. Its earlier target and diagnostic VOID attempts retain their
original receipts and verdicts. This study closes no stable task and changes
no simulated timestamp, service model or default configuration.

CORE-68 retains independent-engine timing; CORE-51 and CORE-54 retain
deployment and frontier obligations. These local sink jobs qualify job
completion compatibility. They do not measure serving time to first token,
time per output token, GPU service or 56-engine throughput.

Before the first source-paired execution, 45 focused software tests passed
in 21.43 seconds and Ruff passed. Software gates remain separate from the
study evidence classes.
