# TRAF-67 worker sizing

Date: 2026-08-26
Branch: `codex/traf67_clean_boundary`
Base: `cd43473`

## Scope

Repeat the already frozen TRAF-66 finite two-child overlap boundary under a
clean exposure protocol. Reuse the committed source ranges, event ledger,
`max(C, P) + min(C, P) / 2` form, component-service envelope, independent
signed-movement calculation and 27 preservation locks without amendment or
refit.

## Exposure budget

- No record data has been accessed before this note.
- Commit a field-addressed reader and access-log contract before the first
  record access.
- Permit only the visible 1K COMP-75 calibration row and log every access.
- Never read a candidate record, or any other record file, as a whole.
- Keep the held-out access ledger empty. Do not access or compare 2K or 4K
  values.
- Do not rerun the scored flagship, change decode pricing, touch TRAF-65 or
  touch the NVLink module.

## Estimated change

Small documentation and harness repetition, expected to reuse the TRAF-66
implementation directly. Anticipated public changes are a committed
field-addressed reader and access protocol, a clean TRAF-67 runner/result,
focused tests, the mechanical registry/index closure, and EOL attributes.
Bulk or scratch output belongs under the task-specific external run root.

## Commit plan

1. Land the exposure protocol and expectations-only surfaces, including the
   field-addressed reader, before any record access.
2. Perform the single permitted field-addressed access and reproduce the
   frozen boundary result and preservation evidence.
3. Close TRAF-67 and move TRAF-66 only as the literal registry wording allows,
   then run all validation gates on each commit.
