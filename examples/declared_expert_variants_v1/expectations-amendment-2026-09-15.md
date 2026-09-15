# Declared expert variant amendment, 2026-09-15

This amendment records the baseline the implementation actually starts from and
nothing else. No frozen cell, literal, refusal, interface, evidence class or
closure condition of the 2026-09-15 freeze is changed, withdrawn or added to.
The original freeze stays in place unchanged.

## Why it exists

The freeze was written before the PLACE-13 slice landed and said that if that
slice landed first, the declared builder, the placement package entry and the
placement registry would carry the digests the PLACE-13 freeze records instead
of the ones in `expectations.json`. PLACE-13 has landed: its work is on the
main branch at `622188a1308b379f2c1c7943e4f45ccfe7583e8f`, so
`declared_sglang_manifest` exists and cell V6 can run. The freeze anticipated
this case and named it; this file supplies the replacement digests so the
baseline is a literal rather than a forward reference.

## Replacement baseline

The implementation starts from the freeze commit
`841dfb380180608f6c18b68ed5ae1e585d9be689`, which is the 2026-09-15 freeze
replayed onto that main branch head. Three of the seven baseline file digests
moved and are replaced by the values in the JSON beside this file; the other
four are unchanged. `docs/modules/placement.md` moved because both the PLACE-13
slice and the freeze commit itself edited the registry, so its recorded digest
is the one the implementation modifies rather than a pre-freeze value.

Two study files join the baseline, because the harness this slice adds runs
both as subprocesses and a change to either would change what the two fatal
identity cells prove: `examples/sglang_declared_layout_v1/run_study.py` and
`tests/test_sglang_declared_layout.py`.

## What does not change

The five reference manifest digests were recomputed on the new base and are
byte for byte the values the freeze records, which is exactly what the freeze
predicted: the PLACE-13 slice adds a second entry point rather than an option,
so it moves no manifest the vLLM builder emits. Both fatal study `--check`
identities, every cell from V1 through V13, the zero scored denominator and
both closure conditions stand as frozen.
