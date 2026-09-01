# TRAF-80 NVLink mechanism alignment sizing

Date: 2026-09-01

## Planning basis

- As-of commit: `65593131a0448d2b33f51018d5972c918dad3493`
- Scope: replace the fixed-packet, sender-timer credit, implicit-class and flat-switch surrogates with the TRAF-79 mechanism boundary; preserve all accepted bypass and direct-mesh behavior; freeze and run the registered sanity study; publish inherited-envelope shifts; close TRAF-80 only if every guard is literal.
- Assumptions: the current A100 candidate profile remains the compatibility input; generation-scoped structural defaults can be added without relabeling any unidentified value as measured; existing frozen consumer records remain unchanged and can be pinned by digest and implementation identity from a new preservation ledger.
- Exclusions: no cluster work, model weights, web access, backend submodule edits, deployment module edits, README edits, H200 collective work, MiniMax work, or deployment-curve scored-lineage edits.
- Owner: NVLink alignment worker on `codex/traf80_nvlink_alignment`.
- Dependencies: TRAF-79 mechanism record and the merged TRAF-65, TRAF-69, TRAF-70, TRAF-72 and TRAF-74 preservation surfaces. TRAF-73 remains the owner of unidentified A100 numeric and policy values.
- First reviewable slice: expectations-only study specification and preservation manifest, committed after this sizing-only commit and before behavioral implementation or the first study run.

## Expected files

Expected modified files:

- `simllm/backends/htsim_nvlink.py`
- `simllm/backends/__init__.py`
- `tests/test_htsim_nvlink.py`
- `docs/modules/traffic.md`
- `docs/design/nvlink-domain-model.md` if its final-status mechanism diagram text needs alignment

Expected created files:

- bounded family `examples/nvlink_mechanism_alignment_v1/` for expectations, preservation pins, the sanity runner, result ledger and plain-language report
- bounded family `tests/test_nvlink_mechanism_alignment_*.py` for freeze, study, result and preservation guards

Existing frozen consumer artifacts and generated figures are inputs and locks. They are not expected to be modified. Mechanically emitted study results and preservation digests are listed but do not count as handwritten lines.

## Handwritten line ranges

- Production: 650 to 1,200 lines.
- Tests and fixtures: 550 to 1,000 lines.
- Studies, configuration and documentation: 450 to 900 lines.

Confidence: low. The dominant uncertainty is preserving the legacy TX and RX event order exactly while moving credit authority to explicit receiver-buffer release events and adding a live NVSwitch crossbar seam. The number of inherited consumer envelopes that need executable rerun adapters may also widen the study family. No hardware, cluster, external repository or waiting work is planned.

## Scope-change rule

Update this note before continuing if implementation needs a new repository interface, touches a fenced module family, cannot preserve a frozen consumer byte for byte, or moves any handwritten category outside its range.

## Completion actuals

Pending. Record the final touched-file count, handwritten and generated line deltas, and explain any range miss before the final delivery commit.
