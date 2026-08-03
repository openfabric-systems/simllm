# simllm.backends and third_party

Invocation and result parsing for the network simulators, plus the pinned
backend submodules.

## Interface

- `HtsimRnicConfig` + `build_htsim_rnic_command` + `run_htsim_rnic`: direct
  GOAL-driven `htsim_rnic` runs (profiles `rnic-nn`, `rnic-nn-fluid`,
  `rnic-cn`; a run is valid only with `physical_quiescence=verified`),
  binary discovered via `SIMLLM_HTSIM_RNIC`, the README build location,
  then `PATH`.
- `FlowCompletion` + `parse_completion_csv`: completion-CSV parsing
  (`profile,flow_id,source,destination,tag,payload_bytes,start_time_ps,completion_time_ps,fct_ps`);
  `RnicRunResult.job_completion_time_ps()` for JCT.
- `simllm.backends.fct.normalized_fct`: per-flow FCT normalized to the
  `rnic-nn` baseline of the identical GOAL, matched by
  (source, destination, tag). Valid for aligned-start flows; for phases
  with model-dependent start stagger use the phase makespan ratio
  (M1 finding F1).
- `HtsimDcqcnConfig` + `run_htsim_dcqcn`: GOAL-driven RoCEv2 DCQCN runs
  over a topology-file ns-tm3 Clos (`htsim_dcqcn_atlahs`, landed via the
  backend DCQCN PR); same completion-CSV schema and quiescence contract.
- `HtsimUecConfig` + `build_htsim_uec_command`: argv construction for
  GOAL-driven `htsim_uec` runs.

## Pinned submodules

| Submodule | Repo | Ref | Provides |
|---|---|---|---|
| `third_party/atlahs` | [ATLAHS-rnic-private](https://github.com/yifeng-ethz/ATLAHS-rnic-private) | `main` | GOAL toolchain (txt2bin, LogGOPSim, goal_gen), validated `htsim_rnic` launcher (`atlahs_entry.py`) |
| `third_party/htsim` | [HTSIM-rnic-private](https://github.com/yifeng-ethz/HTSIM-rnic-private) | `main` | UEC htsim, RNIC model series, `htsim_rnic` executable |

As of 2026-08-03 both launcher and wiring PRs are merged, so the RNIC
profiles run from the pinned `main` refs:

```
cmake -S third_party/htsim/htsim/sim -B build/htsim -DCMAKE_BUILD_TYPE=Release
cmake --build build/htsim --parallel
build/htsim/datacenter/htsim_rnic -goal trace.bin -linkspeed_bps 400000000000 -rnic_profile rnic-cn
```

Changes to the backends go through their own repos on
`<YYYY_MM_DD>/simllm-addon` branches; SimLLM only bumps pins.

## Status

`htsim_rnic` invocation, completion parsing and FCT normalization landed
with M1 (BACK-1, BACK-3 closed). The end-to-end test runs them for real
wherever the backend toolchain is built (it self-skips otherwise), and the
M1 sanity studies exercise the full pipeline: 15 of 18 pre-registered
checks pass, the six fluid workload-A configurations and four workload-B
runs to zero picosecond residual, and the three failures are traced to
mis-registrations, not defects (findings F1-F3 in examples/m1/RESULTS.md).

## Open tasks

- BACK-2: LogGOPSim invocation helper for fast flow-level sweeps.
- BACK-4: default multi-QP striping of large inter-node flows in the
  traffic layer (tag-distinct subflows), motivated by the DCQCN ECMP
  collision study: leaf-affine permutations collide under static flow
  hashing on an 8-path Clos (16 of 32 flows at 3.5x FCT, seed-invariant);
  4-way striping cut the tail to 2.05x.

## Backend-repo follow-ups (tracked here, executed in their repos)

- HTSIM-1: `rnic-ss` (Slingshot-like) profile wiring; the runtime factory
  rejects it with a clear error until the slingshot runtime lands. Its CLI
  options are already parsed so the flag ABI is stable.
- HTSIM-2: goodput/state/queue trace flags for `rnic-cn`; they need trace
  hooks in the reviewed runtime first.
- HTSIM-3: GOAL-driven DCQCN profile (`htsim_dcqcn_atlahs`).
- HTSIM-4: GOAL parser hardening and the checked-in `txt2bin` build target.
- ATLAHS-1: correct the vendored-fallback wording (the vendored htsim tree
  cannot satisfy the resolver) and pin a known-good HTSIM commit.
