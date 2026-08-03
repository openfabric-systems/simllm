# simllm.backends and third_party

Invocation and result parsing for the network simulators, plus the pinned
backend submodules.

## Interface

- `HtsimUecConfig` + `build_htsim_uec_command`: argv construction for
  GOAL-driven `htsim_uec` runs (implemented, tested).
- Planned: the same for `htsim_rnic` (profiles `rnic-nn`, `rnic-nn-fluid`,
  `rnic-cn`) and completion-CSV parsing
  (`profile,flow_id,source,destination,tag,payload_bytes,start_time_ps,completion_time_ps,fct_ps`).

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

`htsim_uec` command builder implemented. RNIC profile invocation and result
parsing not started (milestone M1).

## Open tasks

- BACK-1: `htsim_rnic` command builder plus completion-CSV parser
  (milestone M1).
- BACK-2: LogGOPSim invocation helper for fast flow-level sweeps.

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
