# Single-switch ASU Tomahawk1 study

This study invokes the native `htsim_ns_tm1` binary directly to compare a
preinstalled, wire-budgeted ASU calendar with PRBS lotteries and unpaced
incast on one modeled C3232C. It compares four 25G ports and one 100G port,
MTU, allocation windows, scheduler ticks, explicit lane collisions and
uncalibrated XPE/cell-accounting sensitivities. Model design and source
provenance are in the paired backend's `docs/rnic-switch-models/tm1.md`.
The published backend is commit `534723c` on
[`codex/asu_tm1_single_switch`](https://github.com/yifeng-ethz/HTSIM-rnic-private/tree/codex/asu_tm1_single_switch).

Build the paired backend with tests enabled and run its complete ctest suite.
Supply the built binary and a bulk output directory explicitly:

```sh
python examples/asu_tm1_single_switch_v1/run_study.py \
  --binary "$HTSIM_TM1_BINARY" --out "$TM1_STUDY_OUTPUT" \
  --backend-root "$HTSIM_SOURCE" --jobs 4
```

For an individual native experiment:

```sh
"$HTSIM_TM1_BINARY" --policy quota_dd --source 4x25 --receiver 4x25 \
  --mtu 1500 --window-us 32 --tick-ns 8 --fraction 0.95 --seed 1
```

`summary.csv` retains every native case. `case-*.json` records exact commands,
`case-*.csv` and stderr retain raw output, and `manifest.json` identifies
binary, runner and expectation hashes. A fatal guard makes the study void;
conservation/capacity guards are not scored behavioral wins. Generated files
belong in the explicit bulk directory, outside Git.

`quota_dd` is a deterministic rotating packet calendar with byte deficit
carry. `quota_prbs` permutes legal credited opportunities. Both constrain
physical source and destination service. `prbs` is a statistical allocation
lottery without hard per-window quotas. `unpaced` starts every source at
line rate. Losing a packet does not count as finishing an application task.

W is allocation-window length. Reference rnic-cn `dwnd` means a one-way
control deadline K. The calendar assumes known demand already installed;
this run does not measure declaration/grant delivery, PTP accuracy, deployed
cut-through latency, receiver resequencing, RC/DCQCN recovery or GPU TTFT/TPOT.
The 256 KiB cap is proposed per physical DATA queue; live NX-OS still uses
dynamic alpha admission. See the frozen expectations and pre-run correction
for physical accounting and interpretation rules.
