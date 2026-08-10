"""DCQCN realism study: ECN-only vs ECN+PFC under seeded a2a, vs the
deterministic profiles.

Runs the mixed lognormal all-to-all on realistic small buffers:
deterministic singles for rnic-nn-fluid, rnic-nn, rnic-cn, and seeded
ensembles for the two DCQCN modes (the ECMP hash and ECN sampler vary by
seed; the deterministic profiles need no ensemble). Emits one completion
CSV per run plus a summary of drops, recovery events and pause metrics,
ready for the CDF-with-shadow plot.

Usage: python run_dcqcn_modes.py [--out DIR] [--seeds 8]
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from run_ladder import mixed_all_to_all_goal  # same frozen workload

from simllm._local_config import path_from_env
from simllm.backends import HtsimRnicConfig, run_htsim_rnic
from simllm.backends.htsim_dcqcn import HtsimDcqcnConfig, run_htsim_dcqcn
from simllm.goal import to_binary
from simllm.workload import LogNormalLengths

HERE = Path(__file__).resolve().parent
TOPO = HERE.parent / "m1" / "topologies" / "clos_64_400g.topo"
G400 = 400_000_000_000
# 1 MiB is where the lossy regime lives for this GOAL: at 4 MiB the ECN
# control alone keeps occupancy below overflow (verified), so the study
# operates at the buffer point where drops and recovery actually occur.
SMALL_BUFFERS = {"-shared_buffer_bytes": "1048576", "-egress_buffer_bytes": "1048576"}

_ = LogNormalLengths  # re-exported context for readers; sizes live in the GOAL


def manifest_counters(result, keys):
    out = dict.fromkeys(keys, 0)
    for line in result.manifest:
        for token in line.split():
            for key in keys:
                if token.startswith(key + "="):
                    try:
                        out[key] = int(token.split("=", 1)[1])
                    except ValueError:
                        pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()
    if args.out is None:
        data_root = path_from_env("SIMLLM_DATA_ROOT")
        if data_root is None:
            parser.error("--out is required when SIMLLM_DATA_ROOT is not set")
        args.out = data_root / "cn_ladder" / "dcqcn_modes"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    goal_bin = to_binary(mixed_all_to_all_goal().write(out / "a2a16.goal"))
    rows = []

    for profile in ("rnic-nn-fluid", "rnic-nn", "rnic-cn"):
        r = run_htsim_rnic(HtsimRnicConfig(
            goal_bin=goal_bin, profile=profile, linkspeed_bps=G400,
            completion_csv=out / f"{profile}.csv",
            topology=TOPO if profile == "rnic-cn" else None))
        med = statistics.median(f.fct_ps for f in r.flows) / 1e6
        rows.append({"run": profile, "seed": "", "median_fct_us": round(med, 3)})
        print(rows[-1])

    counter_keys = ["ns_tm3_dropped_packets", "silent_rtos", "ecn_marked_packets",
                    "loss_rate_cuts", "dcqcn_pfc_pause_frames",
                    "dcqcn_pfc_paused_wall_ps", "dcqcn_pfc_max_cascade_depth"]
    for mode, extra in (("ecn-only", {"-pfc": "off"}), ("ecn-pfc", {"-pfc": "on"})):
        for seed in range(1, args.seeds + 1):
            flags = dict(SMALL_BUFFERS)
            flags.update(extra)
            flags["-seed"] = str(seed)
            flags["-ecn_seed"] = str(seed)
            r = run_htsim_dcqcn(HtsimDcqcnConfig(
                goal_bin=goal_bin, topology=TOPO, link_bps=G400,
                completion_csv=out / f"dcqcn-{mode}-s{seed}.csv",
                extra_flags=flags))
            med = statistics.median(f.fct_ps for f in r.flows) / 1e6
            counters = manifest_counters(r, counter_keys)
            rows.append({"run": f"dcqcn-{mode}", "seed": seed,
                         "median_fct_us": round(med, 3), **counters})
            print(rows[-1])

    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(out / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} runs -> {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
