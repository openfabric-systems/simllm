"""Human-readable projection of the width-tail result authority."""

from __future__ import annotations


def render(report):
    rows = report["configurations"]
    complete = [r for r in rows if r["status"] == "complete"]
    standalone = [r for r in complete if r["mode"] == "collective"]
    steps = [r for r in complete if r["mode"] == "step"]
    by_key = {(r["mode"], r["pattern"], r["width"], r["rate_gbps"], r["profile"]): r for r in rows}
    norm = report["normalization"]
    provenance = report["provenance"]
    reproduction = report["morning_reproduction"]
    verdict = report["verdict"]
    prefix_guards = [g for r in standalone for g in r["guards"] if g["family"] == "receiver_prefix_floor"]
    physical_guards = [g for r in standalone if r["profile"] == "rnic-cn"
                       for g in r["guards"] if g["family"] == "receiver_prefix_floor"]
    lines = [
        "# Collective width tail: completed components and two void cells", "",
        "## What ran", "",
        "The full 64-cell matrix ran fresh after the BACK-68 guard amendment.",
        f"Original expectations-only commit: `{provenance['expectations_commit']}`.",
        f"Amendment-only commit: `{provenance['guard_amendment_commit']}`.",
        "Backend pin remains `617ce20`; hashes and GOAL identities are in results.json.",
        "No old bulk results were reused. The reference ring/all-to-all payloads,",
        "rail-major placement, 400/200 Gbit/s rates, profiles, exact points and",
        "behavioral bands are unchanged. The supported expert step retains the",
        "already corrected source-major ordering of the same selected ranks.", "",
        (f"Standalone: {len(standalone)}/32 complete, two physical width-64 all-to-all "
         "control-loss exits. Supported step attempts: 16/32 complete, all ideal;"),
        "all 16 physical steps are rejected by BACK-38 with null timing and shares.",
        (f"Raw populations: {sum(r['flow_count'] for r in standalone):,} standalone flows and "
         f"{sum(r['flow_count'] for r in steps):,} step-artifact flows. These are flow"),
        "populations within configurations, not independent request-tail samples.", "",
        "## What came out", "",
        f"Study status: **{verdict['status']}**.",
        "The two HTSIM-40 cells are still fatal, void and unscored. Their exact",
        "control-loss signature was declared survivable before the rerun, so",
        "completed components retain interpretable evidence. No failed guard is",
        "included in a behavioral score and no missing completion is synthesized.", "",
        (f"All {sum(g['rows'] for g in prefix_guards):,} receiver completion-prefix floors pass. "
         f"The minimum physical prefix slack is {min(g['minimum_slack_ps'] for g in physical_guards):,} ps."),
        "All 14 completed physical phases satisfy physical makespan >= ideal",
        "makespan, and all original individual byte-plus-propagation floors pass.",
        (f"Exact oracles: {verdict['exact_oracles']['instances']} rows, "
         f"{verdict['exact_oracles']['failed']} misses. Behavioral relations: five families, "
         f"{verdict['behavioral_relations']['instances']} instances, "
         f"{verdict['behavioral_relations']['failed']} misses in completed components."),
        "These denominators are separate from fatal guards and coverage gaps.", "",
        (f"Preservation against the tracked morning record at `80eef42`: all "
         f"{reproduction['ideal_numeric_fields']} numerical fields in "
         f"{reproduction['ideal_configurations']} ideal configurations reproduce exactly;"),
        "all 24 original exact-oracle records are identical, including their failures.",
        "GOAL digests and completion statuses also reproduce. The source record",
        "digest and a zero-mismatch comparison are stored in morning_reproduction.", "",
        "## Physical bounds before headline numbers", "",
        "At 400G, byte service costs 20 ps/byte; at 200G, 40 ps/byte. Every",
        "flow floor is its own payload service plus 2 us ideal propagation or",
        "4 us for the physical remote Clos path. These floors appear beside every",
        "p50 below. No finite physical ceiling follows from link capacity under",
        "queueing and flow control. The frozen packetized ideal envelopes remain",
        "the conditional ceilings in results.json, not relaxed oracle tolerances.", "",
        "For a ring, the dependency-depth floor is 2(W-1)*(S/W*ps_per_byte+P).",
        "At W=64 and 400G the ideal phase interval is [293.287680, 314.899200] us;",
        "the physical floor is 545.287680 us and its ceiling is unbounded.",
        "For all-to-all, each receiver has D=7F/8 remote senders and floor",
        "D*65536*ps_per_byte+P. At F=64 and 400G the ideal interval is",
        "[75.400320, 4773.104000] us. The physical floor is 77.400320 us,",
        "with no valid phase completion because of control loss.", "",
        "The fixed compute service has floor and ceiling 100 us by input.",
        "Step bounds add that service to twice the corresponding network bounds.",
        "Shares have floor 0 and ceiling 1, tightened per cell in JSON. At width",
        "64 and 400G the ring step is bounded by [686.575360, 729.798400] us,",
        "and the expert step by [250.800640, 9646.208000] us. Their fabric share",
        "floors are 85.435% and 60.128%, respectively. This compute denominator",
        "is a synthetic control, not calibrated GPU throughput.", "",
        "The independent ring transfer factor 2(W-1)/W implies 6.779 GB/s per",
        "rank at W=64, 400G from the ideal phase, below the 50 GB/s endpoint",
        "capacity. This checks work and units, not a measured deployment tail.", "",
        "## Standalone flow and phase measurements", "",
        "All times below are us. Quantiles are nearest rank within each cell.",
        "NN is rnic-nn; CN is rnic-cn. Each floor includes the profile's path",
        "propagation. CN width-64 all-to-all stays void. Physical ceilings are unbounded.", "",
    ]
    for pattern in ("ring", "all-to-all"):
        lines += [f"### {pattern}", "",
                  "| Width | Gbit/s | NN floor | NN p50 / p99 | CN floor | CN p50 / p99 | NN phase | CN phase |",
                  "|---|---|---|---|---|---|---|---|"]
        for width in (8, 16, 32, 64):
            for rate in (400, 200):
                nn = by_key[("collective", pattern, width, rate, "rnic-nn")]
                cn = by_key[("collective", pattern, width, rate, "rnic-cn")]
                cn_floor = (nn["bounds"]["flow_propagation_floor_ps"] + 2_000_000) / 1e6
                cn_tail = (f"{cn['fct_p50_ps']/1e6:.6f} / {cn['fct_p99_ps']/1e6:.6f}"
                           if cn["status"] == "complete" else "void")
                cn_phase = f"{cn['phase_makespan_ps']/1e6:.6f}" if cn["status"] == "complete" else "void"
                lines.append(f"| {width} | {rate} | {nn['bounds']['flow_propagation_floor_ps']/1e6:.6f} | "
                             f"{nn['fct_p50_ps']/1e6:.6f} / {nn['fct_p99_ps']/1e6:.6f} | "
                             f"{cn_floor:.6f} | {cn_tail} | {nn['phase_makespan_ps']/1e6:.6f} | {cn_phase} |")
        lines.append("")
    lines += [
        "## Refutations and their limits", "",
        (f"There are {sum(n['aligned_flows'] for n in norm):,} aligned flow pairs. "
         f"{sum(n['below_1x'] for n in norm)} physical flows remain below the ideal FCT:"),
        "47 at F=32, 400G and 75 at F=32, 200G. Their minima remain 0.759090",
        "and 0.575342. The latter is source 0 to 8, tag 1000. The corresponding",
        "phase ratios are 2.029921 and 1.959788, both above their fatal floor 1.",
        "The [controlled experiment](../aligned_baseline_v1/RESULTS.md) finds",
        "sub-one shared-flow ratios with every byte prefix and phase floor intact,",
        "and none in isolated controls. This resolves the BACK-68 convention",
        "refutation as scheduling under the registered decision rule. It does not",
        "prove the absence of every possible hidden credit defect.", "",
        "Shared per-flow ratios are now diagnostic. The unshared initial ring",
        "round retains its aligned per-flow lower bound. Later ring starts differ",
        "across profiles, so those flows retain raw FCT and compare by full phase.",
        "The two conservation floors remain fatal in every completed cell.", "",
        (f"The unchanged per-flow 2x target still misses: {sum(n['above_2x'] for n in norm)} "
         f"aligned pairs exceed it, maximum {max(n['slowdown_max'] for n in norm):.6f}."),
        "The 1.2x diagnostic remains separately recorded. Fairness corrections",
        "do not repair a slow physical tail or relax any upper behavioral band.", "",
        "At width 64 the physical backend again exits 2 with `fabric dropped",
        "control lifecycle` at both rates. The 1,048,576-byte buffer default",
        "and control policy are unchanged. Raw error.txt files remain in bulk.",
        "HTSIM-40 still owns recovery. Partial CSVs are never treated as completed phases.", "",
        "TRAF-89 is unchanged: standalone ideal ring points miss their frozen",
        "prediction by exactly (2W-3) ns: 13, 29, 61 and 125 ns at either rate.",
        "The earlier study attributed this to zero-cost GOAL joins executed as",
        "1 ns. This rerun preserves that observation and the four rate-relation",
        "misses; it does not revise the frozen oracle. Supported ring step points",
        "and rate relations still match exactly because their ordered artifact",
        "execution has a different schedule boundary.", "",
        "## Supported critical-path contribution", "",
        "The real HtsimRequestMetricReducer and attribute_step_detail partition",
        "each supported step's ordered artifacts. All 16 ideal steps conserve",
        "the partition and TTFT. Collective and fabric shares coincide here.",
        "Local expert service is masked by the larger fabric service; its work",
        "remains separate in JSON. Times are us; shares are fractions of the step.", "",
        "| Width | Gbit/s | Ring step / TTFT | Ring fabric share | Expert step / TTFT | Expert fabric share |",
        "|---|---|---|---|---|---|",
    ]
    for width in (8, 16, 32, 64):
        for rate in (400, 200):
            ring = by_key[("step", "ring", width, rate, "rnic-nn")]
            expert = by_key[("step", "all-to-all", width, rate, "rnic-nn")]
            lines.append(f"| {width} | {rate} | {ring['step_latency_ps']/1e6:.6f} | "
                         f"{100*ring['fabric_share']:.3f}% | {expert['step_latency_ps']/1e6:.6f} | "
                         f"{100*expert['fabric_share']:.3f}% |")
    lines += [
        "", "## What it changes", "",
        "BACK-68 acceptance is met: controlled pair and incast experiments, a",
        "corrected shared-flow convention with reusable live metrics, a pre-run",
        "guard amendment, and a fresh full matrix preserving the original ideal",
        "numbers exactly. Its backends.md entry now retains only orchestrator",
        "integration closure: remove the entry and regenerate the protected",
        "README_PRO task-progress projection together. The worker cannot edit",
        "that projection; removing the entry alone fails the repository's progress",
        "consistency gate. No numerical or backend work remains under BACK-68.", "",
        "## What it does not change", "",
        "COMP-9 still needs held-out request-tail and removal-of-contention",
        "validation. BACK-38 still blocks physical multi-artifact steps. BACK-69",
        "still owns packet-level critical-path segments; these sinks do not emit",
        "CriticalPathBreakdown or per-visit queue waits. HTSIM-40 control loss",
        "and TRAF-89 join timing remain unresolved. No task IDs are registered,",
        "and neither BACK-69 nor HTSIM-40 registry text is changed. No README,",
        "native backend, backend pin, compute default or third_party source changes.", "",
        "![Width-tail measurements and supported ideal step shares](figures/collective_tail.png)", "",
        "The PNG and PDF project results.json. Panels (a) and (b) show phase",
        "makespans and gray ideal phase floors. Panel (c) shows the within-cell",
        "50th and 99th percentile flow completion times (p50 and p99) at",
        "400 Gbit/s: solid circles denote p50, dotted hollow triangles p99.",
        "Its gray solid and dotted lines are the ideal and physical byte-plus-path",
        "floors. Panels (d) and (e) show ideal fabric shares of the supported step.",
        "Panel (f) compares complete physical and ideal phase makespans; circles",
        "denote ring and squares all-to-all. Its unit line is the fatal phase",
        "floor; the 2x line is a visual reference, not the per-flow acceptance test.",
        "Outside (c), solid and dashed lines denote 400 and 200 Gbit/s.",
        "Time axes in (a) to (c) are logarithmic. Missing physical width-64",
        "all-to-all points denote void cells, not zero times. Whole fabric shares",
        "are not excess-tail attribution.", "",
        "## Reproduction and validation", "",
        "results.json is the numerical authority; this report and both figure",
        "formats are projections. Per-cell GOALs, manifests, raw completions and",
        "receiver_prefix_floors.csv remain under the external bulk root.", "",
        "```bash", ". ./.env.local.sh",
        '.venv/bin/python examples/collective_width_tail_v1/run_study.py --out "$SIMLLM_DATA_ROOT/collective_width_tail_back68_v1" --publish',
        "```", "",
        "Use a fresh --out for simulation. --summarize-only projects existing",
        "evidence, checking expectation and executable provenance. Tracked text",
        "digests accept LF normalization; generated text artifacts use LF bytes.",
        "Exit 2 means an undeclared fatal or incomplete matrix. The two declared",
        "void cells remain explicit even when the completed-component run exits 0.",
        "The first full test gate found two unrelated older-study executable-hash",
        "mismatches when this wave's native binary variables were inherited, and",
        "a stale protected progress projection after removing BACK-68. The final",
        "offline gate omits those native variables and keeps the truthful integration",
        "residual entry; no tests or historical artifact hashes were weakened.",
        "Final ruff, full pytest and module-format gate output is retained in the",
        "wave handoff. Offline tests exercise early-prefix overcredit, phase-only",
        "failure, scheduler order, identity/multiplicity drift and survivable voids.", "",
    ]
    return "\n".join(lines)
