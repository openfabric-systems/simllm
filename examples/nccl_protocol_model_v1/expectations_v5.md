# Source chooser and timing-method scope, frozen before revision

The first independent capture of candidate 0e44eca3 is retained as a valid
refutation of its coverage and precision targets. It is now development data
for this revision. Neither its old score nor the original candidate parameter
file is overwritten. The next validation uses another unseen payload grid.

Replace the observed 16-KiB protocol bracket with the library's selection
operator: choose the Ring protocol minimizing its recorded latency plus byte
service estimate. Recover those software cost coefficients from the existing
diagnostic logs, using only AllReduce, float32, one-pipeline-operation entries.
These are NCCL's own predicted times, not hardware measurements. Verify the
recovered lines against the recorded estimates and all automatic callbacks.
Retain at most a 16-byte ambiguity for integer rounding. The chooser cost is
used only to select the protocol, never as the measured service cost.

Keep the previously fitted GPU-work coefficients, apparent startup and source
geometry unchanged. Replace the one cross-protocol method allowance with one
nonnegative intercept and physical-service coefficient per protocol. Identify
these offsets from paired automatic reference and original-harness medians
in the original dense capture and first validation capture. Software polling
and launch behavior differ between protocols, so one global offset need not
transfer between their kernels. Do not claim these effective offsets uniquely
identify host scheduling, polling or clock behavior.

The reference prediction remains the physical/GPU-work/synchronization model.
The original-harness prediction adds the corresponding method offset. Compare
each timing method with its own prediction; the plotted solid center is the
original GPU-event method, matching the user's original figure. Include the
reference as a separately labeled dashed prediction when both are shown.
The deterministic uncertainty band is the union of the two method estimates,
expanded by the frozen maximum-relative-residual plus 90th-percentile repeat
spread rule, separately by protocol. The calibration inventory now includes
all original automatic medians, the fixed-protocol controls, and all first
validation medians. They are explicitly development data, not validation for
this revision. No software-work cost is refitted to them.

After committing the new parameters, collect five independent process repeats
on each architecture for the 60 sizes 286720 through 4153344 bytes in increments
of 65536. Keep the original timing block and the reference benchmark settings.
Use the same reference library and binary identities. The parameter file must
precede every new validation process. The second grid does not overlap the
original dense grid or the first validation grid. No second-grid value may
enter the model or envelope. The precision bars remain 10 percent for the
reference and 15 percent for the original method, at least 99 percent coverage
per curve, and no more than 40 percent full band width. Target all medians.

The timing-method control matrix from the first capture remains diagnostic.
Its warmup, iteration, worker-reuse and buffer-rotation controls did not identify
one general A100 cause under the frozen effect criterion. Report that negative
result. The GPU visibility probe is an illustration, not a fitted NCCL RTT.
