# Timing-method startup scope, frozen before the next fit

This amendment replaces the inherited tiny-message startup in the reference
prediction. Source inspection distinguishes the inherited CUDA-event boundary
from nccl-tests host-wall timing, while each uses a different iteration count
and worker lifecycle. A startup taken from one method is not a physical floor
for the other. The physical encoded-byte floor remains unchanged.

Estimate one nonnegative apparent startup per architecture and width from
the frozen LL calibration rows, jointly with their data and synchronization
costs. It includes unresolved launch and fixed primitive overhead and is not
identified as pure host launch time. Use that same startup for LL128 and Simple
fits, whose remaining protocol-specific overhead is explicit. No new payload
anchor or measured validation value enters the fit.

Add the intercept column first, then channel bytes, instruction rounds and
synchronization units in that order. Retain a column only if it increases
matrix rank on the LL calibration design. A collinear constant synchronization
term is absorbed into the apparent startup and its separate coefficient is
zero by parameterization, not by a claim that hardware synchronization is free.
Record which parameters remain identifiable. Use the same deterministic
bounded solver and all previously frozen accuracy, coverage and band-width
bars. Preserve the prior candidate and its failures as development evidence.

Add 8-byte and 1-KiB rows to the new timing-method controls, under both warmup
and iteration counts. These are independent diagnostic checks of boundary
transfer. They are not added to the fit or used to enlarge the model band.
The fresh off-grid validation still starts after the candidate parameters are
committed. Any mechanism revision after reading that validation needs a new
freeze and new validation measurements.
