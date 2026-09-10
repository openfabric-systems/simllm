# Revised cost composition, frozen before implementation and refit

This amendment supersedes the three-coefficient additive service form in
expectations.md. The initial candidate's source geometry is retained. Its
retrospective cost fit and excessive uncertainty width are preserved as a
refuted candidate, not erased or counted as acceptance. Both forms remain
informed by already observed hardware; no public preregistration is claimed.

The revised form models pipeline overlap between link serialization and GPU
data work using their maximum. It adds an explicit cost for synchronization
slices, including the empty Simple slice that still executes barriers and
counter publication. The old form counted empty slices but assigned them no
cost. These changes apply to every size, with no new payload timing anchors.

The four nonnegative costs per architecture/width/protocol are maximum-channel
encoded-MiB service, instruction-round service, synchronization-slice service,
and nonempty Simple fence/publication service. Generic LL and LL128 primitive
calls each count as a synchronization unit; Simple executes two slice bodies
per primitive, including empty ones. The fixed tiny-message floor remains.

Total reference time is the tiny-message floor plus synchronization and fence
service plus the maximum of physical encoded-byte service and GPU data work.
Use bounded nonnegative least squares for this piecewise-smooth composition,
with deterministic initial seeds. Record active-column rank and fitted costs;
no claim of unique identification follows merely from a small residual.

The envelope rule and all accuracy, width, coverage and hardware acceptance
bars remain unchanged. Calibration residuals are computed separately for each
selected protocol because a forced inactive protocol does not identify the
uncertainty of a different active protocol. The method allowance remains
shared across a machine/width curve. Protocol-specific radii use only the
same frozen calibration cells belonging to that protocol. Every retained or
fresh point is still scored and drawn. No withheld residual sets a radius.

The RTT sensitivity partitions the composite synchronization budget, which
includes slice and publication work, before adding any excess. It must never
charge a second transport authority or turn an unidentifiable split into a
measured polling constant. The newly captured GPU handshake remains an
independent illustration of the scale of a visibility exchange.
