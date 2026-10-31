# Satellite Dynamics Simulator — development roadmap

This document follows the milestone sequence adopted after the 2026-10-30
repository/legacy-output audit. A passing regression test **does not** imply
physical validity: each subsystem needs independent checks of its governing
physics. The older notebook-stage names and the `v0.1-inertia` Git tag may
still appear in Git history; do not reuse or move published tags.

| Milestone | Purpose | Exit criterion |
|---|---|---|
| **M0 — Baseline** | Reproducibility, setup, CI, golden fixtures | Checks pass; legacy baseline and limitations archived |
| **M1 — Interfaces** | Frames, quaternion conventions, module contracts, reaction semantics | Contracts unit-tested, reviewed by team, CI green |
| **M2 — Mass properties** | Canonical mass inventory; stowed/partial/full deployment; explicit COM | Total mass consistent across mission configurations and inertia independently verified |
| **M3 — Attitude** | Rigid-bus Euler dynamics, quaternion propagation, torque input | Conservation and analytic rigid-body tests pass |
| **M4 — Environment** | Orbit, gravity/geomagnetic field, plasma-relative motion | Frames, time coverage and reference cases verified |
| **M5 — EDT current** | OML extraction, electrical boundary validation, Lorentz loading | Current orientation/boundaries and simple force cases verified |
| **M6 — Tether mechanics** | Correct zero-rest-length tension, physical masses, discretization, solver | Mechanics-only tests pass and discretization/time-step convergence demonstrated |
| **M7 — Tether outputs** | Full node/force/current history and diagnostics | Saved results reproduce numerical/physical diagnostics |
| **M8 — Mechanical coupling** | Bus translation/attitude ↔ tether reaction, no EDT loads | Momentum balance and coupling convergence validated |
| **M9 — Electrodynamic coupling** | Geometry-dependent current and distributed Lorentz forces | Current-on/off and analytic loading checks pass |
| **M10 — Numerical integration strategy** | Compare monolithic/staggered/reduced modes | Chosen solver has documented error/convergence and performance envelope |
| **M11 — Orbital mission analysis** | Orbit change/deorbit behavior, if in mission scope | Orbital energy/momentum and reference cases verified |
| **M12 — Visualization** | Read saved trajectories; plot and animate results | Figures produced without modifying physics models |
| **M13 — Release & validation** | End-to-end scenarios, documentation, reproducibility | Independent user can install, run, reproduce and inspect results |

M2/M3 and M4/M5/M6 can proceed concurrently once M1 contracts are reviewed.
M8 starts with **zero electromagnetic current** to isolate mechanical exchange.
M9 introduces electrodynamics only after mechanical coupling passes tests.

## Known issues (not fixed in M1)

1. The archived tether run collapses from an initially deployed configuration;
   this is **legacy behavior**, not a validated physical target.
2. The tether's smoothed tension law produces a nonzero tensile force at
   nominal zero extension in the archived parameter set.
3. Legacy mass scaling changes transient behavior; distinguish quasi-static
   and physical transient simulations.
4. The electrical coordinate/current sign needs validation against the actual
   cathode/collector boundary conditions and geometry.
5. The archived magnetic-field table clamps after its time range; future field
   sampling must not silently reuse stale values.
6. The inertia demo does not conserve total mass between the illustrative
   stowed and deployed configurations; M2 must reconcile component masses.
7. **Do not substitute instantaneous total assembly inertia** into rigid-bus
   Euler equations for a flexible tether: model the tether separately and apply
   its reaction on the bus.

## Golden-data policy

- Keep M0 references immutable unless reviewed with a documented justification,
  original input data, new checksums, and independent validation.
- Archived outputs are regression records **only** until their physics is checked.
- Any new numerical model must carry its own validated reference cases.

See [interfaces.md](interfaces.md), [m1-interface-spec.md](m1-interface-spec.md),
and [m0-baseline.md](m0-baseline.md).
