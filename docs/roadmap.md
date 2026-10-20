# Roadmap

Each stage ends with a tag. A stage is done when its tests pass and its output reproduces the frozen legacy numbers
(or a documented, intentional change).

| Stage | Tag | Content | Status |
|---|---|---|---|
| 0 | `v0.0` | repo, `core`, conventions, golden-data workflow | in progress |
| 1 | `v0.1-inertia` | `inertia`: building blocks, configs 1 and 2, particle inertia, validation | skeleton delivered |
| 2 | `v0.2-environment` | `environment` (orbit, WMM, tables) and `edt_current` (OML profile), built in parallel | |
| 3 | `v0.3-tether` | `tether`: chain dynamics, settle detection, results file, tether force at attachment | |
| 4 | `v0.4-viz` | `viz`: plots/animation from the results file | |
| 5a | `v0.5-torque` | tether force -> torque about the **system** COM at prescribed attitude | |
| 5b | `v0.6-attitude` | `attitude_dynamics`: Euler equations + quaternions, time-varying inertia | |
| 6 | `v1.0-coupled` | `coupled`: tether + attitude, physical masses | |

## Golden-data policy
* Configs 1 and 2 are frozen now (`tests/fixtures/golden_inertia.json`, generated from the original notebook).
* Config 3 results are frozen as **"legacy behaviour"** only after the open items marked (!) below are resolved, so we
  don't enshrine a bug.

## Open items found while reading the notebooks
1. (!) **Current-profile orientation.** The ODE has I=0 at x=0 and the cathode boundary at x_max, but segment positions
   are measured from the bus. With a cathode on the bus this looks mirrored (zero current at the bus, peak at the tip).
   Check by plotting `I_seg[0, :]`; verify against your solver notebook.
2. **Motional EMF uses inertial velocity**; the plasma corotates with Earth (several percent effect).
3. **B(t) clamps after one orbit**; attitude studies need B(t) for arbitrary t.
4. **`mass_scale` makes the tether solver quasi-static** (fine for shape, not for transients). Coupled stage needs `mass_scale = 1`.
5. **Exporter recomputes forces** after the fact (finite-difference COM velocity); one function should serve both the RHS and the export.
6. **One-way coupling today:** attachment is prescribed, so tether force is never applied to the bus; inertia is about the bus COM, not the system COM.
7. Housekeeping: config fields set via `hasattr`; `jac_sparsity=None` despite the builder; docstring says Radau but default is BDF; legacy v2 export block has 3 mis-indented lines.

## Validation cases planned for Stages 5-6
Torque-free rigid-body precession; conservation of angular momentum; gravity-gradient pitch libration at sqrt(3)·n for a
dumbbell; tether-off limit reproduces the pure rigid-body result.

## Decision pending (Stage 6): coupling strategy
Monolithic stiff integrator vs staggered co-simulation vs reduced-order tether. Interfaces are kept neutral so any can be tried.
