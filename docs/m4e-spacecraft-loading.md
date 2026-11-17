# M4e — Spacecraft-side electrodynamic loading and disturbance interfaces

## Scope

M4e **does not propagate an orbit**. This milestone extracts instantaneous external
forces on a prescribed tether shape for later use by the flexible tether solver
and, eventually, the orbital team's Orekit simulator.

- Current coordinate `s`: arc length from the CubeSat attachment (`s=0`) toward
  the far zenith end (`s=L`). Positive **conventional** current is bus-to-tip;
  conductor electron flow is toward the bus.
- Inertial frame `I` is an abstract shared right-handed frame; all external
  providers must explicitly agree on its convention. Vectors must be in SI.
- Node ordering is attachment-to-tip, in `TetherState(r_nodes_I,v_nodes_I)`.
- The CNT cathode sits on the bus but its emitted-electron pointing direction
  is not used in the conductor Lorentz-force law. This code assumes neither
  the emission direction nor the entire plasma return circuit is established.
- The first electrical snapshot takes **imposed** bus-to-plasma potential,
  uniform electron density and temperature, and an average longitudinal
  motional field. The 1D OML solver returns a current **demand**, not a
  self-consistent electrical operating point. CNT checks compare that demand
  against **assumed** gate-voltage/current/power limits. Matching limits is
  necessary but never sufficient for feasible operation. The digitized CNT
  cathode data are approximate and unverified for flight hardware.

## Interfaces

```python
from edtsim.electrodynamics import (solve_prescribed_bias_snapshot,
                                  evaluate_distributed_loading)
```

`solve_prescribed_bias_snapshot(...)` yields a `SpatialCurrentProfile` with
`segment_currents_A(t_s, midpoint_arc_lengths_m)` and cathode-demand diagnostics.
The profile never extrapolates outside its `[0,L]` domain.

`evaluate_distributed_loading(time_s, tether_state, field, current_source,
reference_point_I_m=..., ...)` computes local and net **external tether** forces.
A `field` supplies `sample(t_s, r_I_m) -> B_I_T`, optionally from a real
geomagnetic adapter; plasma and neutral velocities are independent callbacks.

At each segment: `dF_L = I(s) dl_I × B_I`. The returned net torque is
`sum((r_mid - r_ref) × (dF_L + dF_drag))` about the **specified reference
point**. It is an equivalent total external moment of the tether's loads,
**not the torque applied to the rigid bus by the tether attachment**.
Attachment reactions require a tether dynamics force/momentum balance and
will be provided by M5/M6 through `core.frames.attachment_reaction_on_bus`.
Do **not** also apply these distributed forces directly to the bus if the
tether dynamics model already contains them.

For local segment velocity `v`, plasma velocity `v_p`:

- Distributed mechanical power: `P_mech = sum(dF_L · v)`.
- Motional EMF power: `P_emf = sum(I ((v-v_p) × B) · dl)`.
- Plasma-motion term: `P_plasma = sum(dF_L · v_p)`.
- Identity: `P_mech + P_emf - P_plasma = 0` (numerical sign test).

The power identity is **not** a plasma-current circuit or energy closure.

## Run a reproducible, illustrative check

```bash
python -m pip install -e '.[dev,dynamics]'
python -m pytest
python scripts/run_m4e_loading_reference.py \
    configs/scenarios/m4e_loading_reference.json \
    --output-dir results/m4e_loading
```

The example explicitly uses a synthetic uniform field/flow, an imposed bus
potential, arbitrary plasma density and temperature, a stationary straight
tether aligned radially away from Earth, and preliminary scalar drag. Its
`summary.json` reports CNT **demand and limit checks**, applied external force
and torque, and power residual. `segment_loading.csv` gives the spatial
current/force/drag arrays. No spacecraft motion is propagated.

## Requirements to extend after M4e

1. Replace uniform plasma assumptions and the averaged EMF with spatial
   plasma and local EMF inputs appropriate to the evolving tether nodes.
2. Close bus charging, collecting conductor, cathode emission, and power
   electronics in one voltage/current balance; use independent references.
3. Include possible ion collection, sheath/geometry validity limits, and
   realistic CNT emitter characterization.
4. Couple distributed loads to a flexible tether solver, which returns
   attachment force and couple *on the bus*. Verify action-reaction,
   energy/momentum conservation and timestep convergence.
5. Define epoch, frame, coordinate and force interfaces for the orbital
   team's integrator; never advance two independent orbital trajectories.

## Validation limits

Deterministic tests cover vector signs, power identity, zero-current behavior,
segment subdivision, geometry and input validation, CNT constraint flags,
OML profile adapter and CLI results. They cannot establish whether the
assumed bus-to-plasma potential is achievable or what the actual QSET current
will be. Existing WMM and IGRF reference tests remain separate from M4e.
