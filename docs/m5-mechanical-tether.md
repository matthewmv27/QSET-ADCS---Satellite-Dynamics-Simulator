# M5 — Physical mechanical tether (standalone)

**Scope:** prescribed attachment (node 0), **physical** transient mass–spring–damper
chain, isolated static-equilibrium option, SI units and inertial frame **I**.
**Out of scope:** orbit propagation, flexible-bus coupling, current, magnetic
fields, distributed electrodynamic loads, reel payout, rigid-bus attitude and
spin, beam bending stiffness, collision/contact, and numerical shape fitting
by inflating masses. Those belong to later integration milestones.

The archived notebook in `legacy/` remains unchanged. It is a *regression
record*, not a trusted transient solver.

## What changes versus the legacy notebook?

The archived solver used `smooth_max0(x,eps)=(x+sqrt(x*x+eps^2))/2`
for *both* extension and spring tension. At nominal rest, with the archived
run parameters `L=10 m`, `N_seg=5`, `EA=5000 N`, and `eps_frac=1e-4`, this
produced approximately **0.404508497 N** of nonphysical tension (a dedicated
test reproduces the number). M5 uses an exactly unilateral axial force law
instead; a numerical machine-precision tolerance only avoids representational
roundoff at nominal rest.

The old simulation also inflated nodal masses when stiffness was high, which
changes the physical transient equations. **M5 never scales masses**. For
quasi-static configurations it uses an independent potential-energy minimizer;
for transients it integrates the equations with the actual nodal masses.

The archived finite-difference Jacobian sparsity was not independently
validated; M5 does **not** pass a sparsity matrix to SciPy. Solver performance
may be slower for very stiff or finely segmented systems. Do not add a pattern
until its nonzero structure is checked against finite differences and the
possible environmental force callbacks.

## Constitutive law

For segment `j` of natural length `ell0 = L/N` and unit tangent `u_j`
from node `j` to `j+1`, with `ell = ||r[j+1]-r[j]||`, and axial extension
rate `dotell = (v[j+1]-v[j]) · u_j`, M5 defines:

```
T_j = 0,                                            ell <= ell0
T_j = max(0, EA/ell0*(ell-ell0) + etaA/ell0*dotell), ell > ell0
F_on_j =  T_j u_j
F_on_j+1 = -T_j u_j
```

`EA` [N] comes from `HomogeneousTether.axial_rigidity_N`, and `etaA` [N s]
is `CableModel.viscous_axial_rigidity_Ns`. This is a Kelvin–Voigt-like
axial dashpot active **only while stretched**, with no compressive force.
The transition is nonsmooth; choose an integration strategy and tolerance
appropriate for this one-sided law. Do not apply axial damping in slack.

## Physical mass, COM and inertia

Each segment carries actual tether mass `dm = rho * A * L / N`.
The trapezoidal lumped node weights are `[dm/2, dm, ..., dm, dm/2]`.
A separate `end_mass_kg` is added to the final node. The anchored half-mass
is retained at node 0, not silently dropped. The sum of node masses equals
`m_tether + m_end`, with no scaling.

At each sampled time, the module produces combined tether+tip COM and an
**approximate point-mass** tensor about that COM in I coordinates.
For a straight uniform slender rod without a tip mass, its transverse inertia
converges to `m*L**2/12` as `N` increases. Cross-sectional intrinsic inertia
is not included here; M2 supports more detailed static component accounting.

## Constraint reactions versus applied external loads

Node 0 follows the caller's prescribed attachment position, velocity, and
acceleration. To avoid subtracting huge orbital radii while integrating
metre-scale cable motion, **relative node positions/velocities** are evolved
internally. History snapshots are returned in I coordinates.

M5 returns **force ON BUS BY TETHER** at the attachment, inferred from node
0 force balance:

```
F_on_bus = F_spring_on_anchor + F_external_at_anchor
           + m_anchor*(g_anchor - a_anchor)
```

There is no free couple in the current pin-attachment model. Later rigid-bus
coupling should use the attachment arm and `attachment_reaction_on_bus`
(`edtsim.core.frames`) to convert it to a bus-frame force/torque. Do not also
apply the same external tether force directly to the bus.

M4e computes distributed **external** Lorentz and drag forces on a prescribed
`TetherState`. A future coupling adapter can map segment forces to the
corresponding nodal forces and call `external_nodal_forces_I(t,TetherState)`.
Such an adapter must consider material arc length versus dynamically stretched
length; the M4e drag evaluator currently validates their approximate equality.
**The mechanical-only reference has current exactly zero** by construction.

## Running M5

Install dependencies:

```bash
python -m pip install -e ".[dev,tether]"
python -m pytest tests/test_m5_mechanical_tether.py tests/test_m5_cli.py -v
```

Transient example:

```bash
python scripts/run_m5_mechanical_reference.py \
    configs/scenarios/m5_mechanical_example.json \
    --output-dir results/m5_reference
```

The example JSON is deliberately **illustrative and compliant**, not QSET
hardware. Replace it with team-reviewed mass, geometry, stiffness and damping
parameters when they become available.

To use the static mode, copy the JSON, change `"mode": "quasi_static"`,
provide constant gravity and a stationary attachment, and select an initial
shape close to the desired equilibrium. This does not produce a transient.
Slack geometries can admit nonunique equilibria; inspect the optimizer residual
and do not treat a stationary result as a demonstrated stable deployment.

### Outputs

Transient mode writes:

- `summary.json` — assumptions, masses, final state and solver diagnostics
- `history.npz` — complete arrays: node position/velocity, masses, actual
  segment lengths, tensions, COM, inertia, energies and attachment reaction
- `history.csv` — energies, force-on-bus, COM and maximum tension versus time
- `segments.csv` — **every segment's actual length and tension** at every sample
- `nodes.csv` — **every node's position, velocity and mass** at every sample

Quasi-static mode writes `summary.json` containing equilibrium positions,
segment lengths/tensions, potential energy and solver convergence diagnostics.
Neither mode silently replaces failed integrations with a final snapshot.

For constant gravity, kinetic, elastic and gravity-potential energy are
computed. **Total mechanical energy is only expected to be conserved for a
fixed anchor, no damping and no other external forces**. A changing anchor,
damping or external loads exchange/dissipate energy; callable gravitational
acceleration fields have no assumed potential and produce `NaN` for the
gravity-potential/total-energy diagnostic.

## M5 test/exit evidence

`tests/test_m5_mechanical_tether.py` covers:

1. Reproduction of legacy 0.404508497-N nominal-rest tension and its removal.
2. Exact zero tension for rest/slack/compression, and zero damping when slack.
3. Two *free* lumped masses: equal-opposite spring force, total momentum and
   energy conservation (and energy dissipation with damping).
4. Nodal mass conservation, tip-mass separation, COM and transverse inertia
   convergence toward the analytical uniform rod.
5. Mechanical-only prescribed-attachment runs, zero-current diagnostics,
   attachment reaction, fixed-gravity equilibrium and physical mass usage.
6. Transient integration step/tolerance sensitivity and segment-count
   convergence against a higher-resolution cable.
7. An independently finite-differenced elastic-potential gradient.
8. Input validation, full time-history and CLI artifact verification.

**Validation boundary:** M5 is only an isolated mechanical chain model. Its
success does not validate the electrodynamic-tether deployment/attitude coupling
or establish a stable free-flying spacecraft. The actual tether's Young's
modulus, damping, length, diameter/width and tip mass remain preliminary.

## Revised 50 m / 100-segment demonstration

The primary demonstration now represents a **50 m tether divided into 100
material segments (101 nodes including the prescribed bus attachment)**. Each
segment has a nominal rest length of **0.5 m**. The previous **2 m / five-
segment** case is preserved as `configs/scenarios/m5_mechanical_small_regression.json`
for quick tests and backwards comparison.

The 50 m example uses a small prescribed axial extension and a smooth 0.4 m
transverse initial bow, with the anchor stationary, no gravity, and **zero
current**. This is a test of the cable's mechanical response and force
transmission, **not** a prediction of 50 m in-orbit deployment or stability.
The deliberately soft illustrative material parameters are not QSET hardware.
They should not be used as a basis for designing the actual tether.

`_forces()` evaluates the same tension-only law over all segments with NumPy
vector operations, preserving the prior scalar law and its numerical
roundoff guard; the large mesh avoids 100 separate Python calls for each
RHS evaluation. Independent scalar/pair-spring tests check equivalence.

Run:

```bash
python scripts/run_m5_mechanical_reference.py \
  configs/scenarios/m5_mechanical_example.json \
  --output-dir results/m5_50m
python scripts/plot_m5_mechanical_reference.py results/m5_50m
```

Plot files include `tether_shapes.png`, `tip_transverse.png`,
`maximum_tension.png`, `attachment_force.png`, and
`mechanical_energy.png`. Their corresponding full numerical data are
stored in `history.npz` and the per-node/per-segment CSV files.

### Mesh refinement at the same 50 m physical geometry

```bash
python scripts/run_m5_mesh_convergence.py \
  configs/scenarios/m5_mechanical_example.json \
  --segments 25 50 100 200 \
  --output-dir results/m5_mesh_study
```

The script evaluates the **same** total length, material parameters, end
mass, initial smooth shape, and integration tolerances at four resolutions.
It writes `mesh_comparison.csv` and `mesh_summary.json` with physical mass,
maximum tension, solver evaluations, and differences in tip trajectory,
final shape and attachment force against the finest mesh.

**Important:** the finest mesh is *only a reference solution*, not exact
truth. Observe whether the differences decrease as resolution increases;
repeat with tighter time tolerances/step sizes if they do not. A bigger
`segments` value by itself does not validate bending, gravity gradients,
deployment mechanics, or the actual spacecraft/tether coupling.

For a meaningful comparison of stiffness and speed, keep `length_m`, Young's
modulus, density, end mass, damping, and the other physical parameters fixed
while changing only `segments`. High-resolution stiff cases may require a
validated sparse Jacobian or other numerical improvements in a later step.

## Orbit-driven straight-tether demonstration (M5 extension)

The 50 m sinusoidally bowed `m5_mechanical_example.json` is retained as a
**mechanical regression case**. For the physically more informative *initially
straight* reference, see [`m5-orbit-driven-demonstrator.md`](m5-orbit-driven-demonstrator.md).
That runner provides a prescribed 500 km two-body circular attachment,
pointwise central gravity, co-rotating initial node velocities, and an optional
**synthetic** Lorentz load through M4e's segment-to-node adapter. The orbit is
not propagated by M5, the magnetic field/current are not mission estimates,
and the CubeSat still does not respond to the attachment reaction (M6).
