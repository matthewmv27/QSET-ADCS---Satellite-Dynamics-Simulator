# M2 — Configuration-dependent assembly mass properties

**Status:** initial tested implementation; hardware-specific values **not** validated.
This module creates static assembly mass-property snapshots in the M1 body
frame, with origin at the **rigid bus COM**. It does not integrate dynamics.

## Scope and mass bookkeeping

Use `edtsim.inertia.mission.MissionMassModel`. Its input is a *single* mass
inventory, deliberately separated into these nonoverlapping categories:

| Inventory item | Included mass | Model |
|---|---|---|
| `bus` | Rigid bus dry structure + internal components already included in bus mass | Box approximation with origin at its COM |
| `spool_hardware` | Reel, core, spindle, supports; **not** the tether itself | Independent component with measured or CAD COM/tensor |
| `tether` | Total conducting tether material | Mass is partitioned into wound and deployed fractions |
| `end_mass` | End mass / tip | Point approximation at stored or deployed position |
| `other_fixed` | Optional fixed payloads not already counted in bus | Explicit mass, COM and COM tensor |

For deployment fraction `f` between 0 and 1:

- `wound_tether_mass = (1-f) * tether.mass`
- `deployed_tether_mass = f * tether.mass`
- `deployed_tether_length = f * tether.length`
- `end_mass` and `spool_hardware` are each counted exactly once
- `total_mass` must be identical for every configuration

At `f=0`, the end mass is at `end_stowed_B`. At `f>0`, the simple straight
configuration assumes the end mass is at the deployed tip. These are **static
snapshots**, not a model of the end mass moving through the deployment hardware.
An abrupt change between `f=0` and infinitesimal deployment is possible and is
not a physical trajectory. For deployment continuity, the mechanical payout
path, stowed cable routing, and tip release geometry must be modelled explicitly.

The remaining wrapped tether is approximated as a uniform annular cylinder of
**fixed** inner/outer radius and height with decreasing mass. Real layer-by-layer
unwinding will change its density/shape and inertia. Replacing this approximation
requires measured winding geometry or spool packing data.

## Reference points and aggregation

Every `MassComponent` has `mass_kg`, `com_B`, and `inertia_com_B`. For an assembly,

```text
m_total = sum(m_i)
r_COM_B = sum(m_i * r_i_B) / m_total
I_bus_B = sum(I_i_COM_B + m_i (|r_i_B|^2 1 - r_i_B r_i_B^T))
I_assembly_COM_B = I_bus_B - m_total (|r_COM_B|^2 1 - r_COM_B r_COM_B^T)
```

`AssemblyMassResult` returns both tensors plus `MassProperties` and an auditable
component ledger. Both tensors are expressed in **body axes**. The reference
point differs and must be retained in reporting, comparisons and downstream use.
Do **not** insert the flexible assembly's time-varying inertia into the bus-only
rigid-body Euler equation; the bus and tether have separate dynamical states.

## Bent tether geometry

Call `model.bent(fraction, nodes_B, reference_lengths_m=None)`.

- Nodes are in body axes and begin at the bus attachment point.
- Each adjacent pair is integrated as an **exact uniform line segment**, not a
  point mass. The end mass is located at the final node.
- Segment masses depend on **material rest-length allocation**, not their
  potentially stretched instantaneous length. By default all segments
  represent equal *unstretched* arc length; override with `reference_lengths_m`
  when nonuniform material discretization is used.
- The sum of the supplied rest lengths must equal the nominal paid-out length.
- Node geometry can be stretched, curved or locally slack. The function only
  computes the mass properties of that geometry; physical validity is assessed
  by the tether mechanical model in M6.
- The M2 approximation treats each segment as infinitesimally thin and does
  not model bending stiffness, local twist, or finite cross-section inertia.

## Examples and checks

From the repository root after installing editable dependencies:

```bash
python -m pip install -e '.[dev]'
python -m pytest
python examples/mass_properties_quickstart.py
python scripts/verify_baseline.py
```

**Analytical validation:** tests cover the rod+tip formula, parallel-axis
identity, component mass accounting at multiple deployment fractions, exact
straight-line equivalence for discretized line segments, and the O(N^-2)
convergence of lumped midpoint approximations toward a continuous rod. They
also test physically invalid inputs, duplicate ledger names, and reference
lengths independent of current segment stretch.

M0's golden fixture and old `examples/inertia_quickstart.py` remain unchanged;
they are deliberately historical regression cases and need not conserve mass.
`examples/mass_properties_quickstart.py` uses a *single* consistent illustrative
inventory; the values are not from the QSET mission's approved hardware CAD.

## Inputs required to call M2 mission-validated

1. Confirm which hardware is included in the bus dry mass and whether the
   spool's supports/reel/core are accounted for elsewhere.
2. Obtain a mass, COM, orientation, and COM inertia tensor for each major rigid
   component, with a clear body frame and reference point. The default box model
   for the bus is a first-order approximation only.
3. Obtain conductor total mass, deployed length, cross-sectional details, and
   measured spool winding-layer dimensions as a function of deployment.
4. Locate the stowed end mass, tether exit/attachment point, and actual payout
   path. Check whether the end mass moves before a taut cable develops.
5. Decide whether the mission configurations are snapshots of mass deployment,
   a continuous dynamic state, or a mixed series of operational stages.
6. Supply actual validation tolerances; the tests currently validate math
   independently but **do not** establish CAD-level accuracy.

Use `configs/mass_inventory_template.csv` to assemble component data. Blank
fields in that template are intentional: no flight hardware properties were
invented for this milestone.
