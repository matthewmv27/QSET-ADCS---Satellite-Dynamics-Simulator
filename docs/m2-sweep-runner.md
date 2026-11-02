# M2 extension — Parameter sweeps and versioned input configurations

**Scope:** static mass-property snapshots using the already-tested `MissionMassModel`.
These results are **not** deployment trajectories, dynamic forces, aerodynamic
loads, or approved QSET flight mass properties. This extension does not alter
`mission.py`, the legacy notebooks, or M0 golden fixtures.

## Run an illustrative study

From the repository root:

```bash
python -m pip install -e '.[dev]'
python scripts/sweep_mass_properties.py configs/scenarios/m2_illustrative.json -o results/m2_illustrative
python -m pytest
```

To also save PNG graphs (optional visualization dependency):

```bash
python -m pip install -e '.[viz]'
python scripts/sweep_mass_properties.py configs/scenarios/m2_illustrative.json -o results/m2_illustrative --plot
```

The example produces **18 cases**: 5 deployment fractions of one spacecraft,
12 cases from a 2-length × 2-tip-mass × 3-fraction design grid, and one bent
configuration. Values are **illustrative only**, not QSET design choices.
Use a different output folder for every study revision. Generated `results/`
files should not be included in source-control commits unless intentionally
reviewed as reference artifacts.

## Input file layout

Use **JSON schema version 1**, as shown in
[`configs/scenarios/m2_illustrative.json`](../configs/scenarios/m2_illustrative.json).
It has these top-level keys:

- `schema_version`: exactly `1`.
- `id`: stable case-set identifier (letters/digits/underscore/hyphen).
- `description`, `revision`: nonempty human descriptions and revision ID.
- `provenance`: `{ "status": ..., "source": ..., "revision": ... }`.
- `parameter_provenance` (optional): overrides the default provenance on
  selected parameter keys; all unspecified inputs inherit global provenance.
- `model`: **one** mass inventory for one spacecraft design.
- `sweeps`: one or more studies referencing the inventory. For `design_grid`
  and `bent_cases`, each sweep **must** also declare its own
  `provenance` (status/source/revision), since hypothetical variations may be
  less reliable than the underlying base hardware parameters.

Provenance `status` must be `illustrative`, `preliminary`, or `verified`. This is
**declared by the author of the input**, not verified by the program. Never
mark values as verified without traceable evidence (such as measured values or
reviewed CAD revision). `source` and `revision` are required so later results
can be traced back to input assumptions.

### Model inventory

| JSON field | Expected inputs | Model meaning |
|---|---|---|
| `bus` | `mass_kg`, `dimensions_m` [x,y,z] | Uniform box approximation, center at bus COM |
| `spool_hardware` | `mass_kg`, `com_B_m`, `inertia_com_B_kgm2` | Hardware only, tether material excluded |
| `tether` | `length_m`, `linear_density_kg_per_m`, `attach_B_m` | Material mass = full length × linear density |
| `end_mass` | `mass_kg`, `stowed_B_m` | Point end mass; deployed at tip |
| `wound_geometry` | `r_inner_m`, `r_outer_m`, `width_m`, `axis_B`, `com_B_m` | Idealized annulus of wound tether |
| `other_fixed` | Optional list of `name`, `mass_kg`, `com_B_m`, `inertia_com_B_kgm2` | Extra fixed components outside bus and spool |

**Reference conventions:** `B` axes are the body frame; its origin is the **rigid
bus COM**. All numbers use SI units, and inertia tensors are 3×3 matrices in
kg·m², about each component's own COM. See `docs/interfaces.md`.

`bus` is a uniform rectangular-box approximation; current M2 code does not yet
accept an arbitrary CAD-derived rigid-bus inertia tensor as a direct substitute.
Other fixed components and spool hardware *do* accept individual COM tensors.
Do not add hardware twice in `bus` and `other_fixed`.

### Sweep types

**1. Deployment sweep** (`kind: "deployment"`)

```json
{"id":"deployment_fraction","kind":"deployment","fractions":[0,0.25,0.5,0.75,1],"direction_B":[0,0,-1]}
```

The entire inventory is fixed. `wound_tether_mass = (1−f) × tether_mass` and
`deployed_tether_mass = f × tether_mass`; total mass is conserved across fractions.

**2. Design grid** (`kind: "design_grid"`)

```json
{"id":"design_comparison","kind":"design_grid", "axes": {"tether.length_m":[5,10], "end_mass.mass_kg":[0.1,0.2]}, "fractions":[0,0.5,1], "direction_B":[0,0,-1],"provenance":{"status":"illustrative","source":"Selected design candidates","revision":"study-v1"}}
```

The runner evaluates the Cartesian product of all axes and fractions.
Supported scalar design axes:

- `bus.mass_kg`
- `tether.length_m`
- `tether.linear_density_kg_per_m`
- `end_mass.mass_kg`
- `spool_hardware.mass_kg`
- `wound_geometry.r_inner_m`, `wound_geometry.r_outer_m`, `wound_geometry.width_m`

The product `length_m × linear_density_kg_per_m` is used to derive tether mass;
changing length at fixed density therefore **changes spacecraft design mass**.
Do **not** interpret design-grid results as one mass-conserving mission history.
Within one design combination, varying only deployment fraction conserves mass.

**3. Specified bent geometry** (`kind: "bent_cases"`)

```json
{"id":"geometry_examples","kind":"bent_cases","cases":[
  {"id":"bent_A","fraction":1,
   "nodes_B_m":[[0,0,-0.15],[3,0,-4.15],[0,0,-8.15]],
   "reference_lengths_m":[5,5]}
],"provenance":{"status":"illustrative","source":"Illustrative bending geometry","revision":"study-v1"}}
```

For each bent case, the first node must be the tether attachment point.
`reference_lengths_m` (when provided) represents **unstretched material
lengths**, whose sum must equal `fraction × tether.length_m`. Tether mass is
assigned by these *rest lengths*, not by possibly stretched node distances.
These are prescribed shapes, not simulated tether dynamics.

## Outputs and reproducibility

In the requested folder the runner saves:

- `mass_properties.csv`: one row per case with scenario and source-status
  labels (both base inventory and sweep-specific), design inputs, total mass, three COM coordinates, **all nine** matrix
  entries for inertia about bus COM and **all nine** about system COM, and
  sorted principal moments; components and overrides as embedded JSON strings.
- `mass_properties.json`: complete per-case input models, body-frame geometry,
  component mass ledgers, tensors, user-declared source/revision metadata,
  assumptions/limitations, and a SHA-256 of the canonical scenario contents.
- With `--plot`, separate PNGs for Ixx, Iyy, and Izz about **system COM** vs
  deployed fraction, for each straight/configuration study. Bent results are
  available in CSV/JSON rather than plotting against fictitious time.

Rows use **body axes**. Distinguish the inertia tensor about *bus* COM from the
tensor about *total assembly* COM. Neither is a rigid-bus-only attitude-solver
inertia when the tether is flexible.

Outputs are deterministic for the same scenario content and implementation;
there are no timestamps that would cause a result to differ on every run.
An invalid scenario or unphysical parameter combination **raises an error**;
the runner does not silently discard failing design cases. The default limit
is 2,000 cases; use `--max-cases` to adjust consciously. Plotting is limited to
16 distinct design curves per sweep, to avoid unreadable plots.

## Important limitations and future interfaces

- The wound tether is modelled as a fixed annulus with declining mass, not
  actual layer-by-layer payout.
- At exactly zero deployment, the end mass uses `stowed_B_m`; it may move
  discontinuously to the modeled tip for an infinitesimal positive fraction.
- The bus is a uniform box; cable and deployed segments are slender zero-radius
  rods; the end mass is a point mass.
- Winding capacity and physical feasibility are **not** checked when design
  sweeps alter tether length or material density while holding spool geometry fixed.
- CAD data and the developing deployment mechanism are not yet available.
- **Atmospheric drag is intentionally absent from M2.** It is an external force
  and torque that depends on atmosphere-relative motion, density, exposed area
  and flow direction. Add drag after environment and tether-force interfaces
  exist (M4/M6), not by modifying these mass-property equations.
- The legacy OML/current-profile model is not invoked. The improved payload
  current model should later be integrated through an interchangeable
  electrodynamics interface (M5).

When preliminary hardware properties become available, create a new scenario
revision, preserve the previous one, update provenance at both scenario and
parameter level, and compare results under a clear design-change report.
