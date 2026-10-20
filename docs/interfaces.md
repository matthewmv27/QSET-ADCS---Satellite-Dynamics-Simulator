# Interfaces and conventions

This is the contract between subpackages. If code and this file disagree, fix one of them in the same commit.

## 1. Conventions

| Topic | Convention |
|---|---|
| Units | SI everywhere internally (m, kg, s, rad, A, T, N). Degrees only at user-input/display boundaries. |
| Frames | **I** = Earth-centred inertial (GCRS). **B** = body frame, origin at the **bus COM**. |
| Attitude | `R_IB` maps body vectors to inertial: `v_I = R_IB @ v_B`. |
| Euler angles | Intrinsic ZYX: `R_IB = Rz(yaw) @ Ry(pitch) @ Rx(roll)`. |
| Inertia tensors | About the **bus COM**, in the **body frame**, unless the name ends in `_I` (inertial axes). For the coupled model use `shift_inertia_to_com` to move to the system COM. |
| Vectors / arrays | A single vector is shape `(3,)`; a set of N is `(N, 3)`; a time history is `(Nt, ...)`. |
| Deployed tether | Config 2 lies along `-z_B` from the attachment point `attach_B`. |
| Variable suffixes | `_B` body frame, `_I` inertial frame, `_com` about own COM. |

## 2. Dependency rule (enforced by `tests/test_architecture.py`)

```
core  <-  inertia  <-  tether  <-  coupled
core  <-  environment  <-----------^
core  <-  edt_current  <-----------^
core  <-  viz
```

A subpackage may import only from the ones to its left. `tether` does **not** import `environment` or
`edt_current`: it receives `B(t)` and `I(s, t)` as arguments (arrays or callables). That keeps astropy/WMM out of the
solver and lets every piece be tested with synthetic inputs.

## 3. Results file (DRAFT, finalised in Stage 3)

Written from the legacy v2 exporter's fields; **shapes and units to be confirmed against the running code** in Stage 3.

| Field | Shape | Meaning |
|---|---|---|
| `t` | `(Nt,)` | time [s] (pseudo-time if `mass_scale != 1`, see `meta`) |
| `r_nodes` | `(Nt, N_nodes, 3)` | tether node positions, inertial frame [m] |
| `x_bus` | `(Nt, 3)` | bus COM position, inertial frame [m] |
| `fL_seg` | `(Nt, N_seg, 3)` | Lorentz force on each segment [N] |
| `I_seg` | `(Nt, N_seg)` | current in each segment [A] |
| `B_eci` | `(Nt, 3)` | geomagnetic field at the bus [T, confirm] |

Planned additions: `r_rel = r_nodes - x_bus` (better for plotting), `R_IB(t)`, bus dimensions, tension at the
attachment node (the force the bus actually feels), and a `meta` block (schema version, units, frame tags, epoch,
orbit, tether parameters, `mass_scale`, code version). Canonical format: `.npz` (numpy only); `.mat` is an optional
adapter written with scipy for MATLAB users.

## 4. Legacy name map

| Legacy notebook name | New location |
|---|---|
| `hat`, `unit`, `Rx`, `Ry`, `Rz`, `euler_zyx_to_R`, `rotation_matrix_from_a_to_b` | `edtsim.core` |
| `BusParams`, `SpoolParams`, `TetherParams`, `PointMassParams` | `edtsim.core` (now validated) |
| `parallel_axis_shift`, `point_mass_inertia_about_origin`, `box_/annular_cylinder_/slender_rod_inertia_about_com`, `inertia_body_to_inertial` | `edtsim.inertia` |
| `moi_config_1_coiled`, `moi_config_2_extended_on_minus_z` | `edtsim.inertia` (config 2 returns a `NamedTuple`) |
| `sample_point_in_annular_cylinder` | `edtsim.inertia` |
| `compute_inertia_from_particles_about_bus_com_in_inertial` | `edtsim.inertia.inertia_from_particles(r, m, r_ref)` (reference point is explicit) |
| orbit, WMM, astropy transforms, B-grid | `edtsim.environment` (Stage 2) |
| OML current-profile solver | `edtsim.edt_current` (Stage 2) |
| tether chain RHS, solver, settle detection | `edtsim.tether` (Stage 3) |
| `input()` wizard `main()` | config files + CLI (later) |
