# Shared interfaces and conventions (M1)

**Status:** M1 contract for future models. The packaged production code currently
implements `core` and `inertia`; `attitude`, `environment`, `edt_current`,
`tether`, and `coupling` are not yet implemented as independent production models.
See `docs/m1-interface-spec.md` for the detailed integration data flow.

These conventions define data exchange, not a validated physical model. If new
code disagrees with this contract, resolve the discrepancy in a reviewed PR.

## 1. Frames, rotations, and units

| Item | Contract |
|---|---|
| Units | SI throughout: m, kg, s, rad, N, N·m, A, T, kg·m²; model-specific plasma units must be converted at boundaries |
| `I` inertial frame | Earth-centered GCRS axes for the orbital/environment interface. Do not silently mix another ECI convention |
| `B` bus body frame | Right-handed orthogonal axes, origin **at rigid bus COM**; assumed fixed to the rigid bus |
| `E` Earth-fixed frame | For magnetic/geographic models only; transform to `I` before exporting to dynamics |
| `R_IB` | Active rotation from body to inertial: `v_I = R_IB @ v_B`, `R_BI = R_IB.T` |
| `q_IB` | Hamilton **scalar-first** quaternion `(w, x, y, z)`, unit length, represents `B -> I` |
| Euler input | Intrinsic ZYX: `R_IB = Rz(yaw) @ Ry(pitch) @ Rx(roll)`; all angles in radians |
| `omega_B` | Bus angular velocity relative to inertial, expressed in body coordinates, rad/s |
| Quaternion rate | `dq_IB/dt = 0.5 * (q_IB ⊗ [0, omega_B])` |
| `t_s` | Seconds elapsed since mission simulation start. The UTC epoch is separate metadata |
| Single vectors | `(3,)` for position, velocity, force; single quaternion `(4,)` |
| Node arrays | `(N_nodes, 3)`; time histories `(Nt, ...)`; node zero is bus attachment |
| Axis suffixes | `_B` in body coordinates, `_I` in inertial; no unsuffixed vector fields in public model contracts |

A quaternion and its negative represent the same orientation. `BusState` checks
that `q_IB` is unit length; normalization is **explicit** (`normalize_quaternion`)
rather than silently changing a state. `R_IB` is not a synonym for a generic ECI
coordinate rotation.

`edtsim.core.rotations` provides legacy Euler matrix utilities;
`edtsim.core.frames` adds quaternion and attachment helpers consistent with those matrices.

## 2. Mass-property frames and reference points

- The existing `moi_config_1_coiled` and `moi_config_2_extended_on_minus_z`
  return tensors **about the bus COM, in body axes**. These are *assembly*
  inertias about a reference point, not automatically rigid-bus inertias.
- A new `MassProperties` snapshot contains total mass, `com_B` (assembly COM
  relative to **bus COM**) and `inertia_com_B` (assembly tensor **about the
  assembly COM**, expressed in body axes).
- The rigid attitude solver must use the rigid **bus** inertia about the bus
  COM. A flexible tether has separate positions, velocities, and forces; its
  contribution to momentum enters through coupling, not by blindly replacing
  bus inertia with a time-dependent assembly inertia.
- Treat a rigidly deployed tether as part of a *single rigid body* only when
  that assumption is explicitly part of the model and documented.
- `inertia_from_particles(r, m, r_ref)` requires all positions and the reference
  point in the same axes; the function itself has no frame metadata.

## 3. Shared M1 data snapshots

| Class | Fields | Meaning |
|---|---|---|
| `BusState` | `r_I`, `v_I`, `q_IB`, `omega_B` | Bus COM state, inertial position/velocity; body quaternion/rate |
| `AttachmentState` | `r_I`, `v_I` | Fixed attachment position and velocity in inertial axes |
| `AttachmentReaction` | `force_on_bus_I`, `couple_on_bus_I` | Tether force (and optional free couple) **on bus**, inertial axes |
| `MassProperties` | `mass_kg`, `com_B`, `inertia_com_B` | Combined rigid/assembly mass properties and explicit reference |
| `TetherState` | `r_nodes_I`, `v_nodes_I` | `(N,3)` positions/velocities in inertial axes, node 0 attached |
| `EnvironmentSample` | `t_s`, `position_I`, `B_I`, `plasma_velocity_I` | Field in tesla; plasma velocity in m/s, inertial |

All are validated data containers, **not solvers**. Arrays are copied and marked
read-only. They can be converted to floating-point arrays inside an integrator,
which manages its own state representation.

## 4. Mechanical attachment interface

Assume a point fixed on the bus at `attach_B` [m] relative to bus COM.

```text
r_attach_I = r_bus_I + R_IB @ attach_B
v_attach_I = v_bus_I + R_IB @ cross(omega_B, attach_B)
```

A rotating/reeling attachment needs an extra relative-velocity term; this
fixed-point interface is not sufficient for deployment out of a moving spool.

The tether returns its reaction **ON THE BUS** as force `F_bus_I` and optional
free couple `M_bus_I`; converting to bus force and torque about bus COM gives:

```text
F_bus_B = R_IB.T @ F_bus_I
Tau_bus_B = cross(attach_B, F_bus_B) + R_IB.T @ M_bus_I
```

The tether experiences the **opposite** attachment reaction. Lorentz forces
are applied to tether segments; do not apply the full distributed force to the
bus *and* its mechanical reaction as two separate inputs.

## 5. Future environment, current, and tether contracts

- **Environment:** sampled `t_s -> B_I` and plasma velocity at each required
  evaluation location; specify model version, UTC start epoch and valid time
  domain. Reject or explicitly regenerate out-of-range field requests, never
  silently clamp the last sample.
- **EDT current:** define geometric arc length `s_m` **from the bus outward**;
  segment `i` goes from node `i` to node `i+1` in that direction. **Signed
  conventional current** is positive along increasing `s_m`. This is a
  software sign convention, not a claim that the actual physical current flows
  outward. Validate cathode polarity, collection boundary conditions and OML
  integration direction before implementing the adapter.
- **Lorentz load:** `dF_I = I_signed * (dell_I × B_I)` applied per segment,
  with `dell_I = r_(i+1)_I - r_i_I` in the positive arc-length direction.
  Segment currents are in A, fields in T, and force in N.
- **Tether:** node zero coincides with the attachment position/velocity for
  the fixed-point interface; the end mass is node `-1` if the chain model
  uses that discretization. A future tether package must document its mass
  lumping, elasticity, and solver modes.

## 6. Dependency boundary

```text
  core <--- inertia            core <--- attitude
    ^           ^                ^
    |           |                |
  environment   tether       future coupling orchestrator
  edt_current   (receives field/current as inputs; no WMM/Astropy imports)

  coupling may depend on core, inertia, attitude, environment, edt_current, tether
  viz reads recorded result data, not private solver internals
```

`tests/test_architecture.py` enforces static package-import boundaries. Heavy
WMM/Astropy dependencies must not be introduced into core, inertia or attitude.

## 7. Trajectory records (provisional; NOT an implemented exporter)

Target `.npz` format, with a `schema_version`, epoch, simulation units, frames,
reference-point metadata, code version, model parameters, and solver options.
Suggested arrays: `t_s (Nt,)`, `r_bus_I (Nt,3)`, `v_bus_I (Nt,3)`,
`q_IB (Nt,4)`, `omega_B (Nt,3)`, `r_nodes_I (Nt,N,3)`,
`v_nodes_I (Nt,N,3)`, `current_seg_A (Nt,N-1)`,
`force_lorentz_seg_I (Nt,N-1,3)`, `force_attach_on_bus_I (Nt,3)`.

Arrays may be absent for simulations that do not include a subsystem; metadata
must tell readers which fields are included. A future task will finalize
serialization and adapters, including optional MATLAB export. Legacy `.mat`
files are not automatically assumed to follow these conventions.
