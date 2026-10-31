# M1 — Physical interface acceptance and integration plan

M1 introduces **contracts and unit-tested transformations only**. No new tether,
attitude, or orbit dynamics have been implemented. This is a minimal
implementation suitable for a separate pull request following M0.

## Why M1 precedes further dynamics

The existing inertia package returns assembly inertia about the bus COM, while
legacy tether nodes use an inertial coordinate system and prescribed attachment
motion. A future two-way simulator needs clear frames and reaction-force
semantics; otherwise a numerically stable simulation can still produce wrong
torques or double-count external loads.

## Module flow

```text
Mission inputs -> mass inventory/configurations -> MassProperties
            -> bus initial state (BusState)
Orbit/environment model ---------------> EnvironmentSample
BusState + attach_B -------------------> AttachmentState
AttachmentState + environment/current --> TetherState, segment loads
Tether loads and boundary constraints --> AttachmentReaction (ON BUS)
BusState + AttachmentReaction ----------> bus wrench (F_B, Tau_B)
Coupled time integrator ---------------> synchronized trajectories/results
```

### Boundary contracts by workstream

| Model | Inputs | Outputs | M1 disposition |
|---|---|---|---|
| Inertia | component mass/geometries, configuration | mass/COM/tensor with explicit frame and point | Existing functions retained; `MassProperties` future adapter |
| Attitude | BusState, rigid bus inertia, torque_B | next BusState | Planned; quaternion convention now implemented/tested |
| Environment | time, epoch, position, orbit inputs | EnvironmentSample | Planned; output snapshot and units defined |
| Current | tether segment geometry, relative plasma velocity, EM parameters | signed `I_seg` and loads | Planned; sign convention defined, physical orientation unresolved |
| Tether | AttachmentState, TetherState, external loads, physical parameters | next TetherState, AttachmentReaction | Planned; sign of reaction now defined |
| Coupling | all above, time/solver settings | synchronized history and diagnostics | Planned; no solver algorithm selected in M1 |

The `BusState` includes bus translation **even if** some initial tests prescribe
an orbit. Constraining the bus position affects momentum-exchange validation.

## Grounded design choices and unresolved decisions

1. `I` uses the **GCRS** convention in the existing documentation; verify the
   environment notebook agrees before extraction. Treat ECEF separately.
2. Quaternion uses scalar-first Hamilton ordering. Test conversions against
   the existing Euler rotation helpers, rather than assuming the notebook uses
   the same quaternion representation (it currently uses prescribed Euler angles).
3. The body frame has origin at the *rigid bus COM*, not the moving total system
   COM. This avoids a moving reference for the bus Euler equations.
4. For dynamic tether coupling, report the **force exerted by the tether on the
   bus**, not the force applied to the tether by the bus. They are opposites.
5. The current geometric coordinate runs from the bus outward; verify the
   cathode/collector configuration before assigning physical direction of flow.
6. The actual tether deployment mechanism and whether the attachment reels
   relative to the bus are **not established** by M1. The fixed-point helper
   is not a general deployment model.
7. Model values used in the example are synthetic and not mission parameters.

## M1 validation cases

- Quaternion identity, `q`/`-q` equivalence, normalization rejection.
- Euler ZYX quaternion rotation agrees with the existing `Rz @ Ry @ Rx` code.
- Quaternion composition order and body angular-rate derivative convention.
- Rotating bus attachment location and velocity include `omega x r` correctly.
- Tether-on-bus reaction produces the expected body-frame force and moment arm.
- State snapshots reject wrong dimensions, nonfinite entries, or non-unit `q`.
- All M0 inertia and archive integrity tests still pass unchanged.

## M1 exit checklist

- [x] Frame, quaternion and unit conventions recorded.
- [x] Shared state and reaction snapshots added with validation.
- [x] Attachment motion/wrench transforms added with analytical unit tests.
- [x] Module dependency boundaries and future input/output records documented.
- [x] No changes to M0 golden physics results or legacy notebook artifacts.
- [ ] M1 branch CI green on the remote after PR opened.
- [ ] Team confirms the chosen conventions before downstream solvers depend on them.

**Next:** M2 builds a consistent mission configuration mass inventory and an
adapter to `MassProperties`; M3 implements a rigid-body propagator using
`BusState` and `quaternion_rate_from_body_omega`. The physical tether corrections
can be developed independently provided the interface is honored.
