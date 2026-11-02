# M3 — Standalone rigid-bus attitude dynamics

## Scope

Propagate attitude quaternion `q_IB` (Hamilton, scalar-first, body to inertial)
and body-frame angular velocity `omega_B` using *rigid-bus* inertia about the
bus COM. Accept external torque in **body axes**, without assuming a specific
orbit, controller, environment, or tether. See `docs/interfaces.md`.

Equations (constant inertia):

- `I_B * omega_dot_B = tau_B - omega_B × (I_B * omega_B)`
- `q_dot_IB = 0.5 * q_IB ⊗ [0, omega_B]`

The integrator uses SciPy `solve_ivp`; the quaternion is normalized at output
samples and for torque callbacks. The raw quaternion ODE is advanced internally.
The integrator has no state constraint stabilization beyond this numerical
normalization; energy and momentum drift should be checked on every scenario.

## Usage

```bash
python -m pip install -e '.[dev,dynamics]'
python examples/attitude_quickstart.py
python -m pytest
```

`propagate_attitude(inertia_B, q0_IB, omega0_B, t_eval_s, torque_B=None)`
returns an `AttitudeHistory` with `t_s`, `q_IB`, `omega_B`, `torque_B`,
`kinetic_energy_J`, and `angular_momentum_I`. `torque_B(t,q,omega)` returns a
3-vector in N m in body axes. The optional SciPy dependency is only imported
when the propagation function is called.

## Verification and limitations

Standalone tests cover stationary attitude, analytic principal-axis rotation,
constant applied torque, torque-free conservation, and invalid inputs.

**Not included:** orbital dynamics, sun-synchronous-orbit construction, J2,
gravity-gradient torque, atmospheric drag, magnetorquers, tether reactions,
flexible assembly inertia, and deployment dynamics. M4/environment and later
integration stages must supply those effects through documented interfaces.

For a flexible tether, do **not** use the instantaneous whole-assembly inertia
as if it were a rigid-bus inertia. Use the rigid bus inertia and mechanical
attachment reaction in the later coupled model.

## Orbit input pending confirmation

The initial mission description referred to a sun-synchronous orbit at
4,000–5,000 km altitude. That is **medium Earth orbit**, not LEO (< 2,000 km).
An alternative interpretation is 400–500 km. No orbit altitude is assumed in
M3. Before M4, confirm the intended altitude and distinguish altitude above
Earth's surface from geocentric orbital radius; SSO inclination depends on
altitude, eccentricity, and chosen J2 convention.
