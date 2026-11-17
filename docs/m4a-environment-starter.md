# M4A — Environment foundation (preliminary, not the final M4)

## Objective and scope

This increment extracts the legacy notebook's **prescribed circular orbital geometry** into a reusable, NumPy-only `environment` package and adds a first-order J2 nodal-drift option for preliminary sun-synchronous-orbit (SSO) studies. It also supplies strict magnetic-field interpolation, a NED-to-ECEF rotation, assumed corotating flow, a `v_rel × B` helper, a segment Lorentz-force helper, and a basic scalar-coefficient drag formula.

It deliberately **does not** implement WMM2025, NRLMSIS 2.0, the 3D spatial geomagnetic field, absolute GCRS/ITRS transforms, real plasma flow, the payload cathode/current model, or a high-fidelity orbit propagator. Do not interpret the example as a mission prediction.

## Frames, units, and caveats

- `CircularReferenceOrbit` uses an **idealized equatorial inertial frame**, Earth spin axis aligned with +Z. Its `raan_deg` at `t=0` is relative to this idealized frame, **not** a launch-time-derived inertial RAAN. No absolute epoch is used.
- `altitude_m` means a constant circular radius minus WGS84 equatorial Earth radius. This is not a geodetic WGS84 height. All quantities are SI units; angles are degrees on public orbit setup and radians in the underlying J2 calculation.
- `sso_inclination_deg(altitude_m)` uses the first-order mean-element J2 nodal-precession relation and the average tropical-year apparent-Sun rate. It assumes near-circular orbit, ignoring drag, higher-order perturbations, and precise Sun geometry.
- `CircularReferenceOrbit(..., precess_raan=True)` *prescribes* RAAN drift and a Keplerian in-plane angle. Its returned `v_I` is the analytical derivative of its `r_I`. The state is **not** a numerical trajectory solving Newton's equation under J2 gravity; do not use it to validate precise long-term deorbit/position predictions.
- `SampledMagneticField` represents `B_I(t)` **on one designated reference trajectory**; it is not a field `B(r,t)` and rejects arbitrary positions. It raises out-of-domain `ValueError` rather than freezing `B` after one orbit.
- `UniformMagneticField` is a synthetic, inertial constant field for deterministic unit tests.
- `ned_to_ecef_matrix` rotates **vectors** without a translation (unlike transforming a field vector as a position).
- `corotating_flow_I` is a simple Earth-spin-locked velocity approximation. Neutral thermosphere winds and plasma drifts need different providers; **neutral-relative** velocity is used for drag and **plasma-relative** velocity for electrodynamic EMF.
- `aerodynamic_drag_force_I` uses a generic scalar `Cd` and projected area. It does not represent free-molecular spacecraft panel aerodynamics or an atmospheric density model.

## Example

```bash
python -m pip install -e '.[dev,dynamics]'
python -m pytest
python examples/environment_quickstart.py
python scripts/run_m4_environment_reference.py \
  configs/scenarios/m4a_synthetic_450km.json \
  --output-dir results/m4a_synthetic
```

The CSV records `t_s`, inertial `r_I`, `v_I`, synthetic `B_I`, and `E_I=(v_tether-v_plasma)×B`; `metadata.json` repeats the input JSON, calculated nominal inclination, period, and warning. No user is expected to read a long nested JSON just to inspect trends; start with the CSV.

## Mandatory validation gates before completing M4

1. Decide orbit epoch, 400–500 km target altitude or range, eccentricity, local time of ascending/descending node (LTAN/LTDN) and actual epoch-specific RAAN; compare a J2 numerical propagation with a trusted independent orbit reference.
2. Implement tested epoch-aware GCRS↔ITRS position/velocity and magnetic-vector rotation with the appropriate Earth orientation data and time scales; never use a position transformation on a magnetic vector without translation removal.
3. Integrate a pinned WMM2025 or other approved field model using geodetic WGS84 lat/lon/height. Confirm nT→T, NED→ITRS→GCRS, time validity, altitude and geomagnetic activity limitations. Check field comparisons against published reference samples and table interpolation convergence. A sampled orbit field is not a full spatial field for long/bent tethers.
4. Integrate an atmosphere provider (e.g., NRLMSIS 2.0) with epoch, solar F10.7 and geomagnetic indices; document density uncertainty and compare references. Distinguish neutral winds from plasma flow, and document projected-area and gas–surface interaction assumptions for drag.
5. Obtain payload team's model and sign conventions: cathode and emitter location, positive current coordinate, collector type (round wire vs tape), bare/insulated region, electrical resistance, sheath/OML assumptions, boundary conditions, and performance/reference solutions. Use an interchangeable current-profile interface. Do not treat the notebook's current-rescaling approximation as an automatically validated solution.
6. Check Lorentz force and electrical power signs in independently solvable test cases; report shooting residual and convergence explicitly. Do not silently replace failed current solutions with zero current.

## Developer note

This is a self-contained **M4A incremental PR**, not "M4 complete." Keep it as a separate feature branch. Do not modify the established M0 baseline fixtures or M1–M3 models to accommodate it.
