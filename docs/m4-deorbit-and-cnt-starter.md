# M4 extension — preliminary 500 km orbital work, CNT emitter, tether design

## Scope and provenance

This implementation was developed against the **updated user-provided cloned ZIP** after M4A was already present. It adds modules and tests; no existing M0–M4A numerical routines, archived goldens, or README are replaced.

The source `CNT Current Profile.py` supplied on 13 Nov 2026 is **not a complete electrodynamic-tether current profile**. It is an empirical *carbon-nanotube field-emission cathode* model built from approximate digitizations of figures of unspecified provenance. Source curves in figures 1–2 may describe a *different cathode* from figure 3. It normalizes an approximate I–V shape to 2.5 mA emitted current at a time-dependent voltage. This normalization does NOT measure electron-transmission efficiency. The wear-law exponent `n=1` and cathode-power factor `0.70` are assumptions, not calibrated flight values; beyond 500 equivalent hours the voltage is extrapolated.

The positive conventional-current coordinate is defined **from CubeSat attachment (s=0) toward the zenith/far tether tip (s=L)**, opposite electron flow toward the CubeSat. With a collecting region near the far tip, a prescribed *constant* longitudinal current is a consistent simplified force input; an extended bare collecting tether requires a nonuniform `I(s)` model. The cathode is mounted on a separate outward-facing bus face and emits electrons into the ionosphere. No circuit closure is assumed by the force demonstrator.

## Added modules

- `edtsim.tether.geometry`: round wire and rectangular tape from dimensions; mass/length, EA, ideal conductor resistance, collection *geometric* perimeter, and geometric projected silhouette. A rectangle's projected area requires a width direction. Sheath enhancement and true free-molecular drag coefficients remain unknown.
- `edtsim.electrodynamics.current`: `ZeroCurrent` and `PrescribedConstantCurrent` with common `segment_currents_A(t_s, midpoints_s_m)` signature, **not** a current-collection solver.
- `edtsim.electrodynamics.cnt_cathode`: `EmpiricalCNTCathode`, in SI units, estimating emitted current vs bias/age, required voltage, cathode-only power, and a simple equivalent-age increment. This does **not** imply an equal tether current. The source model's current/voltage curves require verification against source papers and payload hardware.
- `edtsim.environment.models`: idealized spatial centered magnetic dipole and constant/exponential neutral-atmosphere density providers. Neither is WMM2025 or NRLMSIS; both exist to permit deterministic sensitivity studies. Magnetic dipole orientation in the sample is a chosen assumption, **not epoch-resolved geography**.
- `edtsim.orbit_dynamics.decay`: propagates a constant-mass point COM with central gravity, J2, simple neutral-relative drag, and distributed prescribed-current `I dl×B` loading along a straight tether constrained to stay radially outward. Integrates drag/EDT work in the state and checks mechanical-energy balance using the full J2 potential.
- `scripts/run_m4_decay_reference.py`: reads a deliberately illustrative scenario and produces short `summary.json` and case-specific CSV timeseries, without a 2,500-line JSON dump.
- `scripts/sweep_m4_tether_geometry.py`: computes static round/tape geometry sweeps; **design properties**, not actual flight geometry or plasma collection.
- `scripts/plot_m4_decay_results.py`: optional three PNG diagnostics from the saved CSVs.

## Run it

From a clean repository checkout with SciPy test dependencies:

```bash
python -m pip install -e '.[dev,dynamics]'
python -m pytest
python scripts/verify_baseline.py
python scripts/sweep_m4_tether_geometry.py \
  configs/scenarios/m4_geometry_sweep.json \
  --output results/m4_geometry_sweep.csv
python scripts/run_m4_decay_reference.py \
  configs/scenarios/m4_500km_assumed_current.json \
  --output-dir results/m4_500km
```

Optional visualization:

```bash
python -m pip install -e '.[viz]'
python scripts/plot_m4_decay_results.py results/m4_500km
```

**Read `summary.json` first**, then compare each case's CSV or plots. The following modes are explicitly separate:

1. `off`: baseline drag-only, no tether current, but assumes the tether is mechanically deployed and contributes geometric drag.
2. `outward_always`: constant *positive* conventional current toward the zenith tip, even during force-favorable or unfavorable orbital phases. Feasibility is **not** established.
3. `inward_always`: reversed current for an **analytical sign check**. It might not be achievable in the passive real circuit.
4. `outward_braking_only`: a **hypothetical current switch-off strategy**; the model turns off positive current when the net instantaneous EDT force would do positive mechanical work. This controller does not establish electrical feasibility.

Columns include inertial position/velocity, total Lorentz/drag force vectors, signed along-track Lorentz force, `work_lorentz_J`, `work_drag_J`, two-body osculating semimajor axis/perigee, `mechanical_energy_J`, and `energy_work_residual_J`.

The **energy diagnostic** is

\[
\Delta E_{\rm mech,J2} - W_{\rm drag} - W_{\rm EDT} \simeq 0.
\]

`E_mech,J2` uses constant system mass and J2 gravitational potential; it is not the simple two-body Kepler energy. The `osculating_a_m` and perigee are instantaneous **two-body orbital elements** and show J2 short-period variation. When comparing cases, compare differences from `off`, not the naive first-to-last element variation alone.

## Interpretation and sign warning

For the **chosen illustrative aligned centered-dipole field** and first-order 500 km SSO geometry, outward current can produce **positive** net orbital work, while the sign-reversed diagnostic current can produce negative net work. This is a mathematical property of that test field/orbit; it is **not** evidence that actual QSET current will accelerate or deorbit. Verify the epoch-resolved field and physically achievable circuit current before making grant-performance claims.

The example's 2.5 mA prescribed current matches the supplied CNT script's 248 V / fresh nominal-emission anchor by **choice**, but the simulation does **not** claim a 2.5 mA tether current is available, nor model cathode voltage/power losses, bare-tether collection, plasma density, CNT ageing through the orbit, switching/control feasibility, electron collection sheaths, or electrical circuit closure. No actual orbit decay time-to-reentry is estimated. Current model failures must never silently become zero current.

## Remaining M4 validation gates

1. **Epoch/LTAN:** determine orbit epoch, RAAN and local solar time; validate a numerical J2 propagator against independent orbit references. The current initial SSO inclination is the *first-order mean-element* value; initial state is treated as osculating for demonstration.
2. **Real geomagnetic field:** implement/test an epoch-aware GCRS↔ITRS frame provider; WGS84 geodetic position, NED field components and vector-only transforms; WMM2025/IGRF validity and nT→T. NOAA WMM2025 is intended for years 2025–2029 and the main-field calculator's nominal altitude range includes 500 km. Magnetic storm limitations must be considered. The numerical idealized centered dipole is **not** a WMM substitute. At present `SampledMagneticField` is only defined on its original reference trajectory and rejects arbitrary spatial requests, appropriately.
3. **Neutral atmosphere:** integrate/pin an NRLMSIS version with actual UTC epoch and solar/geomagnetic drivers; validate density references, neutral wind assumptions, and bus/tether free-molecular drag coefficients and projected areas. A static 500 km exponential density profile cannot predict decay across solar-cycle changes.
4. **Plasma/electrical:** obtain electron density/temperature/composition, collection geometry and circuit polarity, bare length, electric conductivity, emitter supply limits, CNT curve provenance and uncertainty. Integrate a nonuniform signed `I(s,t)` provider with a separate CNT emission-limit constraint and independent power balance. Use `v_tether-v_plasma` for motional EMF. Compare source currents against the verified payload paper/model; do not infer them from the cathode curve alone.
5. **Mechanics:** no tether tension, variable shape, spacecraft attitude, spool deployment, moving COM, deformation, tip dynamics, full 6DOF coupling or oscillations are modelled. All are later milestones. Bus projected area and tape width direction are illustrative conventions.
6. **Grant**: demonstrate negative *orbit-averaged* EDT work from a physically supported current and Earth-field model under transparent assumptions; any single-step braking arrow or prescribed reversed current is **not sufficient** evidence.

All dimensions/fields in the examples are deliberately illustrative and must not be labelled approved QSET design parameters.
