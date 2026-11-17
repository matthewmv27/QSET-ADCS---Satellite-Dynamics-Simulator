# M4d — Geomagnetic field and Lorentz-force validation

This increment validates **vector conventions, reference-field components, and algebraic power balances**. It is not a validated QSET deorbit-performance model. It does **not** connect the preliminary OML and CNT emission models into a self-consistent plasma/electrical circuit.

## Conventions

- Tether coordinate `s=0` at the CubeSat; `s=L` at the zenith tip. Positive conventional current travels bus -> tip; electrons in the conductor travel tip -> bus.
- `r`, `v`, `B`, `dl` use the simulator inertial frame, metres / m/s / tesla. For the real Earth providers this frame is assumed GCRS, while the circular orbit is an idealized equatorial inertial approximation.
- `e_r = r / |r|`, `e_v = v / |v|`, `e_h = e_r × e_v`. For straight radial tether, positive conventional current and positive `B · e_h` imply **negative** Lorentz force along `v`: `F_along = -sum(I_j ds B_j · e_h)`.
- Positive `F_along` does *not* imply a numerical error; it means the prescribed current and field produce instantaneous along-track acceleration.

## Power identities and interpretation

A discretized tether segment has `dF = I dl × B`. For local segment velocity `v_s` and plasma velocity `v_p`, the motional field is `(v_s - v_p) × B`. Thus:

`sum(dF · v_s) + sum[I ((v_s - v_p) × B) · dl] = sum(dF · v_p)`.

This is a sign / coordinate *identity* and not a demonstration of physical circuit closure. For nonzero plasma motion, it is **incorrect** to assume that mechanical power is always the negative of motional-EMF power. And `F_total · v_COM` (orbital translational work) need not equal distributed power when different segments move at different velocities.

## Run the orbit-level sign audit

```bash
python scripts/run_m4d_field_audit.py \
  configs/scenarios/m4d_field_audit.json \
  --output-dir results/m4d_field_audit
```

Produces `samples.csv` with along-track force and power terms at each sample, and `summary.json` with integrated work and residuals. `centered_dipole` is synthetic, intentionally available with only NumPy. Values in the supplied JSON are **illustrative**.

To run using Earth **main-field** coefficients after installing the optional dependencies:

```bash
python -m pip install -e '.[dev,dynamics,earth-models]'
python scripts/run_m4d_field_audit.py configs/scenarios/m4d_field_audit.json \
    --model wmm2025 --output-dir results/m4d_wmm2025
python scripts/run_m4d_field_audit.py configs/scenarios/m4d_field_audit.json \
    --model igrf14 --output-dir results/m4d_igrf14
python -m pytest tests/test_m4d_field_force_validation.py -v -rs
```

`summary.json` records whether the main field is synthetic or a real coefficient-based model. **A main-field model is not a storm-time field**. The orbit is prescribed; the integral of `F · v` is *work evaluated on that trajectory*, not an integrated orbital decay or a guarantee of net braking.

## Independent NOAA WMM2025 component benchmarks

Six test vectors are copied from NOAA NCEI **WMM2025 Test Values** (2025.0, 0 and 100 km WGS84 ellipsoid height; input geodetic lat/longitude). See `tests/fixtures/wmm2025_noaa_reference.json`. The test directly compares the adapter's north, east, and down components in nT. It executes **only** when Astropy and `wmm-calculator` are installed. This is a software component benchmark, not a tolerance on actual geomagnetic-field error. NOAA source: https://www.ncei.noaa.gov/sites/g/files/anmtlf171/files/2025-02/WMM2025testvalues.pdf

Tests for the IGRF14 east/north/up mapping and WMM north/east/down mapping can run without the optional libraries using controlled mock data. A truly independent **numeric IGRF14** reference at ~500 km, plus an independently verified GCRS/ITRS transformation, remains to be added before full M4 field validation is complete. Likewise, WMM 2025 fixtures at 0/100 km do **not** independently establish error bounds at 500 km.

## Release gates (not all satisfied by this patch)

1. Deterministic analytical Lorentz sign, current-reversal, local and COM work tests pass.
2. Independent NOAA WMM2025 **signed-component** tests run and pass with actual `wmm-calculator` installed.
3. IGRF14 numerical component references checked at relevant high-altitude positions and epochs.
4. GCRS/ITRS frame transformations independently checked, with model and IERS/EOP versions recorded.
5. Orbit-sweep results evaluated under a physically possible current model (later electrical integration). Re-examine mostly positive predicted work **without** forcing a desired result.
6. The grant-oriented orbital-decay study distinguishes aerodynamic drag, Lorentz work, and assumed-current limitations.

Never claim M4d establishes deorbit feasibility solely because a torque/force identity passes.
