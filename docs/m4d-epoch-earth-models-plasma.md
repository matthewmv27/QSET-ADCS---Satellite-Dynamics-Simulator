# M4d — Epoch sweeps, real geomagnetic/atmosphere adapters, and OML collection

**Status:** Preliminary. This increment does not close the CNT–tether–plasma electrical circuit and does not validate a flight deorbit prediction. It is designed around the **500 km** preliminary LEO-SSO reference and accepts sweepable epochs, inclinations, RAAN, LTAN/LTDN, and solar-activity inputs.

## Dependencies and external-model verification

For the 1-D OML plasma reference only:

```bash
python -m pip install -e '.[dev,dynamics]'
python scripts/run_m4_plasma_reference.py configs/scenarios/m4_plasma_example.json --output-dir results/m4_plasma
```

For IGRF-14, WMM2025, WGS84/GCRS/ITRS transformations and NRLMSIS 2.0:

```bash
python -m pip install -e '.[dev,dynamics,earth-models]'
python -m pytest tests/test_m4_optional_model_integration.py -rs
python scripts/run_m4_epoch_sweeps.py configs/scenarios/m4_epoch_sweep_example.json --output-dir results/m4_epochs
```

**Check that the optional real-model smoke tests actually RUN rather than report SKIPPED**. The project development environment may not include these model libraries. Unit tests using stubbed third-party APIs check that conversions and parameters are correct, but are not a replacement for the real-library tests.

The `earth-models` extra includes `astropy`, `ppigrf` (IGRF-14 default), `wmm-calculator` (WMM2025), and `pymsis` (NRLMSIS 2.0). Full models are only imported when selected. The existing centered-dipole and exponential-density providers remain unchanged for deterministic regression.

### Earth-frame convention

The new real-field providers interpret input position `position_I_m` as **geocentric GCRS position in metres** at elapsed time `t_s` after the ISO-8601 UTC epoch. `EarthFrame` uses Astropy GCRS→ITRS, WGS84 geodetic lat/lon/alt, and converts magnetic NED vectors to ITRS and then GCRS (nT→T). Rotation of a magnetic-field vector is NOT the translation of a position. Astropy's automatic IERS downloads are disabled; missing/up-to-date Earth-orientation parameters may limit pointing accuracy and must be checked before flight analysis. The older idealized circular reference states are only approximately represented by GCRS axes; orbital initial states need rigorous epoch-of-elements conventions for precision navigation.

- `GeomagneticMainField(epoch, model='igrf14')` returns the main-field vector in Tesla at an arbitrary spatial point and elapsed time. `ppigrf.igrf` returns **east, north, up**, converted here to NED. `wmm-calculator` returns **north, east, down**. WMM's adapter conservatively restricts altitude to 0–850 km and epoch to 2025–2029; its implementation uses date without fractional day for coefficient evaluation (negligible annual secular change for the short reference horizon, not an interpolation error bound). IGRF-14 in this adapter has a 1900–2030 epoch range.
- `MSIS2AtmosphericDensity` uses **NRLMSIS v2.0 explicitly** through `pymsis.calculate`, with supplied daily `F10.7`, 81-day mean `F10.7a`, and an approximate **seven-bin constant Ap history**. This prevents invisible network-dependent solar-index downloads. Density is kg/m³. The neutral-flow calculation elsewhere still assumes atmospheric co-rotation; winds are not represented.
- These models do not account for external magnetospheric current systems, transient geomagnetic storms, plasma drifts, solar-wind fields, ion composition, or actual spacecraft charging.

## Orbital parameters are linked, not independent

The orbit-sweep input file is `configs/scenarios/m4_epoch_sweep_example.json`.

```json
"orbit_sweep": {
  "epochs_utc": ["2026-11-15T12:00:00Z", "2027-06-21T12:00:00Z"],
  "altitudes_m": [500000],
  "inclinations": ["sso", 97.5],
  "plane_sweeps": [
    {"type": "ltan_hours", "values": [6, 10.5]},
    {"type": "raan_deg", "values": [0, 90]},
    {"type": "ltdn_hours", "values": [18]}
  ]
}
```

Each plane-sweep group makes an **independent family of cases**. It is NOT valid to vary RAAN and LTAN independently within a single epoch case. For an epoch with apparent geocentric Sun right ascension `alpha_sun`, the reference relation is:

```
LTAN [hours] = (12 + (RAAN_deg - alpha_sun_deg)/15) mod 24
LTDN [hours] = (LTAN + 12) mod 24
RAAN_deg = (alpha_sun_deg + 15*(LTAN_hours-12)) mod 360
```

These relations use the Sun's right ascension **at the configured epoch** as an approximation, not the precise computed epoch of the next equatorial crossing. For `"inclinations": ["sso"]`, first-order J2 theory derives the inclination needed for the **mean** nodal rate; explicit angles such as `97.5` are allowed but **not guaranteed sun synchronous**. This is not a mission-accurate ephemeris or a long-term frozen-orbit prediction.

The environment configuration defines `solar_activity_cases` with labels and `f107`, `f107a`, `ap`, all treated as **illustrative** values. The runner evaluates a cross-product of epochs, altitudes, inclinations and *independent plane-case groups*, then for each case uses each activity scenario.

### Outputs and what they mean

```bash
python scripts/run_m4_epoch_sweeps.py configs/scenarios/m4_epoch_sweep_example.json --output-dir results/m4_epochs
```

Produces `summary.csv` and `summary.json`. Results include resolved RAAN and LTAN/LTDN, SSO-derived inclination and J2 node rate, geomagnetic magnitude, modelled density range, and trapezoidal approximation to the Lorentz/drag work on the **prescribed reference path**. **Reference-path work does not propagate orbital decay.** A negative value of `lorentz_work_J` indicates opposing work *under the assumed current*, not electrical feasibility.

To enable numerical translation (expensive, at most **four combinations** per call):

```bash
python scripts/run_m4_epoch_sweeps.py configs/scenarios/m4_epoch_sweep_example.json --propagate --output-dir results/m4_decay_epochs
```

Reduce the epoch/plane/activity lists first to no more than four combinations. This integrates central gravity + J2 + drag + *assumed* Lorentz force over the requested short duration; it reports `delta_osculating_a_m`. The tether is still a prescribed straight radial line, and the osculating semimajor axis has periodic J2 oscillations. Use cumulative work and matched controls to interpret the trend. Current feedback from plasma/CNT remains **absent**. The runner is deliberately not called from GitHub Actions CI because real models and full perturbation propagation are expensive.

## Preliminary electron-collection model

The planned hardware uses the CNT cathode on the CubeSat and a collecting tether with its far end facing away from Earth. Arc length `s` increases **from the bus to the zenith tip**; conventional current is positive in that direction. Electrons collected at the far/outer bare portion move toward the bus.

The new `PrescribedBusPotentialOML` model solves:

\[\frac{d I}{d s}=-j_e(V),\qquad
\frac{dV}{ds}=E_m-\frac{I}{\sigma A}.\]

Boundary conditions: **fixed bus tether-to-plasma potential V(0)** and **no electrical current past the far tip, I(L)=0**. The collecting source `j_e(V)` uses the long/thin cylindrical OML electron approximation (including a thermal-bias continuation), and the tape uses a perimeter upper-bound, not measured tape collection. A high-positive-bias envelope (`ThinCollectorOMLEnvelope`) is also available for back-of-envelope checks. The OML approximation's validity depends on collector radius or tape width relative to plasma Debye length; neither solution establishes true flight applicability without actual plasma conditions.

```bash
python scripts/run_m4_plasma_reference.py configs/scenarios/m4_plasma_example.json --output-dir results/m4_plasma
```

Outputs `current_profile.csv` (arc length, signed conventional current, conductor-plasma bias, electron collection per metre) and `summary.json` (bus current, end residual, ohmic losses, approximate CNT required gate voltage and check against an *assumed* supply limit). The cathode gate voltage is **not** the same physical quantity as `V(0)`. The emitter model is based on approximate digitized plots without the source raw data; it is **not calibrated**. No instantaneous ionosphere provider, plasma return potential, ion current, electron sheath correction, contactor impedance, self-consistent emitter/plasma closure, thermal change or deployment dynamics are implied.

For a complete EDT electrical circuit, we still need: validated plasma electron density, temperature, ion composition/drift; conductive and insulated geometry; actual tether potential relative to plasma and motional EMF; source/return boundaries; CNT I–V curve, operating limits and power supply; and independent current/voltage test cases. Do not feed the preliminary current directly into the grant prediction as though the current were guaranteed.

## Source and limitation notes

- NOAA WMM: https://www.ncei.noaa.gov/products/world-magnetic-model
- WMM calculation package: https://github.com/CIRES-Geomagnetism/wmm
- IGRF 14 ppigrf API: https://github.com/IAGA-VMOD/ppigrf
- pymsis NRLMSIS: https://swxtrec.github.io/pymsis/reference/generated/pymsis.calculate.html
- Electron collection reference (NASA): https://ntrs.nasa.gov/api/citations/20030106418/downloads/20030106418.pdf
- 1-D collection coupled with OML and ohmic resistance: https://e-archivo.uc3m.es/rest/api/core/bitstreams/4a3e7a0e-9008-450c-bc53-0a4816700a08/content

**Project status:** this increment establishes swappable real environmental providers, computational epoch sweeps, and a testable *preliminary* electron-collection solver. It does not complete M4's electrical validation or prove the grant's deorbit criterion. Before making performance claims, run real third-party integration tests, compare external magnetic/density reference outputs, and validate current closure.
