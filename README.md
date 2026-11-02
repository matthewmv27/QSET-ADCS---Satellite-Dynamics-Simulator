# QSET ADCS — Satellite Dynamics Simulator

Python package for developing and validating models of a CubeSat with an electrodynamic tether.
The long-term target is a coupled spacecraft attitude, tether and electrodynamics simulator.
**At present, only the `core` and `inertia` subpackages are implemented and covered by the packaged test suite.** Orbit/environment, OML current and tether-chain work remain in legacy notebooks and are not yet validated packaged modules. Attitude and two-way coupling have not been implemented.

## Quick start

Use Python **3.9 or later** as declared in `pyproject.toml`. The M0 baseline was verified with Python 3.13.5. On macOS/Linux, from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest
python examples/inertia_quickstart.py
python scripts/verify_baseline.py
```

On Windows (PowerShell), replace the activation command with `.venv\Scripts\Activate.ps1`. The base development setup installs NumPy and pytest; it does **not** require Astropy, the World Magnetic Model, or MATLAB. The optional dependency groups in `pyproject.toml` are reserved for future modules: `tether`, `environment`, `viz`, and `all`.

## What currently works

| Location | Scope | Status |
|---|---|---|
| `src/edtsim/core/` | Validated parameter containers and rotation/math helpers | Implemented |
| `src/edtsim/inertia/` | Basic inertia tensors, stowed/straight-deployed configurations and particle summation | Implemented |
| `tests/` | Unit tests, architecture checks and frozen inertia regression tests | Passing locally at M0 |
| `examples/inertia_quickstart.py` | Stowed and straight-deployed demonstration | Runs locally at M0 |
| `legacy/` | Research notebooks and one archived simulation run | Historical, **not** physics-validated |
| `docs/` | Existing interface/roadmap notes and baseline procedures | In progress |

The example values are illustrative, **not approved mission mass properties**. In particular, the sample stowed and deployed inputs do not conserve total component mass and should not be treated as consistent successive configurations of one spacecraft.

## Running checks

```bash
python -m pytest                    # Existing and new unit/regression tests
python examples/inertia_quickstart.py
python scripts/verify_baseline.py  # Verify archived artifact SHA-256 checksums
```

The baseline verification script checks the archived files byte-for-byte. It does not execute the notebooks and does not claim their scientific validity. GitHub Actions runs these checks for pushes and pull requests. See [docs/m0-baseline.md](docs/m0-baseline.md) for exact baseline details, limitations and how to change an archive intentionally.

## Repository layout

```text
src/edtsim/core/         Shared numerical utilities and parameter validation
src/edtsim/inertia/      Moment-of-inertia calculations
examples/               Small, runnable demonstrations
tests/                  Python tests and archived regression fixtures
scripts/                Golden-data generator and archive verification
legacy/                 Unmodified research notebooks and saved run01 output
docs/                   Conventions, roadmap and baseline/review documentation
.github/workflows/       Automated test workflow
```

## Development rules

1. Use SI units throughout model internals; state coordinate frame and reference point with every mass property. Current conventions: [docs/interfaces.md](docs/interfaces.md).
2. Do not silently change golden data or legacy notebook outputs. If you deliberately replace a reference, document its scientific and software justification, source, and new checksums in the same pull request.
3. Every new model needs independent physical validation, not just agreement with legacy numerical outputs.
4. Keep optional heavy dependencies out of the `core` and `inertia` imports.
5. Add focused unit tests and a runnable example when introducing a new subpackage.

The M1 foundation defines shared coordinate-frame conventions, physical state structures, and module interfaces. See docs/interfaces.md and docs/m1-interface-spec.md. Run python examples/interfaces_quickstart.py to demonstrate the new interfaces.

## M2 parameter sweeps — running studies and reading results

A **parameter sweep** evaluates the mass-properties model repeatedly with
systematically varied inputs. This helps compare hypothetical designs without
rewriting Python code. M2 computes **static configuration snapshots**, not
orbital motion, aerodynamic drag, electrical current, or deployment dynamics.

### 1. Input JSON versus output JSON

- **Input scenario**: `configs/scenarios/m2_illustrative.json` — the file you **edit**.
  It defines the baseline component inventory, parameter provenance, and the
  sweeps to run. Its numerical values are **illustrative**, not QSET flight data.
- **Output results**: `results/m2_illustrative/mass_properties.json` — the large
  file the runner **generates**. Each case repeats the full input model and
  computed mass properties for traceability. It is normal for it to be thousands
  of lines long. Do not edit it to configure the next run.
- **Output summary**: `results/m2_illustrative/mass_properties.csv` — one row per
  evaluated case, suited for Excel, Python, and quick comparisons.
- **Optional plots**: `deployment_fraction_Ixx.png`, `design_comparison_Iyy.png`,
  etc. — diagonal inertia about the **whole assembly COM**, in body axes.

### 2. Run the example

From the repository root, with the project installed:

```bash
python -m pip install -e '.[dev,viz]'
python scripts/sweep_mass_properties.py \
  configs/scenarios/m2_illustrative.json \
  --output-dir results/m2_illustrative --plot
```

The example runs **18 cases**: five deployment states, twelve hardware-design
combinations, and one prescribed bent tether. To create your own study, copy
`configs/scenarios/m2_illustrative.json`, assign a new `id` and `revision`, edit
its `model` and `sweeps`, and use a new `--output-dir` to avoid overwriting
another study's results. See [M2 sweep reference](docs/m2-sweep-runner.md) for
all supported fields and validation rules.

### 3. What to edit in an input scenario

| JSON key | Meaning |
|---|---|
| `model.bus` | Uniform-box bus mass (`mass_kg`) and dimensions (`dimensions_m`) |
| `model.spool_hardware` | Fixed deployment hardware, **not** wound tether mass |
| `model.tether` | Total available length, mass per unit length, and attachment position |
| `model.end_mass` | Tip mass and stowed position |
| `model.wound_geometry` | Approximate location and shape of wound tether on its spool |
| `sweeps[].kind` | `deployment`, `design_grid`, or `bent_cases` |
| `provenance` | Whether inputs are `illustrative`, `preliminary`, or `verified`, with source and revision |

**Deployment sweeps** change `fractions` between 0 (fully stowed) and 1
(fully deployed) while conserving the total mass of *one* design.

**Design grids** evaluate all combinations in `axes`, e.g. two tether lengths ×
two tip masses × three deployment fractions = twelve cases. Changing total
length at constant linear density changes the total tether mass and compares
*different designs*, not a mass-conserving trajectory.

**Bent cases** accept a list of prescribed 3-D tether-node positions in body
coordinates (`nodes_B_m`), optionally with reference segment lengths. They are
geometry snapshots, not shapes predicted by a dynamics solver.

### 4. How to interpret `mass_properties.json`

The top level includes `scenario_id`, `scenario_revision`,
`declared_provenance`, `source_configuration_sha256`, `limitations`, and
`cases`. Most of the file is the `cases` array. Each case includes:

| Result key | Meaning / units |
|---|---|
| `case_id`, `sweep_id`, `kind` | Which calculation produced the entry |
| `deployment_fraction` | Deployed tether fraction (0 to 1), **not elapsed time** |
| `design_overrides` | Design variables changed from the baseline input |
| `geometry` | Straight direction or prescribed bent nodes |
| `input_model`, `component_masses_kg` | The actual inputs and mass breakdown used for that case |
| `mass_kg` | Total mass of the specified assembly |
| `com_B_m` | Combined center of mass measured from **rigid bus COM**, in body axes [m] |
| `inertia_about_bus_com_B_kgm2` | Full 3×3 assembly tensor about the **bus COM** [kg·m²] |
| `inertia_about_system_com_B_kgm2` | Full 3×3 assembly tensor about the **assembly COM** [kg·m²] |
| `principal_moments_system_com_kgm2` | Sorted eigenvalues of the latter tensor [kg·m²] |

`Ixx`, `Iyy`, and `Izz` are diagonal entries; off-diagonal terms show products
of inertia and should not be discarded. Always state **which reference point**
is used. Neither assembly tensor is automatically the rigid-bus inertia needed
by the later flexible-body attitude solver.

Instead of reading thousands of lines manually, first open the CSV, then use
this small Python snippet to inspect the JSON cases:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('results/m2_illustrative/mass_properties.json')
data = json.loads(p.read_text())
print('Scenario:', data['scenario_id'], 'Cases:', len(data['cases']))
for case in data['cases']:
    if case['kind'] == 'deployment':
        I = case['inertia_about_system_com_B_kgm2']
        print(f"{case['case_id']:30s} "
              f"deployed={case['deployment_fraction']:.0%} "
              f"mass={case['mass_kg']:.3f} kg "
              f"Ixx={I[0][0]:.6g} kg·m²")
PY
```

The output contains a source-configuration SHA-256 to identify which input
scenario generated it. This is **traceability**, not proof that the physical
assumptions are correct. The declared `verified` status is entered by the
scenario author, not independently certified by the software.

