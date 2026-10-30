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

## What works today

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
