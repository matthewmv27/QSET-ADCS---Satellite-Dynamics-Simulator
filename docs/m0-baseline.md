# M0 — Baseline and reproducibility record

**Scope:** Freeze existing behavior; document and automatically check the codebase. Do not modify the physical models in M0.

## Source and starting point

- Source: user-provided ZIP of the cloned repository, reviewed 2026-10-30.
- ZIP's Git `main` HEAD: `c1e451bd53224c0856037ca33a786046b641ae91` (`Add legacy simulation run 01`).
- The source archive had a tracked `tests/.DS_Store` macOS metadata file; it has no scientific purpose. It should be removed from Git tracking as an M0 housekeeping change.
- Repository package: `edtsim`, currently version `0.1.0` in `pyproject.toml`.

## Verified locally, before M0 edits

| Check | Result |
|---|---|
| `python -m pytest -q` | 73 passed |
| `PYTHONPATH=src python examples/inertia_quickstart.py` | Completed successfully (stowed and extended inertia printed) |
| Python | 3.13.5 |
| NumPy | 2.3.5 |
| SciPy | 1.17.0 (installed, not required for packaged inertia tests) |
| pytest | 9.0.2 |
| setuptools | 82.0.1 |
| Astropy / `wmm_calculator` | Not installed; legacy environment/OML not exercised |

These values are environment observations, **not** a fully locked dependency set. Exact third-party dependency versions may change when fresh environments install the version ranges in `pyproject.toml`.

## Archived numerical references

`tests/fixtures/baseline_manifest.json` contains SHA-256 and byte counts for:

1. `tests/fixtures/golden_inertia.json` — existing inertia regression fixture.
2. `legacy/MOI_model_current_profile_only_export_matlab_with_forces_FIXED.ipynb` — source notebook.
3. `legacy/Tether_Numerical_Solver-6.ipynb` — earlier tether notebook.
4. `legacy/runs/run01/MOI_model_current_profile_only_export_matlab_with_forces_FIXED_executed.ipynb` — archived executed notebook.
5. `legacy/runs/run01/console_outputs.txt` — saved console output.
6. `legacy/runs/run01/tether_nodes.mat` — saved node history.

Check integrity with:

```bash
python scripts/verify_baseline.py
```

This is a **byte-integrity check**, not notebook execution, scientific validation, or proof that the saved tether trajectory is representative of a deployed tether. In particular, the saved `run01` is known to show severe shortening/collapse and must **not** be treated as a physically valid acceptance target for a future tether solver.

### Golden fixture provenance limitation

The `golden_inertia.json` provenance names an original notebook, `MOI_model_current_profile_only__1_.ipynb`, SHA-256 `96f81f21fd301ff226bae58811303b7805b527906f2d05d88531a2dedef13d97`. **That exact notebook was not in the uploaded ZIP.** The generator at `scripts/generate_golden_inertia.py` requires the original notebook and therefore cannot presently regenerate the fixture from its cited source. The existing fixture still supports behavioral regression tests, but its original-input provenance remains to be independently verified if the notebook is recovered.

## Inertia smoke test is not mission validation

`examples/inertia_quickstart.py` uses synthetic inputs. Its 0.6 kg stowed spool versus 0.5 kg deployed tether does not describe a mass-conserving deployment configuration as written. M2 must reconcile the component inventory before any actual mission mass property claim.

## Verification after the M0 edits

- `python -m pytest -ra`: **74 passed**, including the new archive-integrity test.
- `python -m pip install --no-deps --no-build-isolation -e .`: editable package built and installed successfully in the available Python environment.
- `python examples/inertia_quickstart.py`: ran after editable installation.
- `python scripts/verify_baseline.py`: six archived artifacts passed SHA-256 verification.
- The verification above was performed using Python 3.13.5. GitHub Actions runs on the other configured Python versions only after the workflow is pushed.

## Automated checks introduced by M0

- `.github/workflows/python-tests.yml`: Python 3.9, 3.11, 3.13 matrix; editable install with only base/dev extras; unit tests; example; archived artifact verification.
- `tests/test_baseline_manifest.py`: verify frozen SHA-256 hashes during `pytest`.
- `scripts/verify_baseline.py`: standalone standard-library integrity verifier.
- `README.md`: local installation, architecture/status, limitations, and development rules.

The CI definition has been added but **cannot be reported as passing on GitHub until the files are pushed and the workflow runs.** If one interpreter fails, investigate compatibility instead of changing scientific data to force a green check.

## Changing a frozen artifact deliberately

1. Open an issue or pull request describing *why* the reference needs updating.
2. Preserve the previous version in Git history, including the original generator inputs/parameters where available.
3. Distinguish intentional physics corrections from accidental numerical drift.
4. Update `tests/fixtures/baseline_manifest.json` with the new SHA-256 and `size_bytes` for that artifact, **in the same reviewed commit**.
5. Re-run the complete test suite and, where appropriate, independent analytical/physics checks.

**Do not auto-regenerate the manifest in CI or on every test run.** That would defeat its purpose as an immutable reference.

## M0 acceptance checklist

- [x] Archived legacy files identified and hashed.
- [x] Legacy status and baseline provenance documented.
- [x] Existing packaged test suite executed locally.
- [x] Existing inertia example executed locally.
- [x] README/setup instructions provided.
- [x] Automatic test workflow created.
- [ ] GitHub Actions green on the remote after a PR/push.
- [ ] A second contributor verifies setup from a fresh checkout.

**M0 exit condition:** All local checks pass, GitHub CI passes on a pull request, and the archive/limitations are documented. No physics changes are required for M0.
