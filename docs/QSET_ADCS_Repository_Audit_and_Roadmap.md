# QSET ADCS — Repository Audit 
**Review scope:** 34 project-owned files retained after excluding installed `.venv` dependencies, `.git` metadata, OS resource forks and caches. Three notebooks were reviewed, including the archived executed duplicate.  
**Execution:** `PYTHONPATH=src python -m pytest -q` **73 passed**; `PYTHONPATH=src python examples/inertia_quickstart.py` successful. SciPy, NumPy, matplotlib and pytest available; Astropy and wmm-calculator unavailable here. Accordingly, full environment-dependent notebook reproduction was not attempted. Saved executed notebook, console output, and `.mat` output were inspected.

## Moving Forward

**Keep the existing `src/edtsim` package and separate models into independently validated subsystems. Prioritize correcting the tether mechanics and standardizing physical interfaces before two-way electrodynamic coupling.** Implement attitude dynamics after validated mass properties rather than waiting for tether visualization. Add mechanically coupled bus/tether dynamics with electrical current disabled, then introduce electrodynamics, then orbital-deorbit studies if required.

There are **two distinct deliverables**: (1) mass properties of fixed configurations including hypothetical flexible configurations, and (2) dynamics of a rigid bus mechanically coupled to a flexible tether. The flexible tether is not part of the bus's rigid inertia tensor in the latter model; its influence is conveyed by momentum and attachment reactions. Do not double-count tether mass or electrodynamic torques.

## Verified project inventory

| File | Review finding / next action |
|---|---|
| `README.md` | One heading only; replace with overview, setup, architecture diagram, run examples, limitations. |
| `pyproject.toml` | Valid setuptools `src` package, Python >=3.9, NumPy base, optional SciPy, astropy/WMM, matplotlib, pytest. Broaden dependency groups to include attitude numerical integrators; pin/record tested versions separately. |
| `.gitignore` | Broad Python ignore, excludes `.venv` and `.mat` with exceptions for saved reference runs. ZIP nevertheless includes virtual environment and Git metadata; exclude these from future review archives. |
| `LICENSE` | Present; retain as legal metadata. |
| `docs/interfaces.md` | Defines SI, GCRS inertial frame, bus-COM body frame, `R_IB`, axis conventions, draft results format. Revise fixed-inertia vs flexible mass-properties distinction, add quaternion, attachment sign, current orientation, 6DOF states, time and event conventions. |
| `docs/roadmap.md` | Stage 0–6 plan; lists several correct legacy concerns. Update order and explicit tests, ensure Stage 0 and Stage 1 statuses match code/tests. |
| `src/edtsim/__init__.py` | Documents future modules but only core/inertia exist; update after modules actually land. |
| `src/edtsim/core/__init__.py` | Public exports; keep explicit and synchronized. |
| `src/edtsim/core/constants.py` | Earth GM and equatorial radius; adequate for simple circular-orbit case. Document the gravity model for all mission studies. |
| `src/edtsim/core/rotations.py` | Cross-product matrix, normalization, elementary rotations, Euler-to-rotation and vector aligner; tested. Add quaternion-to-matrix and quaternion-derivative utilities with consistent conventions. |
| `src/edtsim/core/params.py` | Dataclasses validate many positive masses/dimensions and 3-vectors. Add validation for finite values, physical configuration relationships, shared mission data, tether electrical geometry and solver controls. |
| `src/edtsim/inertia/__init__.py` | Public API for inertia functions; preserve. |
| `src/edtsim/inertia/primitives.py` | Correct familiar uniform box, annular cylinder, slender rod, point masses, parallel-axis shifts and frame rotation; analytical tests pass. These are **rigid-geometry mass-property** utilities. |
| `src/edtsim/inertia/configs.py` | Config 1 = box+coiled annulus+stowed end point; Config 2 = box+straight rod along `-z_B`+tip. Neither ensures whole-assembly mass conservation or outputs assembly COM. Add named configuration input and complete mass-property report. |
| `src/edtsim/inertia/particles.py` | Vectorized point-mass tensor about an explicit reference point; tested. For dynamic nodes use correct lumped masses and positions, and combine with bus carefully. |
| `src/edtsim/inertia/sampling.py` | Volume-uniform random point in annular cylinder, reproducible seed; tested. Suitable for uncertainty/demo studies, not a substitute for an actual known tip-mass stowage location. |
| `src/edtsim/inertia/validation.py` | Principal moments and simple physical plausibility check; pass existing tests. Add symmetry, mass conservation and component-inventory invariants. |
| `examples/inertia_quickstart.py` | Executed successfully; placeholder 3U case has total stowed mass 4.75 kg versus deployed mass 4.65 kg. Make demo mass-conserving and clearly label examples as illustrative. |
| `scripts/generate_golden_inertia.py` | AST-extracts legacy primitive functions and generates fixtures. Documentation's example notebook filename is not in uploaded ZIP; fixture provenance references another notebook/hash not matching included legacy source. Preserve the fixture but document provenance and regenerate only with a verified original. |
| `tests/conftest.py` | Independent annular-cylinder particle quadrature and fixture; useful. |
| `tests/test_architecture.py` | Tests permitted package imports for **existing** subpackages. New `attitude`, `coupling`, etc. require updating allowed mapping; current graph does not yet enforce contracts between new modules. |
| `tests/test_golden_inertia.py` | Legacy regression fixture comparisons for 3 synthetic configuration cases and primitive/rotation/particle functions; useful but regression alone cannot verify physical correctness. |
| `tests/test_inertia_configs.py` | Analytical and particle comparisons for stowed/deployed straight model; passes; add cross-config mass equality, COM and partially deployed tests. |
| `tests/test_inertia_particles.py` | Point-mass, translation-invariance, shape and equivalence tests; passes. |
| `tests/test_inertia_primitives.py` | Analytical shapes, transformations, numerical quadrature, parallel axis; passes. |
| `tests/test_inertia_validation_and_sampling.py` | Eigenvalue/physicality and random sample properties; passes. |
| `tests/test_params.py` | Input coercion and parameter constraint checks; passes. |
| `tests/test_rotations.py` | Rotation direction and orthogonality tests; passes. |
| `tests/fixtures/golden_inertia.json` | Three synthetic cases with saved expected inertias and provenance. All three change mass between stowed and deployed: 4.75→4.65, 3.70→3.60 and 2.80→2.75 kg. Keep as behavior fixtures, *not* mission-conserving configurations. |
| `legacy/Tether_Numerical_Solver-6.ipynb` | Multiple cells implementing dimensionless OML shooting, dimensional rectangular-tape model, current/power/force length/bias sweeps. Shape and sign need correlation with actual spacecraft cathode configuration. The sweep uses simple `P=I_avg V_bias` and `F=I_avg LB` approximations; do not present as full electrical power budget or vector force model. |
| `legacy/MOI_model_current_profile_only_export_matlab_with_forces_FIXED.ipynb` | 1456-line source cell; contains all original mass-property functions, prescribed circular orbit, astropy/WMM field table, dimensional OML current solver, tether lumped-mass RHS, stiff integration, `input()` wizard and MAT export. Extract tested modules rather than copying entire notebook wholesale. |
| `legacy/runs/run01/MOI_model_current_profile_only_export_matlab_with_forces_FIXED_executed.ipynb` | Executed copy of the 1456-line notebook with saved outputs; preserve as reproducibility evidence, not as independent source implementation. |
| `legacy/runs/run01/console_outputs.txt` | Full parameter log for a 30-second example: 4 kg bus, 10 m / 0.1 kg tether, 0.2 kg end mass, five segments, 400 km circular orbit, 50 V bias, 10 V cathode, mass_scale=6.33, BDF integrator. |
| `legacy/runs/run01/tether_nodes.mat` | Saved 61 time steps, `(61,6,3)` node positions, `(61,5,3)` Lorentz forces, segment currents, orbit and magnetic field. **Quantitative finding:** tether tip-to-attachment separation falls from 10 m to 0.001662 m in 30 s. See critical findings below. |

## Critical findings, ranked

### P0 — The saved tether simulation physically collapses

`legacy/runs/run01/tether_nodes.mat` has `(Nt,N_nodes,3)=(61,6,3)` at 0.5-second increments. The attachment-to-tip separation is 10 m at `t=0`, 8.762865 m at 5 s, 6.191251 m at 10 s, 2.748166 m at 15 s, 0.440771 m at 20 s, 0.027089 m at 25 s, and **0.001662 m at 30 s**. It is invalid to interpret the final inertia as a normally deployed 10 m tether.

In `tether_chain_dynamics_rhs`, default `smooth_max0` for positive `eps` is `0.5*(x+sqrt(x*x+eps*eps))`, hence is positive even at zero extension. With saved run's `EA=5000 N`, `N_seg=5`, `L=10 m`, `smooth_eps_frac=1e-4`, `c=10 Ns/m`, zero relative velocity, and nominal unstretched length, the two smooth clamps produce **~0.40451 N nominal tension**, not zero. This is enormous relative to saved Lorentz segment forces (~1e-8 N) and is consistent with observed collapse, although a full isolated study is needed to attribute all time-dependent behavior. The mass inflation to `6.33` also changes acceleration. Fix tension law and write force-at-rest regression tests before treating any legacy dynamic result as physically valid.

### P0 — Automatic dynamic mass scaling

`run_bent_tether_sim` sets `sim.mass_scale=max(sim.mass_scale, (T_target/T_fast)^2)` whenever `T_fast<T_target`; `tether_chain_dynamics_rhs` multiplies masses by this factor while the returned inertia uses unscaled physical masses. Therefore solver time evolution is nonphysical if scaling is active. Restrict the option to a clearly named quasi-static shape-finder mode; true transient modes must use physical masses.

### P0 — Implicit missing bus reaction and 6DOF physics

`tether_chain_dynamics_rhs` prescribes bus orbit, has zero attachment velocity due to assumed constant attitude, and applies segment spring forces only to tether particles. First segment reaction on the bus is missing. To conserve momentum for a freely evolving system, include rigid bus translation and rotation, the tether-node degrees of freedom and equal/opposite attachment forces, then test against the exact zero-external-force limit. Never add the full tether Lorentz torque directly to the bus **and** mechanical transmitted force; that risks double counting.

### P0/P1 — Boundary orientation and electrical power closure

`compute_dimless_current_profile` integrates `i(0)=0`, collecting current through increasing `x`, whereas the chain midpoint distances `seg_mid_s_m` are measured from bus toward tether tip. This yields current growing toward tip in the sample, while a bus-located cathode would generally require the current boundary to be interpreted in the opposite direction. Resolve with a physics diagram and verified cathode position/conventional-current signs, not by reversing arrays blindly. The current model is a simplified OML approximation and may not correctly handle inactive/passive/reversed-polarity regimes.

### P1 — Current profile is not coupled to instantaneous geometry

The legacy current profile is solved once using mean absolute motional EMF over the orbital table and then scaled by the scalar projection of **bus orbital velocity** crossed with B on an **initial fixed tether direction**. It does not use instantaneous changing segment direction, tether velocity, plasma-relative velocity, or plasma-density changes. Linear scaling of an OML solution need not reproduce a re-solved nonlinear current distribution. Replace with per-step or reduced-order validated electrical solve; define performance strategy later.

### P1 — Mass conservation across configurations

`moi_config_1_coiled` uses `spool.mass` and `m_end_stowed`; `moi_config_2_extended_on_minus_z` uses `tether.mass` and `m_end_deployed`, without guaranteeing consistency. All archived synthetic fixtures have mismatch. Explicitly model hardware spool mass separately from deployed/wound tether material, retain stable spacecraft dry mass, and use one canonical component inventory across all stages. Return mass, center of mass and 3×3 inertia about both bus COM and assembly COM.

### P1 — Inconsistent tether mass lumping / inertia consistency

In chain RHS, `N_seg-1` internal nodes each receive `tether.mass/(N_seg-1)` and the last node gets end mass. A uniform cable's continuous mass is not generally represented accurately by this allocation at coarse resolution, and the last node receives no tether mass. Define an element/consistent or properly lumped mass discretization, with the node at the bus attachment treated explicitly. Verify discrete inertia approaches the closed-form uniform rod with increasing segments, and that mass is counted only once across bus/tether.

### P1 — Jacobian sparsity builder is unused **and incomplete**

`build_jac_sparsity(Np)` handles self and previous-neighbor forces, but acceleration of an interior node also depends on the **next** neighbor through the adjacent spring. `run_bent_tether_sim` passes `jac_sparsity=None` regardless. Correct the pattern against finite-difference/automatic-differentiation structure tests before enabling sparsity; an incorrect sparsity matrix can invalidate derivatives and solver performance.

### P1 — Field sampling stops after one orbit

`make_B_eci_callable` interpolates B over one orbital period and returns the last value for all later times; longer studies freeze the geomagnetic field. Also, the `itrs_vec_to_gcrs_vec` uses position-like `SkyCoord` transformations on a magnetic vector; check proper vector-only rotation and unit consistency independently. Astropy/WMM could not be re-executed in this audit environment, so this is a verification item rather than a demonstrated coordinate-transform bug.

### P1 — Silent failure of electrical solve

`run_bent_tether_sim` catches any exception in current-profile construction, logs a warning, and substitutes zero current, while ultimately returning `success=True`. This can produce plausible-looking but meaningless electrodynamic results. Distinguish explicit zero-current control cases from numerical failures; fail fast by default, or return status with error and prohibit using invalid runs as successful.

### P1 — Result-state API is too small

The integrator creates an entire `(time,state)` history, and `.mat` includes node positions, currents, B, Lorentz segment forces, and bus center, but returned Python dictionary includes only one settled/final snapshot plus metadata. It lacks node velocities, tensions, attachment reactions, power, and convergence diagnostics; the MATLAB file path is hard-coded (`tether_nodes.mat`). Define a `SimulationResult` model and save time histories with explicit units/frames/schema and run identifiers. Avoid recomputing physical force differently for visualization.

### P2 — Data provenance and governance

Golden inertia fixture refers to notebook `MOI_model_current_profile_only__1_.ipynb` with SHA256 `96f81f...`; the included legacy source has a different name/hash. Preserve reference data; recover exact original if available, or document verified equivalence of extracted functions, and keep fixture-generation in controlled maintenance. Ensure Git tags and `pyproject.toml` versions align. Use PRs with physics-specific reviews and GitHub Actions tests.

## Required physics contracts

**Frame conventions:** maintain `I=GCRS inertial`, `B=body frame at bus COM`, `R_IB` maps body to inertial; explicitly add body-to-inertial quaternion order/sign and time epoch. `x_bus_I` and `v_bus_I` describe bus COM translation; body-frame inertia for rigid bus is fixed when its rigid mass is unchanged.

**Attachment:** for point fixed on bus, `x_attach_I=x_bus_I+R_IB r_attach_B`; `v_attach_I=v_bus_I+R_IB(omega_B × r_attach_B)`. The tether returns `F_on_bus_I`, and the bus torque is `tau_attach_B=r_attach_B × R_IB^T F_on_bus_I`. Sign and application point are contractual, not inferred at call sites.

**Mass properties:** `MassProperties(mass, com, inertia, frame, reference_point, components, configuration_id)`. The **assembled** inertia of a rigidly constrained extended tether is useful for static configuration analysis. Do not substitute this total rigidified tensor for the bus's own inertia inside a flexible-system rigid-bus Euler equation; internal tether motion enters via reaction forces and angular momentum.

**Electrodynamics:** define signed current coordinate `s`, cathode location, conventional-current direction, electrical potential and EMF orientation. Distributed Lorentz force is `dF=I(s,t) dl × B`, applied to tether elements. Compute power and current closure consistently, and ensure model scope supports only the claims made about deployment/deorbiting.

**Results:** retain `t`, `x_bus`, `v_bus`, `q_IB`, `omega_B`, `r_nodes`, `v_nodes`, `segment_tension`, `attachment_force`, `I_seg`, `B`, `fL_seg`, optional `I_inertia`, solver diagnostics and `meta` (source commit, SI units, frames, epoch, physical model, numerical tolerances, mission config). If `.npz` plus JSON metadata is chosen, provide a MATLAB export adapter.


## Revised implementation roadmap (ordered by dependency)

### Milestone M0 — Freeze, reproduce and audit (first PR)

- [ ] Record ZIP snapshot, source commit, Python and library versions; exclude `.venv`/`.git` from future distributable archives.
- [ ] Preserve 73 passing tests as baseline with GitHub Actions; run standalone inertia quickstart in CI.
- [ ] Preserve original notebooks and run01 artifacts unchanged; add reproducibility notes. Recover original notebook behind golden fixture, or mark fixture ancestry uncertain.
- [ ] Map actual functions to target packages; document all current physical limitations.
- [ ] Add a minimal README with setup, tests, model scope and limitations.

**Exit:** fresh checkout installs and passes current 73 tests; reference artifact provenance is explained; no changes to physics yet.

### Milestone M1 — Physics and interface specification (can overlap M0)

- [ ] Lock SI units, reference points, frame and quaternion conventions and signing of tether current/force.
- [ ] Define `SpacecraftState`, `MassProperties`, `TetherState`, `EnvironmentSample`, `CurrentProfile`, `SimulationResult` data contracts.
- [ ] Declare whether initial target simulates deployed-only motion or deployment transition, and whether orbital motion is prescribed or integrated.
- [ ] Define acceptance tolerances and conservation metrics per model, then adjust them based on solver-convergence studies.

**Exit:** all modules can exchange data without hidden coordinate/units assumptions.

### Milestone M2 — Mass properties completion

- [ ] Separate fixed bus, remaining spool hardware, wound tether, deployed tether and end mass in common mass budget.
- [ ] Make stowed, partially deployed, fully deployed straight, and arbitrary bent geometry configurable and consistent.
- [ ] Return combined COM and tensor about both bus COM and system COM with explicit frames.
- [ ] Verify mass is conserved across configurations and tensors agree with analytical reference shapes; compare lumped tether against continuous rod as resolution increases.
- [ ] Update examples/golden cases with **new mission-consistent fixtures** while preserving existing legacy fixtures unchanged for regression.

**Exit:** validated single mission model returns physically consistent mass properties for every explicitly defined configuration.

### Milestone M3 — Standalone rigid-bus attitude dynamics (parallel to M4)

- [ ] Add quaternion math for `R_IB` and body rate propagation with sign/unit tests.
- [ ] Implement `I_B dω/dt=τ_B−ω×(I_B ω)` for fixed bus inertia; separate torque providers from integrator.
- [ ] Test identity/zero rate, axis-aligned rotation, constant torque, torque-free angular momentum and kinetic energy conservation, quaternion norm.
- [ ] Plot attitude, Euler angles if helpful, quaternion and body rates. Optional FlightGear integration only after verification.

**Exit:** a validated rotation simulator can run with zero, known and prescribed external torques.

### Milestone M4 — Validate and package environment and EDT current (parallel to M3/M5)

- [ ] Extract circular orbit and magnetic-field code into `environment` with deterministic tests on synthetic fields.
- [ ] Test ECI/ECEF frames and rotation-only magnetic-field-vector transformations using independent reference values; make field time-domain explicit and never silently clamp beyond domain.
- [ ] Use plasma-relative velocity and validate WMM applicability to orbit/epoch; document field interpolation errors.
- [ ] Extract OML shooting and separate rectangular-tape/cylindrical cross-section geometries, including their collecting perimeter and resistance.
- [ ] Resolve cathode location, boundary conditions and current coordinate; verify shooting residual, no-collection/zero-current scenarios and an independent reference case.
- [ ] Replace electrical silent-fallback with explicit modes and errors. Test reference/power sign and current variations under changed EMF/geometry; add optional cached profiles only after convergence validation.

**Exit:** the current model returns trusted current distribution with documented assumptions and an independent electrodynamic force test.

### Milestone M5 — Correct and package the mechanical tether

- [ ] Write isolated 1-spring / 2-mass tests (rest length, slack, compression, damping, momentum, energy) and reproduce the unwanted 0.4045-N rest tension.
- [ ] Fix tension-only regularization to yield exactly zero tension for unstretched/slack stationary segments; avoid artificial damping outside tension regime.
- [ ] Implement physically consistent cable mass distribution; test mass, COM and inertia against a uniform rod as `N_seg` increases.
- [ ] Separate optional **quasi-static equilibrium** mode from **physical transient** dynamics; disable mass inflation in transient mode.
- [ ] Fix or remove invalid Jacobian sparsity; compare numerically against actual finite differences. Validate solver stiffness/convergence versus `N_seg`, timestep and tolerance.
- [ ] Return full time history, actual segment lengths, tension, mechanical energy, attachment force and all solver diagnostics.
- [ ] Run mechanical-only tether cases under constant/known gravity and prescribed attachment, with current set exactly to zero.

**Exit:** a tether at nominal rest length does not collapse under phantom tension and physical transients converge numerically.

### Milestone M6 — Two-way **mechanical** bus/tether coupling

- [ ] Compute attachment position/velocity from propagated spacecraft attitude and bus translation.
- [ ] Return equal-and-opposite tether/bus attachment force and its moment about bus COM.
- [ ] Couple physical bus six-degree-of-freedom state with tether node states for free-body verification. If a prescribed bus orbit is retained for reduced mode, mark external orbit constraint explicitly.
- [ ] Test zero-tether limit reduces to M3, zero-force total linear/angular momentum conservation, and equal/opposite impulse transfer.
- [ ] Compare monolithic solver with staggered-coupling prototype across exchange-step sizes; select on accuracy/stability evidence.

**Exit:** a rotating bus moves its tether attachment and tether force back-reacts on bus without double counting.

### Milestone M7 — Electrodynamic two-way coupling

- [ ] Evaluate magnetic/plasma environment at simulation time and appropriate spatial sample points.
- [ ] Solve signed current along instantaneous tether geometry (or validated approximation), with appropriate relative velocity.
- [ ] Apply segment Lorentz forces and other modeled loads to tether nodes; close cathode/plasma circuit in line with selected model.
- [ ] Verify force vector `I dl×B` against simple uniform field and uniform-current case; check net force, moments, power/energy direction and sign flips.
- [ ] Compare reduced electrical scaling against actual re-solved OML cases; do not assume mean-EMF scaling remains valid.
- [ ] Capture complete time history and run uncertainty/sensitivity studies.

**Exit:** verified attitude–tether–current feedback for selected mission mode, including tests for disabling each subsystem.

### Milestone M8 — Orbit/deorbit dynamics (conditional on mission scope)

- [ ] Add translational orbit propagation with external forces on all bodies as necessary, rather than claiming orbital decay from prescribed circular orbit.
- [ ] Evaluate orbit-energy and angular-momentum change, distinguish drag/motional power and imposed supply power, and validate direction against known simple cases.
- [ ] Add electrodynamic tether + geomagnetic mission scenarios and long-time environmental interpolation with explicit validity checks.

**Exit:** only if this stage is completed and validated can the simulator estimate actual deorbit rate rather than merely tether/attitude behavior.

### Milestone M9 — Visualization, study automation and release

- [ ] Plot bus attitude/rates, tether shapes/velocities/tension, currents, Lorentz forces, energy/momentum diagnostics and COM/inertia state.
- [ ] Produce reproducible animation of **full saved time histories**; keep `.mat` MATLAB adapter optional.
- [ ] Build scenario configs, sweep scripts, seeded experiments, and reference mission data with explicit versioning.
- [ ] Validate clean installation on new machine and run complete CI; review assumptions/accuracy with subsystem leads; cut a release after validation evidence is recorded.

**Exit:** reproducible versioned simulator, user documentation and validated scenario studies.

## GitHub project structure and actionable issues

Create **one GitHub Project** with columns `Backlog`, `Ready`, `In Progress`, `In Review`, `Blocked`, `Done` and tracks `Foundations`, `Mass Properties`, `Attitude`, `Environment/Current`, `Tether`, `Coupling`, `Validation/Visualization`. Group issues under milestone or parent issue. Use these top-level parent issues:

1. `P0 Repository baseline / reproducibility` — CI, frozen fixtures, baseline README, source documentation.
2. `P0 Frame and state contracts` — reference frames, quaternion and force sign conventions, result schemas.
3. `P1 Mass properties and configuration conservation` — canonical components, partial deployment, COM/inertia, tests.
4. `P1 Standalone attitude propagation` — quaternion ODE, torques, conservation tests.
5. `P1 Environment integration` — orbit, WMM, time samples, transforms, plasma-relative motion.
6. `P0 OML/current coordinate audit` — cathode location, signed boundary, electrical closure.
7. `P0 Tether mechanics defect reproduction and fix` — zero-extension tension, collapse regression, physical mass, discretization, Jacobian.
8. `P1 Two-way mechanical coupling` — 6DOF attachment reaction, momentum/energy tests.
9. `P1 Electrodynamic coupling` — segment Lorentz loading, currents/geometry, convergence.
10. `P2 Deorbit/orbital evolution` — if required, orbital propagation and deorbit validation.
11. `P2 Output schema, visualization, MATLAB adapter` — reproducible histories, diagnostics, animations.
12. `P2 Integrated scenarios, documentation and release` — mission parameter inventory, acceptance test report, packaged release.

**First four pull requests (recommended):** (1) baseline+CI+README without changing physics; (2) `MassProperties` / frame/state contract and documentation; (3) zero-tension reproduction + corrected mechanical law with isolated unit tests; (4) mass-conserving configurations + rigid attitude solver in small independently reviewable changes. Environmental/OML extraction can proceed in parallel with the latter two.

## Critical integration tests to add

- Zero applied torque/zero body rate leaves attitude unchanged.
- Torque-free rigid body conserves inertial angular momentum, kinetic energy, quaternion norm to convergence tolerance.
- Total mass, COM and inertia of equivalent stowed/deployed component inventories are consistent.
- A rest-length tether with zero relative velocity has zero tension and no internally induced collapse.
- No-force coupled bus/tether conserves total linear and angular momentum; attachment forces are equal/opposite.
- Tether with current disabled generates exactly zero Lorentz force, and coupled result matches mechanical-only baseline.
- Known constant current, straight tether, uniform B yields vector Lorentz force `I L×B`.
- Reversing current or B reverses Lorentz-force direction.
- Current-profile boundary values match the **actual** cathode/collector coordinates, not simply the mathematical integration coordinate.
- Out-of-range environment time sample fails explicitly or invokes verified regeneration; never silently freezes B.
- Stiff-solver accuracy converges with tighter tolerances and more tether segments; sparse Jacobian matches numerical sparsity.
- Converting saved results to/from MATLAB retains time, units, frames and force sign.
- If orbit/deorbit stage enabled: orbital energy response agrees in sign with work done by the applied external force.

