"""M5 prescribed-500-km-orbit tether demonstration (no orbital propagation).

Three comparisons: straight/zero current, straight/small in-plane angle,
straight/assumed-current synthetic-field M4e loads. Node 0 follows a
prescribed EXACT two-body circular trajectory, including analytical
velocity and acceleration. Every cable node feels central gravity; the
reference orbit does not precess. Outputs are resolved in the instantaneous
radial/along-track/cross-track frame rather than raw 6800-km ECI positions.

Usage:
  python scripts/run_m5_orbit_driven_reference.py \\
    configs/scenarios/m5_orbit_driven_500km.json --output-dir results/m5_orbit
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from edtsim.core.constants import MU_EARTH
from edtsim.environment.orbit import CircularReferenceOrbit, sso_inclination_deg
from edtsim.environment.fields import UniformMagneticField
from edtsim.electrodynamics.current import PrescribedConstantCurrent
from edtsim.electrodynamics.tether_adapter import prescribed_loading_callback
from edtsim.tether import simulate_cable
try:
    from scripts.run_m5_mechanical_reference import _model
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from run_m5_mechanical_reference import _model


SCENARIOS = ('straight_gravity', 'tilted_gravity', 'straight_lorentz')


def circular_orbit_from_config(config):
    """Create prescribed two-body orbit. J2 nodal precession is forbidden here."""
    orbit_spec = config['orbit']
    if orbit_spec.get('precess_raan', False) is not False:
        raise ValueError('M5 orbit-driven demo requires precess_raan=false: two-body circular acceleration must be consistent with prescribed motion')
    alt = float(orbit_spec['altitude_m'])
    inclination = orbit_spec['inclination_deg']
    if inclination == 'sso':
        inclination = sso_inclination_deg(alt)
    return CircularReferenceOrbit(
        altitude_m=alt, inclination_deg=float(inclination),
        raan_deg=float(orbit_spec.get('raan_deg', 0.)),
        argument_latitude_deg=float(orbit_spec.get('argument_latitude_deg', 0.)),
        precess_raan=False)


def central_gravity_I(t_s, position_I_m):
    """Two-body Earth gravity (identical law for attachment and every node)."""
    position = np.asarray(position_I_m, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError('position_I_m must be finite 3-vector')
    radius = np.linalg.norm(position)
    if radius <= 0:
        raise ValueError('central gravity undefined at Earth centre')
    return -MU_EARTH * position / radius**3


def orbital_frame(orbit, t_s):
    """Right-handed (radial, along-track, orbit-normal) inertial unit axes."""
    r, v = orbit.state(t_s)
    er = r / np.linalg.norm(r)
    vt = v - (v @ er) * er
    et = vt / np.linalg.norm(vt)
    eh = np.cross(er, et)
    return r, v, er, et, eh


class TranslatedCentralGravity:
    """Earth gravity for inertial coordinates translated by fixed origin."""

    def __init__(self, origin_I_m):
        self.origin_I_m = np.asarray(origin_I_m, float).copy()

    def sample_nodes(self, t_s, positions_relative_I_m):
        positions = np.asarray(positions_relative_I_m, float) + self.origin_I_m
        radii = np.linalg.norm(positions, axis=1)
        if not np.all(np.isfinite(radii)) or np.any(radii <= 0):
            raise ValueError('invalid position in vectorized central gravity')
        return -MU_EARTH * positions / radii[:, None]**3

    def __call__(self, t_s, position_relative_I_m):
        return central_gravity_I(t_s, np.asarray(position_relative_I_m) + self.origin_I_m)


def circular_attachment(orbit, origin_I_m=None):
    """Prescribed (r,v,a), translated origin to reduce cancellation."""
    origin = np.zeros(3) if origin_I_m is None else np.asarray(origin_I_m, float)
    def attached(t_s):
        r, v = orbit.state(t_s)
        return r - origin, v, central_gravity_I(t_s, r)
    return attached


def straight_initial_state(model, orbit, angle_deg=0., extension_fraction=0., origin_I_m=None):
    """Initially straight tether, with co-rotating velocities, not static nodes.

    Angle is a *rigid tilt* in the orbital plane, not an initial bow.
    The exact straight baseline has angle=0 and zero axial extension.
    """
    theta = np.deg2rad(float(angle_deg))
    ext = float(extension_fraction)
    if not np.isfinite(theta) or not np.isfinite(ext) or ext < 0:
        raise ValueError('angle and nonnegative extension must be finite')
    ar, av, er, et, eh = orbital_frame(orbit, 0.)
    u = np.linspace(0., 1., model.segments + 1)
    direction = np.cos(theta)*er + np.sin(theta)*et
    offsets = (u * model.tether.length_m * (1. + ext))[:, None] * direction
    omega_vec = orbit.mean_motion_rad_s * eh
    origin = np.zeros(3) if origin_I_m is None else np.asarray(origin_I_m, float)
    r0 = (ar - origin) + offsets
    v0 = av + np.cross(np.broadcast_to(omega_vec, offsets.shape), offsets)
    # Make the fixed attachment position and velocity exactly identical to
    # the prescribed orbit, even for arbitrary orbit-plane orientations.
    r0[0] = ar - origin
    v0[0] = av
    return r0, v0


def local_history(orbit, history, origin_I_m=None):
    """Node offsets in LVLH radial/along-track/cross-track coordinates."""
    origin = np.zeros(3) if origin_I_m is None else np.asarray(origin_I_m, float)
    local = np.zeros_like(history.r_nodes_I_m)
    force_local = np.zeros_like(history.attachment_force_on_bus_I_N)
    for j, t in enumerate(history.t_s):
        ar, _, er, et, eh = orbital_frame(orbit, float(t))
        basis = np.stack([er, et, eh], axis=1)
        local[j] = (history.r_nodes_I_m[j] - (ar - origin)) @ basis
        force_local[j] = history.attachment_force_on_bus_I_N[j] @ basis
    return local, force_local


def _write_csv(path, headings, rows):
    with Path(path).open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headings)
        writer.writerows(rows)


def run(config_path, output_dir, *, cases=SCENARIOS, segments_override=None):
    data = json.loads(Path(config_path).read_text())
    if segments_override is not None:
        if isinstance(segments_override, bool) or not isinstance(segments_override, int) or segments_override < 1:
            raise ValueError('segments_override must be a positive integer')
        data['mechanics']['segments'] = segments_override
    model = _model(data)
    orbit = circular_orbit_from_config(data)
    duration = float(data['integration']['duration_s'])
    samples = int(data['integration']['samples'])
    if not np.isfinite(duration) or duration <= 0 or samples < 2:
        raise ValueError('positive duration and at least two samples required')
    ts = np.linspace(0., duration, samples)
    if not cases or len(set(cases)) != len(cases) or set(cases) - set(SCENARIOS):
        raise ValueError('cases must be unique subset of '+str(SCENARIOS))
    current = float(data['electrodynamics']['assumed_signed_uniform_current_A'])
    magnetic = float(data['electrodynamics']['synthetic_field_orbit_normal_T'])
    tilt_deg = float(data['initial']['libration_perturbation_deg'])
    extension = float(data['initial'].get('extension_fraction', 0.))
    if not all(np.isfinite(x) for x in (current, magnetic, tilt_deg)):
        raise ValueError('current, field and tilt must be finite')
    if current == 0 or magnetic == 0:
        # Zero current/field is acceptable for no-loading controls, but make
        # nonzero input explicit for the example's current-on comparison.
        if 'straight_lorentz' in cases:
            raise ValueError('straight_lorentz case requires nonzero assumed current and field')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    origin = orbit.state(0.)[0].copy()
    attachment = circular_attachment(orbit, origin)
    gravity_provider = TranslatedCentralGravity(origin)
    _, _, _, _, eh0 = orbital_frame(orbit, 0.)
    field = UniformMagneticField(tuple(magnetic * eh0))  # deliberately synthetic inertial B
    load = prescribed_loading_callback(field, PrescribedConstantCurrent(current))
    results = {}
    for name in cases:
        angle = tilt_deg if name == 'tilted_gravity' else 0.
        r0, v0 = straight_initial_state(model, orbit, angle, extension, origin)
        has_ed = (name == 'straight_lorentz')
        history = simulate_cable(
            model, ts, r0, v0,
            attachment=attachment, gravity_I=gravity_provider,
            external_nodal_forces_I=load if has_ed else None,
            method=data['integration'].get('method', 'DOP853'),
            rtol=float(data['integration'].get('rtol', 1e-8)),
            atol=float(data['integration'].get('atol', 1e-10)),
            max_step_s=float(data['integration'].get('max_step_s', np.inf)))
        local, force_local = local_history(orbit, history, origin)
        tip_angle_deg = np.rad2deg(np.arctan2(local[:, -1, 1], local[:, -1, 0]))
        # Difference from straight chord between anchor and actual tip in
        # co-rotating axes, not merely the tip's angular displacement.
        u = np.linspace(0, 1, model.segments + 1)
        bow = np.linalg.norm(local - u[None, :, None] * local[:, -1:, :], axis=2)
        case_dir = out / name
        case_dir.mkdir(exist_ok=True)
        np.savez_compressed(case_dir/'history.npz',
            t_s=history.t_s, positions_LVLH_m=local,
            positions_translated_I_m=history.r_nodes_I_m,
            inertial_origin_Earth_centered_m=origin,
            velocities_I_m_s=history.v_nodes_I_m_s,
            segment_lengths_m=history.actual_segment_lengths_m,
            segment_tensions_N=history.segment_tensions_N,
            attachment_force_on_bus_I_N=history.attachment_force_on_bus_I_N,
            attachment_force_LVLH_N=force_local,
            node_masses_kg=history.node_masses_kg,
            tip_libration_deg=tip_angle_deg,
            maximum_chord_bow_m=bow.max(axis=1))
        _write_csv(case_dir/'history.csv',
            ['time_s', 'tip_radial_m', 'tip_alongtrack_m', 'tip_crosstrack_m',
             'tip_angle_deg', 'max_chord_bow_m', 'max_tension_N',
             'force_bus_radial_N', 'force_bus_alongtrack_N', 'force_bus_crosstrack_N'],
            [[t, *tip, phi, bow_max, np.max(tension), *f]
             for t,tip,phi,bow_max,tension,f in zip(
                ts, local[:, -1, :], tip_angle_deg, bow.max(axis=1),
                history.segment_tensions_N, force_local)])
        # At selected times, preserve every node's geometry for plots and
        # spatial comparisons without requiring a full 3D viewer.
        snapshots = [0, len(ts)//4, len(ts)//2, 3*len(ts)//4, len(ts)-1]
        _write_csv(case_dir/'shape_samples.csv',
            ['time_s', 'node_index', 'radial_m', 'alongtrack_m', 'crosstrack_m'],
            [[ts[j], k, *point] for j in snapshots
             for k, point in enumerate(local[j])])
        results[name] = dict(
            electrical_loading_assumed=has_ed,
            initial_angular_offset_deg=angle,
            initial_tip_LVLH_m=local[0, -1].tolist(),
            final_tip_LVLH_m=local[-1, -1].tolist(),
            initial_max_tension_N=float(history.segment_tensions_N[0].max()),
            max_tension_N=float(history.segment_tensions_N.max()),
            max_chord_bow_m=float(bow.max()),
            final_tip_libration_deg=float(tip_angle_deg[-1]),
            final_attachment_force_LVLH_N=force_local[-1].tolist(),
            mass_kg=model.total_mass_kg,
            solver=history.solver_diagnostics)
    summary = dict(
        milestone='M5', study='orbit_driven_prescribed_attachment',
        provenance=data.get('provenance', 'unspecified'),
        source_config=str(config_path),
        orbit=dict(altitude_m=orbit.altitude_m, inclination_deg=orbit.inclination_deg,
                   raan_deg=orbit.raan_deg, period_s=orbit.period_s,
                   precess_raan=False, gravity='central_mu_over_r_squared'),
        inertial_positions_use_fixed_translation=True,
        inertial_origin_earth_centered_I_m=origin.tolist(),
        tether=dict(length_m=model.tether.length_m, segments=model.segments,
                    mass_kg=model.total_mass_kg, mass_scale_used=1.0),
        electrical=dict(assumed_current_A=current, synthetic_field_T=magnetic,
                        actual_CNT_OML_circuit_closed=False),
        results=results,
        limitations=[
            'PRESCRIBED Kepler circular orbit (two-body, fixed RAAN), no orbit propagation or J2 precession',
            'SSO inclination defines orientation only; over a short run this is not a true Sun-synchronous orbit propagation',
            'Straight cable follows prescribed attachment; spacecraft does not respond to reaction force',
            'Synthetic spatially uniform inertial magnetic field and imposed constant current, not a real Earth/plasma prediction',
            'Tether is tension-only with no bending stiffness, reel payout, or CNT/OML circuit closure',
            'Energy in inertial frame includes orbital kinetic energy; do not interpret as closed-system energy conservation'])
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cases', nargs='+', default=list(SCENARIOS), choices=SCENARIOS)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir, cases=args.cases), indent=2))
