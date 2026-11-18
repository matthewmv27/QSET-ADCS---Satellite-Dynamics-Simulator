"""Compare actual physical 50-m tether transients across mesh resolutions.

No mass scaling or electrical loads. Reports differences against the finest
mesh, NOT a proof of convergence to the true flexible-cable solution.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from edtsim.tether import constant_attachment, simulate_cable
from scripts.run_m5_mechanical_reference import _model, _initial


def run_mesh_study(config_path, output_dir, segment_counts=(25, 50, 100, 200)):
    config = json.loads(Path(config_path).read_text())
    if config.get('mode', 'transient') != 'transient':
        raise ValueError('mesh study requires transient mode')
    if config['environment'].get('current_A', 0) != 0:
        raise ValueError('mesh study requires exactly zero current')
    if not isinstance(segment_counts, (list, tuple)) or len(segment_counts) < 2:
        raise ValueError('provide at least two segment counts')
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in segment_counts):
        raise ValueError('segment counts must be positive integers')
    if list(segment_counts) != sorted(set(segment_counts)):
        raise ValueError('segment counts must be unique and sorted ascending')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    integration = config['integration']
    ts = np.linspace(0, float(integration['duration_s']), int(integration['samples']))
    if not np.all(np.isfinite(ts)) or len(ts) < 2 or ts[-1] <= 0:
        raise ValueError('invalid integration time grid')
    gravity = np.asarray(config['environment']['gravity_I_m_s2'], float)
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
        raise ValueError('gravity must be a finite three-vector')
    attach_data = config['attachment']
    attachment = constant_attachment(attach_data['position_I_m'],
        attach_data.get('velocity_I_m_s'), attach_data.get('acceleration_I_m_s2'))
    histories = {}
    for n in segment_counts:
        case = json.loads(json.dumps(config))
        case['mechanics']['segments'] = n
        model = _model(case)
        r0, v0 = _initial(model, case)
        histories[n] = (model, simulate_cable(
            model, ts, r0, v0, attachment=attachment, gravity_I=gravity,
            method=integration.get('method', 'DOP853'),
            rtol=integration.get('rtol', 1e-8), atol=integration.get('atol', 1e-10),
            max_step_s=integration.get('max_step_s', np.inf)))
    ref_n = segment_counts[-1]
    ref = histories[ref_n][1]
    ref_u = np.linspace(0., 1., ref_n + 1)
    ref_last = ref.r_nodes_I_m[-1]
    rows = []
    for n, (model, history) in histories.items():
        this_u = np.linspace(0., 1., n + 1)
        this_last = history.r_nodes_I_m[-1]
        interpolated = np.column_stack([
            np.interp(ref_u, this_u, this_last[:, axis]) for axis in range(3)])
        shape_rms = float(np.sqrt(np.mean(np.sum((interpolated - ref_last)**2, axis=1))))
        tip_time_rms = float(np.sqrt(np.mean(np.sum(
            (history.r_nodes_I_m[:, -1] - ref.r_nodes_I_m[:, -1])**2, axis=1))))
        rows.append(dict(
            segments=n, nodes=n + 1, rest_segment_length_m=float(model.rest_length_m),
            physical_mass_kg=float(model.total_mass_kg),
            maximum_tension_N=float(history.segment_tensions_N.max()),
            final_tip_x_m=float(this_last[-1,0]),
            final_tip_y_m=float(this_last[-1,1]),
            final_tip_z_m=float(this_last[-1,2]),
            tip_time_rms_error_m_vs_finest=tip_time_rms,
            final_shape_rms_error_m_vs_finest=shape_rms,
            final_attachment_force_error_N_vs_finest=float(np.linalg.norm(
                history.attachment_force_on_bus_I_N[-1]
                - ref.attachment_force_on_bus_I_N[-1])),
            nfev=int(history.solver_diagnostics['nfev']),
            mass_scale_used=float(history.solver_diagnostics['mass_scale_used'])))
    with open(out/'mesh_comparison.csv','w',newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary=dict(milestone='M5', study='50m_mesh_refinement',
        source_config=str(config_path), reference_segments=ref_n,
        reference_is_not_analytical_truth=True, rows=rows,
        limitations=['Finite-mesh comparison is not proof of physical accuracy',
            'Tether material and geometry are illustrative',
            'Prescribed attachment, no orbital or current coupling',
            'Tension-only cable has no bending stiffness'])
    (out/'mesh_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--segments', nargs='+', type=int, default=[25,50,100,200])
    args=parser.parse_args()
    print(json.dumps(run_mesh_study(args.config,args.output_dir,args.segments),indent=2))
