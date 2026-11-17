"""Run illustrative spacecraft-side tether-loading diagnostic (no orbit propagation).

python scripts/run_m4e_loading_reference.py configs/scenarios/m4e_loading_reference.json \
    --output-dir results/m4e_loading
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from edtsim.core.states import TetherState
from edtsim.electrodynamics import solve_prescribed_bias_snapshot, evaluate_distributed_loading
from edtsim.environment.flows import motional_electric_field_I
from edtsim.tether import (RoundWire, RectangularTape, HomogeneousTether)


class UniformField:
    def __init__(self, value):
        self.B = np.asarray(value, dtype=float)
        if self.B.shape != (3,) or not np.all(np.isfinite(self.B)):
            raise ValueError('B_I_T must be a finite 3-vector')

    def sample(self, t_s, position_I_m):
        return self.B


def _constant_velocity(vector):
    v = np.asarray(vector, dtype=float)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError('Flow must be finite 3-vector')
    return lambda t, pos: v


def run(config_path, output_dir):
    cfg = json.loads(Path(config_path).read_text())
    if cfg.get('provenance',{}).get('status') != 'illustrative':
        raise ValueError('M4e reference scenario must explicitly say illustrative')
    if cfg['environment'].get('mode') != 'synthetic_uniform_field_and_flow':
        raise ValueError('M4e example only supports explicitly synthetic constant environment')
    shape = cfg['tether']['cross_section']
    if shape['type'] == 'round':
        section = RoundWire(shape['diameter_m'])
    elif shape['type'] == 'tape':
        section = RectangularTape(shape['width_m'], shape['thickness_m'])
    else:
        raise ValueError('Unknown cross-section type')
    tcfg = cfg['tether']
    tether = HomogeneousTether(section,tcfg['length_m'],tcfg['density_kg_m3'],
                              tcfg['young_modulus_Pa'],tcfg['conductivity_S_m'],
                              tcfg['bare_fraction'])
    geom = cfg['geometry']
    state = TetherState(geom['nodes_I_m'],geom['node_velocities_I_m_s'])
    dl = np.diff(state.r_nodes_I,axis=0)
    length = np.linalg.norm(dl,axis=1)
    if not np.isclose(sum(length), tether.length_m, rtol=1e-9, atol=1e-6):
        raise ValueError('Material length does not agree with prescribed nodes')
    midpoint = (state.r_nodes_I[1:]+state.r_nodes_I[:-1])/2
    velocities = (state.v_nodes_I[1:]+state.v_nodes_I[:-1])/2
    env = cfg['environment']
    field = UniformField(env['B_I_T'])
    plasma_velocity = _constant_velocity(env['plasma_velocity_I_m_s'])
    neutral_velocity = _constant_velocity(env['neutral_velocity_I_m_s'])
    rho = float(env['neutral_density_kg_m3'])
    if not np.isfinite(rho) or rho < 0:
        raise ValueError('Invalid neutral density')
    emf_per_m = np.array([np.dot(motional_electric_field_I(v,plasma_velocity(0,p),field.sample(0,p)),axis)/ds
                          for v,p,axis,ds in zip(velocities,midpoint,dl,length)])
    mean_emf = float(np.dot(emf_per_m,length)/np.sum(length))
    spread = float(np.max(np.abs(emf_per_m-mean_emf)))
    e = cfg['electrical']
    if e.get('mode') != 'prescribed_bus_bias_OML_with_CNT_capacity_diagnostic':
        raise ValueError('Only diagnostic prescribed-bias OML mode is implemented')
    c = cfg['cathode']
    solution = solve_prescribed_bias_snapshot(tether,
          electron_density_m3=e['electron_density_m3'],
          electron_temperature_K=e['electron_temperature_K'],
          bus_plasma_bias_V=e['bus_plasma_bias_V'],
          motional_E_along_V_m=mean_emf,
          available_gate_voltage_V=c['available_gate_voltage_V'],
          equivalent_age_h=c['equivalent_age_h'],
          max_cathode_power_W=c['max_cathode_power_W'],
          max_emission_current_A=c['max_emission_current_A'],nodes=int(e['nodes']))
    t = cfg['time_s']
    ref = cfg['bus_reference_point_I_m']
    width = geom.get('tape_width_direction_I')
    result = evaluate_distributed_loading(t,state,field,solution.profile,
                      reference_point_I_m=ref,
                      plasma_velocity_provider=plasma_velocity,
                      tether=tether, neutral_velocity_provider=neutral_velocity,
                      neutral_density_provider=lambda time,pos:rho,
                      drag_coefficient=cfg['aerodynamics']['drag_coefficient'],
                      tape_width_direction_I=width)
    status=solution.cathode_check
    summary = {
      'milestone':'M4e', 'input_config':str(config_path),
      'source_status':'illustrative',
      'modelling_scope':'EXTERNAL tether loading, not force/torque transmitted to bus',
      'material_length_m':float(sum(length)),
      'current_at_bus_A':status.current_required_A,
      'required_cnt_gate_voltage_V':status.gate_voltage_required_V,
      'estimated_cnt_cathode_power_W':status.estimated_cathode_power_W,
      'within_assumed_cathode_limits':status.supports_assumed_demand,
      'operating_point_closed':False,
      'assumed_bus_to_plasma_bias_V':e['bus_plasma_bias_V'],
      'mean_motional_E_along_tether_V_m':mean_emf,
      'maximum_segment_E_deviation_V_m':spread,
      'oml_tip_current_residual_A':float(solution.oml_solution.tip_current_residual_A),
      'net_external_tether_force_I_N':result.net_tether_external_force_I_N.tolist(),
      'net_external_tether_torque_about_ref_I_Nm':result.net_tether_external_torque_about_ref_I_Nm.tolist(),
      'distributed_lorentz_power_W':result.lorentz_power_local_W,
      'motional_emf_power_W':result.motional_emf_power_W,
      'plasma_motion_power_W':result.plasma_motion_power_W,
      'power_identity_residual_W':result.power_identity_residual_W,
      'distributed_neutral_drag_power_W':result.drag_power_local_W,
      'warnings': list(status.warnings) + [
         'Prescribed OML current is a hypothetical electrical loading, even when CNT limit check passes.',
         'Spatially variable motional field is reduced to a length-weighted average in OML solution.',
         'Resulting external tether force/torque is NOT the bus attachment reaction.',
         'No translation or attitude propagation; no validation of natural tether orientation.'
      ]
    }
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (out/'segment_loading.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['segment','midpoint_x_I_m','midpoint_y_I_m','midpoint_z_I_m',
                    'current_A','lorentz_fx_I_N','lorentz_fy_I_N','lorentz_fz_I_N',
                    'drag_fx_I_N','drag_fy_I_N','drag_fz_I_N'])
        for i,(mid,current,m,d) in enumerate(zip(result.segment_midpoints_I_m,
               result.segment_currents_A,result.lorentz_forces_I_N,result.drag_forces_I_N)):
            w.writerow([i,*map(float,mid),float(current),*map(float,m),*map(float,d)])
    return summary


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('results/m4e_loading'))
    a = parser.parse_args()
    print(json.dumps(run(a.config,a.output_dir),indent=2))
