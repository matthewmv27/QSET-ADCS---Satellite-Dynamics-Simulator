"""Standalone M5 physical-tether reference run (no orbit or electrical current).

From repository root:
  python scripts/run_m5_mechanical_reference.py \\
    configs/scenarios/m5_mechanical_example.json --output-dir results/m5_reference
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from edtsim.tether import (RoundWire, RectangularTape, HomogeneousTether,
    CableModel, constant_attachment, simulate_cable, solve_static_equilibrium)


def _model(data):
    shape = data['tether']['cross_section']
    if shape['type'] == 'round':
        cross_section = RoundWire(shape['diameter_m'])
    elif shape['type'] == 'tape':
        cross_section = RectangularTape(shape['width_m'], shape['thickness_m'])
    else:
        raise ValueError("cross_section.type must be 'round' or 'tape'")
    physical = HomogeneousTether(
        cross_section, data['tether']['length_m'], data['tether']['density_kg_m3'],
        data['tether']['young_modulus_Pa'], data['tether']['conductivity_S_m'],
        bare_fraction=data['tether'].get('bare_fraction', 1.0))
    return CableModel(physical, data['mechanics']['segments'],
        data['mechanics'].get('end_mass_kg',0.),
        data['mechanics'].get('viscous_axial_rigidity_Ns',0.))


def _initial(model, data):
    anchor = np.asarray(data['attachment']['position_I_m'],float)
    direction = np.asarray(data['initial']['direction_I'],float)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)) or np.linalg.norm(direction)==0:
        raise ValueError("initial.direction_I must be finite, nonzero 3-vector")
    direction = direction/np.linalg.norm(direction)
    bow = np.asarray(data['initial'].get('midpoint_bow_I_m', [0,0,0]),float)
    if bow.shape != (3,) or not np.all(np.isfinite(bow)):
        raise ValueError('midpoint_bow_I_m must be a finite 3-vector')
    if abs(float(bow @ direction)) > 1e-10:
        raise ValueError('midpoint bow must be perpendicular to initial tether direction')
    ext = float(data['initial'].get('extension_fraction',0.))
    if not np.isfinite(ext) or ext<0:
        raise ValueError('extension_fraction must be finite and nonnegative')
    u=np.linspace(0,1,model.segments+1)
    positions = anchor + u[:,None]*model.tether.length_m*(1+ext)*direction + np.sin(np.pi*u)[:,None]*bow
    vel = np.asarray(data['attachment'].get('velocity_I_m_s',[0,0,0]),float)
    if vel.shape != (3,) or not np.all(np.isfinite(vel)):
        raise ValueError('attachment velocity must be finite 3-vector')
    initial_velocity = np.broadcast_to(vel, positions.shape).copy()
    return positions,initial_velocity


def _csv(path, columns, rows):
    with open(path,'w',newline='') as f:
        writer=csv.writer(f);writer.writerow(columns);writer.writerows(rows)


def run(config_path, output_dir):
    data=json.loads(Path(config_path).read_text())
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    model=_model(data)
    r0,v0=_initial(model,data)
    gravity=np.asarray(data['environment']['gravity_I_m_s2'],float)
    if gravity.shape!=(3,) or not np.all(np.isfinite(gravity)):
        raise ValueError('gravity_I_m_s2 must be a finite 3-vector')
    if float(data['environment'].get('current_A',0)) != 0:
        raise ValueError('M5 mechanical reference requires current_A exactly 0; M4e integration belongs to M6')
    mode=data.get('mode','transient')
    common=dict(milestone='M5', mode=mode, provenance=data.get('provenance','unspecified'),
        source_config=str(config_path), mass_kg=model.total_mass_kg,
        tether_mass_kg=model.tether.mass_kg, end_mass_kg=model.end_mass_kg,
        segments=model.segments, mass_scale_used=1.0,
        electrical_current_A=0.,
        limitations=['Fixed/prescribed attachment, bus translation/attitude not integrated',
                     'Lumped nodal masses; no transverse bending stiffness or bending moments',
                     'No electromagnetic, plasma or aerodynamic forces in this reference',
                     'Illustrative parameters are not verified mission hardware'])
    if mode=='quasi_static':
        if np.linalg.norm(np.asarray(data['attachment'].get('velocity_I_m_s',[0,0,0]),float))>0 or np.linalg.norm(np.asarray(data['attachment'].get('acceleration_I_m_s2',[0,0,0]),float))>0:
            raise ValueError('quasi_static requires a stationary attachment')
        result=solve_static_equilibrium(model,r0,data['attachment']['position_I_m'],gravity)
        common.update(positions_I_m=result.r_nodes_I_m.tolist(),
            segment_lengths_m=result.actual_segment_lengths_m.tolist(),
            tensions_N=result.segment_tensions_N.tolist(),
            force_on_bus_I_N=result.attachment_force_on_bus_I_N.tolist(),
            potential_J=result.total_potential_J,
            solver=result.solver_diagnostics)
    elif mode=='transient':
        integration=data['integration']
        ts=np.linspace(0,float(integration['duration_s']),int(integration['samples']))
        attach=constant_attachment(data['attachment']['position_I_m'],
            data['attachment'].get('velocity_I_m_s'),data['attachment'].get('acceleration_I_m_s2'))
        result=simulate_cable(model,ts,r0,v0,attachment=attach,gravity_I=gravity,
            method=integration.get('method','Radau'),rtol=integration.get('rtol',1e-7),
            atol=integration.get('atol',1e-9),max_step_s=integration.get('max_step_s',np.inf))
        np.savez_compressed(out/'history.npz',t_s=result.t_s,
            r_nodes_I_m=result.r_nodes_I_m,v_nodes_I_m_s=result.v_nodes_I_m_s,
            node_masses_kg=result.node_masses_kg,
            actual_segment_lengths_m=result.actual_segment_lengths_m,
            segment_tensions_N=result.segment_tensions_N,
            kinetic_energy_J=result.kinetic_energy_J,elastic_energy_J=result.elastic_energy_J,
            gravity_potential_J=result.gravity_potential_J,
            total_mechanical_energy_J=result.total_mechanical_energy_J,
            com_I_m=result.com_I_m,inertia_about_com_I_kgm2=result.inertia_about_com_I_kgm2,
            attachment_force_on_bus_I_N=result.attachment_force_on_bus_I_N)
        _csv(out/'history.csv',
            ['time_s','kinetic_J','elastic_J','gravity_potential_J','mechanical_energy_J',
             'attachment_force_x_N','attachment_force_y_N','attachment_force_z_N',
             'com_x_m','com_y_m','com_z_m', 'max_tension_N'],
            [[t,k,e,g,total,*fr,*com,max(tau)] for t,k,e,g,total,fr,com,tau in zip(
                result.t_s,result.kinetic_energy_J,result.elastic_energy_J,
                result.gravity_potential_J,result.total_mechanical_energy_J,
                result.attachment_force_on_bus_I_N,result.com_I_m,result.segment_tensions_N)])
        _csv(out/'segments.csv',
            ['time_s','segment_index','actual_length_m','rest_length_m','tension_N'],
            [[t,j,l,model.rest_length_m,tau] for t,lengths,forces in zip(
                result.t_s,result.actual_segment_lengths_m,result.segment_tensions_N)
                for j,(l,tau) in enumerate(zip(lengths,forces))])
        _csv(out/'nodes.csv',
            ['time_s','node_index','mass_kg','r_x_m','r_y_m','r_z_m','v_x_m_s','v_y_m_s','v_z_m_s'],
            [[t,j,result.node_masses_kg[j],*r,*v] for t,rs,vs in zip(
                result.t_s,result.r_nodes_I_m,result.v_nodes_I_m_s)
                for j,(r,v) in enumerate(zip(rs,vs))])
        common.update(samples=len(ts), max_tension_N=float(result.segment_tensions_N.max()),
            final_tip_I_m=result.r_nodes_I_m[-1,-1].tolist(),
            final_force_on_bus_I_N=result.attachment_force_on_bus_I_N[-1].tolist(),
            energy_conservation_applicable=result.solver_diagnostics['energy_conservation_applicable'],
            solver=result.solver_diagnostics,
            saved_files=['history.npz','history.csv','segments.csv','nodes.csv'])
    else:
        raise ValueError("mode must be 'transient' or 'quasi_static'")
    (out/'summary.json').write_text(json.dumps(common,indent=2,allow_nan=False)+'\n')
    return common


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='M5 standalone physical tether reference, no current')
    parser.add_argument('config')
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.config,args.output_dir),indent=2))
