"""M4 reduced-order 500-km study. Run from the repository root.

python scripts/run_m4_decay_reference.py configs/scenarios/m4_500km_assumed_current.json --output-dir results/m4_500km

No realistic ionospheric current closure, epoch, WMM, MSIS, active controller,
or tether dynamics are implied. This is a reproducible sensitivity/reference run.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from edtsim.core.constants import R_EARTH_EQ
from edtsim.environment import (CircularReferenceOrbit, sso_inclination_deg,
                                 CenteredDipoleMagneticField,
                                 ConstantAtmosphericDensity, ExponentialAtmosphericDensity)
from edtsim.tether import RoundWire, RectangularTape, HomogeneousTether
from edtsim.electrodynamics import ZeroCurrent, PrescribedConstantCurrent, EmpiricalCNTCathode
from edtsim.orbit_dynamics import RadialTetherForces, propagate_orbit_decay


ALLOWED = {'off', 'outward_always', 'inward_always', 'outward_braking_only'}


def _number(value, name, positive=False, nonnegative=False):
    if type(value) not in (float,int) or not np.isfinite(value):
        raise ValueError(name+' must be finite JSON number')
    f = float(value)
    if positive and f <= 0 or nonnegative and f < 0:
        raise ValueError(name+' must be positive' if positive else name+' must be nonnegative')
    return f


def _read_config(path):
    def no_duplicates(pairs):
        result = {}
        for key,value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key: '+key)
            result[key]=value
        return result
    cfg = json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=no_duplicates,
                     parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON constant '+x)))
    required = {'id','provenance','orbit','vehicle','tether','magnetic_field','atmosphere','assumed_current','integration','runs','cathode_estimate'}
    if set(cfg)!=required:
        raise ValueError('scenario keys missing/unknown: '+str(sorted(set(cfg)^required)))
    if cfg['provenance']['status']!='illustrative':
        raise ValueError('this example runner only accepts explicitly illustrative scenario status')
    if not isinstance(cfg['runs'],list) or not cfg['runs'] or len(set(cfg['runs']))!=len(cfg['runs']) or set(cfg['runs'])-ALLOWED:
        raise ValueError('runs must be unique permitted mode labels')
    return cfg


def run_scenario(config_path, output_dir):
    cfg = _read_config(config_path)
    orbit_cfg = cfg['orbit']
    altitude = _number(orbit_cfg['initial_altitude_m'],'initial_altitude_m',positive=True)
    if not 400e3<=altitude<=500e3:
        raise ValueError('reference mission scenario altitude must be 400–500 km')
    if orbit_cfg['inclination'] != 'first_order_sso':
        raise ValueError('reference scenario only supports first_order_sso inclination')
    inc = sso_inclination_deg(altitude)
    # Initial osculating TWO-BODY state; J2 is integrated thereafter.
    orbit = CircularReferenceOrbit(altitude,inc,raan_deg=_number(orbit_cfg['raan_deg'],'raan_deg'),
                                   argument_latitude_deg=_number(orbit_cfg['argument_latitude_deg'],'argument_latitude_deg'),
                                   precess_raan=False)
    r0,v0 = orbit.state(0)
    material = cfg['tether']
    shape = material['cross_section']
    if shape['type']=='round':
        cross = RoundWire(_number(shape['diameter_m'],'diameter_m',positive=True))
    elif shape['type']=='tape':
        cross = RectangularTape(_number(shape['width_m'],'width_m',positive=True),
                                _number(shape['thickness_m'],'thickness_m',positive=True))
    else:
        raise ValueError('cross_section.type must be round or tape')
    tether = HomogeneousTether(cross,_number(material['length_m'],'length_m',positive=True),
              _number(material['density_kg_m3'],'density_kg_m3',positive=True),
              _number(material['young_modulus_Pa'],'young_modulus_Pa',positive=True),
              _number(material['conductivity_S_m'],'conductivity_S_m',positive=True),
              _number(material['bare_fraction'],'bare_fraction',nonnegative=True))
    Bcfg = cfg['magnetic_field']
    if Bcfg['type'] != 'centered_dipole':
        raise ValueError('this example runner only supports a synthetic centered dipole')
    field = CenteredDipoleMagneticField(_number(Bcfg['equatorial_field_T'],'equatorial_field_T',positive=True),
                                tuple(Bcfg['dipole_axis_I_at_t0']),
                                _number(Bcfg['spin_rad_s'],'spin_rad_s'))
    acfg = cfg['atmosphere']
    if acfg['type']=='constant':
        density = ConstantAtmosphericDensity(_number(acfg['density_kg_m3'],'density_kg_m3',nonnegative=True))
    elif acfg['type']=='exponential':
        density = ExponentialAtmosphericDensity(
            _number(acfg['reference_altitude_m'],'reference_altitude_m',positive=True),
            _number(acfg['reference_density_kg_m3'],'reference_density_kg_m3',positive=True),
            _number(acfg['scale_height_m'],'scale_height_m',positive=True),
            _number(acfg['min_altitude_m'],'min_altitude_m',positive=True),
            _number(acfg['max_altitude_m'],'max_altitude_m',positive=True))
    else:
        raise ValueError('atmosphere.type must be constant or exponential')
    vehicle = cfg['vehicle']
    dry_mass = _number(vehicle['dry_mass_kg'],'dry_mass_kg',positive=True)
    bus_area = _number(vehicle['bus_projected_area_m2'],'bus_projected_area_m2',nonnegative=True)
    bus_cd = _number(vehicle['bus_Cd'],'bus_Cd',nonnegative=True)
    tether_cd = _number(vehicle['tether_Cd'],'tether_Cd',nonnegative=True)
    count = cfg['integration']['samples']
    segments = cfg['integration']['segments']
    if type(count) is not int or not 3<=count<=100000:
        raise ValueError('samples must be int in [3,100000]')
    if type(segments) is not int or not 1<=segments<=200:
        raise ValueError('segments must be int in [1,200]')
    orbits = _number(cfg['integration']['orbits'],'orbits',positive=True)
    if orbits>20:
        raise ValueError('restrict illustrative run to 20 orbits')
    t_eval = np.linspace(0,orbits*orbit.period_s,count)
    current = _number(cfg['assumed_current']['magnitude_A'],'assumed_current.magnitude_A',nonnegative=True)
    cath = EmpiricalCNTCathode()
    voltage = _number(cfg['cathode_estimate']['bias_voltage_V'],'bias_voltage_V',positive=True)
    age = _number(cfg['cathode_estimate']['equivalent_age_h'],'equivalent_age_h',nonnegative=True)
    cath_capacity = cath.emitted_current_A(voltage,age)
    out = Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    base_meta = {
       'scenario_id':cfg['id'],'inputs':cfg,'inclination_deg':inc,
       'initial_altitude_m':altitude,'period_s':orbit.period_s,
       'tether_mass_kg':tether.mass_kg,'total_assumed_system_mass_kg':dry_mass+tether.mass_kg,
       'tether_linear_density_kg_m':tether.linear_density_kg_m,
       'tether_area_m2':cross.area_m2,'tether_perimeter_m':cross.perimeter_m,
       'tether_axial_rigidity_N':tether.axial_rigidity_N,'tether_resistance_ohm':tether.resistance_ohm,
       'cathode_emission_capacity_A_at_assumed_bias':cath_capacity,
       'cathode_empirical_support_warnings':cath.support_warnings(voltage,age),
       'assumed_current_exceeds_empirical_cathode_capacity':current>cath_capacity,
       'warnings':[
           'ALL input numbers are illustrative; this is not a QSET flight prediction.',
           'The cathode model estimates emission capacity ONLY; no tether collection/circuit closure is solved.',
           'Prescribed currents, even below CNT capacity, have NOT been established achievable.',
           'Straight radial tether shape and tip orientation are externally imposed; no flexible dynamics.',
           'Centered dipole is NOT WMM2025 and synthetic atmosphere is NOT NRLMSIS.',
           'The orbit has a chosen orientation in an ideal inertial frame; no epoch/LTAN is determined.',
           'J2 potential energy minus integrated external work is the integration diagnostic.',
           'Osculating two-body elements vary within each orbit due to J2.'
       ]
    }
    summaries=[]
    for mode in cfg['runs']:
        if mode=='off': profile=ZeroCurrent()
        else: profile=PrescribedConstantCurrent(-current if mode=='inward_always' else current)
        forces=RadialTetherForces(field,density,profile,tether,dry_mass,
                                 bus_Cd=bus_cd,bus_projected_area_m2=bus_area,
                                 tether_Cd=tether_cd,segments=segments,
                                 braking_only=(mode=='outward_braking_only'))
        history=propagate_orbit_decay(r0,v0,t_eval,forces,max_step_s=60)
        delta_mech = history.mechanical_energy_J-history.mechanical_energy_J[0]
        residual = delta_mech-history.work_drag_J-history.work_lorentz_J
        summary={
           'mode':mode,
           'initial_altitude_m':float(np.linalg.norm(history.r_I_m[0])-R_EARTH_EQ),
           'final_radius_altitude_m':float(np.linalg.norm(history.r_I_m[-1])-R_EARTH_EQ),
           'final_osculating_semimajor_axis_m':float(history.osculating_semimajor_axis_m[-1]),
           'delta_mechanical_energy_J':float(delta_mech[-1]),
           'lorentz_work_J':float(history.work_lorentz_J[-1]),
           'drag_work_J':float(history.work_drag_J[-1]),
           'max_abs_energy_work_residual_J':float(np.max(np.abs(residual))),
           'min_along_track_lorentz_N':float(np.min(history.along_track_lorentz_N)),
           'max_along_track_lorentz_N':float(np.max(history.along_track_lorentz_N)),
           'fraction_of_samples_with_braking_force':float(np.mean(history.along_track_lorentz_N < -1e-16)),
           'samples':len(history.t_s),
        }
        summaries.append(summary)
        path=out/(mode+'.csv')
        with path.open('w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['t_s','radius_altitude_m','osculating_perigee_m','osculating_a_m',
                             'r_x_m','r_y_m','r_z_m','v_x_m_s','v_y_m_s','v_z_m_s',
                             'F_lorentz_x_N','F_lorentz_y_N','F_lorentz_z_N',
                             'F_drag_x_N','F_drag_y_N','F_drag_z_N','F_lorentz_along_N',
                             'work_lorentz_J','work_drag_J','mechanical_energy_J',
                             'energy_work_residual_J','mean_segment_current_A'])
            for i,t in enumerate(history.t_s):
                writer.writerow([float(t),float(np.linalg.norm(history.r_I_m[i])-R_EARTH_EQ),
                    float(history.perigee_altitude_m[i]),float(history.osculating_semimajor_axis_m[i]),
                    *history.r_I_m[i].tolist(),*history.v_I_m_s[i].tolist(),
                    *history.lorentz_force_I_N[i].tolist(),*history.drag_force_I_N[i].tolist(),
                    float(history.along_track_lorentz_N[i]),float(history.work_lorentz_J[i]),
                    float(history.work_drag_J[i]),float(history.mechanical_energy_J[i]),
                    float(residual[i]),float(np.mean(history.segment_currents_A[i]))])
    final={'metadata':base_meta,'cases':summaries}
    (out/'summary.json').write_text(json.dumps(final,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return final


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('configuration',type=Path)
    ap.add_argument('--output-dir',type=Path,default=Path('results/m4_500km'))
    args=ap.parse_args()
    result=run_scenario(args.configuration,args.output_dir)
    print('Illustrative scenario:',result['metadata']['scenario_id'])
    print('Cathode-capacity estimate A:',result['metadata']['cathode_emission_capacity_A_at_assumed_bias'])
    for case in result['cases']:
        print('{:<23s} Lorentz work = {:+.6g} J; drag work = {:+.6g} J; min F_along = {:+.6g} N'.format(
            case['mode'],case['lorentz_work_J'],case['drag_work_J'],case['min_along_track_lorentz_N']))
    print('CSV and summary.json saved to',args.output_dir)
