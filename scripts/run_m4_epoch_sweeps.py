"""Sweep UTC epochs / SSO inclination / RAAN or LTAN/LTDN with REAL optional
IGRF14/WMM2025 and NRLMSIS2.0 providers.

Example:
  pip install -e '.[dev,dynamics,earth-models]'
  python scripts/run_m4_epoch_sweeps.py configs/scenarios/m4_epoch_sweep_example.json --output-dir results/m4_epochs

Reference mode (default): samples prescribed J2-precessing *reference* orbit
and integrates approximate along-track work (NOT free orbital decay).
--propagate: numerically integrates two-body+J2+drag+assumed Lorentz over the
same short horizon, very computationally expensive with high-fidelity models.
No live plasma circuit; all currents are prescribed assumptions.
"""
from __future__ import annotations
import argparse
import csv
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path
import numpy as np
from edtsim.environment import CircularReferenceOrbit
from edtsim.environment.orbit_sweeps import make_orbit_cases
from edtsim.environment.earth_models import GeomagneticMainField, MSIS2AtmosphericDensity
from edtsim.tether import RoundWire, RectangularTape, HomogeneousTether
from edtsim.electrodynamics import PrescribedConstantCurrent
from edtsim.orbit_dynamics import RadialTetherForces, propagate_orbit_decay
from edtsim.core.constants import R_EARTH_EQ


def _load_config(path):
    def unique(pairs):
        obj={}
        for key,val in pairs:
            if key in obj: raise ValueError('duplicate JSON key '+key)
            obj[key]=val
        return obj
    cfg=json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                   parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON constant')))
    expected={'id','provenance','orbit_sweep','environment','vehicle','tether','current','sampling'}
    if set(cfg)!=expected:
        raise ValueError('unexpected/missing keys: '+str(sorted(set(cfg)^expected)))
    if cfg['provenance']['status']!='illustrative':
        raise ValueError('reference sweep must be identified as illustrative')
    sampling=cfg['sampling']
    if set(sampling)!={'steps_per_orbit','segment_count','orbits'}:
        raise ValueError('invalid sampling keys')
    if type(sampling['steps_per_orbit']) is not int or not 4<=sampling['steps_per_orbit']<=720:
        raise ValueError('steps_per_orbit must be int [4,720]')
    if type(sampling['segment_count']) is not int or not 1<=sampling['segment_count']<=50:
        raise ValueError('segment_count must be int [1,50]')
    if type(sampling['orbits']) not in (int,float) or not 0<sampling['orbits']<=10:
        raise ValueError('orbits must be positive <=10')
    return cfg


def _tether(cfg):
    cs=cfg['cross_section']
    if cs['type']=='round':
        section=RoundWire(cs['diameter_m'])
    elif cs['type']=='tape':
        section=RectangularTape(cs['width_m'],cs['thickness_m'])
    else:
        raise ValueError('unknown cross_section type')
    return HomogeneousTether(section,cfg['length_m'],cfg['density_kg_m3'],
       cfg['young_modulus_Pa'],cfg['conductivity_S_m'],cfg['bare_fraction'])


def _build_providers(case, environment):
    field=GeomagneticMainField(case.epoch_utc,environment['magnetic_model'])
    density=MSIS2AtmosphericDensity(case.epoch_utc,environment['f107'],
                                    environment['f107a'],environment['ap'])
    return field,density


def run_sweeps(config_path, output_dir, *, propagate=False,
               solar_ra_provider=None, provider_factory=None):
    cfg=_load_config(config_path)
    kwargs={} if solar_ra_provider is None else {'solar_ra_provider':solar_ra_provider}
    cases=make_orbit_cases(cfg['orbit_sweep'],**kwargs)
    env_cfg=cfg['environment']
    activities=env_cfg.get('solar_activity_cases')
    if not isinstance(activities,list) or not activities:
        raise ValueError('environment.solar_activity_cases must be a nonempty list')
    for activity in activities:
        if set(activity)!={'label','f107','f107a','ap'}:
            raise ValueError('activity case must have label, f107, f107a, ap')
    variants=list(product(cases,activities))
    if propagate and len(variants)>4:
        raise ValueError('--propagate limited to <=4 cases due to model evaluation cost')
    tether=_tether(cfg['tether'])
    current_value=float(cfg['current']['assumed_uniform_A'])
    if not np.isfinite(current_value):
        raise ValueError('assumed current must be finite')
    dry_mass=float(cfg['vehicle']['dry_mass_kg'])
    if not np.isfinite(dry_mass) or dry_mass<=0:
        raise ValueError('invalid dry mass')
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,(case,activity) in enumerate(variants):
        env_case={**env_cfg, **activity}
        field,density=(_build_providers(case,env_case) if provider_factory is None
                      else provider_factory(case,env_case))
        orbit=CircularReferenceOrbit(case.altitude_m,case.inclination_deg,
            raan_deg=case.raan_deg,precess_raan=not propagate)
        samples=cfg['sampling']['steps_per_orbit']
        orbit_count=cfg['sampling']['orbits']
        t_eval=np.linspace(0.,orbit.period_s*orbit_count,int(np.ceil(samples*orbit_count))+1)
        forces=RadialTetherForces(field,density,PrescribedConstantCurrent(current_value),
            tether,dry_mass,bus_Cd=cfg['vehicle']['bus_Cd'],
            bus_projected_area_m2=cfg['vehicle']['bus_projected_area_m2'],
            tether_Cd=cfg['vehicle']['tether_Cd'],segments=cfg['sampling']['segment_count'])
        if propagate:
            r0,v0=orbit.state(0)
            history=propagate_orbit_decay(r0,v0,t_eval,forces,max_step_s=120)
            power=np.einsum('ij,ij->i',history.lorentz_force_I_N, history.v_I_m_s)
            work=float(history.work_lorentz_J[-1])
            dragwork=float(history.work_drag_J[-1])
            delta_a=float(history.osculating_semimajor_axis_m[-1]-history.osculating_semimajor_axis_m[0])
            positions=history.r_I_m
            densities=np.array([density.sample(t,r) for t,r in zip(t_eval,positions)])
            magnetic=np.array([field.sample(t,r) for t,r in zip(t_eval,positions)])
        else:
            rs,vs=np.array([orbit.state(t) for t in t_eval]).transpose((1,0,2))
            FL=[];FD=[]
            for t,r,v in zip(t_eval,rs,vs):
                fl,fd,_=forces.forces_I(t,r,v)
                FL.append(fl);FD.append(fd)
            FL,FD=np.asarray(FL),np.asarray(FD)
            power=np.einsum('ij,ij->i',FL,vs)
            integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            work = float(integrate(power, t_eval))
            dragwork = float(integrate(np.einsum('ij,ij->i', FD, vs), t_eval))
            delta_a=None  # prescribed reference orbit has no physical decay
            densities=np.array([density.sample(t,r) for t,r in zip(t_eval,rs)])
            magnetic=np.array([field.sample(t,r) for t,r in zip(t_eval,rs)])
        result={**asdict(case), 'case_id':f'epoch_{i:04d}',
                'field_model':env_cfg['magnetic_model'],
                'activity_case':activity['label'],
                'f107':activity['f107'], 'f107a':activity['f107a'], 'ap':activity['ap'],
                'atmosphere_model':'NRLMSIS2.0',
                'assumed_current_A':current_value,
                'is_integrated_decay':bool(propagate),
                'mean_field_magnitude_T':float(np.mean(np.linalg.norm(magnetic,axis=1))),
                'min_density_kg_m3':float(np.min(densities)),
                'max_density_kg_m3':float(np.max(densities)),
                'lorentz_work_J':work,'drag_work_J':dragwork,
                'delta_osculating_a_m':delta_a,
                'negative_lorentz_work':bool(work<0),
                'model_status':'preliminary; prescribed-current and fixed radial tether; not circuit closure'}
        rows.append(result)
    (out/'summary.json').write_text(json.dumps({'scenario':cfg['id'],'cases':rows},indent=2,allow_nan=False)+'\n')
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path('results/m4_epochs'))
    p.add_argument('--propagate',action='store_true',help='integrate translation; max four cases')
    a=p.parse_args()
    rows=run_sweeps(a.config,a.output_dir,propagate=a.propagate)
    print(f'{len(rows)} cases written to {a.output_dir}. All results are illustrative until electrical closure.')
