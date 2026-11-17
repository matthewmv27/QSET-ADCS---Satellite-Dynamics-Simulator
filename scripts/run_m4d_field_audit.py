"""M4d reference-orbit Lorentz sign/power audit; not an orbit-decay prediction.

Run: python scripts/run_m4d_field_audit.py configs/scenarios/m4d_field_audit.json \
         --output-dir results/m4d_field_audit
Use --model igrf14 or --model wmm2025 only after installing earth-models.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from edtsim.environment.orbit import CircularReferenceOrbit, sso_inclination_deg
from edtsim.environment.models import CenteredDipoleMagneticField
from edtsim.environment.earth_models import GeomagneticMainField
from edtsim.environment.flows import corotating_flow_I
from edtsim.environment.epoch_frames import utc_datetime
from edtsim.environment.m4d_audit import sample_orbit_audit


def run(config_path, output_dir, model_override=None):
    with open(config_path,encoding='utf-8') as f:
        config=json.load(f)
    epoch=str(config['epoch_utc'])
    utc_datetime(epoch)
    altitude=float(config['altitude_m'])
    if not np.isfinite(altitude) or altitude<=0:
        raise ValueError('altitude_m must be positive')
    inc=config['inclination_deg']
    if inc=='sso': inc=sso_inclination_deg(altitude)
    orbit=CircularReferenceOrbit(altitude,float(inc),
          float(config['raan_deg']),float(config.get('argument_latitude_deg',0)))
    model=model_override or config['field_model']
    if model=='centered_dipole':
        field=CenteredDipoleMagneticField()
        assumption='Synthetic centered dipole: NOT a real geomagnetic prediction'
    elif model in ('wmm2025','igrf14'):
        field=GeomagneticMainField(epoch,model)
        assumption=('Earth main-field model; excludes storm/disturbance field. '
                    'Reference orbit is idealized inertial Cartesian, assumed GCRS')
    else:
        raise ValueError('field_model must be centered_dipole, wmm2025 or igrf14')
    count=int(config.get('samples_per_orbit',81))
    periods=float(config.get('orbits',1.))
    if count<3 or count>100000 or not np.isfinite(periods) or periods<=0:
        raise ValueError('invalid samples_per_orbit or orbits')
    t=np.linspace(0,periods*orbit.period_s,count)
    flow=config.get('plasma_flow','stationary_inertial')
    if flow=='stationary_inertial':
        plasma=None
    elif flow=='corotation_approximation':
        plasma=lambda t,r:corotating_flow_I(r)
    else:
        raise ValueError('plasma_flow must be stationary_inertial or corotation_approximation')
    current=float(config['outward_current_A'])
    length=float(config['tether_length_m'])
    segments=int(config['segments'])
    rows=sample_orbit_audit(orbit,field,t,current,length,segments,plasma)
    def integrate(x):
        a=np.array(x,dtype=float)
        return float((np.trapezoid if hasattr(np,'trapezoid') else np.trapz)(a,t))
    summary={
        'milestone':'M4d', 'model':model, 'source_config':str(config_path),
        'epoch_utc':epoch, 'inclination_deg_resolved':orbit.inclination_deg,
        'raan_deg':orbit.raan_deg, 'altitude_m':altitude,
        'assumed_signed_current_A':current, 'tether_length_m':length,
        'plasma_flow':flow, 'samples':len(rows), 'span_s':float(t[-1]),
        'orbit_normal_field_projection_definition':'B dot (rhat x vhat)',
        'negative_along_track_samples':sum(x['along_track_force_N']<0 for x in rows),
        'positive_along_track_samples':sum(x['along_track_force_N']>0 for x in rows),
        'lorentz_work_at_bus_orbit_velocity_J':integrate([x['power_com_W'] for x in rows]),
        'distributed_mechanical_work_J':integrate([x['power_distributed_W'] for x in rows]),
        'motional_emf_integral_J':integrate([x['motional_emf_power_W'] for x in rows]),
        'plasma_motion_work_J':integrate([x['plasma_motion_power_W'] for x in rows]),
        'max_abs_identity_residual_W':max(abs(x['power_identity_residual_W']) for x in rows),
        'limitations':[assumption,
           'Prescribed orbit/geometry/current; no orbital propagation in this audit',
           'Positive conventional current is bus to tip and may not be achievable',
           'Algebraic force/EMF power identity does not close plasma/CNT circuit'],
    }
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'samples.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('results/m4d_field_audit'))
    parser.add_argument('--model',choices=['centered_dipole','wmm2025','igrf14'])
    args=parser.parse_args()
    summary=run(args.config,args.output_dir,args.model)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
