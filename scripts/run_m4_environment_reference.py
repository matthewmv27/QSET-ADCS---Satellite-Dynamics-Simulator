"""Run illustrative M4A environment reference and export human-readable CSV.

Requires only NumPy. Explicitly rejects real WMM/atmosphere claims.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from edtsim.environment import (CircularReferenceOrbit, sso_inclination_deg,
                                UniformMagneticField, corotating_flow_I,
                                motional_electric_field_I)


def run(config_path, output_dir):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding='utf8'))
    if config.get('schema_version') != 'm4a-synthetic-v1':
        raise ValueError('Expected schema_version=m4a-synthetic-v1')
    orbit_cfg = config['orbit']
    altitude = float(orbit_cfg['altitude_m'])
    mode = orbit_cfg['inclination_mode']
    if mode != 'sso_j2_approx':
        raise ValueError('This starter runner only supports sso_j2_approx')
    if config['magnetic_field']['model'] != 'synthetic_uniform':
        raise ValueError('M4A runner supports only synthetic_uniform, not a real geomagnetic field')
    if config['flow']['model'] != 'assumed_earth_corotation':
        raise ValueError('M4A runner supports only assumed_earth_corotation')
    duration = float(config['output']['duration_s'])
    step = float(config['output']['step_s'])
    if not (np.isfinite(duration) and duration > 0 and np.isfinite(step) and step > 0):
        raise ValueError('duration_s and step_s must be positive and finite')
    n = round(duration/step)
    if not np.isclose(n * step, duration, atol=1e-8, rtol=0) or n > 100000:
        raise ValueError('Duration must be a whole number of steps, maximum 100000')
    incl = sso_inclination_deg(altitude)
    orbit = CircularReferenceOrbit(
        altitude_m=altitude, inclination_deg=incl,
        raan_deg=float(orbit_cfg['raan_deg']),
        argument_latitude_deg=float(orbit_cfg['argument_latitude_deg']),
        precess_raan=orbit_cfg['precess_raan'],
    )
    provider = UniformMagneticField(tuple(config['magnetic_field']['B_I_T']))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in np.linspace(0, duration, n + 1):
        r, v = orbit.state(float(t))
        B = provider.sample(float(t), r)
        plasma_v = corotating_flow_I(r)
        emf = motional_electric_field_I(v, plasma_v, B)
        rows.append({
            't_s': float(t),
            **{'r_'+axis+'_I_m':float(r[k]) for k,axis in enumerate('xyz')},
            **{'v_'+axis+'_I_m_s':float(v[k]) for k,axis in enumerate('xyz')},
            **{'B_'+axis+'_I_T':float(B[k]) for k,axis in enumerate('xyz')},
            **{'E_'+axis+'_I_V_m':float(emf[k]) for k,axis in enumerate('xyz')},
        })
    with (output_dir / 'environment_samples.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        'model_kind':'prescribed_secular_J2_reference_plus_synthetic_uniform_B',
        'warning':'Not a high-fidelity orbit, measured atmospheric/plasma flow, WMM result, or QSET flight prediction.',
        'input_config':config,
        'computed_inclination_deg':incl,
        'nominal_period_s':orbit.period_s,
        'nominal_j2_nodal_rate_deg_day':float(np.rad2deg(orbit.nodal_rate_rad_s)*86400),
        'samples':len(rows),
    }
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n',encoding='utf8')
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',help='M4A illustrative JSON scenario')
    p.add_argument('--output-dir', default='results/m4a_synthetic')
    args=p.parse_args()
    result=run(args.config,args.output_dir)
    print(f"Generated {result['samples']} synthetic samples; SSO inclination={result['computed_inclination_deg']:.4f} deg")
    print(f"Files saved in {args.output_dir}; results are NOT verified Earth-field/drag predictions")


if __name__ == '__main__':
    main()
