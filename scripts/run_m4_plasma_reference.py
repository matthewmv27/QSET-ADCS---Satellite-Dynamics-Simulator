"""Generate PRELIMINARY 1-D OML collection/cathode capacity comparison.

python scripts/run_m4_plasma_reference.py configs/scenarios/m4_plasma_example.json --output-dir results/m4_plasma
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from edtsim.electrodynamics import EmpiricalCNTCathode
from edtsim.electrodynamics.oml_collection import PrescribedBusPotentialOML
from edtsim.tether import HomogeneousTether, RoundWire, RectangularTape


def run(config_path, output_dir):
    config=json.loads(Path(config_path).read_text())
    if config.get('provenance',{}).get('status')!='illustrative':
        raise ValueError('plasma study must label all parameters illustrative')
    t=config['tether']
    shape=t['cross_section']
    if shape['type']=='round':
        cs=RoundWire(shape['diameter_m'])
    elif shape['type']=='tape':
        cs=RectangularTape(shape['width_m'],shape['thickness_m'])
    else:
        raise ValueError('cross section must be round or tape')
    tether=HomogeneousTether(cs,t['length_m'],t['density_kg_m3'],t['young_modulus_Pa'],
                              t['conductivity_S_m'],t['bare_fraction'])
    e=config['plasma']
    solver=PrescribedBusPotentialOML(tether,e['electron_density_m3'],e['electron_temperature_K'],
                       config['electrical']['bus_plasma_bias_V'],
                       config['electrical']['motional_E_along_V_m'])
    solution=solver.solve(nodes=int(config['numerical']['nodes']))
    emitter=EmpiricalCNTCathode()
    voltage=emitter.required_voltage_V(solution.bus_emission_current_required_A,
                                      config['cathode']['equivalent_age_h'])
    maximum=config['cathode']['available_gate_voltage_V']
    result={'warning':'PRELIMINARY: assigned bus-to-plasma bias; no self-consistent CNT/plasma circuit closure',
      'current_at_bus_A':solution.bus_emission_current_required_A,
      'current_at_tip_residual_A':solution.tip_current_residual_A,
      'cathode_gate_voltage_required_V_empirical':voltage,
      'cathode_gate_voltage_limit_V':maximum,
      'gate_voltage_within_assumed_limit':bool(voltage<=maximum),
      'ohmic_loss_W':solution.ohmic_loss_W,
      'total_tether_resistance_ohm':tether.resistance_ohm,
      'bare_fraction':tether.bare_fraction}
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    with (out/'current_profile.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['s_m','conventional_current_A','tether_plasma_bias_V',
                                   'collected_electron_A_per_m'])
        for row in zip(solution.s_m,solution.conventional_current_A,
                       solution.tether_plasma_bias_V,solution.collected_electron_current_per_m_A_m):
            w.writerow([float(x) for x in row])
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path('results/m4_plasma'))
    args=p.parse_args()
    print(json.dumps(run(args.config,args.output_dir),indent=2))
