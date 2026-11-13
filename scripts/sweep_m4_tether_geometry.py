"""Sweep calculated physical tether properties (no deployment dynamics/current closure).

python scripts/sweep_m4_tether_geometry.py configs/scenarios/m4_geometry_sweep.json --output results/m4_geometry_sweep.csv
"""
import argparse
import csv
import json
from pathlib import Path

from edtsim.tether import RoundWire, RectangularTape, HomogeneousTether


def geometry_rows(config):
    if config.get('provenance',{}).get('status')!='illustrative':
        raise ValueError('geometry sweep requires explicitly illustrative inputs')
    material=config['material']
    candidates=config['candidates']
    lengths=config['lengths_m']
    if not isinstance(candidates,list) or not candidates or not isinstance(lengths,list) or not lengths:
        raise ValueError('provide at least one candidate and one length')
    rows=[]
    ids=set()
    for candidate in candidates:
        name=candidate['id']
        if not isinstance(name,str) or not name or name in ids:
            raise ValueError('candidate IDs must be nonempty unique strings')
        ids.add(name)
        if candidate['type']=='round':
            cross=RoundWire(candidate['diameter_m'])
        elif candidate['type']=='tape':
            cross=RectangularTape(candidate['width_m'],candidate['thickness_m'])
        else:
            raise ValueError('unsupported cross section '+str(candidate['type']))
        for L in lengths:
            tether=HomogeneousTether(cross,L,material['density_kg_m3'],
                         material['young_modulus_Pa'],material['conductivity_S_m'],
                         material['bare_fraction'])
            rows.append({
                'design_id':name,'cross_section':candidate['type'],
                'length_m':tether.length_m,'material_area_m2':cross.area_m2,
                'geometric_perimeter_m':cross.perimeter_m,
                'bare_length_m':tether.bare_length_m,
                'linear_density_kg_m':tether.linear_density_kg_m,
                'tether_mass_kg':tether.mass_kg,
                'axial_rigidity_N':tether.axial_rigidity_N,
                'electrical_resistance_ohm':tether.resistance_ohm,
                'provenance_status':'illustrative'})
    return rows


def run_geometry_sweep(config_path,output):
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8'))
    rows=geometry_rows(cfg)
    path=Path(output)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('--output',type=Path,default=Path('results/m4_geometry_sweep.csv'))
    args=parser.parse_args()
    rows=run_geometry_sweep(args.config,args.output)
    print('Saved',len(rows),'illustrative geometry cases to',args.output)
