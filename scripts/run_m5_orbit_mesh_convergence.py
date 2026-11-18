"""Study mesh dependence of M5 orbit-driven *prescribed attachment* runs.

Compares final LVLH shape and tip/attachment time histories to finest mesh;
this is numerical convergence evidence, not proof of mission accuracy.

Example:
  python scripts/run_m5_orbit_mesh_convergence.py \\
     configs/scenarios/m5_orbit_driven_500km.json \\
     --segments 25 50 100 200 --output-dir results/m5_orbit_mesh
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scripts.run_m5_orbit_driven_reference import run


def run_study(config_path, output_dir, *, segment_counts=(25, 50, 100, 200),
              cases=('straight_gravity','straight_lorentz')):
    if (len(segment_counts) < 2 or list(segment_counts) != sorted(set(segment_counts))
            or any(type(n) is not int or n < 1 for n in segment_counts)):
        raise ValueError('segment counts must be unique increasing positive integers')
    if len(cases) < 1:
        raise ValueError('at least one case required')
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    for n in segment_counts:
        run(config_path,output/f'mesh_{n}',cases=cases,segments_override=n)
    ref_n=segment_counts[-1]
    ref_u=np.linspace(0,1,ref_n+1)
    rows=[]
    for case in cases:
        with np.load(output/f'mesh_{ref_n}'/case/'history.npz') as reference:
            r_ref=reference['positions_LVLH_m'].copy()
            f_ref=reference['attachment_force_LVLH_N'].copy()
        for n in segment_counts:
            with np.load(output/f'mesh_{n}'/case/'history.npz') as data:
                r=data['positions_LVLH_m']
                f=data['attachment_force_LVLH_N']
                tensions=data['segment_tensions_N']
                masses=data['node_masses_kg']
                u=np.linspace(0,1,n+1)
                r_final=np.column_stack([np.interp(ref_u,u,r[-1,:,axis]) for axis in range(3)])
                delta=r_final-r_ref[-1]
                rows.append(dict(case=case,segments=n,nodes=n+1,
                    mass_kg=float(masses.sum()),
                    max_tension_N=float(np.max(tensions)),
                    final_shape_rms_difference_m_vs_finest=float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))),
                    tip_history_rms_difference_m_vs_finest=float(np.sqrt(np.mean(np.sum(
                        (r[:,-1]-r_ref[:,-1])**2,axis=1)))),
                    attachment_history_rms_difference_N_vs_finest=float(np.sqrt(np.mean(np.sum(
                        (f-f_ref)**2,axis=1))))))
    with (output/'mesh_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(milestone='M5',study='orbit_driven_mesh_refinement',
                 source_config=str(config_path),reference_segments=ref_n,
                 finest_mesh_is_not_analytical_truth=True,rows=rows,
                 limitations=['No rigid-bus response or orbit propagation',
                              'Synthetic B and prescribed current; not a flight prediction',
                              'Shape convergence does not imply tension or force convergence',
                              'One short reference trajectory is not a validated mission envelope'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--segments',nargs='+',type=int,default=[25,50,100,200])
    parser.add_argument('--cases',nargs='+',default=['straight_gravity','straight_lorentz'])
    args=parser.parse_args()
    print(json.dumps(run_study(args.config,args.output_dir,
              segment_counts=args.segments,cases=args.cases),indent=2))
