"""Plot M4 illustrative orbital-work and osculating-element comparisons.

python -m pip install -e '.[viz]'
python scripts/plot_m4_decay_results.py results/m4_500km
"""
import argparse
import csv
from pathlib import Path
import numpy as np


def read_history(path):
    with Path(path).open(newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows:
        raise ValueError('empty M4 output CSV')
    return {k:np.asarray([float(row[k]) for row in rows]) for k in rows[0]}


def make_plots(results_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(results_dir)
    histories={p.stem:read_history(p) for p in sorted(directory.glob('*.csv'))
               if p.stem in ('off','outward_always','inward_always','outward_braking_only')}
    if 'off' not in histories:
        raise ValueError('off.csv baseline missing from M4 results')
    off=histories['off']
    figs=[]
    fig,ax=plt.subplots(figsize=(9,5))
    for key,h in histories.items():
        ax.plot(h['t_s']/3600.,h['work_lorentz_J'],label=key)
    ax.set(xlabel='Elapsed time [h]',ylabel='Integrated Lorentz work [J]',
           title='Illustrative 500 km orbit: assumed-current electrodynamic work')
    ax.grid(alpha=.3);ax.legend(fontsize=8);fig.tight_layout()
    p=directory/'m4_lorentz_work.png';fig.savefig(p,dpi=150);plt.close(fig);figs.append(p)
    fig,ax=plt.subplots(figsize=(9,5))
    for key,h in histories.items():
        ax.plot(h['t_s']/3600.,(h['osculating_a_m']-off['osculating_a_m']),label=key)
    ax.set(xlabel='Elapsed time [h]',ylabel='Osculating semimajor axis difference vs off [m]',
           title='Illustrative 500 km orbit: difference from identical drag-only baseline')
    ax.grid(alpha=.3);ax.legend(fontsize=8);fig.tight_layout()
    p=directory/'m4_delta_a_relative_to_off.png';fig.savefig(p,dpi=150);plt.close(fig);figs.append(p)
    fig,ax=plt.subplots(figsize=(9,5))
    for key,h in histories.items():
        ax.plot(h['t_s']/3600.,h['F_lorentz_along_N'],label=key)
    ax.set(xlabel='Elapsed time [h]',ylabel='Along-track Lorentz force [N]',
           title='Illustrative 500 km orbit: signed force versus orbital motion')
    ax.grid(alpha=.3);ax.legend(fontsize=8);fig.tight_layout()
    p=directory/'m4_along_track_force.png';fig.savefig(p,dpi=150);plt.close(fig);figs.append(p)
    return figs


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_dir',type=Path)
    args=parser.parse_args()
    for output in make_plots(args.results_dir):print('Saved',output)
