"""Plot LVLH histories from M5 orbit-driven mechanical tether demonstration.

Usage:
  python scripts/plot_m5_orbit_driven.py results/m5_orbit
"""
import argparse
from pathlib import Path
import numpy as np


def plot_results(directory):
    import matplotlib.pyplot as plt

    out=Path(directory)
    cases=('straight_gravity','tilted_gravity','straight_lorentz')
    records={}
    for case in cases:
        with np.load(out/case/'history.npz') as result:
            records[case]={key:result[key] for key in result.files}
    names={'straight_gravity':'Radial, zero current',
           'tilted_gravity':'Small initial tilt, zero current',
           'straight_lorentz':'Radial, synthetic EDT current'}
    t=records[cases[0]]['t_s']

    fig,ax=plt.subplots(figsize=(8,4.5))
    for case in ('straight_gravity','straight_lorentz'):
        x=records[case]['positions_LVLH_m']
        ax.plot(t, 1000*x[:,-1,1],label=names[case])
    ax.set(xlabel='Time (s)',ylabel='Tip along-track displacement (mm)',
           title='Initially straight 50 m tether: gravity-gradient vs EDT load')
    ax.grid(True,alpha=.3);ax.legend();fig.tight_layout()
    fig.savefig(out/'tip_alongtrack.png',dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.5))
    for case in cases:
        ax.plot(t,records[case]['tip_libration_deg'],label=names[case])
    ax.set(xlabel='Time (s)',ylabel='Tip angle relative to radial (degrees)',
           title='Tether-tip libration in prescribed LVLH frame')
    ax.grid(True,alpha=.3);ax.legend();fig.tight_layout()
    fig.savefig(out/'tip_libration.png',dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.5))
    for case in cases:
        tens=records[case]['segment_tensions_N']
        ax.plot(t, 1e3*tens.max(axis=1),label=names[case])
    ax.set(xlabel='Time (s)',ylabel='Maximum tension (mN)',
           title='Tether tension under prescribed orbital motion')
    ax.grid(True,alpha=.3);ax.legend();fig.tight_layout()
    fig.savefig(out/'maximum_tension.png',dpi=170);plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(11,4.2),sharex=True)
    for axis, case in zip(axs,('straight_gravity','straight_lorentz')):
        data=records[case]['positions_LVLH_m']
        for idx in (0,len(t)//2,len(t)-1):
            axis.plot(data[idx,:,0],1000*data[idx,:,1],label=f't={t[idx]:g}s')
        axis.set(xlabel='Radial distance from attachment (m)',
                 ylabel='Along-track offset (mm)',title=names[case])
        axis.grid(True,alpha=.3);axis.legend()
    fig.suptitle('Initially straight shape; deformations emerge during integration')
    fig.tight_layout();fig.savefig(out/'tether_shapes_LVLH.png',dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.5))
    for case in ('straight_gravity','straight_lorentz'):
        f=records[case]['attachment_force_LVLH_N'][:,1]
        ax.plot(t,1e6*f,label=names[case])
    ax.set(xlabel='Time (s)',ylabel='Attachment force along-track (μN)',
           title='Force on prescribed bus attachment (not bus attitude response)')
    ax.grid(True,alpha=.3);ax.legend();fig.tight_layout()
    fig.savefig(out/'attachment_alongtrack.png',dpi=170);plt.close(fig)
    return ['tip_alongtrack.png','tip_libration.png','maximum_tension.png',
            'tether_shapes_LVLH.png','attachment_alongtrack.png']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_directory')
    args=parser.parse_args()
    for file in plot_results(args.results_directory):
        print('Saved',Path(args.results_directory)/file)
