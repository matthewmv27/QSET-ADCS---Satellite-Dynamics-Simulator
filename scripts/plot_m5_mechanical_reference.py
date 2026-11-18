"""Save plots from an existing M5 history.npz (no re-integration)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def plot_history(result_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(result_dir)
    with np.load(out/'history.npz') as history:
        t = history['t_s']
        r = history['r_nodes_I_m']
        tension = history['segment_tensions_N']
        force = history['attachment_force_on_bus_I_N']
        energy = history['total_mechanical_energy_J']
        saved=[]
        fig,ax=plt.subplots()
        for i in sorted(set([0,len(t)//4,len(t)//2,3*len(t)//4,len(t)-1])):
            # x and y are inertial coordinates for this illustrative setup.
            ax.plot(r[i,:,0],r[i,:,1],label=f'{t[i]:.2f} s')
        ax.set(xlabel='Inertial X (m)',ylabel='Inertial Y (m)',
               title='M5 prescribed 50 m tether: sampled shapes')
        ax.legend();ax.grid(True,alpha=.3); fig.tight_layout()
        p=out/'tether_shapes.png'; fig.savefig(p,dpi=160); saved.append(str(p));plt.close(fig)
        fig,ax=plt.subplots()
        ax.plot(t,r[:,-1,1]-r[0,-1,1]);
        ax.set(xlabel='Time (s)',ylabel='Tip Y change (m)',title='Tip transverse displacement')
        ax.grid(True,alpha=.3);fig.tight_layout()
        p=out/'tip_transverse.png';fig.savefig(p,dpi=160);saved.append(str(p));plt.close(fig)
        fig,ax=plt.subplots()
        ax.plot(t,np.max(tension,axis=1));
        ax.set(xlabel='Time (s)',ylabel='Maximum segment tension (N)',title='Maximum cable tension')
        ax.grid(True,alpha=.3);fig.tight_layout()
        p=out/'maximum_tension.png';fig.savefig(p,dpi=160);saved.append(str(p));plt.close(fig)
        fig,ax=plt.subplots()
        for axis,name in enumerate(('X','Y','Z')):
            ax.plot(t,force[:,axis],label=name)
        ax.set(xlabel='Time (s)',ylabel='Force on bus (N)',title='Attachment reaction')
        ax.legend();ax.grid(True,alpha=.3);fig.tight_layout()
        p=out/'attachment_force.png';fig.savefig(p,dpi=160);saved.append(str(p));plt.close(fig)
        fig,ax=plt.subplots()
        ax.plot(t,energy)
        ax.set(xlabel='Time (s)',ylabel='Total mechanical energy (J)',
               title='Mechanical energy (damped illustrative case)')
        ax.grid(True,alpha=.3);fig.tight_layout()
        p=out/'mechanical_energy.png';fig.savefig(p,dpi=160);saved.append(str(p));plt.close(fig)
    return saved

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir')
    args=parser.parse_args()
    for path in plot_history(args.result_dir): print('Saved', path)
