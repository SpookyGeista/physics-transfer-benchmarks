"""Readable display of the unchanged PI spread diagnostic, including zero spread.

The x axis is linear near zero and logarithmic farther out. No artificial
epsilon is added to variances, no cases are discarded, and no data are changed.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source=ROOT/'results/pi_spread_diagnostic/episode_spread.csv'
    frame=pd.read_csv(source);out=ROOT/'reports/figures/pi_spread'
    if out.exists():raise FileExistsError('Existing display retained')
    if not (frame.groupby(['particles','field']).size()==432).all():raise ValueError('Incomplete cases')
    out.mkdir(parents=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.9),layout='constrained')
    for ax,field,scale,unit,linear in zip(axes,('position','friction'),(1e6,1.),('µm','N'),(1.,.0001)):
        selected=frame[frame.field==field];allx=np.sqrt(selected.predicted_variance)*scale
        ally=np.sqrt(selected.mse)*scale
        for p,color in zip((256,1024,4096),('#b68a40','#3778ad','#007f86')):
            d=selected[selected.particles==p];x=np.sqrt(d.predicted_variance)*scale;y=np.sqrt(d.mse)*scale
            ax.scatter(x,y,s=10,alpha=.4,color=color,label=f'P={p}',linewidths=0)
        ax.set_xscale('symlog',linthresh=linear,linscale=.6);ax.set_yscale('log')
        ax.set_xlim(0,max(allx.max(),linear*10)*1.25)
        ax.set_ylim(ally[ally>0].min()/1.5,ally.max()*1.4)
        lower=max(ax.get_ylim()[0],linear*.01);upper=min(ax.get_xlim()[1],ax.get_ylim()[1])
        axis=np.geomspace(lower,upper,100);ax.plot(axis,axis,'--',color='#777',lw=1)
        ax.set_xlabel(f'Predicted RMS spread [{unit}]');ax.set_ylabel(f'Realized episode RMSE [{unit}]')
        ax.set_title(field.capitalize(),weight='bold');ax.grid(alpha=.13)
    axes[0].legend(frameon=False,fontsize=8,loc='upper left')
    fig.suptitle('Predicted spread and realized error differ',weight='bold',fontsize=12)
    fig.supxlabel('All 432 development cases per P, including zero predicted variance.\n'
                  'Horizontal axes are linear near zero; dashed curve indicates equal scales.',fontsize=8)
    for suffix in ('png','pdf'):fig.savefig(out/f'spread_vs_error.{suffix}',dpi=220)
    plt.close(fig)
    (out/'provenance.json').write_text(json.dumps(dict(sources_sha256={str(source):sha(source),
        str(Path(__file__).resolve()):sha(__file__)},display_only=True,all_episodes_retained=True,
        x_axis_linear_thresholds=dict(position_um=1.,friction_N=.0001)),indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
