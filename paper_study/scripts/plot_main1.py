"""Plot the three locked Main1 contrasts without choosing a favorable subset."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'results/statistics_main1')
    p.add_argument('--out',type=Path,default=ROOT/'reports/figures/main1')
    args=p.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError('Choose a fresh figure directory')
    result=json.loads((args.data/'results.json').read_text())
    expected=['aging_assimilation_ratio','aging_minus_relaxing_relative_gain','aging_probe_vs_hold_ratio']
    if [c['contrast'] for c in result['contrasts']]!=expected:raise ValueError('Unexpected primary contrasts')
    out.mkdir(parents=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    labels=['Aging: assimilation / prior MSE',
            'Relative assimilation gain: aging minus relaxing',
            'Aging: active probe / equal-duration hold MSE']
    fig,axes=plt.subplots(3,1,figsize=(7.1,6.1),layout='constrained')
    for index,(ax,c,label) in enumerate(zip(axes,result['contrasts'],labels)):
        lo,hi,point=[c[k] for k in ('lower_99','upper_99','point')]
        threshold=0 if index==1 else .9
        xmin=min(lo,point,threshold,0);xmax=max(hi,point,threshold,1)
        pad=.08*max(xmax-xmin,.1);xmin-=pad;xmax+=pad
        ax.set_xlim(xmin,xmax);ax.set_ylim(-.55,.55);ax.set_yticks([])
        ax.axvspan(threshold if index==1 else xmin,xmax if index==1 else threshold,
                   color='#007f86',alpha=.08)
        ax.axvline(threshold,color='#777',ls='--',lw=1)
        color='#007f86' if c['criterion_passed'] else '#715244'
        ax.hlines(0,lo,hi,color=color,lw=2.3);ax.vlines([lo,hi],-.06,.06,color=color,lw=1.5)
        ax.scatter([point],[0],c=color,s=40,zorder=3)
        ax.set_title(f'{index+1}. {label}',loc='left',fontsize=10,weight='bold')
        ax.text(.99,.87,f'{point:.3f} [{lo:.3f}, {hi:.3f}]',transform=ax.transAxes,ha='right',fontsize=9)
        ax.set_xlabel('Difference of relative MSE reductions' if index==1 else 'Ratio of pooled MSEs')
        ax.grid(axis='x',alpha=.15)
    fig.suptitle('Held-out replication: all three predeclared contrasts',fontsize=13,weight='bold')
    fig.supxlabel('99% crossed-block percentile intervals; 8 model banks × 8 history blocks.\n'
                  'Shading marks the predeclared criterion region; approximate coverage.',fontsize=8.5)
    for suffix in ('png','pdf'):fig.savefig(out/f'primary_contrasts.{suffix}',dpi=220)
    plt.close(fig)
    components=result['components']
    labels=['Aging\nprobe + assimilation','Aging\nprobe + prior',
            'Relaxing\nprobe + assimilation','Relaxing\nprobe + prior','Aging\nhold + assimilation']
    values=np.array([c['rms_over_all_samples_mm'] for c in components])*1000
    fig,ax=plt.subplots(figsize=(7.1,3.7),layout='constrained')
    bars=ax.bar(np.arange(5),values,color=['#007f86','#c97d1b','#007f86','#c97d1b','#6b7e8d'],width=.68)
    ax.set_xticks(np.arange(5),labels,fontsize=8);ax.set_ylabel('Pooled future position RMSE [µm]')
    ax.set_ylim(0,max(values)*1.25);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    for b,value in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,value+max(values)*.025,f'{value:.1f}',ha='center',fontsize=9)
    ax.set_title('Absolute losses underlying the primary contrasts',weight='bold',fontsize=12)
    fig.supxlabel('150 ms preparation; noisy local calibration and x/v/N sensors; 250 ms forecast.\n'
                  'Square root of the pooled MSE, not the mean of episode RMSEs.',fontsize=8.5)
    for suffix in ('png','pdf'):fig.savefig(out/f'absolute_losses.{suffix}',dpi=220)
    plt.close(fig)
    sources={str(path.resolve()):sha(path) for path in [args.data/'results.json',args.data/'analysis_lock.json',Path(__file__)]}
    (out/'provenance.json').write_text(json.dumps(dict(sources_sha256=sources,
        output_sha256={str(path):sha(path) for path in out.iterdir() if path.is_file()}),indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
