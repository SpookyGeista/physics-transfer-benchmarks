"""Descriptive post-fit figure; no checkpoint selection or parameter fitting."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lock import ROOT, digest, verify


def main():
    source_lock=verify()
    rows=list(csv.DictReader((ROOT/'results/summary/fits.csv').open()))
    cases=['balanced','wide_scales','compliant','stiff_high_capacity']
    labels=['Balanced','Wide scales','Compliant','Stiff / high capacity']
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.2),gridspec_kw={'width_ratios':[1.,1.25]},layout='constrained')
    for i,case in enumerate(cases):
        chosen=[r for r in rows if r['case']==case and r['condition']=='noisy']
        values=np.array([float(r['validation_rmse_clean_N'])*1000 for r in chosen])
        axes[0].scatter(np.full(3,i)+[-.10,0,.10],values,color='#1368a8',s=32)
        axes[0].plot([i-.2,i+.2],[values.mean()]*2,color='#222222',lw=2)
    axes[0].set_xticks(range(4),labels,rotation=20,ha='right')
    axes[0].set_ylabel('Validation force RMSE against clean target [mN]')
    axes[0].set_ylim(0,2.7)
    axes[0].set_title('Noisy calibration: all 12 fitted models')
    axes[0].grid(axis='y',alpha=.2)
    path=ROOT/'results/models/balanced/r61001/noisy/validation_predictions.npz'
    clean_path=ROOT/'results/data/balanced/r61001/clean/val.npz'
    with np.load(path) as data:
        predicted,measured=data['prediction'][0],data['measured'][0]
    with np.load(clean_path) as data:
        clean=data['Ft'][0];dt=float(data['dt'])
    time=np.arange(len(clean))*dt
    axes[1].plot(time,measured,color='.75',lw=.65,label='Noisy validation measurement')
    axes[1].plot(time,clean,color='#202020',lw=1.5,label='Clean reference')
    axes[1].plot(time,predicted,color='#d85c22',ls='--',lw=1.25,label='Fitted model')
    axes[1].set_xlabel('Time [s]');axes[1].set_ylabel('Mean tangential force [N]')
    axes[1].set_title('First validation profile: balanced, seed 61001')
    axes[1].legend(loc='best',frameon=False,fontsize=8)
    axes[1].grid(alpha=.2)
    fig.suptitle('Classical three-stop identification from local measurements',fontsize=13)
    out=ROOT/'results/figures';out.mkdir(exist_ok=True)
    for extension in ('png','pdf'):
        target=out/f'local_identification.{extension}'
        if target.exists():raise RuntimeError('refusing figure overwrite')
        fig.savefig(target,dpi=180)
    plt.close(fig)
    (out/'provenance.json').write_text(json.dumps(dict(design_lock_sha256=source_lock,
        plotting_source_sha256=digest(__file__), selection='all noisy fits; first fixed case/seed/profile only for illustration',
        inputs={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'results/summary/fits.csv',path,clean_path)},
        outputs={p.name:digest(p) for p in out.iterdir() if p.suffix in ('.png','.pdf')}),indent=2)+'\n')


if __name__=='__main__':main()
