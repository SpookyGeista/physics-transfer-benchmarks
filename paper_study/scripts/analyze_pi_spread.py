"""Descriptive model-spread versus realized-error diagnostic on all fixed P cases.

No calibrated interval, population confidence, future-based tuning or success
criterion is introduced. All 432 predeclared development episodes at each P
are retained. A variance forecast is compared with its squared realized error.
"""
from pathlib import Path
import argparse
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/pi_spread_diagnostic')
    args=parser.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError('Retain previous diagnostics; choose fresh output')
    out.mkdir(parents=True);rows=[];hashes={str(Path(__file__).resolve()):sha(__file__)}
    for particles in (256,1024,4096):
        evaluation=ROOT/f'results/costed_pi_convergence_v1/p{particles}'
        if json.loads((evaluation/'process.json').read_text())['status']!='complete':raise ValueError('Incomplete evaluation')
        manifest=json.loads((evaluation/'artifact_manifest.json').read_text())
        files=sorted((evaluation/'predictions/noisy').rglob('*.npz'))
        if len(files)!=12:raise ValueError('Expected the complete fixed convergence subset')
        hashes[str(evaluation/'artifact_manifest.json')]=sha(evaluation/'artifact_manifest.json')
        for path in files:
            relative=path.relative_to(evaluation/'predictions/noisy')
            native=ROOT/'results/costed_pi_development'/relative
            native_meta=json.loads(native.with_suffix('.json').read_text())
            if sha(native)!=native_meta['npz_sha256']:raise ValueError('Native hash mismatch')
            expected=manifest['outputs_sha256'].get(str(path.resolve()))
            if expected is None or sha(path)!=expected:raise ValueError('Prediction hash mismatch')
            with np.load(native,allow_pickle=False) as truth:
                end=int(truth['probe_steps']);position=truth['next_state'][:,end:,0]
                force=truth['friction'][:,end:]
            with np.load(path,allow_pickle=False) as forecast:
                prediction=forecast['filtered_particles_trajectory_assimilation_forecast']
                variance=forecast['filtered_particles_trajectory_assimilation_variance']
                ess=forecast['weight_trace_ess'][:,-1]
                maxweight=forecast['weight_trace_max_weight'][:,-1]
            for field,index,target in [('position',0,position),('friction',3,force)]:
                mse=np.mean((prediction[...,index]-target)**2,axis=1)
                average_variance=variance[...,index].mean(axis=1)
                if np.any(average_variance<0):raise ValueError('Negative saved variance')
                for episode in range(len(mse)):
                    rows.append(dict(particles=particles,batch=str(relative),episode=episode,field=field,
                        mse=float(mse[episode]),predicted_variance=float(average_variance[episode]),
                        ess=float(ess[episode]),maximum_weight=float(maxweight[episode])))
            hashes.update({str(p.resolve()):sha(p) for p in (native,native.with_suffix('.json'),path)})
    frame=pd.DataFrame(rows);frame.to_csv(out/'episode_spread.csv',index=False)
    summaries=[]
    for (particles,field),group in frame.groupby(['particles','field']):
        summaries.append(dict(particles=int(particles),field=field,episodes=len(group),
            pooled_rms_error=float(np.sqrt(group.mse.mean())),
            rms_predicted_spread=float(np.sqrt(group.predicted_variance.mean())),
            ratio_pooled_mse_to_mean_variance=float(group.mse.mean()/group.predicted_variance.mean()) if group.predicted_variance.mean()>0 else None,
            positive_error_with_exact_zero_variance=int(((group.mse>0)&(group.predicted_variance==0)).sum()),
            median_ess=float(group.ess.median())))
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.8),layout='constrained')
    for ax,field,scale,unit in zip(axes,('position','friction'),(1e6,1.),('µm','N')):
        for particles,color in zip((256,1024,4096),('#b68a40','#3778ad','#007f86')):
            d=frame[(frame.particles==particles)&(frame.field==field)]
            x=np.sqrt(d.predicted_variance)*scale;y=np.sqrt(d.mse)*scale
            # Zero-spread points remain in the table and are explicitly omitted
            # only from logarithmic axes; no epsilon changes the numerical data.
            keep=(x>0)&(y>0)
            ax.scatter(x[keep],y[keep],s=8,alpha=.42,color=color,label=f'P={particles}')
        ax.set_xscale('log');ax.set_yscale('log')
        ax.set_xlabel(f'Predicted RMS spread [{unit}]');ax.set_ylabel(f'Realized episode RMSE [{unit}]')
        lo=min(ax.get_xlim()[0],ax.get_ylim()[0]);hi=max(ax.get_xlim()[1],ax.get_ylim()[1])
        ax.plot([lo,hi],[lo,hi],ls='--',color='#888',lw=.8)
        ax.grid(alpha=.13);ax.set_title(field.capitalize(),weight='bold')
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Fixed-model spread is not an error guarantee',weight='bold',fontsize=12)
    fig.supxlabel('432 development episodes per particle count; one fitted bank and one history block.\n'
                  'Dashed line indicates equal scales, not a required per-episode calibration identity.',fontsize=8)
    for suffix in ('png','pdf'):fig.savefig(out/f'spread_vs_error.{suffix}',dpi=220)
    plt.close(fig)
    if any(sha(p)!=h for p,h in hashes.items()):raise ValueError('Input changed during analysis')
    (out/'provenance.json').write_text(json.dumps(dict(sources_sha256=hashes,
        interpretation='Descriptive moments only; no Gaussian credible intervals or population significance test',
        summaries=summaries),indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
