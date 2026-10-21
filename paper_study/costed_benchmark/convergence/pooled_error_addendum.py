"""Clarify pooled quadratic loss versus mean episode RMSE; no model reruns."""
from pathlib import Path
import csv,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path.insert(0,str(HERE.parent))
import evaluate
NAMES=('position_m','velocity_m_s','normal_N','friction_N')


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    particle=REPO/'paper_study/results/costed_pi_convergence_v1/analysis/episode_metrics.csv'
    mle=REPO/'paper_study/results/costed_pi_mle_v1/episodes.csv'
    two=REPO/'paper_study/results/costed_pi_mle_resolution_v1/episodes.csv'
    peers,base,double=read(particle),read(mle),read(two)
    result=[]
    for scope,indices in [('full432',None),('fixed72',{0,7,14,21,28,35})]:
        groups={}
        for row in peers:
            if indices is None or int(row['episode']) in indices:
                groups.setdefault(f"P{row['particles']}/{row['estimator']}",[]).append(row)
        groups['bounded_mle_start1']=[row for row in base if indices is None or int(row['episode']) in indices]
        if indices is not None:groups['bounded_mle_two_start']=double
        for method,rows in groups.items():
            expected=432 if indices is None else 72
            if len(rows)!=expected:raise ValueError('Comparison scope mismatch')
            entry=dict(scope=scope,method=method,episodes=len(rows))
            for name in NAMES:
                values=np.array([float(row['rmse_'+name]) for row in rows])
                entry['pooled_mse_'+name]=float(np.mean(values**2))
                entry['pooled_rmse_'+name]=float(np.sqrt(np.mean(values**2)))
                entry['mean_episode_rmse_'+name]=float(np.mean(values))
            result.append(entry)
    out=REPO/'paper_study/results/costed_pi_pooled_error_addendum_v1'
    if out.exists():raise FileExistsError('Retain existing addendum')
    out.mkdir()
    with (out/'comparison.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
    evaluate.write_json(out/'summary.json',dict(role='Explicit quadratic-loss addendum; original clearly named mean-RMSE reports retained',
        definition='All episodes have160 future steps. Pooled MSE=mean_i(RMSE_i squared); pooled RMSE=sqrt(pooled MSE). Mean episode RMSE is different.',
        source_sha256={str(p):evaluate.sha(p) for p in (particle,mle,two,Path(__file__).resolve())},comparisons=result),exclusive=True)
    lines=['# Pooled quadratic-error addendum','',
        'The primary error comparison is pooled MSE and its square root. All episodes have the same 160-step future, so pooled MSE equals the average of squared episode RMSEs. Averaging episode RMSE directly gives a different descriptive quantity. Earlier report columns explicitly labeled mean RMSE retain that meaning; this addendum does not overwrite them.','',
        '| Scope | Method | Pooled x RMSE (mm) | Mean episode x RMSE (mm) | Pooled Ft RMSE (N) |','|---|---|---:|---:|---:|']
    for r in result:lines.append(f"| {r['scope']} | {r['method']} | {r['pooled_rmse_position_m']*1000:.5f} | {r['mean_episode_rmse_position_m']*1000:.5f} | {r['pooled_rmse_friction_N']:.6f} |")
    lines+=['','Only compare methods within the same scope. The72-case scope uses precisely indices0,7,14,21,28,35 per batch. No model is rerun, no outcome is excluded, and no independent-replicate confidence interval is inferred. Particle spread/ESS does not establish calibration, nor does a pooled variance ratio justify a universal under- or overdispersion claim.','']
    (out/'README.md').write_text('\n'.join(lines))
    evaluate.write_json(out/'artifact_manifest.json',dict(outputs_sha256={str(p):evaluate.sha(p) for p in out.iterdir() if p.is_file()}),exclusive=True)
    for r in result:
        if r['method'] in ('bounded_mle_start1','bounded_mle_two_start','P4096/trajectory_assimilation'):
            print(r['scope'],r['method'],r['pooled_rmse_position_m'],r['pooled_rmse_friction_N'])


if __name__=='__main__':main()
