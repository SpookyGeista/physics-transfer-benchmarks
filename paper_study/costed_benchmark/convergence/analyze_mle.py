"""Descriptive paired MLE/particle comparison; no model or start selection."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_mle
import evaluate
CHANNELS=run_mle.CHANNELS


def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    lock,lock_sha=run_mle.verify(root);process=run_mle.read(root/'process.json')
    if process['status']!='complete' or process['episodes']!=432:raise ValueError('MLE run incomplete')
    manifest=run_mle.read(root/'artifact_manifest.json');evaluate.verify_hashes(manifest['outputs_sha256'])
    if manifest['lock_sha256']!=lock_sha:raise ValueError('Manifest lock mismatch')
    particle=Path(lock['source_particle_directory']);convergence=particle.parent
    prior_summary=run_mle.read(convergence/'analysis/summary.json')
    evaluate.verify_hashes(run_mle.read(convergence/'analysis/artifact_manifest.json')['outputs_sha256'])
    rows=read_csv(root/'episodes.csv');peers=read_csv(convergence/'analysis/episode_metrics.csv')
    if len(rows)!=432:raise ValueError('Incomplete MLE metric inventory')
    paired=[]
    key=lambda r:(r['normal_world'],r['friction_world'],r['probe'],r['episode_id'])
    mle={key(row):row for row in rows}
    if len(mle)!=432:raise ValueError('Nonunique MLE episodes')
    aggregate=[]
    for estimator,p in [('bounded_initial_state_mle',None),*[(e,str(p)) for p in (256,1024,4096) for e in
            ('prior_propagation','trajectory_assimilation','privileged_full_history')]]:
        group=rows if p is None else [row for row in peers if row['particles']==p and row['estimator']==estimator]
        if len(group)!=432:raise ValueError('Unpaired comparator inventory')
        row=dict(estimator=estimator,particles=p,episodes=len(group))
        for channel in CHANNELS:
            row[f'mean_rmse_{channel}']=float(np.mean([float(r[f'rmse_{channel}']) for r in group]))
            row[f'mean_signed_error_{channel}']=float(np.mean([float(r[f'bias_{channel}']) for r in group]))
        aggregate.append(row)
        if p is not None:
            for peer in group:
                own=mle[key(peer)]
                item={**{k:own[k] for k in ('normal_world','friction_world','probe','episode_id')},
                    'comparator':estimator,'comparator_particles':int(p)}
                for channel in CHANNELS:
                    item[f'mle_rmse_{channel}']=float(own[f'rmse_{channel}'])
                    item[f'comparator_rmse_{channel}']=float(peer[f'rmse_{channel}'])
                    item[f'mle_minus_comparator_{channel}']=item[f'mle_rmse_{channel}']-item[f'comparator_rmse_{channel}']
                paired.append(item)
    cases=sorted({(r['normal_world'],r['friction_world'],r['probe']) for r in rows})
    cells=[]
    for normal,case,probe in cases:
        group=[r for r in rows if (r['normal_world'],r['friction_world'],r['probe'])==(normal,case,probe)]
        row=dict(normal_world=normal,friction_world=case,probe=probe,episodes=len(group))
        for channel in CHANNELS:
            row[f'mean_rmse_{channel}']=float(np.mean([float(r[f'rmse_{channel}']) for r in group]))
            row[f'mean_signed_error_{channel}']=float(np.mean([float(r[f'bias_{channel}']) for r in group]))
        cells.append(row)
    seconds=np.array([float(r['fit_seconds']) for r in rows])
    objectives=np.array([[float(r['initial_objective']),float(r['best_objective'])] for r in rows])
    resources=[]
    for relative in lock['selected_batches']:
        meta=run_mle.read((root/'predictions'/relative).with_suffix('.json'))
        resources.append({k:meta[k] for k in ('incremental_fit_seconds','incremental_forecast_seconds','source_particle_inference_plus_forecast_seconds')})
    summary=dict(scope='Descriptive one-seed development comparison; different box versus empirical-history priors; no optimality/calibration claim',
        episodes=432,batches=12,all_rows_retained=True,optimizer_success_episodes=process['optimizer_success_episodes'],
        budget_exhausted_episodes=process['budget_exhausted_episodes'],wall_seconds=process['seconds'],
        fit_seconds=dict(mean=float(seconds.mean()),median=float(np.median(seconds)),p90=float(np.quantile(seconds,.9)),maximum=float(seconds.max())),
        incremental_fit_seconds=sum(r['incremental_fit_seconds'] for r in resources),
        incremental_forecast_seconds=sum(r['incremental_forecast_seconds'] for r in resources),
        source_particle_inference_plus_forecast_seconds=sum(r['source_particle_inference_plus_forecast_seconds'] for r in resources),
        cost_scope='Cached P4096 start is not free; source time includes both filtering and unnecessary particle future forecast, not measured filter-only time.',
        residual_calls=dict(mean=float(np.mean([int(r['residual_calls']) for r in rows])),maximum=max(int(r['residual_calls']) for r in rows)),
        objective=dict(initial_mean=float(objectives[:,0].mean()),best_mean=float(objectives[:,1].mean()),
            fraction_strictly_improved=float(np.mean(objectives[:,1]<objectives[:,0])),max_increase=float(np.max(objectives[:,1]-objectives[:,0]))),
        aggregate_metrics=aggregate,cells=cells,source_lock_sha256=lock_sha,analysis_source_sha256=evaluate.sha(__file__))
    fractions={}
    for estimator in ('trajectory_assimilation','privileged_full_history'):
        for p in (256,1024,4096):
            group=[r for r in paired if r['comparator']==estimator and r['comparator_particles']==p]
            fractions[f'{estimator}/P{p}']={channel:float(np.mean([r[f'mle_minus_comparator_{channel}']<0 for r in group])) for channel in CHANNELS}
    summary['fraction_episodes_MLE_lower_RMSE']=fractions
    out=root/'analysis'
    if out.exists():raise FileExistsError('Retain existing analysis output')
    out.mkdir();write_csv(out/'paired_comparison.csv',paired);write_csv(out/'cell_metrics.csv',cells)
    evaluate.write_json(out/'summary.json',summary,exclusive=True)
    plot(out,aggregate)
    lines=['# Separate bounded initial-state MLE comparison','',summary['scope']+'.','',
        f"All 432 episodes completed. Optimizer success: {summary['optimizer_success_episodes']}/432; budget exhaustion: {summary['budget_exhausted_episodes']}/432. Total incremental evaluation wall time: {summary['wall_seconds']:.2f} s. Mean/median/p90/max per-episode fit times: {seconds.mean():.4f}/{np.median(seconds):.4f}/{np.quantile(seconds,.9):.4f}/{seconds.max():.4f} s.",'',
        '| Estimator | P | Mean position RMSE (µm) | Mean Ft RMSE (N) | Mean signed N error (N) |','|---|---:|---:|---:|---:|']
    for row in aggregate:
        lines.append(f"| {row['estimator']} | {row['particles'] or 'continuous'} | {row['mean_rmse_position_m']*1e6:.3f} | {row['mean_rmse_friction_N']:.6f} | {row['mean_signed_error_normal_N']:.6f} |")
    lines+=['',f"Model-state optimization cost {summary['incremental_fit_seconds']:.2f} s and one-point future propagation {summary['incremental_forecast_seconds']:.2f} s. The previously computed P4096 starting-posterior pipeline cost {summary['source_particle_inference_plus_forecast_seconds']:.2f} s including particle future predictions. That start is not free, and the combined source timing is not a measurement of filter-only overhead.",'',
        'MLE uses one maximum-posterior grid start and a broader uniform box of possible initial states. It can compensate fixed parameter bias through an inferred state and need not recover a physically replayable initial history. No global optimum, equal-prior comparison or calibrated uncertainty is claimed. The privileged known-history comparator uses fitted coefficients and is not a true-state oracle.','',
        'All state estimates used only causal public x/v/N and commands. The source hypothesis/weight arrays were read without their future forecasts or scores. Truth was introduced only after prediction. Solver failures are fail-loud and every completed episode is retained.','',
        'The separately predeclared higher-budget two-start check uses fixed indices 0,7,14,21,28,35 in every batch; results belong to a different locked output, not a replacement of these base results.','']
    (out/'README.md').write_text('\n'.join(lines))
    run_mle.verify(root)
    evaluate.write_json(out/'artifact_manifest.json',dict(outputs_sha256={str(p):evaluate.sha(p) for p in out.iterdir() if p.is_file()},mle_lock_sha256=lock_sha),exclusive=True)
    print(json.dumps({k:summary[k] for k in ('episodes','wall_seconds','budget_exhausted_episodes','optimizer_success_episodes','aggregate_metrics')},indent=2))


def plot(out,aggregate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    selected=[r for r in aggregate if r['estimator']=='trajectory_assimilation']
    mle=next(r for r in aggregate if r['estimator']=='bounded_initial_state_mle')
    privileged=next(r for r in aggregate if r['estimator']=='privileged_full_history')
    for ax,channel,scale,label in zip(axes,['position_m','friction_N'],[1e6,1],['Mean future position RMSE (µm)','Mean future friction RMSE (N)']):
        ax.plot([256,1024,4096],[r[f'mean_rmse_{channel}']*scale for r in selected],'-o',label='Finite-prior assimilation')
        ax.axhline(mle[f'mean_rmse_{channel}']*scale,color='tab:green',label='Bounded MLE, one start')
        ax.axhline(privileged[f'mean_rmse_{channel}']*scale,color='tab:orange',linestyle='--',label='Privileged known history')
        ax.set_xscale('log',base=2);ax.set_xticks([256,1024,4096],labels=['256','1024','4096'])
        ax.set_xlabel('Particles in approximate comparator');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Fixed noisy development subset: 432 paired episodes')
    fig.tight_layout();fig.savefig(out/'mle_comparison.png',dpi=170);fig.savefig(out/'mle_comparison.pdf')
    plt.close(fig)


if __name__=='__main__':main()
