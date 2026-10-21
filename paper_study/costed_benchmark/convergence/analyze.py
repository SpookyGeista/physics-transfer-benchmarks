"""Apply the previously frozen finite-prior resolution diagnostic criteria."""
from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import evaluate

PVALUES=(256,1024,4096)
CHANNELS=('position_m','velocity_m_s','normal_N','friction_N')
TOL=np.array([.00005,.002,.15,.05])
ESTIMATORS=('prior_propagation','trajectory_assimilation','privileged_full_history')


def load_json(path):return json.loads(Path(path).read_text())


def write_csv(path,rows):
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    all_lock=load_json(root/'convergence_lock.json')
    evaluate.verify_hashes(all_lock['evaluation_locks_sha256'])
    if evaluate.sha(HERE/'protocol.md')!=all_lock['protocol_sha256']:
        raise ValueError('Declared diagnostic protocol changed')
    products={};locks={};processes={};all_metrics=[]
    for p in PVALUES:
        destination=root/f'p{p}';lock,lock_sha=evaluate.verify_lock(destination)
        process=load_json(destination/'process.json')
        if process['status']!='complete' or process['rows']!=1296 or process['expected_rows']!=1296:
            raise ValueError(f'Incomplete resolution run {p}')
        manifest=load_json(destination/'artifact_manifest.json')
        if manifest['evaluation_lock_sha256']!=lock_sha:raise ValueError('Output manifest lock mismatch')
        evaluate.verify_hashes(manifest['outputs_sha256'])
        with (destination/'episodes.csv').open() as stream:rows=list(csv.DictReader(stream))
        if len(rows)!=1296:raise ValueError('Metric row count mismatch')
        all_metrics.extend(rows);locks[p]=lock;processes[p]=process
        products[p]={}
        for relative in lock['selected_batches']:
            path=destination/'predictions/noisy'/relative
            meta=load_json(path.with_suffix('.json'))
            if meta['npz_sha256']!=evaluate.sha(path) or meta['status']!='complete':
                raise ValueError('Prediction artifact mismatch')
            with np.load(path,allow_pickle=False) as stored:arrays={k:stored[k] for k in stored.files}
            products[p][relative]=(meta,arrays)
    if not all(locks[p]['selected_batches']==locks[256]['selected_batches'] for p in PVALUES):
        raise ValueError('Unpaired batch subset')
    out=root/'analysis'
    if out.exists():raise FileExistsError('Retain existing analysis; use a separately versioned analysis source/output')
    out.mkdir()
    differences=[];episode_metrics=[];privileged_delta=np.zeros(4)
    for relative in locks[256]['selected_batches']:
        # Truth is used here only to score already persisted predictions.
        native=Path(locks[256]['data_directory'])/relative
        metadata=load_json(native.with_suffix('.json'))
        with np.load(native,allow_pickle=False) as truth:
            end=int(truth['probe_steps'])
            target=np.concatenate((truth['next_state'][:,end:],truth['normal'][:,end:,None],
                truth['friction'][:,end:,None]),axis=2)
        predictions={p:products[p][relative][1]['filtered_particles_trajectory_assimilation_forecast'] for p in PVALUES}
        d12=np.sqrt(np.mean((predictions[256]-predictions[1024])**2,axis=1))
        d24=np.sqrt(np.mean((predictions[1024]-predictions[4096])**2,axis=1))
        ppriv=products[256][relative][1]['filtered_particles_privileged_full_history_forecast']
        for p in PVALUES[1:]:
            privileged_delta=np.maximum(privileged_delta,np.max(abs(products[p][relative][1][
                'filtered_particles_privileged_full_history_forecast']-ppriv),axis=(0,1)))
        for i,factor in enumerate(metadata['factor_rows']):
            identity=dict(normal_world=metadata['normal_world'],friction_world=metadata['friction_world'],
                probe=metadata['probe']['name'],episode=i,episode_id=factor['episode_id'],profile_index=factor['profile_index'])
            row=dict(identity)
            for c,name in enumerate(CHANNELS):
                row[f'difference_256_1024_{name}']=float(d12[i,c])
                row[f'difference_1024_4096_{name}']=float(d24[i,c])
            row['all_channels_within_tolerance']=bool(np.all(d24[i]<=TOL))
            differences.append(row)
        for p in PVALUES:
            stored=products[p][relative][1]
            weight=evaluate.weight_diagnostics(stored['weights'])
            # Supplementary representation audit: merge initial deflections
            # equal to 1e-12 m. This never affects inference or stability gates.
            unique_states,labels=np.unique(np.round(stored['prior_states'],12),axis=0,return_inverse=True)
            masses=np.zeros((len(target),len(unique_states)))
            for i in range(len(target)):np.add.at(masses[i],labels,stored['weights'][i])
            grouped_ess=1./np.sum(masses*masses,axis=1)
            for estimator in ESTIMATORS:
                predicted=stored[f'filtered_particles_{estimator}_forecast']
                error=predicted-target
                rmse=np.sqrt(np.mean(error**2,axis=1));bias=np.mean(error,axis=1)
                sd=(np.sqrt(np.mean(stored[f'filtered_particles_{estimator}_variance'],axis=1))
                    if estimator!='privileged_full_history' else None)
                # This ratio contrasts errors with finite-prior spread only;
                # it is not a calibrated coverage/confidence calculation.
                for i,factor in enumerate(metadata['factor_rows']):
                    row=dict(normal_world=metadata['normal_world'],friction_world=metadata['friction_world'],
                        probe=metadata['probe']['name'],episode=i,episode_id=factor['episode_id'],particles=p,estimator=estimator,
                        posterior_ess=float(weight['ess'][i]),posterior_ess_fraction=float(weight['ess'][i]/p),
                        posterior_entropy=float(weight['entropy'][i]),posterior_max_weight=float(weight['max_weight'][i]),
                        prior_unique_initial_states_1e12m=len(unique_states),
                        grouped_initial_state_mass_ess=float(grouped_ess[i]))
                    for c,name in enumerate(CHANNELS):
                        row[f'rmse_{name}']=float(rmse[i,c]);row[f'bias_{name}']=float(bias[i,c])
                        row[f'finite_prior_rms_sd_{name}']=None if sd is None else float(sd[i,c])
                    episode_metrics.append(row)
    if len(differences)!=432 or len(episode_metrics)!=3888:raise ValueError('Unpaired episode inventory')
    write_csv(out/'episode_trajectory_differences.csv',differences)
    write_csv(out/'episode_metrics.csv',episode_metrics)
    cells=[]
    identities=sorted({(row['normal_world'],row['friction_world'],row['probe']) for row in episode_metrics})
    for normal,case,probe in identities:
        for p in PVALUES:
            for estimator in ESTIMATORS:
                selected=[row for row in episode_metrics if (row['normal_world'],row['friction_world'],row['probe'],row['particles'],row['estimator'])==(normal,case,probe,p,estimator)]
                if len(selected)!=36:raise ValueError('Incomplete diagnostic cell')
                row=dict(normal_world=normal,friction_world=case,probe=probe,particles=p,estimator=estimator,count=36)
                for c,name in enumerate(CHANNELS):
                    row[f'mean_rmse_{name}']=float(np.mean([x[f'rmse_{name}'] for x in selected]))
                    row[f'mean_bias_{name}']=float(np.mean([x[f'bias_{name}'] for x in selected]))
                row['median_posterior_ess']=float(np.median([x['posterior_ess'] for x in selected]))
                cells.append(row)
    write_csv(out/'cell_metrics.csv',cells)
    cell_failures=[]
    for normal,case,probe in identities:
        selected={row['particles']:row for row in cells if (row['normal_world'],row['friction_world'],row['probe'],row['estimator'])==(normal,case,probe,'trajectory_assimilation')}
        for c,name in enumerate(CHANNELS):
            a,b=selected[1024][f'mean_rmse_{name}'],selected[4096][f'mean_rmse_{name}']
            allowed=.05*max(b,TOL[c])
            if abs(a-b)>allowed:
                cell_failures.append(dict(normal_world=normal,friction_world=case,probe=probe,channel=name,
                    P1024_mean_rmse=a,P4096_mean_rmse=b,absolute_change=abs(a-b),allowed_change=allowed))
    trajectory_summary={}
    for c,name in enumerate(CHANNELS):
        a=np.array([row[f'difference_256_1024_{name}'] for row in differences])
        b=np.array([row[f'difference_1024_4096_{name}'] for row in differences])
        stats=lambda v:dict(median=float(np.median(v)),p90=float(np.quantile(v,.90)),p95=float(np.quantile(v,.95)),maximum=float(np.max(v)))
        trajectory_summary[name]=dict(P256_P1024=stats(a),P1024_P4096=stats(b),fraction_second_smaller=float(np.mean(b<a)),tolerance=float(TOL[c]))
    collapsed={};aggregates=[]
    for p in PVALUES:
        selected=[row for row in episode_metrics if row['particles']==p and row['estimator']=='trajectory_assimilation']
        ess=np.array([row['posterior_ess'] for row in selected]);mw=np.array([row['posterior_max_weight'] for row in selected])
        collapsed[p]=dict(ess_median=float(np.median(ess)),ess_min=float(np.min(ess)),ess_fraction_median=float(np.median(ess/p)),
            fraction_ess_le_2=float(np.mean(ess<=2)),fraction_max_weight_ge_09=float(np.mean(mw>=.9)),
            maximum_weight=float(np.max(mw)),mean_entropy=float(np.mean([row['posterior_entropy'] for row in selected])),
            grouped_initial_state_mass_ess_median=float(np.median([row['grouped_initial_state_mass_ess'] for row in selected])),
            prior_unique_initial_states_min=min(row['prior_unique_initial_states_1e12m'] for row in selected),
            prior_unique_initial_states_max=max(row['prior_unique_initial_states_1e12m'] for row in selected))
        for estimator in ESTIMATORS:
            group=[row for row in episode_metrics if row['particles']==p and row['estimator']==estimator]
            row=dict(particles=p,estimator=estimator)
            for name in CHANNELS:
                row[f'mean_rmse_{name}']=float(np.mean([x[f'rmse_{name}'] for x in group]))
                row[f'mean_signed_error_{name}']=float(np.mean([x[f'bias_{name}'] for x in group]))
                row[f'mean_abs_episode_bias_{name}']=float(np.mean([abs(x[f'bias_{name}']) for x in group]))
                if estimator!='privileged_full_history':row[f'mean_finite_prior_rms_sd_{name}']=float(np.mean([x[f'finite_prior_rms_sd_{name}'] for x in group]))
            aggregates.append(row)
    rank_changes=[]
    for normal,case,probe in identities:
        for name in CHANNELS:
            rankings={p:sorted(ESTIMATORS,key=lambda estimator:next(row[f'mean_rmse_{name}'] for row in cells if
                (row['normal_world'],row['friction_world'],row['probe'],row['particles'],row['estimator'])==(normal,case,probe,p,estimator))) for p in PVALUES}
            if rankings[256]!=rankings[1024] or rankings[1024]!=rankings[4096]:
                rank_changes.append(dict(normal_world=normal,friction_world=case,probe=probe,channel=name,rankings=rankings))
    fraction=float(np.mean([row['all_channels_within_tolerance'] for row in differences]))
    summary=dict(scope='Exploratory one-seed paired numerical diagnostic; no independent-replicate CI or uncertainty calibration',
        episodes_per_P=432,batches_per_P=12,particles=PVALUES,fit_calibration='noisy',sensor='noisy',
        forecast_start='filtered_particles',timings_seconds={p:processes[p]['seconds'] for p in PVALUES},
        trajectory_difference=trajectory_summary,posterior_weight_concentration=collapsed,aggregate_metrics=aggregates,
        practical_stability=dict(fraction_all_channels_within_tolerance=fraction,criterion_95_percent_passed=fraction>=.95,
            cell_rmse_change_criterion_passed=not cell_failures,failed_cell_channels=cell_failures,
            both_passed=fraction>=.95 and not cell_failures),
        privileged_forecast_max_absolute_difference_across_P=dict(zip(CHANNELS,map(float,privileged_delta))),
        method_rank_changes=rank_changes,protocol_sha256=all_lock['protocol_sha256'],
        supplementary_representation_audit='Initial PI states rounded to 1e-12 m only for duplicate-mass ESS; no inference or prespecified gate changes.',
        evaluation_locks_sha256=all_lock['evaluation_locks_sha256'],analysis_source_sha256=evaluate.sha(__file__))
    evaluate.write_json(out/'summary.json',summary,exclusive=True)
    plot(out,episode_metrics,trajectory_summary)
    report(out,summary)
    for p in PVALUES:evaluate.verify_lock(root/f'p{p}')
    evaluate.verify_hashes(all_lock['evaluation_locks_sha256'])
    evaluate.write_json(out/'artifact_manifest.json',dict(outputs_sha256={str(p):evaluate.sha(p) for p in out.iterdir() if p.is_file()},
        convergence_lock_sha256=evaluate.sha(root/'convergence_lock.json')),exclusive=True)
    print(json.dumps(summary['practical_stability'],indent=2))
    print('ANALYSIS COMPLETE',out)


def plot(out,rows,trajectory_summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7.5))
    pairs=sorted({(row['normal_world'],row['friction_world']) for row in rows})
    for normal,case in pairs:
        values=[]
        for p in PVALUES:
            selected=[row for row in rows if (row['normal_world'],row['friction_world'],row['particles'],row['estimator'])==(normal,case,p,'trajectory_assimilation')]
            values.append([np.mean([r['rmse_position_m'] for r in selected])*1e6,
                np.mean([r['rmse_friction_N'] for r in selected]),np.median([r['posterior_ess_fraction'] for r in selected])])
        a=np.array(values);label=f'{normal} / {case}'
        for ax,col in zip([axes[0,0],axes[0,1],axes[1,0]],range(3)):
            ax.plot(PVALUES,a[:,col],'-o',label=label,markersize=4)
    axes[0,0].set_ylabel('Mean future position RMSE (µm)')
    axes[0,1].set_ylabel('Mean future friction RMSE (N)')
    axes[1,0].set_ylabel('Median particle ESS / P')
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:
        ax.set_xscale('log',base=2);ax.set_xticks(PVALUES,labels=[str(p) for p in PVALUES]);ax.set_xlabel('Particles P');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    for name,entry in trajectory_summary.items():
        axes[1,1].plot([0,1],[entry['P256_P1024']['p95']/entry['tolerance'],entry['P1024_P4096']['p95']/entry['tolerance']],'-o',label=name)
    axes[1,1].axhline(1,color='black',linewidth=.8,linestyle='--')
    axes[1,1].set_xticks([0,1],['256 → 1024','1024 → 4096'])
    axes[1,1].set_ylabel('95th percentile trajectory change / tolerance')
    axes[1,1].legend(fontsize=7);axes[1,1].grid(alpha=.2)
    fig.suptitle('Finite-prior resolution: fixed noisy development subset',fontsize=14)
    fig.text(.5,.01,'432 paired episodes from six histories in one development seed; descriptive only, no independent-replicate confidence intervals.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.03,1,.96])
    fig.savefig(out/'resolution_diagnostic.png',dpi=170)
    fig.savefig(out/'resolution_diagnostic.pdf')
    plt.close(fig)


def report(out,s):
    lines=['# Finite-prior resolution diagnostic','',s['scope']+'.','',
        'Fixed subset: four balanced normal/PI pairings × three probes × 36 episodes, noisy fitted models and noisy sensors. Only filtered-particle forecast starts. All three layouts were frozen before P256 scoring.','',
        '| P | Runtime (s) | Median ESS | Median ESS/P | ESS ≤ 2 | max weight ≥ 0.9 |',
        '|---:|---:|---:|---:|---:|---:|']
    for p in PVALUES:
        d=s['posterior_weight_concentration'][p]
        lines.append(f"| {p} | {s['timings_seconds'][p]:.1f} | {d['ess_median']:.2f} | {d['ess_fraction_median']:.4f} | {d['fraction_ess_le_2']:.1%} | {d['fraction_max_weight_ge_09']:.1%} |")
    lines+=['','| P | Estimator | Mean x RMSE (µm) | Mean Ft RMSE (N) | Mean signed N error (N) |','|---:|---|---:|---:|---:|']
    for row in s['aggregate_metrics']:
        lines.append(f"| {row['particles']} | {row['estimator']} | {row['mean_rmse_position_m']*1e6:.3f} | {row['mean_rmse_friction_N']:.6f} | {row['mean_signed_error_normal_N']:.6f} |")
    gate=s['practical_stability']
    lines+=['',f"The prespecified simultaneous trajectory-tolerance fraction is {gate['fraction_all_channels_within_tolerance']:.2%} (required ≥95%): {'PASS' if gate['criterion_95_percent_passed'] else 'FAIL'}. The separate cell-mean RMSE-change check is {'PASS' if gate['cell_rmse_change_criterion_passed'] else 'FAIL'}; {len(gate['failed_cell_channels'])} cell/channel combinations exceed their fixed allowance. Combined practical stability: {'PASS' if gate['both_passed'] else 'FAIL'}.",'',
        f"Method ordering changes in {len(s['method_rank_changes'])} of 48 cell/channel combinations across P. Full details and thresholds are in summary.json; no preferred ordering was selected.",'',
        'Particle ESS and finite-prior predictive spread do not establish calibrated uncertainty. Repeated reachable states can inflate particle ESS; parameter bias is outside the fixed state-only ensemble. Signed errors and absolute episode biases are provided separately to prevent cancellation from hiding errors. Larger P is not ground truth.','',
        'The privileged known-history prediction is a fixed-model comparator, not an oracle. Its cross-P invariance is checked numerically in summary.json. The current evaluator assumes preparation and motion dt both equal 0.0025 s; timestep-refinement reuse needs a separate explicit prefix-time guard or implementation. This diagnostic does not change that assumption.','',
        'Files: summary.json, episode_metrics.csv, episode_trajectory_differences.csv, cell_metrics.csv, resolution_diagnostic.png/.pdf and artifact_manifest.json. Source, native inputs, all model files and prior banks were verified before and after each run and again during analysis.','']
    (out/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
