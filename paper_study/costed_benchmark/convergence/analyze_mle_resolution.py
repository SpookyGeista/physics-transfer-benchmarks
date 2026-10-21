"""Apply the predeclared two-start MLE numerical diagnostic criteria."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_mle_resolution as runner
import evaluate
NAMES=('position_m','velocity_m_s','normal_N','friction_N')
TOL=np.array([.00005,.002,.15,.05])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve()
    lock,digest=runner.verify(root);process=json.loads((root/'process.json').read_text());manifest=json.loads((root/'artifact_manifest.json').read_text())
    if process['status']!='complete' or process['episodes']!=72 or manifest['lock_sha256']!=digest:raise ValueError('Incomplete/mismatched diagnostic')
    evaluate.verify_hashes(manifest['outputs_sha256'])
    with (root/'episodes.csv').open() as f:rows=list(csv.DictReader(f))
    if len(rows)!=72:raise ValueError('Episode inventory mismatch')
    rmse=np.array([[float(r['rmse_'+n]) for n in NAMES] for r in rows]);base=np.array([[float(r['base_rmse_'+n]) for n in NAMES] for r in rows])
    changes=np.array([[float(r['forecast_change_'+n]) for n in NAMES] for r in rows])
    failures=[]
    for normal,case,probe in sorted({(r['normal_world'],r['friction_world'],r['probe']) for r in rows}):
        mask=np.array([(r['normal_world'],r['friction_world'],r['probe'])==(normal,case,probe) for r in rows])
        if mask.sum()!=6:raise ValueError('Incomplete diagnostic cell')
        newer,older=rmse[mask].mean(axis=0),base[mask].mean(axis=0)
        for c,name in enumerate(NAMES):
            allowed=.05*max(newer[c],TOL[c])
            if abs(newer[c]-older[c])>allowed:failures.append(dict(normal_world=normal,friction_world=case,probe=probe,channel=name,
                base_mean_rmse=float(older[c]),two_start_mean_rmse=float(newer[c]),allowed_change=float(allowed)))
    fraction=float(np.mean(np.all(changes<=TOL,axis=1)))
    objectives=np.array([[float(r['base_objective']),float(r['selected_objective'])] for r in rows])
    candidate_differences=[];candidate_objectives=[]
    for relative in lock['selected_batches']:
        with np.load(root/'predictions'/relative,allow_pickle=False) as data:
            candidate_differences.extend(np.sqrt(np.mean((data['candidate_forecasts'][:,0]-data['candidate_forecasts'][:,1])**2,axis=1)))
            candidate_objectives.extend(data['candidate_objectives'])
    candidate_differences=np.asarray(candidate_differences);candidate_objectives=np.asarray(candidate_objectives)
    near_loss=abs(np.diff(candidate_objectives,axis=1)[:,0])<=.001*np.maximum(1.,np.min(candidate_objectives,axis=1))
    separated=np.any(candidate_differences>TOL,axis=1)
    summary=dict(scope='Separate descriptive optimization-resolution diagnostic; same72episodes, no global optimum or population convergence claim',
        episodes=72,starts=144,wall_seconds=process['seconds'],successful_starts=process['successful_starts'],budget_cases=process['budget_cases'],
        zero_start_selected_episodes=sum(int(r['chosen_start'])==1 for r in rows),
        mean_base_rmse=dict(zip(NAMES,map(float,base.mean(axis=0)))),mean_two_start_rmse=dict(zip(NAMES,map(float,rmse.mean(axis=0)))),
        fraction_future_RMSE_improved=dict(zip(NAMES,map(float,np.mean(rmse<base,axis=0)))),
        fraction_past_objective_strictly_improved=float(np.mean(objectives[:,1]<objectives[:,0])),
        mean_base_objective=float(objectives[:,0].mean()),mean_two_start_objective=float(objectives[:,1].mean()),
        maximum_objective_increase=float(np.max(objectives[:,1]-objectives[:,0])),
        trajectory_change_p95=dict(zip(NAMES,map(float,np.quantile(changes,.95,axis=0)))),
        stability=dict(fraction_all_channels_within_tolerance=fraction,criterion_95_percent_passed=fraction>=.95,
            cell_mean_RMSE_criterion_passed=not failures,failed_cell_channels=failures,both_passed=fraction>=.95 and not failures),
        supplementary_candidate_ambiguity=dict(definition='Exploratory: relative loss gap<=0.001*max(1,min loss), future RMS difference exceeds a declared engineering tolerance.',
            near_equal_past_loss_episodes=int(near_loss.sum()),near_equal_loss_different_future_episodes=int(np.sum(near_loss&separated)),
            candidate_future_difference_p95=dict(zip(NAMES,map(float,np.quantile(candidate_differences,.95,axis=0))))),
        fit_seconds=dict(mean=float(np.mean([float(r['total_fit_seconds']) for r in rows])),maximum=max(float(r['total_fit_seconds']) for r in rows)),
        lock_sha256=digest,analysis_source_sha256=evaluate.sha(__file__))
    out=root/'analysis'
    if out.exists():raise FileExistsError('Retain existing analysis')
    out.mkdir();evaluate.write_json(out/'summary.json',summary,exclusive=True)
    lines=['# Higher-budget two-start MLE diagnostic','',summary['scope']+'.','',
        f"All 72 episodes and 144 starts completed; {summary['successful_starts']} successful starts, {summary['budget_cases']} budget cases, {summary['wall_seconds']:.2f} s total wall time. The fixed zero start was selected in {summary['zero_start_selected_episodes']}/72 episodes based only on past likelihood.",'',
        '| Channel | Base mean RMSE | Two-start mean RMSE | Fraction future RMSE improved |','|---|---:|---:|---:|']
    for name in NAMES:lines.append(f"| {name} | {summary['mean_base_rmse'][name]:.8g} | {summary['mean_two_start_rmse'][name]:.8g} | {summary['fraction_future_RMSE_improved'][name]:.1%} |")
    lines+=['',f"Practical trajectory stability fraction: {fraction:.2%} (required ≥95%); cell-mean RMSE check has {len(failures)} failures. Combined criterion: {'PASS' if summary['stability']['both_passed'] else 'FAIL'}.",'',
        f"Mean past objective changes from {summary['mean_base_objective']:.6g} to {summary['mean_two_start_objective']:.6g}. This is not a guarantee of better future prediction. Candidate forecasts were computed only after the past-objective start choice was fixed.",'',
        'The diagnostic covers one fixed episode per history with a balanced geometry/sign diagonal, not every episode. Both variants retain the same broader box support. Neither solver-success flags nor this engineering stability test establish a global optimum or physical state identifiability. Existing base results and all previous locks remain unchanged.','',
        'summary.json contains supplementary near-equal-loss/different-future candidate counts. Their explicit exploratory threshold is not a new confirmatory gate. Every optimizer attempt and selected candidate is preserved in prediction artifacts.','']
    (out/'README.md').write_text('\n'.join(lines));runner.verify(root)
    evaluate.write_json(out/'artifact_manifest.json',dict(lock_sha256=digest,outputs_sha256={str(p):evaluate.sha(p) for p in out.iterdir() if p.is_file()}),exclusive=True)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
