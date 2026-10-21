"""Freeze then evaluate one-decision policies without native simulator imports."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/paper-study-numba')
import sys
sys.dont_write_bytecode=True
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
import argparse
import json
import time
import numpy as np
import pandas as pd
from paper_study.decision_study.common import HERE,BRANCHES,steps,sha,write_json,fresh_output,verify_hashes,validate_config
from paper_study.decision_study.inference import (model_paths,load_model,prior_path,load_prior,observable_prefix,
    filter_prefix,forecast_branch,branch_values,select_policies,actual_forecast)

PUBLIC_RAW=('u','d','d_dot','amplitude','geometry','excitation_sign','observations','observation_steps')


def source_paths():
    return sorted(HERE.glob('*.py'))+sorted((HERE/'tests').glob('*.py'))+[
        HERE/'protocol.md',HERE/'config.json',REPO/'paper_study/costed_benchmark/geometry.py',
        REPO/'research_next/src/state_id.py']


def validate_native_branch(path,metadata,cfg):
    """Factor/clock completeness, no prediction or target score calculation."""
    seed=metadata['preparation_seed'];nw=metadata['normal_world'];fw=metadata['friction_world']
    branch=metadata['branch'];common,later,horizon=steps(cfg)
    release=common if branch=='release' else later
    if seed not in cfg['preparation_seeds'] or [nw,fw] not in cfg['compositions'] or branch not in BRANCHES:
        raise ValueError('Unrecognized native factor labels')
    expected_rows=[dict(episode=e,preparation_seed=seed,preparation_index=p,geometry=g,
        amplitude=cfg['cam_amplitude_m'],excitation_sign=sign,
        episode_id=f'decision-p{seed}-h{p}-g{g}-s{sign:+d}')
        for e,(p,g,sign) in enumerate((p,g,sign) for p in range(6) for g in (0,1,2) for sign in (-1,1))]
    if metadata['rows']!=expected_rows:raise ValueError('Native factor order/product incomplete')
    if metadata['reference_updates']!=release+horizon or metadata['expected_steps']!=release+horizon:
        raise ValueError('Native material clock mismatch')
    if list(metadata['sensor_sigma'])!=cfg['sensor_sigma']:
        raise ValueError('Native sensor condition mismatch')
    if Path(path).parts[-3:]!=(f'p{seed}',f'{nw}_{fw}',branch+'.npz'):
        raise ValueError('Native path/metadata mismatch')
    with np.load(path,allow_pickle=False) as raw:
        if raw['u'].shape!=(36,release+horizon) or raw['observations'].shape!=(36,release//10+1,3):
            raise ValueError('Native episode/time dimensions incomplete')
        if not np.array_equal(raw['observation_steps'],np.arange(0,release+1,10)):
            raise ValueError('Native observation clock mismatch')
        if int(raw['common_step'])!=common or int(raw['release_step'])!=release:
            raise ValueError('Native decision/release event mismatch')
        for key in ('preparation_seed','preparation_index','geometry','amplitude','excitation_sign','episode'):
            expected=np.array([r[key] for r in expected_rows])
            if not np.array_equal(raw[key],expected):raise ValueError('Native factor array mismatch: '+key)


def freeze(data,out):
    data=Path(data).resolve()
    if not data.is_relative_to(HERE):raise ValueError('Only this separate study native output is allowed')
    if json.loads((data/'process.json').read_text())['status']!='complete':raise ValueError('Native data must be complete')
    generation=json.loads((data/'generation_lock.json').read_text());cfg=generation['config'];validate_config(cfg)
    verify_hashes(generation['sources_sha256'])
    seeds=cfg['preparation_seeds'][:1] if generation['smoke'] else cfg['preparation_seeds']
    pairs=[['maxwell','boucwen']] if generation['smoke'] else cfg['compositions']
    expected=[data/f'p{s}/{n}_{f}/{b}.npz' for s in seeds for n,f in pairs for b in BRANCHES]
    if set(data.rglob('*.npz'))!=set(expected):raise ValueError('Unexpected/missing native branch files')
    for p in expected:
        meta=json.loads(p.with_suffix('.json').read_text())
        if (meta['status']!='complete' or not meta['prefix_audit']['bitwise_identical'] or
            meta['npz_sha256']!=sha(p) or meta['expected_steps']!=meta['recorded_steps']):
            raise ValueError(f'Incomplete or corrupted native input: {p}')
        validate_native_branch(p,meta,cfg)
    priorlock=REPO/cfg['prior_bank_directory']/'prior_lock.json'
    prior_meta=json.loads(priorlock.read_text());verify_hashes(prior_meta['sources_sha256'])
    sources={str(p):sha(p) for p in source_paths()}
    model_prior={str(priorlock):sha(priorlock)}
    for nw,fw in cfg['compositions']:
        for p in (*model_paths(nw,fw,cfg),prior_path(nw,fw,cfg)):model_prior[str(p)]=sha(p)
    inputfiles=[p for p in data.rglob('*') if p.is_file()]
    destination=fresh_output(out)
    lock=dict(role='Exploratory decision evaluation frozen before scores',config=cfg,
        sources_sha256=sources,model_prior_sha256=model_prior,prior_sources_sha256=prior_meta['sources_sha256'],
        inputs_sha256={str(p):sha(p) for p in inputfiles},generation_sources_sha256=generation['sources_sha256'],
        data_directory=str(data),triplets=[dict(seed=s,normal=n,friction=f) for s in seeds for n,f in pairs])
    write_json(destination/'evaluation_lock.json',lock)
    write_json(destination/'process.json',dict(status='frozen'))
    return destination


def verify_lock(out):
    out=Path(out);lock=json.loads((out/'evaluation_lock.json').read_text())
    for key in ('sources_sha256','model_prior_sha256','prior_sources_sha256','inputs_sha256','generation_sources_sha256'):
        verify_hashes(lock[key])
    return lock


def branch_scores(prediction,arrays,cfg):
    end=int(arrays['release_step']);common,_,_=steps(cfg)
    target=np.concatenate((arrays['next_state'][:,end:],arrays['normal'][:,end:,None],arrays['friction'][:,end:,None]),axis=2)
    if prediction.shape!=target.shape or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError('Finite complete prediction/target with identical horizon required')
    mse=np.mean((prediction-target)**2,axis=1)
    return dict(position_mse_m2=mse[:,0],velocity_mse=mse[:,1],normal_mse_N2=mse[:,2],friction_mse_N2=mse[:,3],
        probe_seconds=np.full(len(target),end*cfg['dt']),
        absolute_work_J=arrays['absolute_work_J'].sum(axis=1),
        probe_absolute_work_J=arrays['absolute_work_J'][:,:end].sum(axis=1),
        extra_probe_work_J=arrays['absolute_work_J'][:,common:end].sum(axis=1),
        native_guard_count=arrays['guard'].sum(axis=1))


def episode_rows(metadata,scores,decisions,values,branches,cfg):
    rows=[];branchrows=[];n=len(metadata['rows'])
    for e,identity in enumerate(metadata['rows']):
        key=dict(preparation_seed=metadata['preparation_seed'],normal_world=metadata['normal_world'],
            friction_world=metadata['friction_world'],episode=e,preparation_index=identity['preparation_index'],
            geometry=identity['geometry'],excitation_sign=identity['excitation_sign'],
            model_seed=cfg['model_seed'],calibration_noise=cfg['calibration_noise'],sensor='noisy')
        for a,branch in enumerate(BRANCHES):
            branchrows.append(dict(**key,branch=branch,**{k:float(v[e]) for k,v in scores[branch].items()},
                planned_risk_m2=float(values['risk_m2'][e,a]),planned_information_nats=float(values['information_nats'][e,a]),
                planned_extra_work_J=float(values['planned_extra_probe_work_J'][e,a]),
                fantasy_mcse_m2=float(values['fantasy_risk_mcse_m2'][e,a]),
                planning_candidate_guard_count=int(branches[branch]['candidate_guard_steps'][e].sum())))
        for budget in cfg['budgets']:
            bname=budget['name']
            for method,probability in decisions[bname]['policies'].items():
                w=probability[e]
                result={k:float(sum(w[a]*scores[b][k][e] for a,b in enumerate(BRANCHES))) for k in scores['release']}
                over=np.array([scores[b]['extra_probe_work_J'][e]/budget['work_scale_J']>budget['max_scaled_work']+1e-14
                               for b in BRANCHES])
                rows.append(dict(**key,budget=bname,method=method,**result,
                    chosen_branch=BRANCHES[int(np.argmax(w))] if np.count_nonzero(w)==1 else 'probability_mixture',
                    probability_release=float(w[0]),probability_hold100=float(w[1]),probability_continue100=float(w[2]),
                    real_extra_work_budget_violation_probability=float(w@over),
                    planned_risk_m2=float(w@values['risk_m2'][e]),
                    planning_candidate_guard_count=float(sum(w[a]*branches[b]['candidate_guard_steps'][e].sum()
                                                            for a,b in enumerate(BRANCHES)))))
    return rows,branchrows


def run(out):
    out=Path(out).resolve();lock=verify_lock(out);cfg=lock['config'];data=Path(lock['data_directory'])
    if json.loads((out/'process.json').read_text())['status']!='frozen':raise ValueError('Use a fresh frozen evaluation, no overwrite/resume')
    started=time.perf_counter();process=dict(status='running',pid=os.getpid(),command=sys.argv);rows=[];branchrows=[]
    (out/'process.json').write_text(json.dumps(process,indent=2)+'\n')
    try:
        for item in lock['triplets']:
            if time.perf_counter()-started>cfg['evaluation_wall_seconds']:raise TimeoutError('CPU evaluation budget exhausted')
            seed,nw,fw=item['seed'],item['normal'],item['friction'];source=data/f'p{seed}/{nw}_{fw}'
            # The planner receives ONLY release's common prefix, never future states/forces/observations.
            with np.load(source/'release.npz',allow_pickle=False) as raw:
                public=observable_prefix({k:raw[k] for k in PUBLIC_RAW},cfg)
            model=load_model(nw,fw,cfg);grid,prior=load_prior(nw,fw,cfg)
            snapshot=filter_prefix(model,grid,prior,public,cfg)
            planning_start=time.perf_counter()
            branches={b:forecast_branch(model,snapshot,public,b,cfg) for b in BRANCHES}
            values=branch_values(snapshot,branches,cfg);decisions=select_policies(values,cfg)
            planning_seconds=time.perf_counter()-planning_start
            dest=out/f'p{seed}/{nw}_{fw}';dest.mkdir(parents=True)
            planned=dict(**values,decision_weights=snapshot['weights'],weight_trace=snapshot['weight_trace'],
                grid=grid,prior=prior,filtered_particle_xv=snapshot['xv'].reshape(len(public['u']),len(grid),2),
                prefix_candidate_work_J=snapshot['prefix_candidate_work_J'])
            for b,detail in branches.items():
                planned[b+'_candidate_future']=detail['future'];planned[b+'_candidate_sensors']=detail['sensors']
                planned[b+'_sensor_steps']=detail['sensor_steps'];planned[b+'_candidate_guards']=detail['candidate_guard_steps']
            for budget,result in decisions.items():
                planned[budget+'_eligible']=result['eligible']
                for method,probability in result['policies'].items():planned[budget+'__'+method]=probability
            with (dest/'planned.npz').open('xb') as f:np.savez_compressed(f,**planned)
            write_json(dest/'planned.json',dict(status='planned_before_target_read',planning_seconds=planning_seconds,
                planned_sha256=sha(dest/'planned.npz'),information_rights='Common prefix only; no actual future branch sensors or truth used'))
            # Only now read branch outcomes; causal assimilation still only receives later probe sensors.
            scores={};prediction_arrays={}
            metadata=json.loads((source/'release.json').read_text())
            for b in BRANCHES:
                with np.load(source/(b+'.npz'),allow_pickle=False) as raw:
                    sensor_input={k:raw[k] for k in ('observations','observation_steps')}
                    prediction,weights=actual_forecast(snapshot,branches[b],sensor_input,cfg)
                    targets={k:raw[k] for k in ('release_step','next_state','normal','friction','absolute_work_J','guard')}
                scores[b]=branch_scores(prediction,targets,cfg)
                prediction_arrays[b+'_forecast']=prediction;prediction_arrays[b+'_weights']=weights
            with (dest/'forecasts.npz').open('xb') as f:np.savez_compressed(f,**prediction_arrays)
            row,brow=episode_rows(metadata,scores,decisions,values,branches,cfg);rows.extend(row);branchrows.extend(brow)
            print(f'DECISION EVAL p{seed} {nw}/{fw}: {len(row)} policy rows, planning {planning_seconds:.3f}s',flush=True)
        verify_lock(out)
        frame=pd.DataFrame(rows);bframe=pd.DataFrame(branchrows)
        expected_contexts=len(lock['triplets'])*36
        if len(frame)!=expected_contexts*4*8 or len(bframe)!=expected_contexts*3:
            raise ValueError('Complete policy/branch row product required')
        identity=['preparation_seed','normal_world','friction_world','episode']
        if frame.duplicated(identity+['budget','method']).any() or bframe.duplicated(identity+['branch']).any():
            raise ValueError('Duplicate scored row keys')
        frame.to_csv(out/'policy_episodes.csv',index=False);bframe.to_csv(out/'branch_episodes.csv',index=False)
        summary=frame.groupby(['normal_world','friction_world','budget','method'],sort=True)[[
            'position_mse_m2','probe_seconds','absolute_work_J','probe_absolute_work_J',
            'real_extra_work_budget_violation_probability','native_guard_count','planning_candidate_guard_count']].mean().reset_index()
        summary['pooled_position_rmse_um']=1e6*np.sqrt(summary.position_mse_m2)
        summary.to_csv(out/'summary.csv',index=False)
        write_json(out/'summary.json',dict(role='Descriptive only: two development history blocks and one fitted bank',
            policy_rows=len(rows),branch_rows=len(branchrows),no_p_values=True,
            uniform_expected='Exact expected loss of randomized action choice, not loss of an averaged trajectory',
            budget='Expected incremental probe-work constraint; realized violations retained; total two-port absolute work reported',
            coefficients_fitted_on='Previously frozen isolated calibration only; no weight updates here'))
        process.update(status='complete',policy_rows=len(rows),branch_rows=len(branchrows))
    except BaseException as exc:
        process.update(status='failed',error=repr(exc));raise
    finally:
        process['seconds']=time.perf_counter()-started
        (out/'process.json').write_text(json.dumps(process,indent=2)+'\n')
        files=[p for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json']
        write_json(out/'artifact_manifest.json',dict(evaluation_lock_sha256=sha(out/'evaluation_lock.json'),
            outputs_sha256={str(p):sha(p) for p in files}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','run'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path)
    args=p.parse_args()
    if args.stage=='freeze':
        if args.data is None:p.error('--data is required for freeze')
        print(freeze(args.data,args.out))
    else:run(args.out)


if __name__=='__main__':main()
