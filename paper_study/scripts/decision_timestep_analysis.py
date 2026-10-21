"""Paired-noise decision timestep diagnostic; no native execution or refitting.

Public preparation prior remains at the canonical fixture clock. All candidate
mechanics, likelihood and forecast clocks use each native dt. Plans for all
grids are saved before the first additional target score. P401/K1024 are fixed
here, independent of pending held-out decision design choices.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/paper-decision-native-v1')
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(ROOT/'scripts'))
import argparse,copy,json,time
import numpy as np
import pandas as pd
from decision_resolution import (prepare_candidates,score_candidates,dump_plan,read_plan,
    branch_values,select_policies,episode_rows,sha,write_json,verify_hashes,BRANCHES)

GRIDS={'h':ROOT/'decision_study/results/native_development_v1',
    'h2':ROOT/'results/decision_refinement_h2','h4':ROOT/'results/decision_refinement_h4'}


def freeze(out):
    if out.exists():raise FileExistsError(out)
    configs={};hashes={str(Path(__file__).resolve()):sha(__file__),
        str(ROOT/'scripts/decision_resolution.py'):sha(ROOT/'scripts/decision_resolution.py')}
    old=json.loads((ROOT/'decision_study/results/evaluation_development_v1/evaluation_lock.json').read_text())
    for k,v in old.items():
        if k.endswith('_sha256') and isinstance(v,dict):hashes.update(v)
    for label,data in GRIDS.items():
        if json.loads((data/'process.json').read_text())['status']!='complete':raise ValueError('Native incomplete')
        gen=json.loads((data/'generation_lock.json').read_text());hashes.update(gen['sources_sha256'])
        cfg=copy.deepcopy(gen['config']);cfg.update(particles=401,fantasies=1024,
            prior_bank_directory='paper_study/results/prior_grid401')
        divisor={'h':1,'h2':2,'h4':4}[label]
        if cfg['dt']!=.0025/divisor or cfg['observation_steps']!=10*divisor:raise ValueError('Clock mismatch')
        expected={data/f"p{i['seed']}/{i['normal']}_{i['friction']}/{b}.npz" for i in old['triplets'] for b in BRANCHES}
        if set(data.rglob('*.npz'))!=expected:raise ValueError('Incomplete grid')
        for p in expected:
            meta=json.loads(p.with_suffix('.json').read_text())
            if meta['npz_sha256']!=sha(p) or meta['status']!='complete' or not meta['prefix_audit']['bitwise_identical']:
                raise ValueError('Native failed/hash mismatch/prefix mismatch')
        configs[label]=cfg
        for p in data.rglob('*'):
            if p.is_file():hashes[str(p)]=sha(p)
    prior=ROOT/'results/prior_grid401'
    info=json.loads((prior/'prior_lock.json').read_text());hashes.update(info['sources_sha256'])
    for p in prior.rglob('*'):
        if p.is_file():hashes[str(p)]=sha(p)
    verify_hashes(hashes);out.mkdir(parents=True)
    write_json(out/'lock.json',dict(role='Descriptive coupled timestep sensitivity, paired noise, fixed external fixture',
        created_unix=time.time(),configs=configs,triplets=old['triplets'],sources_inputs_sha256=hashes,
        criteria='Report all method/budget gains and ranking/choice changes. No selected stability threshold or new population test.',
        plan_before_target_scope='All24 plans persisted before additional outcomes are scored'))


def native_changes(out,lock):
    rows=[]
    for item in lock['triplets']:
        rel=Path(f"p{item['seed']}/{item['normal']}_{item['friction']}")
        for branch in BRANCHES:
            arrays={}
            for label,data in GRIDS.items():
                with np.load(data/rel/(branch+'.npz')) as z:
                    end=int(z['release_step']);divisor={'h':1,'h2':2,'h4':4}[label]
                    arrays[label]=dict(x=z['next_state'][:,end+divisor-1::divisor,0],
                        u=z['u'][:,::divisor],guard=int(z['guard'].sum()))
            for label in ('h2','h4'):
                error=arrays[label]['x']-arrays['h']['x']
                rows.append(dict(**item,branch=branch,grid=label,native_pooled_drift_um=float(np.sqrt(np.mean(error**2))*1e6),
                    max_force_change_N=float(np.max(abs(arrays[label]['u']-arrays['h']['u']))),native_guards=arrays[label]['guard']))
    pd.DataFrame(rows).to_csv(out/'native_changes.csv',index=False)


def run(out):
    lock=json.loads((out/'lock.json').read_text());verify_hashes(lock['sources_inputs_sha256']);start=time.perf_counter()
    if (out/'process.json').exists():raise FileExistsError('Never overwrite a previous attempt')
    process=dict(status='running',pid=os.getpid(),started_unix=time.time());write_json(out/'process.json',process)
    try:
        for label,cfg in lock['configs'].items():
            for item in lock['triplets']:
                rel=Path(f"p{item['seed']}/{item['normal']}_{item['friction']}");t=time.perf_counter()
                _,snapshot,branches=prepare_candidates(GRIDS[label]/rel,item['normal'],item['friction'],cfg)
                values=branch_values(snapshot,branches,cfg);decisions=select_policies(values,cfg)
                dump_plan(out/label/rel,values,decisions,snapshot,cfg,time.perf_counter()-t)
                print('TIMESTEP PLANNED',label,rel,flush=True)
        rows=[];branchrows=[]
        for label,cfg in lock['configs'].items():
            for item in lock['triplets']:
                rel=Path(f"p{item['seed']}/{item['normal']}_{item['friction']}");dest=out/label/rel
                values,decisions=read_plan(dest,cfg)
                _,snapshot,branches=prepare_candidates(GRIDS[label]/rel,item['normal'],item['friction'],cfg)
                scores,predictions,metadata=score_candidates(GRIDS[label]/rel,snapshot,branches,cfg)
                with (dest/'forecasts.npz').open('xb') as f:np.savez_compressed(f,**predictions)
                r,b=episode_rows(metadata,scores,decisions,values,branches,cfg)
                rows.extend(dict(grid=label,**v) for v in r);branchrows.extend(dict(grid=label,**v) for v in b)
        frame=pd.DataFrame(rows);branches=pd.DataFrame(branchrows)
        if len(frame)!=27648 or len(branches)!=2592:raise ValueError('Incomplete scored factorial')
        frame.to_csv(out/'policy_episodes.csv',index=False);branches.to_csv(out/'branch_episodes.csv',index=False)
        summary=frame.groupby(['grid','budget','method'])[['position_mse_m2','extra_probe_work_J','probe_seconds',
            'real_extra_work_budget_violation_probability']].mean().reset_index()
        summary['pooled_position_rmse_um']=1e6*np.sqrt(summary.position_mse_m2)
        summary.to_csv(out/'summary.csv',index=False)
        native_changes(out,lock)
        key=['preparation_seed','normal_world','friction_world','episode','budget','method']
        baseline=frame[frame.grid=='h'].set_index(key).sort_index();changes=[]
        for label in ('h2','h4'):
            other=frame[frame.grid==label].set_index(key).sort_index()
            if not other.index.equals(baseline.index):raise ValueError('Unpaired timestep comparison')
            for (budget,method),a in baseline.groupby(level=['budget','method']):
                b=other.loc[a.index];ma=float(a.position_mse_m2.mean());mb=float(b.position_mse_m2.mean())
                changes.append(dict(grid=label,budget=budget,method=method,pooled_rmse_change_um=1e6*(np.sqrt(mb)-np.sqrt(ma)),
                    mse_difference_m2=mb-ma,mse_ratio=mb/ma if ma else None,
                    changed_choices=int((a.chosen_branch!=b.chosen_branch).sum()),contexts=len(a)))
        pd.DataFrame(changes).to_csv(out/'policy_changes.csv',index=False)
        verify_hashes(lock['sources_inputs_sha256']);process.update(status='complete',rows=len(frame),branch_rows=len(branches))
    except BaseException as exc:process.update(status='failed',error=repr(exc));raise
    finally:
        process['seconds']=time.perf_counter()-start;(out/'process.json').write_text(json.dumps(process,indent=2)+'\n')
        write_json(out/'manifest.json',dict(outputs_sha256={str(p):sha(p) for p in out.rglob('*') if p.is_file()}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','run'])
    p.add_argument('--out',type=Path,default=ROOT/'results/decision_comparative_timestep_v1');a=p.parse_args()
    out=a.out.resolve()
    if not out.is_relative_to(ROOT/'results') or out==ROOT/'results':raise ValueError('Require results subdirectory')
    if a.stage=='freeze':freeze(out)
    else:run(out)


if __name__=='__main__':main()
