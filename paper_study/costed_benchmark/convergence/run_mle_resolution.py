"""Separate locked two-start/higher-budget MLE resolution diagnostic."""
from pathlib import Path
import argparse,csv,json,sys,time,os
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(HERE))
import run_mle
import evaluate
from bounded_mle import fit_initial_state,forecast

INDICES=np.array([0,7,14,21,28,35],dtype=int)
CONFIG=dict(run_mle.DEFAULTS,max_nfev=120,max_residual_calls=512,max_seconds_per_episode=6.)


def verify(out):
    lock=run_mle.read(out/'resolution_lock.json')
    for key in ('sources_sha256','inputs_sha256','generation_sources_sha256'):evaluate.verify_hashes(lock[key])
    return lock,evaluate.sha(out/'resolution_lock.json')


def freeze(base,out):
    original,original_sha=run_mle.verify(base)
    process=run_mle.read(base/'process.json');manifest=run_mle.read(base/'artifact_manifest.json')
    if process['status']!='complete' or process['episodes']!=432 or manifest['lock_sha256']!=original_sha:
        raise ValueError('Complete unchanged base MLE required')
    evaluate.verify_hashes(manifest['outputs_sha256'])
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a fresh resolution output')
    if out==REPO/'paper_study/results' or not out.is_relative_to(REPO/'paper_study/results'):
        raise ValueError('Use a fresh subdirectory of paper_study/results')
    sources=dict(original['sources_sha256']);inputs=dict(original['inputs_sha256'])
    for path in (Path(__file__).resolve(),HERE/'mle_resolution_protocol.md'):sources[str(path)]=evaluate.sha(path)
    inputs.update(manifest['outputs_sha256'])
    for name in ('mle_lock.json','process.json','artifact_manifest.json'):inputs[str(base/name)]=evaluate.sha(base/name)
    out.mkdir(parents=True,exist_ok=True)
    evaluate.write_json(out/'resolution_lock.json',dict(role='Exploratory MLE budget/start diagnostic fixed before scores',
        config=CONFIG,episode_indices=INDICES,expected_episodes=72,base_directory=str(base),
        base_lock_sha256=original_sha,data_directory=original['data_directory'],
        source_particle_directory=original['source_particle_directory'],selected_batches=original['selected_batches'],
        sources_sha256=sources,inputs_sha256=inputs,generation_sources_sha256=original['generation_sources_sha256'],
        initializations=['best causal P4096 initial particle','fixed relaxed q=0'],selection='minimum past objective, first start wins ties',
        created_unix=time.time(),command=sys.argv),exclusive=True)
    evaluate.write_json(out/'process.json',dict(status='locked',lock_sha256=evaluate.sha(out/'resolution_lock.json')),exclusive=True)
    verify(out);print('RESOLUTION LOCKED',out,evaluate.sha(out/'resolution_lock.json'),flush=True)


def subset(public,indices):
    return {key:(value if key in ('probe_steps','observation_steps') else value[indices].copy()) for key,value in public.items()}


def run(out):
    lock,lock_sha=verify(out)
    if run_mle.read(out/'process.json')!={'status':'locked','lock_sha256':lock_sha}:raise ValueError('Unused lock required')
    rows=[];artifacts={};completed=0;active=None;started=time.perf_counter();indices=np.asarray(lock['episode_indices'])
    process=dict(status='running',pid=os.getpid(),lock_sha256=lock_sha,started_unix=time.time(),command=sys.argv)
    evaluate.write_json(out/'process.json',process)
    try:
        for relative in lock['selected_batches']:
            active=relative;native=Path(lock['data_directory'])/relative;metadata=run_mle.read(native.with_suffix('.json'))
            with np.load(native,allow_pickle=False) as stored:
                raw={key:stored[key] for key in (*evaluate.PUBLIC_KEYS,'observations','observation_steps','probe_steps')}
            inputs=subset(evaluate.observable_inputs(raw),indices)
            public_meta={key:metadata[key] for key in evaluate.PUBLIC_META}
            source=Path(lock['source_particle_directory'])/'predictions/noisy'/relative
            with np.load(source,allow_pickle=False) as stored:
                initial=stored['prior_states'].copy();weight=stored['weights'][indices].copy()
            model=evaluate.load_model(metadata['friction_world'],metadata['normal_world'],'noisy',pi_seed=61001,normal_seed=1,
                dt=metadata['dt'],mass=metadata['mass_kg'],iterations=40)
            candidates=[fit_initial_state(model,inputs,public_meta,initial,weight,config=lock['config']),
                fit_initial_state(model,inputs,public_meta,np.zeros((1,3)),np.ones((len(indices),1)),config=lock['config'])]
            objectives=np.array([[d['best_objective'] for d in result['diagnostics']] for result in candidates]).T
            choice=np.argmin(objectives,axis=1)
            selected={'config':lock['config']}
            for key in ('initial_pi_state','normalized_state','boundary_xv'):
                value=np.stack([c[key] for c in candidates],axis=1)
                selected[key]=value[np.arange(len(indices)),choice]
            selected['boundary_state']={}
            for key in ('normal','pi'):
                value=np.stack([c['boundary_state'][key] for c in candidates],axis=1)
                selected['boundary_state'][key]=value[np.arange(len(indices)),choice]
            # Choice is final before any candidate future is computed.
            candidate_future=[];solver=[]
            for candidate in candidates:
                future,diagnostic=forecast(model,inputs,candidate,return_diagnostics=True)
                candidate_future.append(future['filtered_particles/bounded_initial_state_mle']);solver.append(diagnostic)
            future=np.stack(candidate_future,axis=1)
            prediction=future[np.arange(len(indices)),choice]
            with np.load(native,allow_pickle=False) as truth:
                end=inputs['probe_steps']
                target=np.concatenate((truth['next_state'][indices,end:],truth['normal'][indices,end:,None],truth['friction'][indices,end:,None]),axis=2)
            base=Path(lock['base_directory'])/'predictions'/relative
            base_meta=run_mle.read(base.with_suffix('.json'))
            with np.load(base,allow_pickle=False) as stored:base_prediction=stored['forecast'][indices]
            error=prediction-target;base_error=base_prediction-target
            if not np.isfinite(error).all():raise FloatingPointError('Nonfinite scored diagnostic')
            rmse=np.sqrt(np.mean(error**2,axis=1));bias=np.mean(error,axis=1)
            base_rmse=np.sqrt(np.mean(base_error**2,axis=1));change=np.sqrt(np.mean((prediction-base_prediction)**2,axis=1))
            batch=[]
            for i,index in enumerate(indices):
                factor=metadata['factor_rows'][index]
                row=dict(normal_world=metadata['normal_world'],friction_world=metadata['friction_world'],probe=metadata['probe']['name'],
                    episode=int(index),episode_id=factor['episode_id'],profile_index=factor['profile_index'],chosen_start=int(choice[i]),
                    base_objective=base_meta['diagnostics'][index]['best_objective'],selected_objective=float(objectives[i,choice[i]]),
                    best_particle_start_objective=float(objectives[i,0]),zero_start_objective=float(objectives[i,1]),
                    total_fit_seconds=sum(c['diagnostics'][i]['seconds'] for c in candidates),
                    budget_cases=sum(c['diagnostics'][i]['budget_exhausted'] for c in candidates),
                    successful_starts=sum(c['diagnostics'][i]['solver_success'] for c in candidates))
                for j,name in enumerate(run_mle.CHANNELS):
                    row[f'rmse_{name}']=float(rmse[i,j]);row[f'base_rmse_{name}']=float(base_rmse[i,j])
                    row[f'bias_{name}']=float(bias[i,j]);row[f'forecast_change_{name}']=float(change[i,j])
                batch.append(row)
            destination=out/'predictions'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as stream:
                np.savez_compressed(stream,episode_indices=indices,initial_pi_state=selected['initial_pi_state'],
                    boundary_xv=selected['boundary_xv'],boundary_pi_state=selected['boundary_state']['pi'],
                    boundary_normal_state=selected['boundary_state']['normal'],forecast=prediction,
                    candidate_forecasts=future,candidate_objectives=objectives,selected_start=choice)
            evaluate.write_json(destination.with_suffix('.json'),dict(status='complete',lock_sha256=lock_sha,npz_sha256=evaluate.sha(destination),
                input_npz_sha256=evaluate.sha(native),base_prediction_sha256=evaluate.sha(base),metrics=batch,
                candidate_diagnostics=[c['diagnostics'] for c in candidates],candidate_solver=solver),exclusive=True)
            for path in (destination,destination.with_suffix('.json')):artifacts[str(path)]=evaluate.sha(path)
            rows.extend(batch);completed+=1;print(f'RESOLUTION {completed}/12 {relative}',flush=True)
        if len(rows)!=72:raise ValueError('Diagnostic episode count mismatch')
        with (out/'episodes.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        artifacts[str(out/'episodes.csv')]=evaluate.sha(out/'episodes.csv')
        if verify(out)[1]!=lock_sha:raise ValueError('Diagnostic lock changed')
        evaluate.write_json(out/'artifact_manifest.json',dict(lock_sha256=lock_sha,outputs_sha256=artifacts),exclusive=True)
        evaluate.write_json(out/'process.json',dict(**{k:v for k,v in process.items() if k!='status'},status='complete',
            seconds=time.perf_counter()-started,batches=completed,episodes=len(rows),optimizer_starts=2*len(rows),
            budget_cases=sum(r['budget_cases'] for r in rows),successful_starts=sum(r['successful_starts'] for r in rows)))
    except BaseException as exc:
        evaluate.write_json(out/'process.json',dict(**{k:v for k,v in process.items() if k!='status'},status='failed',
            error=repr(exc),failed_batch=active,seconds=time.perf_counter()-started,completed_batches=completed,
            completed_episodes=len(rows),partial_outputs_sha256=artifacts));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','run','verify'])
    parser.add_argument('--base',type=Path,default=REPO/'paper_study/results/costed_pi_mle_v1')
    parser.add_argument('--out',type=Path,default=REPO/'paper_study/results/costed_pi_mle_resolution_v1')
    args=parser.parse_args();out=args.out.resolve()
    if args.stage=='freeze':freeze(args.base.resolve(),out)
    elif args.stage=='run':run(out)
    else:print('VERIFIED',verify(out)[1])


if __name__=='__main__':main()
