"""Exclusive source/input lock and execution for the separate bounded MLE."""
from pathlib import Path
import argparse
import csv
import json
import sys
import time
import os

import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(HERE.parent))
import evaluate
from bounded_mle import DEFAULTS,fit_initial_state,forecast

CHANNELS=('position_m','velocity_m_s','normal_N','friction_N')


def read(path):return json.loads(Path(path).read_text())


def check_particle_run(path):
    lock,lock_sha=evaluate.verify_lock(path)
    process=read(path/'process.json');manifest=read(path/'artifact_manifest.json')
    if process['status']!='complete' or process['rows']!=1296 or process['fit_batches']!=12:
        raise ValueError('Complete fixed-subset P4096 result required')
    if lock['config']['particles']!=4096 or lock['config']['calibration_noise']!=['noisy'] or lock['config']['forecast_starts']!=['filtered_particles']:
        raise ValueError('Use the declared noisy/filtered P4096 start source')
    if manifest['evaluation_lock_sha256']!=lock_sha:raise ValueError('Source particle manifest mismatch')
    evaluate.verify_hashes(manifest['outputs_sha256'])
    return lock,lock_sha


def freeze(particle,out):
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a fresh MLE output directory')
    if out==REPO/'paper_study/results' or not out.is_relative_to(REPO/'paper_study/results'):
        raise ValueError('Use a fresh subdirectory of paper_study/results')
    parent,particle_lock_sha=check_particle_run(particle)
    convergence=read(particle.parent/'convergence_lock.json')
    evaluate.verify_hashes(convergence['evaluation_locks_sha256'])
    # Capture all inference dependencies/model weights as well as the new files.
    sources=dict(parent['sources_sha256'])
    for path in (HERE.parent/'bounded_mle.py',HERE.parent/'test_bounded_mle.py',Path(__file__).resolve(),HERE/'mle_protocol.md'):
        sources[str(path)]=evaluate.sha(path)
    inputs=dict(parent['input_sha256'])
    for name in ('evaluation_lock.json','process.json','artifact_manifest.json'):
        inputs[str(particle/name)]=evaluate.sha(particle/name)
    inputs[str(particle.parent/'convergence_lock.json')]=evaluate.sha(particle.parent/'convergence_lock.json')
    for relative in parent['selected_batches']:
        for suffix in ('.npz','.json'):
            path=(particle/'predictions/noisy'/relative).with_suffix(suffix)
            inputs[str(path)]=evaluate.sha(path)
    evaluate.verify_hashes(sources);evaluate.verify_hashes(inputs)
    out.mkdir(parents=True,exist_ok=True)
    evaluate.write_json(out/'mle_lock.json',dict(role='Separate exploratory bounded MLE, before native-data state fits',
        config=dict(DEFAULTS),source_particle_directory=str(particle),particle_lock_sha256=particle_lock_sha,
        data_directory=parent['data_directory'],selected_batches=parent['selected_batches'],
        sources_sha256=sources,inputs_sha256=inputs,generation_sources_sha256=parent['generation_sources_sha256'],
        pi_model_seed=61001,normal_model_seed=1,calibration_noise='noisy',forecast_start='filtered_particles',
        source_particles=4096,starting_artifact_keys=['prior_states','weights'],expected_episodes=432,
        command=sys.argv,created_unix=time.time()),exclusive=True)
    evaluate.write_json(out/'process.json',dict(status='locked',lock_sha256=evaluate.sha(out/'mle_lock.json')),exclusive=True)
    print('MLE LOCKED',out,evaluate.sha(out/'mle_lock.json'),flush=True)


def verify(out):
    path=out/'mle_lock.json';lock=read(path)
    for key in ('sources_sha256','inputs_sha256','generation_sources_sha256'):evaluate.verify_hashes(lock[key])
    return lock,evaluate.sha(path)


def run(out):
    lock,lock_sha=verify(out)
    previous=read(out/'process.json')
    if previous!={'status':'locked','lock_sha256':lock_sha}:raise ValueError('Run requires an unused MLE lock')
    started=time.perf_counter();rows=[];artifacts={};completed=0;active_batch=None
    process=dict(status='running',pid=os.getpid(),started_unix=time.time(),lock_sha256=lock_sha,command=sys.argv)
    evaluate.write_json(out/'process.json',process)
    try:
        for relative in lock['selected_batches']:
            active_batch=relative
            native=Path(lock['data_directory'])/relative;meta=read(native.with_suffix('.json'))
            with np.load(native,allow_pickle=False) as stored:
                raw={key:stored[key] for key in (*evaluate.PUBLIC_KEYS,'observations','observation_steps','probe_steps')}
            inputs=evaluate.observable_inputs(raw)
            public_meta={key:meta[key] for key in evaluate.PUBLIC_META}
            source=Path(lock['source_particle_directory'])/'predictions/noisy'/relative
            source_meta=read(source.with_suffix('.json'))
            if source_meta['input_npz_sha256']!=evaluate.sha(native):raise ValueError('Starting posterior batch mismatch')
            with np.load(source,allow_pickle=False) as stored:
                # Only past-derived initial hypotheses and weights are loaded.
                prior_states=stored['prior_states'].copy();weights=stored['weights'].copy()
            model=evaluate.load_model(meta['friction_world'],meta['normal_world'],'noisy',
                pi_seed=lock['pi_model_seed'],normal_seed=lock['normal_model_seed'],dt=meta['dt'],mass=meta['mass_kg'],iterations=40)
            tic=time.perf_counter()
            fitted=fit_initial_state(model,inputs,public_meta,prior_states,weights,config=lock['config'])
            fit_seconds=time.perf_counter()-tic
            tic=time.perf_counter()
            predictions,solver=forecast(model,inputs,fitted,starts=('filtered_particles',),return_diagnostics=True)
            forecast_seconds=time.perf_counter()-tic
            prediction=predictions['filtered_particles/bounded_initial_state_mle']
            # Native force/state targets are introduced only after the forecast.
            with np.load(native,allow_pickle=False) as stored:
                end=inputs['probe_steps']
                target=np.concatenate((stored['next_state'][:,end:],stored['normal'][:,end:,None],stored['friction'][:,end:,None]),axis=2)
                guards=stored['guard'].sum(axis=1)
            error=prediction-target
            if not np.isfinite(error).all():raise FloatingPointError('Nonfinite MLE scoring target/output')
            rmse=np.sqrt(np.mean(error**2,axis=1));bias=np.mean(error,axis=1)
            batch=[]
            for i,factor in enumerate(meta['factor_rows']):
                diagnostic=fitted['diagnostics'][i]
                row=dict(normal_world=meta['normal_world'],friction_world=meta['friction_world'],
                    probe=meta['probe']['name'],episode=i,episode_id=factor['episode_id'],profile_index=factor['profile_index'],
                    estimator='bounded_initial_state_mle',calibration_noise='noisy',sensor=meta['sensor'],
                    forecast_start='filtered_particles',starting_posterior_particles=4096,
                    optimizer_status=diagnostic['status'],optimizer_success=diagnostic['solver_success'],
                    budget_exhausted=diagnostic['budget_exhausted'],optimizer_nfev=diagnostic['optimizer_nfev'],
                    residual_calls=diagnostic['residual_calls'],fit_seconds=diagnostic['seconds'],
                    initial_objective=diagnostic['initial_objective'],best_objective=diagnostic['best_objective'],
                    native_guard_count=int(guards[i]))
                for c,name in enumerate(CHANNELS):row[f'rmse_{name}']=float(rmse[i,c]);row[f'bias_{name}']=float(bias[i,c])
                batch.append(row)
            destination=out/'predictions'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as stream:
                np.savez_compressed(stream,initial_pi_state=fitted['initial_pi_state'],normalized_state=fitted['normalized_state'],
                    boundary_xv=fitted['boundary_xv'],boundary_pi_state=fitted['boundary_state']['pi'],
                    boundary_normal_state=fitted['boundary_state']['normal'],forecast=prediction)
            evaluate.write_json(destination.with_suffix('.json'),dict(status='complete',lock_sha256=lock_sha,
                input_npz_sha256=evaluate.sha(native),source_posterior_npz_sha256=evaluate.sha(source),
                source_particle_inference_plus_forecast_seconds=source_meta['inference_forecast_seconds'],
                incremental_fit_seconds=fit_seconds,incremental_forecast_seconds=forecast_seconds,
                npz_sha256=evaluate.sha(destination),diagnostics=fitted['diagnostics'],solver=solver,metrics=batch),exclusive=True)
            for path in (destination,destination.with_suffix('.json')):artifacts[str(path)]=evaluate.sha(path)
            rows.extend(batch);completed+=1
            print(f'MLE {completed}/12 {relative}',flush=True)
        if len(rows)!=lock['expected_episodes']:raise ValueError('Incomplete MLE episode inventory')
        with (out/'episodes.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        artifacts[str(out/'episodes.csv')]=evaluate.sha(out/'episodes.csv')
        if verify(out)[1]!=lock_sha:raise ValueError('MLE lock changed during run')
        evaluate.write_json(out/'artifact_manifest.json',dict(lock_sha256=lock_sha,outputs_sha256=artifacts),exclusive=True)
        evaluate.write_json(out/'process.json',dict(**{k:v for k,v in process.items() if k!='status'},status='complete',
            seconds=time.perf_counter()-started,ended_unix=time.time(),batches=completed,episodes=len(rows),
            budget_exhausted_episodes=sum(row['budget_exhausted'] for row in rows),
            optimizer_success_episodes=sum(row['optimizer_success'] for row in rows)))
    except BaseException as exc:
        evaluate.write_json(out/'process.json',dict(**{k:v for k,v in process.items() if k!='status'},status='failed',
            seconds=time.perf_counter()-started,error=repr(exc),failed_batch=active_batch,
            completed_batches=completed,completed_episodes=len(rows),
            partial_outputs_sha256=artifacts))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','run','verify'])
    parser.add_argument('--particle',type=Path,default=REPO/'paper_study/results/costed_pi_convergence_v1/p4096')
    parser.add_argument('--out',type=Path,default=REPO/'paper_study/results/costed_pi_mle_v1')
    args=parser.parse_args();out=args.out.resolve()
    if args.stage=='freeze':freeze(args.particle.resolve(),out)
    elif args.stage=='run':run(out)
    else:print('MLE VERIFIED',verify(out)[1])


if __name__=='__main__':main()
