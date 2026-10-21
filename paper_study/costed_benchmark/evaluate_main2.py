"""Prospectively authorized held-out Main2 evaluator; no reference imports.

This is a new version, leaving the frozen development evaluator untouched.
The test authorization must bind this source, config and all three fitted banks
before any native test input is opened. `freeze` requires complete generation,
creates public fitted-only priors and freezes all evaluation inputs. Three bank
workers have independent exclusive output directories. P1024 is a fixed-budget
finite-prior approximation, not a converged/calibrated Bayesian guarantee.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/paper-study-numba')
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,csv,json,time
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(HERE))
import evaluate as development
from inference import load_model,model_paths,build_prior
from bounded_mle import fit_initial_state,forecast as mle_forecast

METHODS=('prior_propagation','trajectory_assimilation','privileged_full_history','bounded_mle_start1','bounded_mle_two_start')
WORLDS=('maxwell','boucwen')
CASES=('balanced','wide_scales','compliant','stiff_high_capacity')
BANKS=[dict(bank=1,pi_seed=61001,normal_seed=1),dict(bank=2,pi_seed=61002,normal_seed=2),dict(bank=3,pi_seed=61003,normal_seed=3)]
CHANNELS=('position_m','velocity_m_s','normal_N','friction_N')


def read(path):return json.loads(Path(path).read_text())
sha=development.sha
write_json=development.write_json
verify_hashes=development.verify_hashes


def validate_config(cfg):
    expected=dict(particles=1024,prior_seed=43001,prior_duration_s=4.,prior_waypoint_limit_m=.025,
        prior_profile_start=0,calibration_noise=['noisy'],forecast_starts=['filtered_particles'],
        numerical_floor=[1e-6,1e-5,1e-4],solver_iterations=40,max_solver_residual_N=1e-7,banks=BANKS,
        methods=list(METHODS),mle_starts=['best causal P1024 initial-state particle','fixed q=0'],
        mle_selection='minimum past objective; first start wins ties')
    for key,value in expected.items():
        if cfg.get(key)!=value:raise ValueError(f'Main2 fixed evaluation mismatch: {key}')
    mle=dict(max_nfev=120,max_residual_calls=512,max_seconds_per_episode=6.,
        ftol=1e-8,xtol=1e-8,gtol=1e-8,numerical_floor=[1e-6,1e-5,1e-4],max_solver_residual_N=1e-7)
    if cfg.get('mle')!=mle:raise ValueError('Main2 fixed MLE budget mismatch')
    if not np.isfinite(cfg['max_wall_seconds_per_bank']) or cfg['max_wall_seconds_per_bank']<=0:
        raise ValueError('Positive finite bank wall budget required')


def source_paths(config_path=HERE/'main2_evaluation.json'):
    """Include these files in the global prospective Main2 sources_sha256."""
    cfg=read(config_path);validate_config(cfg)
    paths=[Path(__file__).resolve(),Path(config_path).resolve(),HERE/'main2_native.json',HERE/'main2_protocol.md',
        HERE/'bounded_mle.py',*development.source_paths()]
    for bank in cfg['banks']:
        for normal in WORLDS:
            for case in CASES:
                paths.extend(model_paths(case,normal,'noisy',bank['pi_seed'],bank['normal_seed']))
    return sorted(set(path.resolve() for path in paths))


def verify_authorization(test_lock,config_path=HERE/'main2_evaluation.json'):
    """Called before opening native generation manifests, samples or metadata."""
    test_lock=Path(test_lock).resolve();cfg=read(config_path);validate_config(cfg)
    lock=read(test_lock)
    if (lock.get('role')!='paper_study_main2_test_authorization' or lock.get('status')!='locked'
        or lock.get('schema_version')!=1 or lock.get('split')!='test'
        or lock.get('config_sha256')!=sha(HERE/'main2_native.json')
        or lock.get('preparation_seeds')!=[53201,53202,53203,53204]):
        raise ValueError('Prospective Main2 authorization mismatch')
    sources=lock.get('sources_sha256',{})
    for path in source_paths(config_path):
        if sources.get(str(path))!=sha(path):raise ValueError(f'Prospective lock omits/mismatches evaluation dependency: {path}')
    verify_hashes(sources)
    return lock,sha(test_lock),cfg


def native_inventory(data,authorization,authorization_sha):
    """Exact complete held-out grid; never silently omit an incomplete batch."""
    data=Path(data).resolve();generation=read(data/'generation_lock.json');process=read(data/'process.json')
    native_cfg=read(HERE/'main2_native.json')
    if (generation.get('test_lock_sha256')!=authorization_sha or generation.get('test_sources_sha256')!=authorization['sources_sha256']
        or generation.get('config')!=native_cfg or generation.get('smoke') is not False):
        raise ValueError('Native generation is not bound to this prospective Main2 authorization')
    verify_hashes(generation['sources_sha256']);verify_hashes(generation['test_sources_sha256'])
    if process.get('status')!='complete' or process.get('batches')!=96 or process.get('episodes')!=3456:
        raise ValueError('Complete96-batch/3456-episode native Main2 required')
    expected={f'p{seed}/{normal}_{case}/{probe}/noisy.npz' for seed in (53201,53202,53203,53204)
        for normal in WORLDS for case in CASES for probe in ('hold_150','gentle_150','strong_300')}
    actual={str(path.relative_to(data)) for path in data.rglob('*.npz')}
    if actual!=expected:raise ValueError(f'Main2 grid mismatch; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
    metadata={}
    for relative in sorted(expected):
        path=data/relative;meta=read(path.with_suffix('.json'))
        if meta.get('status')!='complete' or meta.get('split')!='test' or sha(path)!=meta['npz_sha256']:
            raise ValueError(f'Incomplete/mismatched Main2 batch: {relative}')
        identity=f"p{meta['preparation_seed']}/{meta['normal_world']}_{meta['friction_world']}/{meta['probe']['name']}/{meta['sensor']}.npz"
        if identity!=relative or meta['friction_family']!='three_positive_stops':raise ValueError('Native identity mismatch')
        if meta['dt']!=.0025 or meta['mass_kg']!=2. or meta['initial_compression_m']!=.02 or meta['future_seconds']!=.4:
            raise ValueError('Main2 timing/material initialization contract mismatch; no implicit prefix refinement allowed')
        steps=round((meta['probe']['duration_s']+.4)/.0025)
        if meta['count']!=36 or meta['expected_steps']!=steps or meta['recorded_steps']!=steps or len(meta['factor_rows'])!=36:
            raise ValueError('Main2 batch completeness mismatch')
        declared_probe=next(p for p in native_cfg['probes'] if p['name']==meta['probe']['name'])
        if meta['probe']!=declared_probe or meta['sensor_parameters']!=native_cfg['sensor_conditions']['noisy']:
            raise ValueError('Main2 recorded probe/sensor configuration mismatch')
        expected_factors=[(h,g,s) for h in range(6) for g in range(3) for s in (-1,1)]
        actual_factors=[(r['profile_index'],r['geometry'],r['excitation_sign']) for r in meta['factor_rows']]
        if actual_factors!=expected_factors or any(r['preparation_seed']!=meta['preparation_seed'] or r['amplitude']!=.003
            or r['episode_id']!=f"test-p{meta['preparation_seed']}-h{r['profile_index']}-g{r['geometry']}-s{r['excitation_sign']:+d}"
            for r in meta['factor_rows']):raise ValueError('Main2 factor inventory/order mismatch')
        with np.load(path,allow_pickle=False) as arrays:
            validate_native_arrays(arrays,meta,native_cfg)
        metadata[relative]=meta
    return generation,metadata


def validate_native_arrays(arrays,meta,native_cfg):
    """Check raw public/scoring shapes and factor clocks against the sidecar.

    This does not inspect latent stop states or preparation commands. It runs
    only after prospective authorization, complete generation and the NPZ hash.
    """
    count=meta['count'];steps=meta['expected_steps']
    probe_steps=round(meta['probe']['duration_s']/meta['dt'])
    observation_steps=np.array(sorted(set(range(0,steps+1,native_cfg['observation_steps']))|{probe_steps,steps}))
    if (np.asarray(arrays['probe_steps']).shape!=() or np.asarray(arrays['probe_steps']).dtype.kind not in 'iu'
        or int(arrays['probe_steps'])!=probe_steps):raise ValueError('Native raw probe clock mismatch')
    observed_steps=np.asarray(arrays['observation_steps'])
    if observed_steps.dtype.kind not in 'iu' or not np.array_equal(observed_steps,observation_steps):
        raise ValueError('Native raw observation clock mismatch')
    for name in ('amplitude','geometry','excitation_sign'):
        actual=np.asarray(arrays[name]);expected=np.array([r[name] for r in meta['factor_rows']])
        if actual.shape!=(count,) or not np.array_equal(actual,expected):raise ValueError(f'Native raw factor mismatch: {name}')
    if np.asarray(arrays['geometry']).dtype.kind not in 'iu':raise ValueError('Native raw geometry must be integer')
    shapes={'state':(count,steps,2),'next_state':(count,steps,2),'observations':(count,len(observation_steps),3),
        **{name:(count,steps) for name in ('u','d','d_dot','normal','friction','guard','actuator_work_J','normal_drive_work_discrete_J')}}
    for name,shape in shapes.items():
        actual=np.asarray(arrays[name])
        if actual.shape!=shape or not np.isfinite(actual).all():raise ValueError(f'Native raw shape/nonfinite mismatch: {name}')


def freeze(data,out,test_lock,config_path=HERE/'main2_evaluation.json'):
    data,out,test_lock=Path(data).resolve(),Path(out).resolve(),Path(test_lock).resolve()
    authorization,authorization_sha,cfg=verify_authorization(test_lock,config_path)
    # No native data has been touched before the authorization check above.
    generation,metadata=native_inventory(data,authorization,authorization_sha)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a fresh Main2 evaluation directory')
    if out==REPO/'paper_study/results' or not out.is_relative_to(REPO/'paper_study/results'):
        raise ValueError('Use a fresh subdirectory of paper_study/results')
    out.mkdir(parents=True,exist_ok=True)
    banks={}
    for bank in cfg['banks']:
        b=bank['bank'];banks[b]={}
        for normal in WORLDS:
            for case in CASES:
                model=load_model(case,normal,'noisy',pi_seed=bank['pi_seed'],normal_seed=bank['normal_seed'],iterations=40)
                states,meta=build_prior(model,particles=1024,seed=43001,profile_start=0,duration=4.,waypoint_limit=.025)
                relative=f'priors/bank{b}/{normal}_{case}.npz';path=out/relative;path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as stream:np.savez_compressed(stream,states=states,weights=np.full(1024,1/1024))
                write_json(path.with_suffix('.json'),dict(**meta,bank=bank,normal_world=normal,friction_world=case,npz_sha256=sha(path)),exclusive=True)
                banks[b][f'{normal}/{case}']=relative
    inputs={str(data/name):sha(data/name) for relative in metadata for name in (relative,str(Path(relative).with_suffix('.json')))}
    for name in ('generation_lock.json','process.json'):inputs[str(data/name)]=sha(data/name)
    inputs[str(test_lock)]=authorization_sha
    priors={str(path):sha(path) for path in (out/'priors').rglob('*') if path.is_file()}
    verify_hashes(authorization['sources_sha256']);verify_hashes(inputs);verify_hashes(priors)
    write_json(out/'evaluation_lock.json',dict(role='Main2 held-out descriptive evaluator',version=1,config=cfg,
        config_path=str(Path(config_path).resolve()),test_authorization_path=str(test_lock),test_authorization_sha256=authorization_sha,
        data_directory=str(data),selected_batches=list(metadata),sources_sha256=authorization['sources_sha256'],
        input_sha256=inputs,prior_sha256=priors,prior_banks=banks,generation_sources_sha256=generation['sources_sha256'],
        expected_episodes_per_bank=3456,expected_rows_per_bank=17280,expected_total_rows=51840,
        command=sys.argv,created_unix=time.time()),exclusive=True)
    digest=sha(out/'evaluation_lock.json')
    for bank in cfg['banks']:
        directory=out/f"bank{bank['bank']}";directory.mkdir()
        write_json(directory/'process.json',dict(status='locked',evaluation_lock_sha256=digest),exclusive=True)
    print('MAIN2 EVALUATION LOCKED',out,digest,flush=True)


def verify(out):
    out=Path(out).resolve();path=out/'evaluation_lock.json';lock=read(path)
    _,authorization_sha,_=verify_authorization(lock['test_authorization_path'],lock['config_path'])
    if authorization_sha!=lock['test_authorization_sha256']:raise ValueError('Test authorization changed')
    for key in ('sources_sha256','input_sha256','prior_sha256','generation_sources_sha256'):verify_hashes(lock[key])
    validate_config(lock['config'])
    return lock,sha(path)


def two_start_mle(model,inputs,public_meta,states,weights,cfg):
    """Choice uses past likelihood only; save both single and selected futures."""
    count=len(weights)
    candidates=[fit_initial_state(model,inputs,public_meta,states,weights,config=cfg['mle']),
                fit_initial_state(model,inputs,public_meta,np.zeros((1,3)),np.ones((count,1)),config=cfg['mle'])]
    objective=np.array([[d['best_objective'] for d in c['diagnostics']] for c in candidates]).T
    choice=np.argmin(objective,axis=1)  # determined before any future propagation
    future=[];solver=[]
    for candidate in candidates:
        predictions,checks=mle_forecast(model,inputs,candidate,return_diagnostics=True)
        future.append(predictions['filtered_particles/bounded_initial_state_mle']);solver.append(checks)
    candidate_forecast=np.stack(future,axis=1)
    means={'bounded_mle_start1':candidate_forecast[:,0],
           'bounded_mle_two_start':candidate_forecast[np.arange(count),choice]}
    details=dict(mle_candidate_forecasts=candidate_forecast,mle_candidate_objectives=objective,mle_selected_start=choice,
        mle_candidate_initial_pi=np.stack([c['initial_pi_state'] for c in candidates],axis=1),
        mle_candidate_boundary_xv=np.stack([c['boundary_xv'] for c in candidates],axis=1),
        mle_candidate_boundary_pi=np.stack([c['boundary_state']['pi'] for c in candidates],axis=1),
        mle_candidate_boundary_normal=np.stack([c['boundary_state']['normal'] for c in candidates],axis=1))
    return means,details,[c['diagnostics'] for c in candidates],solver


def score(predictions,truth,metadata,bank,detail,mle_diagnostics):
    """Per-episode MSE is retained so the primary pooled loss is unambiguous."""
    end=int(truth['probe_steps']);target=np.concatenate((truth['next_state'][:,end:],truth['normal'][:,end:,None],truth['friction'][:,end:,None]),axis=2)
    if not np.isfinite(target).all():raise FloatingPointError('Nonfinite scoring target; batch retained as failed')
    if set(predictions)!=set(METHODS):raise ValueError('Main2 requires all five methods')
    diagnostic=development.weight_diagnostics(detail['weights']);rows=[]
    for method,prediction in predictions.items():
        if prediction.shape!=target.shape or not np.isfinite(prediction).all():raise FloatingPointError('Invalid/nonfinite forecast; no dropping')
        error=prediction-target;mse=np.mean(error**2,axis=1);bias=np.mean(error,axis=1)
        for i,factor in enumerate(metadata['factor_rows']):
            row=dict(bank=bank['bank'],pi_model_seed=bank['pi_seed'],normal_model_seed=bank['normal_seed'],calibration_noise='noisy',
                normal_world=metadata['normal_world'],friction_world=metadata['friction_world'],preparation_seed=metadata['preparation_seed'],
                probe=metadata['probe']['name'],probe_duration_s=metadata['probe']['duration_s'],sensor='noisy',episode=i,
                episode_id=factor['episode_id'],profile_index=factor['profile_index'],geometry=int(truth['geometry'][i]),
                excitation_sign=factor['excitation_sign'],method=method,forecast_start='filtered_particles',
                future_steps=target.shape[1],native_guard_count=int(truth['guard'][i].sum()),
                posterior_particle_ess=float(diagnostic['ess'][i]),posterior_particle_max_weight=float(diagnostic['max_weight'][i]),
                posterior_particle_entropy=float(diagnostic['entropy'][i]),mle_selected_start=int(detail['mle_selected_start'][i]),
                mle_start1_objective=mle_diagnostics[0][i]['best_objective'],mle_start2_objective=mle_diagnostics[1][i]['best_objective'],
                mle_start1_budget_exhausted=mle_diagnostics[0][i]['budget_exhausted'],mle_start2_budget_exhausted=mle_diagnostics[1][i]['budget_exhausted'],
                mle_start1_optimizer_success=mle_diagnostics[0][i]['solver_success'],mle_start2_optimizer_success=mle_diagnostics[1][i]['solver_success'])
            for c,name in enumerate(CHANNELS):
                row[f'mse_{name}']=float(mse[i,c]);row[f'rmse_{name}']=float(np.sqrt(mse[i,c]));row[f'bias_{name}']=float(bias[i,c])
            for name in ('actuator_work_J','normal_drive_work_discrete_J'):
                values=truth[name][i,:end]
                row['probe_signed_'+name]=float(np.sum(values));row['probe_absolute_'+name]=float(np.sum(abs(values)))
            rows.append(row)
    return rows


def reuse_batch(destination,lock_sha,native_sha):
    metadata=read(destination.with_suffix('.json'))
    if (metadata.get('status')!='complete' or metadata.get('evaluation_lock_sha256')!=lock_sha
        or metadata.get('input_npz_sha256')!=native_sha or metadata.get('npz_sha256')!=sha(destination)
        or len(metadata.get('metrics',[]))!=180):raise ValueError('Existing bank batch fails locked resume validation')
    return metadata['metrics'],metadata


def preserve_failed_attempt(directory,previous):
    """Retain the full previous process before resume can overwrite it."""
    history=dict(previous.get('prior_attempts_sha256',{}));verify_hashes(history)
    if previous.get('status')!='failed':raise ValueError('Only failed attempts may be archived for resume')
    path=Path(directory)/'attempts'/f'attempt_{len(history)+1:03d}.json'
    path.parent.mkdir(exist_ok=True)
    write_json(path,previous,exclusive=True)
    history[str(path)]=sha(path)
    return history


def run(out,bank_id,*,resume=False):
    out=Path(out).resolve();lock,lock_sha=verify(out);cfg=lock['config']
    bank=next((b for b in cfg['banks'] if b['bank']==bank_id),None)
    if bank is None:raise ValueError('Unknown calibration bank')
    directory=out/f'bank{bank_id}';previous=read(directory/'process.json')
    if previous.get('evaluation_lock_sha256')!=lock_sha:raise ValueError('Bank output lock mismatch')
    if previous.get('status')!='locked' and not(resume and previous.get('status')=='failed'):
        raise ValueError('Use an unused locked bank or explicit resume of a failed bank; running/complete banks are preserved')
    prior_outputs=previous.get('partial_outputs_sha256',{}) if resume else {}
    if resume:verify_hashes(prior_outputs)
    claim=directory/'.worker_claim'
    with claim.open('x') as stream:stream.write(str(os.getpid()))
    history={}
    try:
        if resume:history=preserve_failed_attempt(directory,previous)
    except BaseException:
        claim.unlink();raise
    started=time.perf_counter();rows=[];outputs=dict(history);completed=0;active=None
    process=dict(status='running',pid=os.getpid(),command=sys.argv,started_unix=time.time(),evaluation_lock_sha256=lock_sha,bank=bank_id)
    process.update(attempt_index=len(history)+1,prior_attempts_sha256=history,
        prior_attempt_seconds=sum(read(path).get('seconds',0.) for path in history),
        seconds_scope='current attempt only; all previous failures/times remain in hashed attempt records')
    write_json(directory/'process.json',process)
    try:
        for relative in lock['selected_batches']:
            active=relative
            if time.perf_counter()-started>cfg['max_wall_seconds_per_bank']:raise TimeoutError('Locked Main2 bank wall budget exceeded')
            native=Path(lock['data_directory'])/relative;metadata=read(native.with_suffix('.json'))
            destination=directory/'predictions'/relative
            if destination.exists() or destination.with_suffix('.json').exists():
                if not resume or not destination.is_file() or not destination.with_suffix('.json').is_file():
                    raise FileExistsError('Incomplete or unrequested existing batch retained')
                if str(destination) not in prior_outputs or str(destination.with_suffix('.json')) not in prior_outputs:
                    raise ValueError('Existing batch is absent from the failed-run artifact manifest')
                batch,_=reuse_batch(destination,lock_sha,sha(native))
                rows.extend(batch)
                for path in (destination,destination.with_suffix('.json')):outputs[str(path)]=sha(path)
                completed+=1;continue
            with np.load(native,allow_pickle=False) as stored:
                public_raw={key:stored[key] for key in (*development.PUBLIC_KEYS,'observations','observation_steps','probe_steps')}
            inputs=development.observable_inputs(public_raw);public_meta={key:metadata[key] for key in development.PUBLIC_META}
            model=load_model(metadata['friction_world'],metadata['normal_world'],'noisy',pi_seed=bank['pi_seed'],normal_seed=bank['normal_seed'],iterations=40)
            prior_path=out/lock['prior_banks'][str(bank_id)][f"{metadata['normal_world']}/{metadata['friction_world']}"]
            with np.load(prior_path,allow_pickle=False) as stored:states,weights=stored['states'].copy(),stored['weights'].copy()
            tic=time.perf_counter();inferred,details=development.infer_and_forecast(model,inputs,public_meta,cfg,states,weights)
            particle_seconds=time.perf_counter()-tic
            predicted={name.split('/')[1]:value for name,value in inferred.items()}
            tic=time.perf_counter();mle,mle_detail,mle_diagnostics,mle_solver=two_start_mle(model,inputs,public_meta,states,details['weights'],cfg)
            mle_seconds=time.perf_counter()-tic;predicted.update(mle);details.update(mle_detail)
            for name,value in mle.items():details[f'filtered_particles_{name}_forecast']=value
            # Privileged branch has sole access to actual preparation commands.
            # Both its prefix and motion are fixed at .0025; reject refinements.
            if metadata['dt']!=model.dt or model.dt!=.0025:raise ValueError('Main2 privileged prefix/motion dt contract violated')
            tic=time.perf_counter();privileged,priv_detail=development.privileged_forecasts(model,inputs,metadata,cfg)
            privileged_seconds=time.perf_counter()-tic
            predicted.update({name.split('/')[1]:value for name,value in privileged.items()});details.update(priv_detail)
            if max(float(details['max_solver_residual_N']),float(details['privileged_max_solver_residual_N']))>cfg['max_solver_residual_N']:
                raise FloatingPointError('Main2 particle/privileged solver failure')
            with np.load(native,allow_pickle=False) as stored:
                truth={key:stored[key] for key in ('next_state','normal','friction','probe_steps','geometry','guard','actuator_work_J','normal_drive_work_discrete_J')}
            batch=score(predicted,truth,metadata,bank,details,mle_diagnostics)
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as stream:np.savez_compressed(stream,**details)
            write_json(destination.with_suffix('.json'),dict(status='complete',evaluation_lock_sha256=lock_sha,bank=bank,
                input_npz_sha256=sha(native),input_metadata_sha256=sha(native.with_suffix('.json')),prior_bank_sha256=sha(prior_path),
                npz_sha256=sha(destination),metrics=batch,mle_diagnostics=mle_diagnostics,mle_solver=mle_solver,
                particle_filter_plus_forecast_seconds=particle_seconds,mle_two_start_incremental_seconds=mle_seconds,
                privileged_replay_plus_forecast_seconds=privileged_seconds,
                cost_scope='Particle time includes filtering and both prior/posterior future moments. MLE shares that causal filter and adds two optimizations; start1 and selection share work.'),exclusive=True)
            for path in (destination,destination.with_suffix('.json')):outputs[str(path)]=sha(path)
            rows.extend(batch);completed+=1;print(f'MAIN2 bank{bank_id} {completed}/96 {relative}',flush=True)
        if len(rows)!=17280:raise ValueError('Incomplete Main2 bank metric inventory')
        with (directory/'episodes.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        outputs[str(directory/'episodes.csv')]=sha(directory/'episodes.csv')
        if verify(out)[1]!=lock_sha:raise ValueError('Main2 evaluation lock changed')
        write_json(directory/'artifact_manifest.json',dict(evaluation_lock_sha256=lock_sha,outputs_sha256=outputs),exclusive=True)
        unique=[r for r in rows if r['method']=='bounded_mle_two_start']
        write_json(directory/'process.json',dict(**{k:v for k,v in process.items() if k!='status'},status='complete',
            seconds=time.perf_counter()-started,ended_unix=time.time(),batches=completed,episodes=len(unique),rows=len(rows),
            optimizer_starts=2*len(unique),budget_cases=sum(r['mle_start1_budget_exhausted']+r['mle_start2_budget_exhausted'] for r in unique)))
    except BaseException as exc:
        write_json(directory/'process.json',dict(**{k:v for k,v in process.items() if k!='status'},status='failed',
            error=repr(exc),failed_batch=active,seconds=time.perf_counter()-started,completed_batches=completed,
            completed_rows=len(rows),partial_outputs_sha256=outputs));raise
    finally:
        claim.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('freeze');p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--test-lock',type=Path,required=True);p.add_argument('--config',type=Path,default=HERE/'main2_evaluation.json')
    p=sub.add_parser('run');p.add_argument('--out',type=Path,required=True);p.add_argument('--bank',type=int,choices=[1,2,3],required=True);p.add_argument('--resume',action='store_true')
    sub.add_parser('verify').add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='freeze':freeze(args.data,args.out,args.test_lock,args.config)
    elif args.command=='run':run(args.out,args.bank,resume=args.resume)
    else:print('MAIN2 EVALUATION VERIFIED',verify(args.out)[1])


if __name__=='__main__':main()
