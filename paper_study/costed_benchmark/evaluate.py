"""Locked exploratory evaluation of finite-prior three-stop state inference.

`freeze` verifies a complete native development run and captures sources, fitted
weights, independent public prior banks and every raw input before scoring.
`run` verifies that snapshot before/after evaluation. Neither stage fits weights.
Regular inference gets an explicit public-input whitelist. The separately named
privileged_full_history comparator replays hidden *commands* with fitted laws;
it is not an oracle or a bound on error. Predictive spread is finite-prior spread,
not a calibrated uncertainty interval. No native simulator is imported here.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/paper-study-numba')
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import time

import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(HERE))
from inference import load_model, model_paths, build_prior, ParticleObserver

STARTS=('filtered_particles','measured_start')
ESTIMATORS=('prior_propagation','trajectory_assimilation','privileged_full_history')
DEFAULT_CONFIG=dict(version='costed-pi-finite-prior-v1',particles=256,prior_seed=43001,
    prior_duration_s=4.,prior_waypoint_limit_m=.025,prior_profile_start=0,
    pi_model_seed=61001,normal_model_seed=1,calibration_noise=['clean','noisy'],
    forecast_starts=list(STARTS),numerical_floor=[1e-6,1e-5,1e-4],
    solver_iterations=40,max_solver_residual_N=1e-7,max_wall_seconds=7200.)
PUBLIC_KEYS=('u','d','d_dot','amplitude','geometry')
PUBLIC_META=('dt','mass_kg','initial_compression_m','sensor_parameters')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    return value


def write_json(path,value,*,exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as stream:
        stream.write(json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n')


def clone_state(state):
    return {k:np.array(v,copy=True) for k,v in state.items()}


def validate_config(cfg):
    for key in ('particles','prior_seed','pi_model_seed','normal_model_seed','solver_iterations'):
        value=cfg[key]
        if isinstance(value,bool) or int(value)!=value or value<1:raise ValueError(f'invalid {key}')
    if cfg['prior_seed']!=43001:
        raise ValueError('This development version fixes the independent public prior seed at 43001')
    if cfg['prior_profile_start']!=0 or cfg['prior_duration_s']!=4. or cfg['prior_waypoint_limit_m']!=.025:
        raise ValueError('Use the declared prefix-stable public history bank')
    if cfg['solver_iterations']!=40:raise ValueError('Use 40 bisections matching the native solver')
    for key,allowed in (('forecast_starts',STARTS),('calibration_noise',('clean','noisy'))):
        values=cfg[key]
        if not values or len(values)!=len(set(values)) or any(v not in allowed for v in values):
            raise ValueError(f'invalid {key}')
    floor=np.asarray(cfg['numerical_floor'])
    if floor.shape!=(3,) or not np.isfinite(floor).all() or np.any(floor<=0):raise ValueError('invalid numerical floor')
    for key in ('max_wall_seconds','max_solver_residual_N'):
        if not np.isfinite(cfg[key]) or cfg[key]<=0:raise ValueError(f'invalid {key}')


def observable_inputs(arrays):
    """Copy only commands, known geometry, and x/v/N observations at/before t0.

    Here t0 is the end of probing, after invoking the known future rate command.
    Thus boundary N may include its viscous response. It is not evidence for an
    earlier stop/continue decision and is never used as the historical clamp N.
    True x/v, forces, latent states, prefix IDs/seed and future sensors are absent.
    """
    end=int(arrays['probe_steps'])
    times=np.asarray(arrays['observation_steps'])
    if times.ndim!=1 or not np.issubdtype(times.dtype,np.integer):
        raise ValueError('integer observation steps required')
    if len(times)==0 or times[0]!=0 or np.any(np.diff(times)<=0):
        raise ValueError('unique ordered observation steps required')
    mask=times<=end
    result={k:np.array(arrays[k],copy=True) for k in PUBLIC_KEYS}
    result.update(probe_steps=end,observation_steps=times[mask].copy(),
                  observations=np.asarray(arrays['observations'])[:,mask].copy())
    if result['u'].ndim!=2:raise ValueError('episode/time command arrays required')
    count,total=result['u'].shape
    if count<1 or not 0<=end<total:raise ValueError('positive episode count and forecast horizon required')
    if not len(result['observation_steps']) or result['observation_steps'][-1]!=end:
        raise ValueError('probe-boundary observation required')
    if result['observations'].shape!=(count,len(result['observation_steps']),3):
        raise ValueError('x/v/N observation shape mismatch')
    if any(result[k].shape!=(count,total) for k in ('d','d_dot')):
        raise ValueError('command shape mismatch')
    if any(result[k].shape!=(count,) for k in ('amplitude','geometry')):
        raise ValueError('geometry/amplitude shape mismatch')
    if not np.issubdtype(result['geometry'].dtype,np.integer) or np.any((result['geometry']<0)|(result['geometry']>2)):
        raise ValueError('geometry must select 0=tanh, 1=sin, 2=bump')
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError('nonfinite public inputs')
    return result


def weight_diagnostics(weight):
    weight=np.asarray(weight)
    log=np.zeros_like(weight)
    np.log(weight,out=log,where=weight>0)
    return dict(ess=1./np.sum(weight*weight,axis=-1),entropy=-np.sum(weight*log,axis=-1),
                max_weight=np.max(weight,axis=-1),positive_particles=np.sum(weight>0,axis=-1))


def mechanical_start(xv,observed,start):
    if start=='filtered_particles':return np.array(xv,copy=True).reshape(-1,2)
    if start=='measured_start':return np.repeat(observed,xv.shape[1],axis=0)
    raise ValueError(start)


def forecast_moments(model,xv,state,inputs,weights):
    """One shared particle propagation; aggregate all weighting rules per step.

    No E*P*T trajectory tensor is retained. This permits a fixed 4096-particle
    discretization check without changing the physical propagation equations.
    """
    count,total=inputs['u'].shape;end=inputs['probe_steps']
    particles=len(xv)//count
    if particles<1 or len(xv)!=count*particles:raise ValueError('invalid forecast particle shape')
    means={name:np.empty((count,total-end,4)) for name in weights}
    variances={name:np.empty((count,total-end,4)) for name in weights}
    xv=np.array(xv,copy=True);state=clone_state(state);max_residual=0.
    amp=np.repeat(inputs['amplitude'],particles);geom=np.repeat(inputs['geometry'],particles)
    for j,t in enumerate(range(end,total)):
        commands=[np.repeat(inputs[key][:,t],particles) for key in ('u','d','d_dot')]
        xv,state,N,Ft,residual=model.step(xv,state,*commands,amp,geom)
        max_residual=max(max_residual,float(np.max(abs(residual))))
        values=np.column_stack((xv,N,Ft)).reshape(count,particles,4)
        for name,weight in weights.items():
            if weight.shape!=(count,particles):raise ValueError('forecast weight shape mismatch')
            mean=np.einsum('ep,epk->ek',weight,values)
            variance=np.einsum('ep,epk->ek',weight,(values-mean[:,None,:])**2)
            means[name][:,j]=mean;variances[name][:,j]=variance
    return means,variances,max_residual


def infer_and_forecast(model,inputs,public_meta,cfg,prior_states,prior=None):
    """No full metadata permitted: only PUBLIC_META and observable_inputs."""
    if set(public_meta)!=set(PUBLIC_META):raise ValueError('regular inference metadata whitelist violated')
    count=inputs['u'].shape[0]
    sensor=public_meta['sensor_parameters']
    sigma=[sensor[key] for key in ('sigma_x_m','sigma_v_m_s','sigma_normal_N')]
    observer=ParticleObserver(model,prior_states,np.zeros((count,2)),
        initial_delta=public_meta['initial_compression_m'],sensor_sigma=sigma,
        numerical_floor=cfg['numerical_floor'],prior=prior)
    indices={int(t):i for i,t in enumerate(inputs['observation_steps'])}
    trace=[observer.weights.copy()]
    for t in range(inputs['probe_steps']):
        observer.advance(*(inputs[k][:,t] for k in ('u','d','d_dot')),
                         inputs['amplitude'],inputs['geometry'])
        if t+1 in indices:
            sample=inputs['observations'][:,indices[t+1]]
            observer.observe(sample[:,0],sample[:,1],sample[:,2],
                d=inputs['d'][:,t+1],d_dot=inputs['d_dot'][:,t+1],
                amplitude=inputs['amplitude'],geometry=inputs['geometry'])
            trace.append(observer.weights.copy())
    snapshot=observer.snapshot()
    weight_trace=np.stack(trace,axis=1)
    details=dict(prior_states=np.array(prior_states,copy=True),weights=snapshot['weights'],
        prior=snapshot['prior'],weight_trace=weight_trace,log_likelihood=snapshot['log_likelihood'],
        observation_steps=inputs['observation_steps'].copy(),effective_sigma=snapshot['effective_sigma'],
        filtered_particle_xv=snapshot['xv'],boundary_observed_xv=inputs['observations'][:,-1,:2].copy(),
        boundary_pi_state=snapshot['state']['pi'].reshape(count,-1,3),
        boundary_normal_state=snapshot['state']['normal'].reshape(count,-1,3))
    details.update({'weight_trace_'+k:v for k,v in weight_diagnostics(weight_trace).items()})
    max_residual=snapshot['max_solver_residual'];predictions={}
    for start in cfg['forecast_starts']:
        xv=mechanical_start(snapshot['xv'],inputs['observations'][:,-1,:2],start)
        means,variances,residual=forecast_moments(model,xv,snapshot['state'],inputs,
            dict(prior_propagation=snapshot['prior'],trajectory_assimilation=snapshot['weights']))
        max_residual=max(max_residual,residual)
        for estimator,mean in means.items():
            name=f'{start}/{estimator}'
            predictions[name]=mean
            details[f'{start}_{estimator}_forecast']=mean
            details[f'{start}_{estimator}_variance']=variances[estimator]
    details['max_solver_residual_N']=np.array(max_residual)
    return predictions,details


def privileged_forecasts(model,inputs,metadata,cfg):
    """Known local command history, fitted parameters; never true active state.

    Unlike ordinary inference this comparator knows the actual preparation v/N
    sequence, including fixture load. It cannot pass this information to another
    estimator. The fresh normal module remains modeled. Parameter bias can make
    this comparator worse than observation-based inference; it is not an oracle.
    """
    commands=metadata['prepared_reference_audit']['commands']
    v,N=[np.asarray(commands[k],dtype=float) for k in ('v','normal')]
    count=inputs['u'].shape[0]
    if v.ndim!=2 or v.shape!=N.shape or len(v)!=count or v.shape[1]<2:
        raise ValueError('invalid privileged prefix commands')
    if not np.isfinite(v).all() or not np.isfinite(N).all() or np.any(N<0):
        raise ValueError('invalid privileged prefix inputs')
    pi=model.pi.initial(count)
    for t in range(v.shape[1]-1):
        _,pi=model.pi.step(pi,v[:,t],N[:,t],model.dt)
    state=model.initial(count,metadata['initial_compression_m'],pi)
    xv=np.zeros((count,2));max_residual=0.
    for t in range(inputs['probe_steps']):
        xv,state,_,_,residual=model.step(xv,state,*(inputs[k][:,t] for k in ('u','d','d_dot')),
                                      inputs['amplitude'],inputs['geometry'])
        max_residual=max(max_residual,float(np.max(abs(residual))))
    predictions={};details=dict(privileged_filtered_xv=xv.copy(),
        privileged_boundary_pi_state=state['pi'].copy(),privileged_boundary_normal_state=state['normal'].copy())
    for start in cfg['forecast_starts']:
        initial=mechanical_start(xv[:,None,:],inputs['observations'][:,-1,:2],start)
        means,_,residual=forecast_moments(model,initial,state,inputs,
                                        {'privileged_full_history':np.ones((count,1))})
        max_residual=max(max_residual,residual)
        mean=means['privileged_full_history']
        predictions[f'{start}/privileged_full_history']=mean
        details[f'{start}_privileged_full_history_forecast']=mean
    details['privileged_max_solver_residual_N']=np.array(max_residual)
    return predictions,details


def score_forecasts(predictions,truth,metadata,calibration_noise,cfg,details):
    """Truth introduced after all predictions; no selective episode dropping."""
    end=int(truth['probe_steps'])
    target=np.concatenate((truth['next_state'][:,end:],truth['normal'][:,end:,None],
                           truth['friction'][:,end:,None]),axis=2)
    if not np.isfinite(target).all():raise ValueError('nonfinite truth retained as failed batch')
    diagnostic=weight_diagnostics(details['weights'])
    rows=[]
    for method,prediction in predictions.items():
        if prediction.shape!=target.shape or not np.isfinite(prediction).all():
            raise ValueError('invalid forecast; retain failed batch')
        start,estimator=method.split('/')
        rmse=np.sqrt(np.mean((prediction-target)**2,axis=1))
        for i in range(len(target)):
            factor=metadata['factor_rows'][i]
            row=dict(normal_world=metadata['normal_world'],friction_world=metadata['friction_world'],
                preparation_seed=metadata['preparation_seed'],probe=metadata['probe']['name'],
                duration_s=metadata['probe']['duration_s'],sensor=metadata['sensor'],episode=i,
                episode_id=factor['episode_id'],profile_index=factor['profile_index'],
                geometry=int(truth['geometry'][i]),excitation_sign=factor['excitation_sign'],
                pi_model_seed=cfg['pi_model_seed'],normal_model_seed=cfg['normal_model_seed'],
                calibration_noise=calibration_noise,particles=cfg['particles'],prior_seed=cfg['prior_seed'],
                filter_kind='joint_sensor_likelihood',forecast_start=start,estimator=estimator,method=method,
                native_guard_count=int(truth['guard'][i].sum()),posterior_ess=float(diagnostic['ess'][i]),
                posterior_entropy=float(diagnostic['entropy'][i]),posterior_max_weight=float(diagnostic['max_weight'][i]))
            row.update(dict(zip(('future_position_rmse_m','future_velocity_rmse_m_s',
                'future_normal_rmse_N','future_friction_rmse_N'),map(float,rmse[i]))))
            rows.append(row)
    return rows


def source_paths():
    return [Path(__file__).resolve(),HERE/'inference.py',HERE/'geometry.py',HERE/'profiles.py',
        REPO/'research_next/src/state_id.py',REPO/'paper_study/multistate_learning/model.py',
        REPO/'paper_study/src/implicit_stop_coupling.py']


def verify_hashes(mapping):
    for path,expected in mapping.items():
        if not Path(path).is_file() or sha(path)!=expected:raise ValueError(f'locked file changed/missing: {path}')


def native_inventory(data):
    """Require exactly the complete declared development grid, even for a subset."""
    data=Path(data).resolve()
    generation=json.loads((data/'generation_lock.json').read_text())
    process=json.loads((data/'process.json').read_text())
    if process.get('status')!='complete':raise ValueError('native generation must be complete before evaluation')
    verify_hashes(generation['sources_sha256'])
    cfg=generation['config'];smoke=generation['smoke']
    seeds=cfg['preparation_seeds'][:1] if smoke else cfg['preparation_seeds']
    normals=['boucwen'] if smoke else cfg['normal_worlds']
    cases=['balanced'] if smoke else cfg['friction_parameter_sets']
    probes=['gentle_050'] if smoke else [p['name'] for p in cfg['probes']]
    sensors=['noisy'] if smoke else list(cfg['sensor_conditions'])
    expected={f'p{s}/{n}_{c}/{p}/{o}.npz' for s in seeds for n in normals for c in cases for p in probes for o in sensors}
    actual={str(p.relative_to(data)) for p in data.rglob('*.npz')}
    if actual!=expected:raise ValueError(f'native inventory mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
    count=cfg['episodes_per_batch']
    if process.get('batches')!=len(expected) or process.get('episodes')!=count*len(expected):
        raise ValueError('native process completeness counts differ from declared grid')
    metadata={}
    for relative in sorted(expected):
        path=data/relative
        meta=json.loads(path.with_suffix('.json').read_text())
        if meta.get('status')!='complete' or sha(path)!=meta['npz_sha256']:
            raise ValueError(f'incomplete/hash-mismatched batch: {relative}')
        if meta['count']!=count or meta['recorded_steps']!=meta['expected_steps'] or len(meta['factor_rows'])!=count:
            raise ValueError(f'incomplete batch dimensions: {relative}')
        expected_relative=f"p{meta['preparation_seed']}/{meta['normal_world']}_{meta['friction_world']}/{meta['probe']['name']}/{meta['sensor']}.npz"
        if expected_relative!=relative or meta['friction_family']!='three_positive_stops' or meta['split']!='dev':
            raise ValueError('batch identity mismatch')
        metadata[relative]=meta
    return generation,metadata


def freeze(data,out,cfg,*,batch_globs=None):
    validate_config(cfg)
    data,out=Path(data).resolve(),Path(out).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError('retain prior outputs; choose a fresh evaluation directory')
    if out==REPO/'paper_study/results' or not out.is_relative_to(REPO/'paper_study/results'):
        raise ValueError('evaluation output must be a fresh subdirectory of paper_study/results')
    generation,metadata=native_inventory(data)
    # Selection is fixed here, before scores. A glob is a named development
    # subset, not a permission to omit failed native batches from inventory.
    import fnmatch
    selected=[name for name in metadata if not batch_globs or any(fnmatch.fnmatchcase(name,pat) for pat in batch_globs)]
    if not selected:raise ValueError('empty predeclared batch selection')
    if cfg['prior_seed'] in generation['config']['preparation_seeds']:
        raise ValueError('public prior bank cannot use a native preparation seed')
    out.mkdir(parents=True,exist_ok=True)
    sources={str(p):sha(p) for p in source_paths()}
    banks={}
    for normal,case in sorted({(metadata[p]['normal_world'],metadata[p]['friction_world']) for p in selected}):
        for noise in cfg['calibration_noise']:
            pi_path,n_path=model_paths(case,normal,noise,cfg['pi_model_seed'],cfg['normal_model_seed'])
            sources.update({str(p):sha(p) for p in (pi_path,n_path)})
            model=load_model(case,normal,noise,pi_seed=cfg['pi_model_seed'],normal_seed=cfg['normal_model_seed'],
                dt=generation['config']['dt'],mass=generation['config']['mass_kg'],iterations=cfg['solver_iterations'])
            states,bank_meta=build_prior(model,particles=cfg['particles'],seed=cfg['prior_seed'],
                profile_start=cfg['prior_profile_start'],duration=cfg['prior_duration_s'],waypoint_limit=cfg['prior_waypoint_limit_m'])
            relative=f'priors/{normal}_{case}/{noise}.npz';path=out/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:np.savez_compressed(stream,states=states,weights=np.full(cfg['particles'],1/cfg['particles']))
            write_json(path.with_suffix('.json'),dict(**bank_meta,normal_world=normal,friction_world=case,
                calibration_noise=noise,pi_model_sha256=sha(pi_path),normal_model_sha256=sha(n_path),
                npz_sha256=sha(path)),exclusive=True)
            banks[f'{normal}/{case}/{noise}']=relative
    inputs={str(data/name):sha(data/name) for relative in metadata for name in (relative,str(Path(relative).with_suffix('.json')))}
    inputs.update({str(data/name):sha(data/name) for name in ('generation_lock.json','process.json')})
    bank_hashes={str(p):sha(p) for p in (out/'priors').rglob('*') if p.is_file()}
    verify_hashes(sources);verify_hashes(inputs);verify_hashes(generation['sources_sha256'])
    write_json(out/'evaluation_lock.json',dict(version=1,role='Exploratory snapshot before any task scoring',
        config=cfg,data_directory=str(data),selected_batches=selected,batch_globs=batch_globs,
        native_inventory_count=len(metadata),sources_sha256=sources,input_sha256=inputs,
        generation_sources_sha256=generation['sources_sha256'],prior_sha256=bank_hashes,prior_banks=banks,
        command=sys.argv,created_unix=time.time(),python=sys.executable,python_version=platform.python_version(),
        information=dict(regular='Only public commands/geometry and causal x/v/N; initial kinematics known zero. No measured N clamp.',
            initial_observation='Initial sample common to all hypotheses; not scored. Probe-none weights remain prior exactly.',
            boundary='Observation uses the already-invoked future d_dot; not usable for a previous stop/continue decision.',
            prior='Uniform mass on fixed independently sampled histories; duplicate states retain probability.',
            inference='Weights only; material and mechanical trajectories unchanged. No resampling or parameter changes.',
            spread='Finite-prior predictive spread; not calibrated posterior uncertainty.',
            privileged='Actual local v/N commands replayed through fitted PI; no true active-task states. Not an oracle.')),
        exclusive=True)
    write_json(out/'process.json',dict(status='locked',pid=os.getpid(),evaluation_lock_sha256=sha(out/'evaluation_lock.json')),
               exclusive=True)
    print('LOCKED',out,sha(out/'evaluation_lock.json'),flush=True)


def verify_lock(out):
    out=Path(out).resolve();path=out/'evaluation_lock.json'
    lock=json.loads(path.read_text())
    validate_config(lock['config'])
    for key in ('sources_sha256','input_sha256','generation_sources_sha256','prior_sha256'):verify_hashes(lock[key])
    return lock,sha(path)


def run(out):
    out=Path(out).resolve();lock,lock_sha=verify_lock(out)
    previous=json.loads((out/'process.json').read_text())
    if previous.get('status')!='locked' or previous.get('evaluation_lock_sha256')!=lock_sha:
        raise ValueError('run requires unused locked output; failed/complete outputs are retained')
    cfg=lock['config'];data=Path(lock['data_directory']);started=time.perf_counter()
    process=dict(status='running',pid=os.getpid(),command=sys.argv,started_unix=time.time(),evaluation_lock_sha256=lock_sha)
    write_json(out/'process.json',process)
    rows=[];output_hashes={};completed=0
    try:
        for relative in lock['selected_batches']:
            path=data/relative;metadata=json.loads(path.with_suffix('.json').read_text())
            with np.load(path,allow_pickle=False) as stored:
                truth={k:stored[k] for k in stored.files}
            if truth['u'].shape!=(metadata['count'],metadata['expected_steps']) or truth['observation_steps'][-1]!=metadata['expected_steps']:
                raise ValueError('native array horizon/dimensions differ from complete metadata')
            inputs=observable_inputs(truth)
            public_meta={key:metadata[key] for key in PUBLIC_META}
            for noise in cfg['calibration_noise']:
                if time.perf_counter()-started>cfg['max_wall_seconds']:raise TimeoutError('declared evaluation wall budget exceeded')
                model=load_model(metadata['friction_world'],metadata['normal_world'],noise,
                    pi_seed=cfg['pi_model_seed'],normal_seed=cfg['normal_model_seed'],dt=metadata['dt'],
                    mass=metadata['mass_kg'],iterations=cfg['solver_iterations'])
                bank=out/lock['prior_banks'][f"{metadata['normal_world']}/{metadata['friction_world']}/{noise}"]
                with np.load(bank,allow_pickle=False) as prior:
                    states,weight=prior['states'].copy(),prior['weights'].copy()
                tic=time.perf_counter()
                predicted,detail=infer_and_forecast(model,inputs,public_meta,cfg,states,weight)
                inference_seconds=time.perf_counter()-tic
                tic=time.perf_counter()
                privileged,extra=privileged_forecasts(model,inputs,metadata,cfg)
                privileged_seconds=time.perf_counter()-tic
                if max(float(detail['max_solver_residual_N']),float(extra['privileged_max_solver_residual_N']))>cfg['max_solver_residual_N']:
                    raise FloatingPointError('modeled implicit solver residual exceeds frozen tolerance')
                predicted.update(privileged)
                scored=score_forecasts(predicted,truth,metadata,noise,cfg,detail)
                destination=out/'predictions'/noise/relative
                destination.parent.mkdir(parents=True,exist_ok=True)
                with destination.open('xb') as stream:np.savez_compressed(stream,**detail,**extra)
                write_json(destination.with_suffix('.json'),dict(status='complete',evaluation_lock_sha256=lock_sha,
                    input_npz_sha256=sha(path),input_metadata_sha256=sha(path.with_suffix('.json')),
                    prior_bank_sha256=sha(bank),npz_sha256=sha(destination),calibration_noise=noise,
                    normal_world=metadata['normal_world'],friction_world=metadata['friction_world'],
                    preparation_seed=metadata['preparation_seed'],probe=metadata['probe']['name'],sensor=metadata['sensor'],
                    count=metadata['count'],rows=len(scored),inference_forecast_seconds=inference_seconds,
                    privileged_seconds=privileged_seconds,metrics=scored),exclusive=True)
                for product in (destination,destination.with_suffix('.json')):output_hashes[str(product)]=sha(product)
                rows.extend(scored);completed+=1
            print(f'EVALUATED PI {completed} fit/batches: {relative}',flush=True)
        expected=sum(json.loads((data/name).with_suffix('.json').read_text())['count'] for name in lock['selected_batches'])
        expected*=len(cfg['calibration_noise'])*len(cfg['forecast_starts'])*len(ESTIMATORS)
        if len(rows)!=expected:raise ValueError('incomplete metric row count')
        with (out/'episodes.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        output_hashes[str(out/'episodes.csv')]=sha(out/'episodes.csv')
        if verify_lock(out)[1]!=lock_sha:raise ValueError('evaluation lock changed during run')
        write_json(out/'artifact_manifest.json',dict(evaluation_lock_sha256=lock_sha,outputs_sha256=output_hashes),exclusive=True)
        write_json(out/'process.json',dict(**{k:v for k,v in process.items() if k!='status'},status='complete',
            seconds=time.perf_counter()-started,ended_unix=time.time(),fit_batches=completed,rows=len(rows),expected_rows=expected))
    except BaseException as exc:
        write_json(out/'process.json',dict(**{k:v for k,v in process.items() if k!='status'},status='failed',
            seconds=time.perf_counter()-started,ended_unix=time.time(),error=repr(exc),
            completed_fit_batches=completed,completed_rows=len(rows),partial_outputs_sha256=output_hashes))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('freeze');p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--particles',type=int,default=256)
    p.add_argument('--calibration-noise',nargs='+',choices=['clean','noisy'],default=['clean','noisy'])
    p.add_argument('--forecast-starts',nargs='+',choices=STARTS,default=list(STARTS))
    p.add_argument('--batch-glob',action='append',help='Fixed named development subset; whole native inventory must still be complete')
    p.add_argument('--max-wall-seconds',type=float,default=7200.)
    for command in ('run','verify'):
        commands.add_parser(command).add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='freeze':
        cfg={**DEFAULT_CONFIG,'particles':args.particles,'calibration_noise':args.calibration_noise,
             'forecast_starts':args.forecast_starts,'max_wall_seconds':args.max_wall_seconds}
        freeze(args.data,args.out,cfg,batch_globs=args.batch_glob)
    elif args.command=='run':run(args.out)
    else:print('VERIFIED',verify_lock(args.out)[1])


if __name__=='__main__':main()
