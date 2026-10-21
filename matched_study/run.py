"""Locked controlled synthetic study; old fitted laws are read only."""
from pathlib import Path
import os,sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/physics-transfer-matched-numba')
sys.dont_write_bytecode=True
import argparse,hashlib,json,time,copy,itertools
import numpy as np
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
sys.path.insert(0,str(ROOT))
import core

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def native(x):
    if isinstance(x,dict):return {str(k):native(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [native(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    return x

def read(p):return json.loads(Path(p).read_text())
def write(p,value,exclusive=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x' if exclusive else 'w') as f:json.dump(native(value),f,indent=2,allow_nan=False);f.write('\n')
def check(mapping):
    for p,h in mapping.items():
        if sha(p)!=h:raise ValueError('Changed bound source/input: '+p)

def plan(kind,dt=None,particles=None):
    base=read(ROOT/'config.json');cfg=copy.deepcopy(base)
    cfg['kind']=kind
    if kind=='development':cfg.update(base['development'])
    if kind in ('refinement','resolution'):
        section=base['refinement' if kind=='refinement' else 'particle_resolution']
        cfg.update({k:v for k,v in section.items() if k not in ('dt','particles')})
        if kind=='refinement':
            if dt not in section['dt']:raise ValueError('Undeclared refinement dt')
            cfg['dt']=dt
        else:
            if particles not in section['particles']:raise ValueError('Undeclared particle budget')
            cfg['particles']=particles
    elif dt is not None or particles is not None:raise ValueError('Overrides limited to declared controls')
    cfg.setdefault('mle_enabled',True)
    if kind=='resolution':cfg['mle_enabled']=False
    if kind=='development':cfg['sign_assignment']=base['sign_assignment'][:cfg['profiles_per_history']]
    cfg['initial_compression_m']=.02
    return cfg

def source_map(cfg):
    paths=list(ROOT.glob('*.py'))+list((ROOT/'tests').glob('*.py'))+[ROOT/'config.json',ROOT/'PROTOCOL.md']
    if cfg['kind']=='development':
        paths=[ROOT/'core.py',ROOT/'run.py',ROOT/'tests/test_core.py',ROOT/'tests/test_run.py',ROOT/'config.json',ROOT/'PROTOCOL.md']
    for directory in ['paper_study/costed_benchmark','paper_study/multistate_learning','research_next/src']:
        paths += list((REPO/directory).rglob('*.py'))
    for bank,case,normal in itertools.product(cfg['banks'],cfg['cases'],cfg['normals']):
        paths += [REPO/f'paper_study/multistate_learning/results/models/{case}/r{61000+bank}/noisy/model.json',
                  REPO/f'research_next/results/round1/models/state_id/{normal}/r{bank}/noisy/blind_selection/model.json']
    return {str(p.resolve()):sha(p) for p in sorted(set(paths))}

def freeze(out,kind,dt=None,particles=None):
    if out.exists():raise FileExistsError('Use a fresh result directory')
    if out.parent!=ROOT/'results':raise ValueError('Direct child of matched_study/results required')
    cfg=plan(kind,dt,particles)
    sources=source_map(cfg)
    if kind!='development':
        preflight=read(ROOT/'reports/preflight.json')
        if preflight['status']!='passed':raise ValueError('Successful preflight required')
        check(preflight['source_sha256'])
        sources[str(ROOT/'reports/preflight.json')]=sha(ROOT/'reports/preflight.json')
    write(out/'lock.json',dict(config=cfg,source_sha256=sources,created_unix=time.time(),
        scope='Fresh controlled synthetic histories. Same fitted law used as truth and estimator; no physical or methodological novelty claim.'))
    print('LOCKED',kind,out,flush=True)

def weights(past,observed,sigma):
    residual=(observed[:,None,:,:]-past[None,:,:,:])/sigma
    logw=-.5*np.sum(residual**2,axis=(2,3));logw-=logw.max(axis=1,keepdims=True)
    w=np.exp(logw);return w/w.sum(axis=1,keepdims=True)

def moments(future,w):
    # Center directly to avoid subtracting two nearly equal second moments.
    mean=np.einsum('ep,fptc->fetc',w,future)
    variance=np.einsum('ep,feptc->fetc',w,(future[:,None]-mean[:,:,None])**2)
    return mean,variance

def noise_draw(cfg,history,indices,sign,samples):
    return np.stack([np.random.default_rng(np.random.SeedSequence(
        [cfg['noise_stream_key'],int(history),int(i),int(sign)+1])).normal(size=(12,3))[:samples] for i in indices])

def save_npz(path,arrays,meta,outputs):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,**arrays,metadata_json=np.array(json.dumps(native(meta),allow_nan=False)))
    outputs[str(path)]=sha(path)

def validate_rollout(result,cfg):
    for key in ('past','future','boundary_xv','boundary_pi','boundary_normal'):
        if not np.isfinite(result[key]).all():raise FloatingPointError('Nonfinite '+key)
    # Core explicitly validates all internal steps and supplies maximum residual.
    if result['diagnostics']['max_solver_residual_N']>cfg['maximum_solver_residual_N']:
        raise FloatingPointError('Declared solver residual limit exceeded')

def run(out):
    lock=read(out/'lock.json');cfg=lock['config'];check(lock['source_sha256'])
    if (out/'process.json').exists():raise FileExistsError('Previous attempt retained; use a fresh directory')
    start=time.perf_counter();base=dict(status='running',started_unix=time.time(),command=sys.argv,
        lock_sha256=sha(out/'lock.json'),kind=cfg['kind'])
    write(out/'process.json',base);outputs={};completed=0;active=None;all_diagnostics=[]
    methods=['prior','posterior']+(['mle_start1','mle_two_start'] if cfg['mle_enabled'] else [])+['privileged_history']
    try:
        for bank,normal,case in itertools.product(cfg['banks'],cfg['normals'],cfg['cases']):
            model=core.load(bank,case,normal,dt=cfg['dt'])
            model_id=f'bank{bank}_{normal}_{case}'
            prior_seed=cfg['prior_seeds'][cfg['banks'].index(bank)]
            prior=core.reachable(model,prior_seed,cfg['particles'])
            preparation_path=out/'preparations'/f'{model_id}_prior.npz'
            save_npz(preparation_path,dict(initial_pi=prior['initial_pi'],normal0=prior['normal0'],
                initial_xv=prior['initial_xv']),{**prior['metadata'],'seed':prior_seed,'diagnostics':prior['diagnostics']},outputs)
            truths={}
            for history in cfg['history_seeds']:
                tr=core.reachable(model,history,cfg['profiles_per_history']);truths[history]=tr
                np.testing.assert_array_equal(prior['normal0'],tr['normal0'])
                save_npz(out/'preparations'/f'{model_id}_h{history}.npz',dict(initial_pi=tr['initial_pi'],normal0=tr['normal0'],
                    initial_xv=tr['initial_xv']),{**tr['metadata'],'seed':history,'diagnostics':tr['diagnostics']},outputs)
            for sign,duration,strength in itertools.product([-1,1],cfg['durations'],cfg['strengths_N']):
                indices=np.flatnonzero(np.asarray(cfg['sign_assignment'])==sign)
                if not len(indices):continue
                arm=f's{sign:+d}_d{round(duration*1000)}_a{strength:g}'
                common=dict(duration=duration,strength=strength,sign=sign,geom=cfg['geometry'],amp=cfg['cam_amplitude_m'],
                    future_duration=cfg['future_duration'])
                ensemble=[]
                for future in cfg['future_amplitudes_N']:
                    q=core.rollout(model,prior['initial_pi'],prior['normal0'],**common,future_amplitude=future)
                    validate_rollout(q,cfg);ensemble.append(q)
                for q in ensemble[1:]:np.testing.assert_array_equal(q['past'],ensemble[0]['past'])
                Q=np.stack([q['future'] for q in ensemble]);past=ensemble[0]['past']
                uniform=np.full((1,cfg['particles']),1/cfg['particles'])
                prior_mean,prior_var=moments(Q,uniform);prior_mean=prior_mean[:,0];prior_var=prior_var[:,0]
                ens_path=out/'ensembles'/model_id/(arm+'.npz')
                ensmeta=dict(model_id=model_id,bank=bank,normal=normal,case=case,sign=sign,duration=duration,strength=strength,
                    dt=cfg['dt'],particles=cfg['particles'],futures=cfg['future_amplitudes_N'],
                    prior_preparation_path=str(preparation_path),prior_preparation_sha256=outputs[str(preparation_path)],
                    diagnostics=[q['diagnostics'] for q in ensemble],commands=common,initial_compression_m=.02)
                arrays=dict(initial_pi=prior['initial_pi'],normal0=prior['normal0'],particle_past=past,
                    particle_past_branches=np.stack([q['past'] for q in ensemble]),particle_future=Q,
                    prior_variance=prior_var,boundary_xv=ensemble[0]['boundary_xv'],boundary_pi=ensemble[0]['boundary_pi'],
                    boundary_normal=ensemble[0]['boundary_normal'],past_times=np.arange(1,past.shape[1]+1)*.025,
                    future_times=np.arange(1,Q.shape[2]+1)*.0025)
                for k,v in ensemble[0].items():
                    if k not in ('past','future','boundary_xv','boundary_pi','boundary_normal','diagnostics') and isinstance(v,np.ndarray):arrays[k]=v
                save_npz(ens_path,arrays,ensmeta,outputs)
                for history,tr in truths.items():
                    active=dict(model=model_id,history=history,arm=arm);E=len(indices)
                    truth=[]
                    for future in cfg['future_amplitudes_N']:
                        q=core.rollout(model,tr['initial_pi'][indices],tr['normal0'],**common,future_amplitude=future)
                        validate_rollout(q,cfg);truth.append(q)
                    for q in truth[1:]:np.testing.assert_array_equal(q['past'],truth[0]['past'])
                    target=np.stack([q['future'] for q in truth]);clean=truth[0]['past']
                    noise=noise_draw(cfg,history,indices,sign,past.shape[1])
                    sensors=list(cfg['sensor_sigma']);sig=np.array([np.hypot(cfg['sensor_sigma'][s],cfg['numerical_floor']) for s in sensors])
                    observed=np.stack([clean+noise*np.array(cfg['sensor_sigma'][s]) for s in sensors])
                    w=np.stack([weights(past,y,sd) for y,sd in zip(observed,sig)])
                    pred=np.empty((len(sensors),len(truth),E,len(methods),target.shape[2],4))
                    var=np.empty((len(sensors),len(truth),E,target.shape[2],4))
                    candidates=np.empty((len(sensors),E,2,3));cost=np.empty((len(sensors),E,2));chosen=np.empty((len(sensors),E),int)
                    candidate_past=np.empty((len(sensors),E,2,past.shape[1],3))
                    candidate_future=np.empty((len(sensors),len(truth),E,2,target.shape[2],4))
                    batch_diag=[]
                    for ni,sensor in enumerate(sensors):
                        pm,pv=moments(Q,w[ni]);var[ni]=pv
                        pred[ni,:,:,0]=np.broadcast_to(prior_mean[:,None],target.shape)
                        pred[ni,:,:,1]=pm
                        if cfg['mle_enabled']:
                            for ei,profile in enumerate(indices):
                                fitted=core.mle(model,observed[ni,ei],sig[ni],prior['initial_pi'],w[ni,ei],prior['normal0'],
                                    duration=duration,strength=strength,sign=sign,geom=cfg['geometry'],amp=cfg['cam_amplitude_m'])
                                if fitted['diagnostics']['max_solver_residual_N']>cfg['maximum_solver_residual_N']:
                                    raise FloatingPointError('MLE trial solver residual exceeded fixed limit')
                                candidates[ni,ei]=fitted['initial_candidates'];cost[ni,ei]=fitted['costs'];chosen[ni,ei]=fitted['chosen']
                                batch_diag.append(dict(sensor=sensor,profile_index=int(profile),**fitted['diagnostics']))
                                for fi,future in enumerate(cfg['future_amplitudes_N']):
                                    mq=core.rollout(model,fitted['initial_candidates'],prior['normal0'],**common,future_amplitude=future)
                                    validate_rollout(mq,cfg)
                                    if fi==0:candidate_past[ni,ei]=mq['past']
                                    else:np.testing.assert_array_equal(mq['past'],candidate_past[ni,ei])
                                    candidate_future[ni,fi,ei]=mq['future']
                                    pred[ni,fi,ei,2]=mq['future'][0];pred[ni,fi,ei,3]=mq['future'][fitted['chosen']]
                        # Explicitly privileged exact-history diagnostic, independently rerun from true state.
                        for fi,future in enumerate(cfg['future_amplitudes_N']):
                            oracle=core.rollout(model,tr['initial_pi'][indices].copy(),tr['normal0'].copy(),**common,future_amplitude=future)
                            np.testing.assert_array_equal(oracle['future'],target[fi]);pred[ni,fi,:,-1]=oracle['future']
                    if not np.isfinite(pred).all():raise FloatingPointError('Nonfinite prediction')
                    meta=dict(**active,model_id=model_id,bank=bank,normal=normal,case=case,sign=sign,duration=duration,strength=strength,
                        dt=cfg['dt'],particles=cfg['particles'],sensors=sensors,futures=cfg['future_amplitudes_N'],methods=methods,
                        ensemble_path=str(ens_path),ensemble_sha256=outputs[str(ens_path)],
                        initial_compression_m=.02,diagnostics=[q['diagnostics'] for q in truth],mle_diagnostics=batch_diag,
                        channel_order=['position_m','velocity_m_s','normal_N','friction_N'],
                        force_convention='mean over each 2.5ms output block; x/v at block end',
                        common_noise_key=[cfg['noise_stream_key'],history,'profile_index','sign+1'])
                    arrays=dict(truth_past=clean,truth_past_branches=np.stack([q['past'] for q in truth]),truth_future=target,
                        observed=observed,noise_standard=noise,effective_sigma=sig,
                        weights=w,predictions=pred,posterior_variance=var,prior_variance=prior_var,
                        profile_indices=indices,truth_initial_pi=tr['initial_pi'][indices],normal0=tr['normal0'],
                        truth_boundary_xv=truth[0]['boundary_xv'],truth_boundary_pi=truth[0]['boundary_pi'],
                        truth_boundary_normal=truth[0]['boundary_normal'])
                    if cfg['mle_enabled']:arrays.update(mle_initial=candidates,mle_cost=cost,mle_chosen=chosen,
                        mle_past=candidate_past,mle_future=candidate_future)
                    for k,v in truth[0].items():
                        if k not in ('past','future','boundary_xv','boundary_pi','boundary_normal','diagnostics') and isinstance(v,np.ndarray):arrays['truth_'+k]=v
                    save_npz(out/'batches'/model_id/f'h{history}'/(arm+'.npz'),arrays,meta,outputs)
                    completed+=1;all_diagnostics.extend(batch_diag)
                    if completed%12==0:print('BATCH',completed,active,'seconds',round(time.perf_counter()-start,1),flush=True)
        check(lock['source_sha256'])
        write(out/'manifest.json',dict(lock_sha256=base['lock_sha256'],outputs_sha256=outputs))
        write(out/'process.json',dict(**{k:v for k,v in base.items() if k!='status'},status='complete',batches=completed,
            files=len(outputs),seconds=time.perf_counter()-start,mle_fits=len(all_diagnostics)),False)
        print('COMPLETE',completed,'batches',flush=True)
    except BaseException as exc:
        write(out/'process.json',dict(**{k:v for k,v in base.items() if k!='status'},status='failed',error=repr(exc),
            active=active,completed_batches=completed,partial_outputs_sha256=outputs,seconds=time.perf_counter()-start),False)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','run','verify'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--kind',choices=['development','main','refinement','resolution'],default='main')
    p.add_argument('--dt',type=float);p.add_argument('--particles',type=int);a=p.parse_args();out=a.out.resolve()
    if a.stage=='freeze':freeze(out,a.kind,a.dt,a.particles)
    elif a.stage=='run':run(out)
    else:check(read(out/'lock.json')['source_sha256']);check(read(out/'manifest.json')['outputs_sha256']);print('VERIFIED')
