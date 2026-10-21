"""Post-analysis paired-curve diagnosis; no refitting, selection change or Isaac."""
from pathlib import Path
import os,sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/physics-transfer-mechanism-numba')
sys.dont_write_bytecode=True
import argparse,csv,hashlib,json,time
import numpy as np
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
OLD=REPO/'paper_study'
sys.path.insert(0,str(OLD/'costed_benchmark'))
import evaluate as base
from inference import load_model
CHANNELS=('position_m','velocity_m_s','normal_N','friction_N')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())


def write(p,d,exclusive=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x' if exclusive else 'w') as f:json.dump(base.jsonable(d),f,indent=2,allow_nan=False);f.write('\n')


def check(mapping):
    for p,h in mapping.items():
        if sha(p)!=h:raise ValueError('Changed bound input: '+p)


def contrast(past,observed,clean,sigma):
    """All residual identities concern the same fixed candidate pair."""
    p=np.asarray(past,dtype=np.longdouble)
    y=np.asarray(observed,dtype=np.longdouble)
    c=np.asarray(clean,dtype=np.longdouble)
    s=np.asarray(sigma,dtype=np.longdouble)
    r1=(p[:,0]-y)/s;d=(p[:,1]-p[:,0])/s;e=(y-c)/s
    axes=(1,2)
    d2=np.sum(d*d,axis=axes)
    linear=2*np.sum(r1*d,axis=axes)
    gap=linear+d2
    clean_gap=2*np.sum(((p[:,0]-c)/s)*d,axis=axes)+d2
    noise=-2*np.sum(e*d,axis=axes)
    objective=np.sum(((p-y[:,None])/s)**2,axis=(2,3))
    # Dot products match the production objective's float64 reduction scope.
    r64=((np.asarray(past)-np.asarray(observed)[:,None])/np.asarray(sigma)).reshape(len(past),2,-1)
    j64=np.array([[np.dot(r,r) for r in pair] for pair in r64])
    r64_extended=np.asarray(r64,dtype=np.longdouble)
    j_same=np.sum(r64_extended*r64_extended,axis=2)
    return dict(distance=np.asarray(np.sqrt(d2),float),distance_squared=np.asarray(d2,float),
        linear_term=np.asarray(linear,float),stable_gap=np.asarray(gap,float),clean_gap=np.asarray(clean_gap,float),
        noise_term=np.asarray(noise,float),j64=j64,
        same_residual_extended_gap=np.asarray(j_same[:,1]-j_same[:,0],float),
        gap_identity_error=np.asarray(abs(gap-(objective[:,1]-objective[:,0])),float),
        noise_identity_error=np.asarray(abs(gap-clean_gap-noise),float))


def replay(model,inputs,initial,delta):
    """Vectorized pairs, same model step and original command clock."""
    count,pairs,_=initial.shape
    if pairs!=2:raise ValueError('Exactly two saved candidates required')
    end=inputs['probe_steps'];steps=inputs['observation_steps'][1:]
    indices={int(t):i for i,t in enumerate(steps)}
    xv=np.zeros((count*2,2));state=model.initial(count*2,delta,initial.reshape(-1,3))
    sampled=np.empty((count,2,len(steps),3));future=[];boundary=None;maximum=0.
    amp=np.repeat(inputs['amplitude'],2);geom=np.repeat(inputs['geometry'],2)
    for t in range(inputs['u'].shape[1]):
        xv,state,N,Ft,residual=model.step(xv,state,*[np.repeat(inputs[k][:,t],2) for k in ('u','d','d_dot')],amp,geom)
        maximum=max(maximum,float(np.max(abs(residual))))
        if t+1 in indices:
            observed_n=model.normal_force(xv,state,np.repeat(inputs['d'][:,t+1],2),np.repeat(inputs['d_dot'][:,t+1],2),amp,geom)
            sampled[:,:,indices[t+1]]=np.column_stack((xv,observed_n)).reshape(count,2,3)
        if t+1==end:
            boundary=dict(xv=xv.reshape(count,2,2).copy(),normal=state['normal'].reshape(count,2,3).copy(),pi=state['pi'].reshape(count,2,3).copy())
        if t>=end:future.append(np.column_stack((xv,N,Ft)).reshape(count,2,4))
    if boundary is None:raise ValueError('Positive preparation window required')
    return sampled,np.stack(future,axis=2),boundary,maximum


def freeze(out):
    if out.exists():raise FileExistsError('Fresh output required')
    if out.parent!=ROOT/'results':raise ValueError('Use a direct child of mechanism_study/results')
    evaluation=OLD/'results/observer_main2'; original=read(evaluation/'evaluation_lock.json')
    bound={}
    for key in ('sources_sha256','input_sha256','prior_sha256','generation_sources_sha256'):bound.update(original[key])
    for p in [evaluation/'evaluation_lock.json',OLD/'locks/main2/test_authorization.json',ROOT/'pilot.py',ROOT/'PROTOCOL.md',ROOT/'analyze.py',ROOT/'tests/test_pilot.py']:
        bound[str(p)]=sha(p)
    batches=[]
    for bank in original['config']['banks']:
        directory=evaluation/f"bank{bank['bank']}"
        if read(directory/'process.json')['status']!='complete':raise ValueError('Incomplete original bank')
        for p in (directory/'process.json',directory/'artifact_manifest.json'):
            bound[str(p)]=sha(p)
        outputs=read(directory/'artifact_manifest.json')['outputs_sha256'];check(outputs)
        bound.update(outputs)
        for relative in original['selected_batches']:
            prediction=directory/'predictions'/relative
            batches.append(dict(bank=bank,native=str(Path(original['data_directory'])/relative),prediction=str(prediction),relative=relative))
    check(bound)
    if len(batches)!=288:raise ValueError('Expected all 288 original batches')
    out.mkdir(parents=True)
    write(out/'analysis_lock.json',dict(role='post-analysis mechanism diagnosis on previously inspected Main2 data',created_unix=time.time(),source_and_input_sha256=bound,batches=batches,
        original_evaluation_lock_sha256=sha(evaluation/'evaluation_lock.json'),expected_pairs=10368,
        interpretation='D<=1 and future separation>=0.1 mm are descriptive scales only; no fresh hypothesis test',longdouble_bits=int(np.finfo(np.longdouble).nmant)))
    write(out/'process.json',dict(status='locked',lock_sha256=sha(out/'analysis_lock.json')))
    print('LOCKED',len(batches),'batches',flush=True)


def run(out):
    lock=read(out/'analysis_lock.json');digest=sha(out/'analysis_lock.json')
    if read(out/'process.json')!={'status':'locked','lock_sha256':digest}:raise ValueError('Unused locked output required')
    check(lock['source_and_input_sha256'])
    started=time.perf_counter();rows=[];outputs={};models={};active=None
    process=dict(status='running',lock_sha256=digest,started_unix=time.time(),command=sys.argv)
    write(out/'process.json',process,False)
    try:
        for b,entry in enumerate(lock['batches']):
            active=entry['prediction'];native=Path(entry['native']);source=Path(active)
            meta=read(native.with_suffix('.json'));saved_meta=read(source.with_suffix('.json'))
            with np.load(native,allow_pickle=False) as f:
                inputs=base.observable_inputs({k:f[k] for k in (*base.PUBLIC_KEYS,'observations','observation_steps','probe_steps')})
                sample=inputs['observation_steps'][1:]
                clean=np.concatenate((f['state'][:,sample],f['normal'][:,sample,None]),axis=2)
                end=inputs['probe_steps'];target=np.concatenate((f['next_state'][:,end:],f['normal'][:,end:,None],f['friction'][:,end:,None]),axis=2)
            with np.load(source,allow_pickle=False) as f:
                initial=f['mle_candidate_initial_pi'];old_future=f['mle_candidate_forecasts'];objective=f['mle_candidate_objectives'];selection=f['mle_selected_start'];sigma=f['effective_sigma']
                saved_boundary={k:f['mle_candidate_boundary_'+suffix] for k,suffix in [('xv','xv'),('pi','pi'),('normal','normal')]}
            bank=entry['bank'];modelkey=(bank['bank'],meta['friction_world'],meta['normal_world'])
            if modelkey not in models:
                models[modelkey]=load_model(meta['friction_world'],meta['normal_world'],'noisy',pi_seed=bank['pi_seed'],normal_seed=bank['normal_seed'],dt=meta['dt'],mass=meta['mass_kg'],iterations=40)
            model=models[modelkey]
            past,future,boundary,solver=replay(model,inputs,initial,meta['initial_compression_m'])
            observed=inputs['observations'][:,1:]
            noise=np.stack([np.random.default_rng(np.random.SeedSequence(key)).normal(size=(281,3))[sample] for key in meta['noise_stream_keys']])
            sensor=np.array([meta['sensor_parameters'][k] for k in ('sigma_x_m','sigma_v_m_s','sigma_normal_N')])
            noise_reconstruction=float(np.max(abs(observed-clean-noise*sensor)))
            difference=contrast(past,observed,clean,sigma)
            future_replay=float(np.max(abs(future-old_future)))
            objective_replay=float(np.max(abs(difference['j64']-objective)))
            boundary_replay=max(float(np.max(abs(boundary[k]-saved_boundary[k]))) for k in boundary)
            if not np.allclose(future,old_future,rtol=1e-10,atol=1e-10):raise ValueError('Future replay mismatch')
            if not np.allclose(difference['j64'],objective,rtol=1e-9,atol=1e-7):raise ValueError('Past objective replay mismatch')
            if noise_reconstruction>1e-11 or solver>1e-7:raise ValueError('Clock/noise/solver reconstruction failed')
            distance_future=np.sqrt(np.mean((future[:,1]-future[:,0])**2,axis=1))
            errors=np.mean((old_future-target[:,None])**2,axis=2)
            distance_past=np.sqrt(np.mean((past[:,1]-past[:,0])**2,axis=1))
            scale=model.pi.alpha*model.prefix_load(meta['initial_compression_m'])
            norms=np.linalg.norm((initial[:,1]-initial[:,0])/scale,axis=1)
            j_stable_choice=(difference['stable_gap']<0).astype(int)
            same_residual_choice=(difference['same_residual_extended_gap']<0).astype(int)
            j64_choice=np.argmin(difference['j64'],axis=1)
            clean_choice=(difference['clean_gap']<0).astype(int)
            for i in range(len(initial)):
                row=dict(bank=bank['bank'],history=meta['preparation_seed'],normal=meta['normal_world'],friction=meta['friction_world'],probe=meta['probe']['name'],episode=i,
                    geometry=int(inputs['geometry'][i]),sign=meta['factor_rows'][i]['excitation_sign'],selected_start=int(selection[i]),
                    original_gain=float(objective[i,0]-objective[i,selection[i]]),normalized_initial_distance=float(norms[i]),
                    past_mahalanobis=float(difference['distance'][i]),past_squared_distance=float(difference['distance_squared'][i]),
                    residual_linear_term=float(difference['linear_term'][i]),stable_objective_gap=float(difference['stable_gap'][i]),
                    clean_objective_gap=float(difference['clean_gap'][i]),direct_noise_gap=float(difference['noise_term'][i]),
                    same_residual_extended_gap=float(difference['same_residual_extended_gap'][i]),
                    same_residual_extended_choice=int(same_residual_choice[i]),
                    stable_choice=int(j_stable_choice[i]),replay_float64_choice=int(j64_choice[i]),clean_fixed_pair_choice=int(clean_choice[i]),
                    replay_objective_max_abs=objective_replay,replay_future_max_abs=future_replay,replay_boundary_max_abs=boundary_replay,
                    noise_reconstruction_max_abs=noise_reconstruction,solver_max_abs_N=solver,
                    objective_identity_error=float(difference['gap_identity_error'][i]),noise_identity_error=float(difference['noise_identity_error'][i]),
                    boundary_x_distance_m=float(abs(boundary['xv'][i,1,0]-boundary['xv'][i,0,0])),
                    boundary_v_distance_m_s=float(abs(boundary['xv'][i,1,1]-boundary['xv'][i,0,1])),
                    boundary_normal_state_l2=float(np.linalg.norm(boundary['normal'][i,1]-boundary['normal'][i,0])),
                    boundary_pi_state_l2_m=float(np.linalg.norm(boundary['pi'][i,1]-boundary['pi'][i,0])))
                for c,name in enumerate(CHANNELS):
                    row['future_distance_'+name]=float(distance_future[i,c]);row['start1_mse_'+name]=float(errors[i,0,c]);row['start2_mse_'+name]=float(errors[i,1,c])
                    if c<3:row['past_distance_'+name]=float(distance_past[i,c])
                rows.append(row)
            dest=out/'paired_past'/f"bank{bank['bank']}"/entry['relative'];dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:np.savez_compressed(f,past_predictions=past,observed=observed,clean_native_samples=clean,effective_sigma=sigma,observation_steps=sample,initial_pi=initial,original_objectives=objective,original_selection=selection,**{'boundary_'+k:v for k,v in boundary.items()})
            outputs[str(dest)]=sha(dest)
            if (b+1)%24==0:print('REPLAY',b+1,'/',len(lock['batches']),flush=True)
        if len(rows)!=10368:raise ValueError('Missing candidate pairs')
        with (out/'pairs.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        outputs[str(out/'pairs.csv')]=sha(out/'pairs.csv')
        check(lock['source_and_input_sha256'])
        write(out/'artifact_manifest.json',dict(lock_sha256=digest,outputs_sha256=outputs))
        write(out/'process.json',dict(**{k:v for k,v in process.items() if k!='status'},status='complete',pairs=len(rows),batches=len(lock['batches']),seconds=time.perf_counter()-started),False)
        print('COMPLETE',len(rows),'pairs',flush=True)
    except BaseException as exc:
        write(out/'process.json',dict(**{k:v for k,v in process.items() if k!='status'},status='failed',error=repr(exc),active_batch=active,completed_pairs=len(rows),partial_outputs_sha256=outputs,seconds=time.perf_counter()-started),False)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','run','verify']);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
    if a.stage=='freeze':freeze(out)
    elif a.stage=='run':run(out)
    else:check(read(out/'analysis_lock.json')['source_and_input_sha256']);print('SOURCE AND INPUT BINDINGS VERIFIED')


if __name__=='__main__':main()
