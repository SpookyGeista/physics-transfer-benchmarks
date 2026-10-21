"""Frozen, existing-data comparison of PI inference across three time grids."""
from pathlib import Path
import os,sys,argparse,csv,json,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/paper-study-numba')
os.environ.setdefault('MPLCONFIGDIR','/tmp/paper-study-mpl')
sys.dont_write_bytecode=True
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(HERE.parent))
import evaluate as base
from inference import load_model,model_paths,build_prior
from bounded_mle import fit_initial_state,forecast
read=lambda path:json.loads(Path(path).read_text())
METHODS=('prior_propagation','trajectory_assimilation','bounded_mle_start1','bounded_mle_two_start')
CHANNELS=('position_m','velocity_m_s','normal_N','friction_N')
DEFAULT_OUT=REPO/'paper_study/results/pi_comparative_timestep_v1'
DATA={1:REPO/'paper_study/results/costed_pi_development',2:REPO/'paper_study/results/pi_native_refinement_h2',4:REPO/'paper_study/results/pi_native_refinement_h4'}


def subset(values,indices):
    return {key:(value if key in ('observation_steps','probe_steps') else value[indices].copy()) for key,value in values.items()}


def common_grid(values,divisor):
    """Endpoint mechanics and interval-mean forces on the canonical h grid."""
    n,steps,c=values.shape
    if c!=4 or steps%divisor:raise ValueError('Four channels and divisible time axis required')
    block=values.reshape(n,steps//divisor,divisor,4)
    return np.concatenate((block[:,:,-1,:2],block[:,:,:,2:].mean(axis=2)),axis=2)


def canonical_prior(model,reconstructed,canonical_q=None,tolerance=1e-10):
    radius=model.pi.alpha*model.prefix_load()
    q=reconstructed/radius
    if not np.isfinite(q).all() or np.any(abs(q)>1.+tolerance):raise ValueError('Invalid normalized reachable prior')
    if canonical_q is None:canonical_q=q.copy()
    difference=float(np.max(abs(q-canonical_q)))
    if difference>tolerance:raise ValueError(f'Prior refinement precheck failed: {difference}')
    return canonical_q.copy(),canonical_q*radius,difference


def sources(cfg):
    paths=[Path(__file__).resolve(),HERE/'config.json',HERE/'protocol.md',HERE/'test_timestep.py',HERE.parent/'bounded_mle.py',*base.source_paths()]
    for normal,case in cfg['compositions']:paths.extend(model_paths(case,normal,'noisy',61001,1))
    return {str(p.resolve()):base.sha(p) for p in sorted(set(paths))}


def verify(out):
    lock=read(out/'lock.json')
    for key in ('sources_sha256','input_sha256','prior_sha256','generation_sources_sha256'):base.verify_hashes(lock[key])
    return lock,base.sha(out/'lock.json')


def check_native(path,cfg,divisor,nativecfg):
    meta=read(path.with_suffix('.json'))
    if meta['status']!='complete' or meta['split']!='dev' or base.sha(path)!=meta['npz_sha256']:raise ValueError('Incomplete/mismatched native input')
    dt=.0025/divisor;probe=round(meta['probe']['duration_s']/dt);total=probe+round(.4/dt)
    if (meta['dt']!=dt or meta['count']!=36 or meta['recorded_steps']!=total or meta['expected_steps']!=total
        or meta['sensor']!='ideal' or any(meta['sensor_parameters'].values()) or meta['preparation_seed']!=33001):raise ValueError('Native diagnostic scope/clock mismatch')
    if meta['mass_kg']!=2. or meta['initial_compression_m']!=.02 or meta['future_seconds']!=.4:raise ValueError('Native initialization mismatch')
    factor=[(h,g,s) for h in range(6) for g in range(3) for s in (-1,1)]
    if [(r['profile_index'],r['geometry'],r['excitation_sign']) for r in meta['factor_rows']]!=factor:raise ValueError('Factor order mismatch')
    with np.load(path,allow_pickle=False) as raw:
        if int(raw['probe_steps'])!=probe or not np.array_equal(raw['observation_steps'],np.arange(0,total+1,10*divisor)):
            raise ValueError('Native physical observation clock mismatch')
        for name in ('geometry','amplitude','excitation_sign'):
            if not np.array_equal(raw[name],[r[name] for r in meta['factor_rows']]):raise ValueError('Native raw factor mismatch')
        if raw['u'].shape!=(36,total) or raw['observations'].shape!=(36,total//(10*divisor)+1,3):raise ValueError('Native shape mismatch')
    # No true prefix command, force or latent array is inspected here.
    if nativecfg['preparation_seconds']!=4. or nativecfg['dt']!=dt or nativecfg['observation_steps']!=10*divisor:
        raise ValueError('Both fixture and coupling clocks must match refinement')
    return meta


def freeze(out):
    cfg=read(HERE/'config.json')
    if out.exists():raise FileExistsError('Use a fresh exclusive diagnostic output')
    if out==REPO/'paper_study/results' or not out.is_relative_to(REPO/'paper_study/results'):raise ValueError('Use results subdirectory')
    source=sources(cfg);inputs={};generation_sources={};selected={}
    for divisor in cfg['divisors']:
        data=DATA[divisor];generation=read(data/'generation_lock.json');process=read(data/'process.json')
        expected=(128,4608) if divisor==1 else (24,864)
        if process.get('status')!='complete' or (process.get('batches'),process.get('episodes'))!=expected or generation['smoke'] is not False:
            raise ValueError('Complete existing native generation required')
        base.verify_hashes(generation['sources_sha256']);generation_sources.update(generation['sources_sha256'])
        for name in ('generation_lock.json','process.json'):inputs[str(data/name)]=base.sha(data/name)
        selected[divisor]=[]
        for normal,case in cfg['compositions']:
            for probe in cfg['probes']:
                relative=f'p33001/{normal}_{case}/{probe}/ideal.npz';path=data/relative
                check_native(path,cfg,divisor,generation['config']);selected[divisor].append(relative)
                for p in (path,path.with_suffix('.json')):inputs[str(p)]=base.sha(p)
    out.mkdir();priorfiles={};precheck=[];canonical={};normal_initial={}
    for divisor in cfg['divisors']:
        for normal,case in cfg['compositions']:
            model=load_model(case,normal,'noisy',dt=.0025/divisor)
            reconstructed,meta=build_prior(model,particles=1024,seed=43001,duration=4.)
            key=f'{normal}_{case}'
            q,applied,error=canonical_prior(model,reconstructed,canonical.get(key),cfg['prior_normalized_tolerance'])
            canonical[key]=q
            initial=model.initial(1)['normal']
            if key in normal_initial and not np.array_equal(initial,normal_initial[key]):raise ValueError('Fresh normal state changed with dt')
            normal_initial[key]=initial
            path=out/f'priors/h{divisor}/{key}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:np.savez_compressed(stream,reconstructed_states=reconstructed,states=applied,normalized_q=q,weights=np.full(1024,1/1024),initial_normal_state=initial)
            record=dict(**meta,divisor=divisor,normal_world=normal,friction_world=case,max_normalized_reconstruction_difference=error,npz_sha256=base.sha(path))
            base.write_json(path.with_suffix('.json'),record,exclusive=True);precheck.append(record)
            for p in (path,path.with_suffix('.json')):priorfiles[str(p)]=base.sha(p)
    base.verify_hashes(source);base.verify_hashes(inputs);base.verify_hashes(generation_sources)
    base.write_json(out/'lock.json',dict(role='Descriptive comparative timestep diagnostic before scores',config=cfg,
        sources_sha256=source,input_sha256=inputs,prior_sha256=priorfiles,generation_sources_sha256=generation_sources,
        selected_batches=selected,data_directories={d:str(DATA[d]) for d in cfg['divisors']},prior_prechecks=precheck,
        expected_episodes=216,expected_batches=36,created_unix=time.time(),command=sys.argv),exclusive=True)
    digest=verify(out)[1];base.write_json(out/'process.json',dict(status='locked',lock_sha256=digest),exclusive=True)
    print('TIMESTEP LOCKED',digest,'max q difference',max(r['max_normalized_reconstruction_difference'] for r in precheck),flush=True)


def infer_cfg(cfg):
    return dict(base.DEFAULT_CONFIG,particles=1024,calibration_noise=['noisy'],forecast_starts=['filtered_particles'],numerical_floor=cfg['numerical_floor'])


def mle_two(model,inputs,public,states,weights,cfg,divisor):
    budget=dict(cfg['mle'],max_seconds_per_episode=cfg['mle']['max_seconds_per_episode']*divisor)
    candidates=[]
    try:
        for initial,prior in ((states,weights),(np.zeros((1,3)),np.ones((len(weights),1)))):
            candidates.append(fit_initial_state(model,inputs,public,initial,prior,config=budget))
    except Exception as exc:
        exc.completed_mle_diagnostics=[c['diagnostics'] for c in candidates]
        raise
    objective=np.array([[d['best_objective'] for d in c['diagnostics']] for c in candidates]).T
    choice=np.argmin(objective,axis=1)
    future=[];solver=[]
    try:
        for candidate in candidates:
            pred,diagnostic=forecast(model,inputs,candidate,return_diagnostics=True)
            future.append(pred['filtered_particles/bounded_initial_state_mle']);solver.append(diagnostic)
    except Exception as exc:
        exc.completed_mle_diagnostics=[c['diagnostics'] for c in candidates]
        raise
    values=np.stack(future,axis=1)
    return {'bounded_mle_start1':values[:,0],'bounded_mle_two_start':values[np.arange(len(weights)),choice]},dict(
        candidate_forecasts=values,candidate_objectives=objective,selected_start=choice,
        candidate_initial_states=np.stack([c['initial_pi_state'] for c in candidates],axis=1)),[c['diagnostics'] for c in candidates],solver


def score(predictions,target,meta,indices,divisor,weights):
    target_common=common_grid(target,divisor);rows=[];concentration=base.weight_diagnostics(weights)
    for method,pred in predictions.items():
        error=pred-target;common_error=common_grid(pred,divisor)-target_common
        if not np.isfinite(error).all():raise FloatingPointError('Nonfinite prediction/score; no dropping')
        for i,index in enumerate(indices):
            factor=meta['factor_rows'][index]
            row=dict(divisor=divisor,normal_world=meta['normal_world'],friction_world=meta['friction_world'],probe=meta['probe']['name'],
                episode=int(index),profile_index=factor['profile_index'],geometry=factor['geometry'],excitation_sign=factor['excitation_sign'],
                method=method,posterior_ess=float(concentration['ess'][i]),posterior_max_weight=float(concentration['max_weight'][i]))
            for c,name in enumerate(CHANNELS):
                row['mse_'+name]=float(np.mean(error[i,:,c]**2));row['rmse_'+name]=float(np.sqrt(row['mse_'+name]));row['bias_'+name]=float(np.mean(error[i,:,c]))
                row['common_mse_'+name]=float(np.mean(common_error[i,:,c]**2))
            rows.append(row)
    return rows


def run(out):
    lock,digest=verify(out);cfg=lock['config'];indices=np.array(cfg['episode_indices'])
    if read(out/'process.json')!={'status':'locked','lock_sha256':digest}:raise ValueError('Unused lock required; no automatic retries')
    started=time.perf_counter();artifacts={};allrows=[];failures=[];complete=0;total_budget=0;total_success=0
    process=dict(status='running',pid=os.getpid(),lock_sha256=digest,started_unix=time.time(),command=sys.argv)
    base.write_json(out/'process.json',process)
    try:
        for divisor in cfg['divisors']:
            for relative in lock['selected_batches'][str(divisor)]:
                active=f'h{divisor}/{relative}';destination=out/'predictions'/f'h{divisor}'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
                phase='public_inputs';tic=time.perf_counter();particle_seconds=mle_seconds=0.;predictions={};details={};rows=[]
                try:
                    if time.perf_counter()-started>cfg['max_wall_seconds']:raise TimeoutError('Diagnostic wall guard')
                    path=Path(lock['data_directories'][str(divisor)])/relative;meta=read(path.with_suffix('.json'))
                    with np.load(path,allow_pickle=False) as raw:
                        inputs=subset(base.observable_inputs({k:raw[k] for k in (*base.PUBLIC_KEYS,'observations','observation_steps','probe_steps')}),indices)
                    public={k:meta[k] for k in base.PUBLIC_META};model=load_model(meta['friction_world'],meta['normal_world'],'noisy',dt=meta['dt'])
                    prior=out/f"priors/h{divisor}/{meta['normal_world']}_{meta['friction_world']}.npz"
                    with np.load(prior,allow_pickle=False) as raw:states,weights=raw['states'].copy(),raw['weights'].copy()
                    phase='particle';pt=time.perf_counter()
                    predicted,detail=base.infer_and_forecast(model,inputs,public,infer_cfg(cfg),states,weights)
                    particle_seconds=time.perf_counter()-pt
                    if float(detail['max_solver_residual_N'])>cfg['max_solver_residual_N']:raise FloatingPointError('Particle implicit solver failure')
                    predictions={k.split('/')[1]:v for k,v in predicted.items()};details=dict(detail)
                    phase='mle';mt=time.perf_counter()
                    mle,md,diagnostics,solver=mle_two(model,inputs,public,states,detail['weights'],cfg,divisor)
                    mle_seconds=time.perf_counter()-mt;predictions.update(mle);details.update(md)
                    total_budget+=sum(d['budget_exhausted'] for ds in diagnostics for d in ds)
                    total_success+=sum(d['solver_success'] for ds in diagnostics for d in ds)
                    phase='scoring'
                    with np.load(path,allow_pickle=False) as truth:
                        end=inputs['probe_steps'];target=np.concatenate((truth['next_state'][indices,end:],truth['normal'][indices,end:,None],truth['friction'][indices,end:,None]),axis=2)
                        command=inputs['u'][:,end:]
                    rows=score(predictions,target,meta,indices,divisor,detail['weights']);allrows.extend(rows)
                    details.update(target=target,common_target=common_grid(target,divisor),future_u=command,episode_indices=indices)
                    for name,pred in predictions.items():details['forecast_'+name]=pred;details['common_forecast_'+name]=common_grid(pred,divisor)
                    with destination.open('xb') as stream:np.savez_compressed(stream,**details)
                    base.write_json(destination.with_suffix('.json'),dict(status='complete',lock_sha256=digest,npz_sha256=base.sha(destination),
                        native_sha256=base.sha(path),metrics=rows,mle_diagnostics=diagnostics,mle_solver=solver,
                        particle_seconds=particle_seconds,mle_incremental_seconds=mle_seconds,total_seconds=time.perf_counter()-tic),exclusive=True)
                    complete+=1
                except Exception as exc:
                    failure=dict(grid=divisor,batch=relative,phase=phase,error=repr(exc),seconds=time.perf_counter()-tic)
                    completed_diagnostics=getattr(exc,'completed_mle_diagnostics',[])
                    failure['completed_mle_diagnostics']=completed_diagnostics
                    total_budget+=sum(d['budget_exhausted'] for ds in completed_diagnostics for d in ds)
                    total_success+=sum(d['solver_success'] for ds in completed_diagnostics for d in ds)
                    failures.append(failure)
                    if details:
                        with destination.open('xb') as stream:np.savez_compressed(stream,**details)
                    base.write_json(destination.with_suffix('.json'),dict(status='failed',lock_sha256=digest,**failure,
                        npz_sha256=base.sha(destination) if destination.exists() else None),exclusive=True)
                for p in (destination,destination.with_suffix('.json')):
                    if p.is_file():artifacts[str(p)]=base.sha(p)
                print('TIMESTEP',active,'complete',complete,'failures',len(failures),flush=True)
        with (out/'episodes.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(allrows[0]) if allrows else ['status']);writer.writeheader();writer.writerows(allrows)
        artifacts[str(out/'episodes.csv')]=base.sha(out/'episodes.csv')
        verify(out)
        base.write_json(out/'artifact_manifest.json',dict(lock_sha256=digest,outputs_sha256=artifacts),exclusive=True)
        base.write_json(out/'process.json',dict(**{k:v for k,v in process.items() if k!='status'},status='complete' if not failures else 'incomplete',
            seconds=time.perf_counter()-started,completed_batches=complete,failed_batches=failures,episodes=complete*6,metric_rows=len(allrows),
            expected_episodes=216,optimizer_successes=total_success,optimizer_budget_cases=total_budget))
    except BaseException as exc:
        base.write_json(out/'process.json',dict(**{k:v for k,v in process.items() if k!='status'},status='failed',error=repr(exc),
            seconds=time.perf_counter()-started,completed_batches=complete,failed_batches=failures,partial_outputs_sha256=artifacts));raise


def analyze(out):
    lock,digest=verify(out);process=read(out/'process.json');manifest=read(out/'artifact_manifest.json');base.verify_hashes(manifest['outputs_sha256'])
    if process['status']!='complete' or process['episodes']!=216:raise ValueError('All216 diagnostic episodes required; failed cases remain recorded')
    with (out/'episodes.csv').open() as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=864:raise ValueError('Expected864 method rows')
    summary=dict(scope='Ideal sensors only; noisy fitted bank1. Same feedback policy, separately realized native commands. Descriptive, not Main2 noisy-sensor convergence proof.',
        lock_sha256=digest,process=process,pooled={},comparisons={},ranks={},paired_gains={})
    for divisor in (1,2,4):
        summary['pooled'][divisor]={}
        for method in METHODS:
            r=[v for v in rows if int(v['divisor'])==divisor and v['method']==method]
            summary['pooled'][divisor][method]={name:dict(mse=float(np.mean([float(v['mse_'+name]) for v in r])),
                pooled_rmse=float(np.sqrt(np.mean([float(v['mse_'+name]) for v in r]))),mean_episode_rmse=float(np.mean([float(v['rmse_'+name]) for v in r])),
                common_grid_mse=float(np.mean([float(v['common_mse_'+name]) for v in r]))) for name in CHANNELS}
    changes={};rank_switches=[];gains={};command_changes={}
    for relative in lock['selected_batches']['1']:
        data={}
        for divisor in (1,2,4):
            with np.load(out/'predictions'/f'h{divisor}'/relative,allow_pickle=False) as raw:
                data[divisor]={k:raw[k] for k in raw.files if k.startswith('common_') or k=='future_u'}
        for low,high in ((1,2),(2,4)):
            key=f'{low}_to_{high}'
            a=data[low]['future_u'].reshape(6,-1,low).mean(axis=2);b=data[high]['future_u'].reshape(6,-1,high).mean(axis=2)
            command_changes.setdefault(key,[]).extend(np.sqrt(np.mean((a-b)**2,axis=1)))
            for method in METHODS:
                diff=data[low]['common_forecast_'+method]-data[high]['common_forecast_'+method]
                changes.setdefault((key,method),[]).extend(np.sqrt(np.mean(diff**2,axis=1)))
        for name_idx,name in enumerate(CHANNELS):
            orders={};losses={}
            for divisor in (1,2,4):
                loss={m:np.mean((data[divisor]['common_forecast_'+m]-data[divisor]['common_target'])[:,:,name_idx]**2,axis=1) for m in METHODS}
                losses[divisor]=loss;orders[divisor]=sorted(METHODS,key=lambda m:float(np.mean(loss[m])))
                for method in METHODS[1:]:
                    for comparator in ('prior_propagation','trajectory_assimilation'):
                        if method==comparator:continue
                        gains.setdefault((divisor,method,comparator,name),[]).extend(loss[comparator]-loss[method])
            summary['ranks'][relative+'/'+name]=orders
            for low,high in ((1,2),(2,4)):
                if orders[low]!=orders[high]:rank_switches.append(dict(cell=relative,channel=name,pair=[low,high],before=orders[low],after=orders[high]))
    tolerance=np.array(lock['config']['engineering_tolerances'])
    for method in METHODS:
        d12=np.asarray(changes[('1_to_2',method)]);d24=np.asarray(changes[('2_to_4',method)])
        rms12=np.sqrt(np.mean(d12**2,axis=0));rms24=np.sqrt(np.mean(d24**2,axis=0))
        summary['comparisons'][method]=dict(rms_h_h2=dict(zip(CHANNELS,rms12.tolist())),rms_h2_h4=dict(zip(CHANNELS,rms24.tolist())),
            refinement_ratio=dict(zip(CHANNELS,(rms24/np.maximum(rms12,1e-30)).tolist())),
            h2_h4_fraction_all_channels_within_tolerance=float(np.mean(np.all(d24<=tolerance,axis=1))))
    for method in METHODS[1:]:
        for comparator in ('prior_propagation','trajectory_assimilation'):
            if method==comparator:continue
            key=f'{method}_versus_{comparator}';summary['paired_gains'][key]={}
            for name in CHANNELS:
                values={d:np.asarray(gains[(d,method,comparator,name)]) for d in (1,2,4)}
                summary['paired_gains'][key][name]=dict(mean_mse_gain={d:float(v.mean()) for d,v in values.items()},
                    mean_gain_sign_stable=len({int(np.sign(v.mean())) for v in values.values()})==1,
                    episode_gain_sign_changes_h2_h4=int(np.sum(np.sign(values[2])!=np.sign(values[4]))))
    summary['rank_switches']=rank_switches
    summary['command_rms_changes_N']={key:float(np.sqrt(np.mean(np.asarray(v)**2))) for key,v in command_changes.items()}
    summary['posterior']={d:dict(median_ess=float(np.median([float(r['posterior_ess']) for r in rows if int(r['divisor'])==d and r['method']=='trajectory_assimilation'])),
        fraction_ess_le2=float(np.mean([float(r['posterior_ess'])<=2 for r in rows if int(r['divisor'])==d and r['method']=='trajectory_assimilation']))) for d in (1,2,4)}
    result=out/'analysis';result.mkdir();base.write_json(result/'summary.json',summary,exclusive=True)
    lines=['# Comparative PI timestep diagnostic','',summary['scope'],'',f"All216 episodes completed in{process['seconds']:.1f}s; optimizer successes{process['optimizer_successes']}/432, budget cases{process['optimizer_budget_cases']}.",'',
        '| Method | h pooled x RMSE (mm) | h2 pooled x RMSE (mm) | h4 pooled x RMSE (mm) | h2/h4 tolerance fraction |','|---|---:|---:|---:|---:|']
    for method in METHODS:
        values=[summary['pooled'][d][method]['position_m']['pooled_rmse']*1000 for d in (1,2,4)]
        lines.append(f"| {method} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {summary['comparisons'][method]['h2_h4_fraction_all_channels_within_tolerance']:.1%} |")
    lines+=['',f"Cell/channel rank switches: {sum(r['pair']==[1,2] for r in rank_switches)} of48 for h→h2 and {sum(r['pair']==[2,4] for r in rank_switches)} of48 for h2→h4.",'',
        'summary.json separately reports per-grid pooled MSE, pooled RMSE, mean episode RMSE, common-grid changes, paired gains and their signs. Particle approximation and optimizer ambiguity remain separate limits. No equivalence, global optimum, posterior calibration or Main2 noisy-sensor convergence is established by this small ideal-sensor diagnostic. The6/12/24s optimizer wall guards deliberately avoid a tighter work budget at finer dt; times are diagnostic resources, not matched Main2 runtime.','']
    (result/'README.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    labels=('Prior propagation','P1024 assimilation','MLE start1','MLE two starts')
    for axis,name,scale,ylabel in ((axes[0],'position_m',1000.,'Pooled position RMSE (mm)'),(axes[1],'friction_N',1.,'Pooled friction RMSE (N)')):
        for method,label in zip(METHODS,labels):
            axis.plot([2.5,1.25,.625],[summary['pooled'][d][method][name]['pooled_rmse']*scale for d in (1,2,4)],'o-',label=label)
        axis.set_xscale('log',base=2);axis.set_xticks([.625,1.25,2.5],['0.625','1.25','2.5']);axis.set_xlabel('Timestep (ms)');axis.set_ylabel(ylabel);axis.grid(alpha=.25)
    axes[1].legend(fontsize=8);fig.suptitle('PI timestep diagnostic: ideal sensors, noisy fitted bank1, 72 episodes/grid',fontsize=11)
    fig.savefig(result/'timestep_comparison.png',dpi=180);fig.savefig(result/'timestep_comparison.pdf');plt.close(fig)
    verify(out)
    base.write_json(result/'artifact_manifest.json',dict(lock_sha256=digest,outputs_sha256={str(p):base.sha(p) for p in result.iterdir() if p.is_file()}),exclusive=True)
    print(json.dumps({k:v for k,v in summary.items() if k in ('comparisons','command_rms_changes_N','posterior')},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','run','analyze','verify']);parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();out=args.out.resolve()
    if args.stage=='freeze':freeze(out)
    elif args.stage=='run':run(out)
    elif args.stage=='analyze':analyze(out)
    else:print('TIMESTEP VERIFIED',verify(out)[1])


if __name__=='__main__':main()
