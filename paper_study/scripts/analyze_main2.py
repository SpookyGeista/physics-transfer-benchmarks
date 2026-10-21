"""Independent descriptive Main2 audit. No estimator or scoring helper imports."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/main2-analysis-mpl')
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
import argparse,hashlib,itertools,json,time
import numpy as np
import pandas as pd
METHODS=('prior_propagation','trajectory_assimilation','privileged_full_history','bounded_mle_start1','bounded_mle_two_start')
CHANNELS=('position_m','velocity_m_s','normal_N','friction_N')
WORLDS=('maxwell','boucwen');CASES=('balanced','wide_scales','compliant','stiff_high_capacity')
SEEDS=(53201,53202,53203,53204);PROBES=('hold_150','gentle_150','strong_300')
IDENTITY=['bank','preparation_seed','normal_world','friction_world','probe','episode']
NATIVE_ID=IDENTITY[1:]
GROUPS=[('global',[]),('composition',['normal_world','friction_world']),('probe',['probe']),
        ('bank',['bank']),('history',['preparation_seed'])]
CONTRASTS=[('assimilation/prior','trajectory_assimilation','prior_propagation'),
           ('MLE1/assimilation','bounded_mle_start1','trajectory_assimilation'),
           ('MLE2/MLE1','bounded_mle_two_start','bounded_mle_start1')]+[
           (m+'/privileged',m,'privileged_full_history') for m in METHODS]
PROTOCOL=ROOT/'paper_study/costed_benchmark/main2_protocol.md'
PLAN=ROOT/'paper_study/reports/main2_analysis_plan.md'
TEST=ROOT/'paper_study/tests/test_main2_analysis.py'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def near(actual,expected,context,*,atol=2e-14,rtol=2e-11):
    a,b=np.asarray(actual),np.asarray(expected)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a,b,atol=atol,rtol=rtol):
        raise ValueError('Numerical mismatch: '+context)
    return float(np.max(abs(a-b))) if a.size else 0.


def truth_scores(prediction,target):
    prediction,target=np.asarray(prediction),np.asarray(target)
    if prediction.shape!=target.shape or prediction.ndim!=3 or prediction.shape[-1]!=4 or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError('Matching finite E,T,4 forecasts/targets required')
    error=prediction-target;mse=np.mean(error*error,axis=1)
    return dict(mse=mse,rmse=np.sqrt(mse),bias=np.mean(error,axis=1))


def validate_inventory(frame):
    expected=pd.MultiIndex.from_product([range(1,4),SEEDS,WORLDS,CASES,PROBES,range(36),METHODS],names=IDENTITY+['method'])
    actual=pd.MultiIndex.from_frame(frame[IDENTITY+['method']])
    if not actual.is_unique or len(actual)!=51840 or set(actual)!=set(expected):raise ValueError('Exact Main2 Cartesian inventory required')
    for name,value in [('calibration_noise','noisy'),('sensor','noisy'),('forecast_start','filtered_particles'),('future_steps',160)]:
        if not (frame[name]==value).all():raise ValueError('Unexpected factor: '+name)
    if not (frame.pi_model_seed==61000+frame.bank).all() or not (frame.normal_model_seed==frame.bank).all():
        raise ValueError('Calibration bank identity mismatch')
    if not (frame.profile_index==frame.episode//6).all() or not (frame.geometry==(frame.episode%6)//2).all() or not (frame.excitation_sign==2*(frame.episode%2)-1).all():
        raise ValueError('Episode factor order mismatch')
    expected_ids=[f'test-p{s}-h{e//6}-g{(e%6)//2}-s{(2*(e%2)-1):+d}' for s,e in zip(frame.preparation_seed,frame.episode)]
    if list(frame.episode_id)!=expected_ids:raise ValueError('Redundant episode ID mismatch')
    for c in CHANNELS:
        for stat in ('mse','rmse','bias'):
            values=frame[stat+'_'+c].to_numpy()
            if not np.isfinite(values).all() or (stat!='bias' and np.any(values<0)):raise ValueError('Invalid score')
        near(frame['rmse_'+c].to_numpy()**2,frame['mse_'+c].to_numpy(),'RMSE square relation')


def aggregate(frame):
    rows=[]
    for scope,groups in GROUPS:
        for values,cell in frame.groupby(groups+['method'],sort=True):
            values=values if isinstance(values,tuple) else (values,)
            identity=dict(zip(groups+['method'],values));n=len(cell)
            for c in CHANNELS:
                mse=float(np.mean(cell['mse_'+c]));rows.append(dict(scope=scope,**identity,channel=c,episodes=n,
                    pooled_mse=mse,pooled_rmse=float(np.sqrt(mse)),mean_episode_rmse=float(np.mean(cell['rmse_'+c])),
                    mean_bias=float(np.mean(cell['bias_'+c]))))
    return pd.DataFrame(rows)


def contrasts(frame):
    rows=[]
    for scope,groups in GROUPS:
        cells=frame.groupby(groups,sort=True) if groups else [((),frame)]
        for values,cell in cells:
            values=values if isinstance(values,tuple) else (values,);identity=dict(zip(groups,values))
            for name,numerator,denominator in CONTRASTS:
                a=cell[cell.method==numerator].set_index(IDENTITY).sort_index()
                b=cell[cell.method==denominator].set_index(IDENTITY).sort_index()
                if not a.index.equals(b.index) or not a.index.is_unique:raise ValueError('Unpaired contrast')
                for c in CHANNELS:
                    x=float(np.mean(a['mse_'+c]));y=float(np.mean(b['mse_'+c]))
                    rows.append(dict(scope=scope,**identity,contrast=name,channel=c,paired_episodes=len(a),
                        numerator_method=numerator,denominator_method=denominator,numerator_mse=x,denominator_mse=y,
                        absolute_mse_difference=x-y,mse_ratio=x/y if y>0 else None,ratio_defined=y>0))
    return pd.DataFrame(rows)


def markdown(frame):
    rows=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join('---' for _ in frame.columns)+' |']
    for values in frame.itertuples(index=False,name=None):
        rows.append('| '+' | '.join((f'{x:.6g}' if np.isfinite(x) else 'undefiniert') if isinstance(x,(float,np.floating)) else str(x) for x in values)+' |')
    return '\n'.join(rows)


def freeze(evaluation,out):
    evaluation,out=Path(evaluation).resolve(),Path(out).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'paper_study/results'):raise ValueError('Fresh results subdirectory required')
    paths=[Path(__file__).resolve(),PROTOCOL,PLAN,TEST]
    out.mkdir(parents=True)
    write(out/'analysis_lock.json',dict(role='Independent implementation of frozen descriptive protocol before analyst score inspection',
        created_unix=time.time(),evaluation_directory=str(evaluation),sources_sha256={str(p):sha(p) for p in paths},
        scopes=GROUPS,contrasts=CONTRASTS,no_inference=True))
    write(out/'process.json',dict(status='locked'))
    print('MAIN2 ANALYSIS LOCKED',out,flush=True)


def verify_hash_maps(maps,checks):
    cache={}
    for label,mapping in maps:
        for p,expected in mapping.items():
            if p not in cache:cache[p]=sha(p)
            if cache[p]!=expected:raise ValueError('Hash mismatch: '+p)
            checks.append(dict(binding=label,path=p,sha256=expected))
    return cache


def read_inputs(evaluation,checks):
    """All bank statuses are checked before any metric or NPZ is read."""
    evaluation=Path(evaluation)
    processes={b:read(evaluation/f'bank{b}/process.json') for b in range(1,4)}
    for b,p in processes.items():
        if p.get('status')!='complete' or p.get('batches')!=96 or p.get('rows')!=17280 or p.get('episodes')!=3456:
            raise ValueError(f'Bank{b} not complete; no partial analysis permitted')
    lock=read(evaluation/'evaluation_lock.json');locksha=sha(evaluation/'evaluation_lock.json')
    authpath=Path(lock['test_authorization_path']);auth=read(authpath)
    if sha(authpath)!=lock['test_authorization_sha256'] or auth['status']!='locked' or auth['role']!='paper_study_main2_test_authorization':
        raise ValueError('Test authorization mismatch')
    if auth['preparation_seeds']!=list(SEEDS) or auth['sources_sha256']!=lock['sources_sha256']:
        raise ValueError('Prospective population/source mismatch')
    if auth['sources_sha256'].get(str(PROTOCOL))!=sha(PROTOCOL):raise ValueError('Protocol missing from test lock')
    data=Path(lock['data_directory']);generation=read(data/'generation_lock.json');nativeprocess=read(data/'process.json')
    if generation['test_lock_sha256']!=sha(authpath) or generation['test_sources_sha256']!=auth['sources_sha256']:
        raise ValueError('Generation authorization binding mismatch')
    if nativeprocess['status']!='complete' or nativeprocess['batches']!=96 or nativeprocess['episodes']!=3456:
        raise ValueError('Native completeness failure')
    cfg=read(ROOT/'paper_study/costed_benchmark/main2_native.json')
    if generation['config']!=cfg or generation['smoke'] is not False:raise ValueError('Native config/scope changed')
    ecfg=read(ROOT/'paper_study/costed_benchmark/main2_evaluation.json')
    if lock['config']!=ecfg:raise ValueError('Evaluation config changed')
    expected=sorted(f'p{s}/{n}_{c}/{p}/noisy.npz' for s in SEEDS for n in WORLDS for c in CASES for p in PROBES)
    if sorted(lock['selected_batches'])!=expected or sorted(str(p.relative_to(data)) for p in data.rglob('*.npz'))!=expected:
        raise ValueError('Exact native file inventory mismatch')
    maps=[('authorization',auth['sources_sha256'])]
    for key in ('sources_sha256','input_sha256','prior_sha256','generation_sources_sha256'):maps.append(('evaluation_'+key,lock[key]))
    maps.append(('generation',generation['sources_sha256']))
    frames=[];manifests={}
    for b in range(1,4):
        if processes[b]['evaluation_lock_sha256']!=locksha:raise ValueError('Bank process lock mismatch')
        manifest=read(evaluation/f'bank{b}/artifact_manifest.json');manifests[b]=manifest
        if manifest['evaluation_lock_sha256']!=locksha:raise ValueError('Bank manifest lock mismatch')
        maps.append((f'bank{b}',manifest['outputs_sha256']))
        actual=sorted(str(p.relative_to(evaluation/f'bank{b}/predictions')) for p in (evaluation/f'bank{b}/predictions').rglob('*.npz'))
        if actual!=expected:raise ValueError('Bank prediction inventory mismatch')
        csvpath=evaluation/f'bank{b}/episodes.csv'
        if str(csvpath) not in manifest['outputs_sha256']:raise ValueError('CSV not manifest-bound')
    controlpaths=[evaluation/'evaluation_lock.json',authpath]+[evaluation/f'bank{b}'/name for b in range(1,4) for name in ('process.json','artifact_manifest.json')]
    maps.append(('completion_controls',{str(p):sha(p) for p in controlpaths}))
    verify_hash_maps(maps,checks)
    for b in range(1,4):frames.append(pd.read_csv(evaluation/f'bank{b}/episodes.csv'))
    frame=pd.concat(frames,ignore_index=True);validate_inventory(frame)
    return lock,frame,processes,cfg,expected,maps


def audit_native(data,cfg,expected):
    cache={};costrows=[];numerics=[]
    for relative in expected:
        path=data/relative;meta=read(path.with_suffix('.json'))
        if meta['npz_sha256']!=sha(path) or meta['status']!='complete' or meta['split']!='test':raise ValueError('Native NPZ binding')
        identity=dict(preparation_seed=meta['preparation_seed'],normal_world=meta['normal_world'],friction_world=meta['friction_world'],probe=meta['probe']['name'])
        if relative!=f"p{identity['preparation_seed']}/{identity['normal_world']}_{identity['friction_world']}/{identity['probe']}/noisy.npz":raise ValueError('Native path identity')
        with np.load(path,allow_pickle=False) as z:
            needed=('state','next_state','normal','friction','probe_steps','u','d','d_dot','net_force','guard',
                'actuator_work_J','normal_drive_work_discrete_J','normal_drive_work_zoh_J','friction_applied_work_J','slope',
                'native_momentum_residual_N','native_position_residual_m','solver_residual_N','normal_updates','observations','observation_steps')
            a={k:z[k].copy() for k in needed}
        e=int(a['probe_steps']);steps=e+160;dt=cfg['dt']
        if a['next_state'].shape!=(36,steps,2) or e!=round(meta['probe']['duration_s']/dt) or meta['recorded_steps']!=steps:
            raise ValueError('Native clocks/shape')
        for k,v in a.items():
            if not np.isfinite(v).all():raise ValueError('Nonfinite native '+k)
        near(a['state'][:,1:],a['next_state'][:,:-1],'Native state chain',atol=1e-12,rtol=0)
        near(a['net_force'],a['u']-a['slope']*a['normal']+a['friction'],'Applied mechanical force balance')
        if not np.array_equal(a['normal_updates'],np.broadcast_to(np.arange(1,steps+1),(36,steps))):raise ValueError('Exactly one normal update per step required')
        dx=a['next_state'][...,0]-a['state'][...,0]
        nres=2*(a['next_state'][...,1]-a['state'][...,1])/dt-a['net_force']
        xres=dx-dt*a['next_state'][...,1]
        near(nres,a['native_momentum_residual_N'],'Newton residual identity')
        near(xres,a['native_position_residual_m'],'Kinematic residual identity')
        near(a['actuator_work_J'],a['u']*dx,'Actual actuator work')
        times=np.arange(steps+1)*dt-e*dt
        dnext=np.where(times>=0,.02+.002*np.sin(2*np.pi*times/.4),.02)[1:]
        near(a['normal_drive_work_discrete_J'],a['normal']*(dnext[None,:]-a['d']),'Discrete normal work',atol=2e-14)
        near(a['normal_drive_work_zoh_J'],a['normal']*a['d_dot']*dt,'ZOH normal work')
        near(a['friction_applied_work_J'],a['friction']*dx,'Applied friction work')
        expected_guards=((abs(a['next_state'][...,0])>.5)|(abs(a['next_state'][...,1])>5)|(a['normal']<.01)).astype(int)
        if not np.array_equal(expected_guards,a['guard']):raise ValueError('Native guard identity')
        if not np.array_equal(a['observation_steps'],np.arange(0,steps+1,10)):raise ValueError('Sensor clock')
        target=np.concatenate((a['next_state'][:,e:],a['normal'][:,e:,None],a['friction'][:,e:,None]),axis=2)
        for i in range(36):
            r=dict(**identity,episode=i,probe_seconds=e*dt,total_seconds=steps*dt,native_guard_count=int(a['guard'][i].sum()))
            for label,name in [('actuator','actuator_work_J'),('normal_drive','normal_drive_work_discrete_J'),
                               ('normal_drive_zoh','normal_drive_work_zoh_J'),('friction','friction_applied_work_J')]:
                for segment,slice_ in [('probe',slice(None,e)),('total',slice(None))]:
                    values=a[name][i,slice_]
                    r[f'{segment}_{label}_signed_J']=float(values.sum());r[f'{segment}_{label}_absolute_J']=float(abs(values).sum())
            r['probe_two_port_absolute_J']=r['probe_actuator_absolute_J']+r['probe_normal_drive_absolute_J']
            r['total_two_port_absolute_J']=r['total_actuator_absolute_J']+r['total_normal_drive_absolute_J']
            costrows.append(r)
        numerics.append(dict(relative=relative,native_guard_count=int(a['guard'].sum()),
            max_abs_newton_residual_N=float(abs(nres).max()),max_abs_kinematic_residual_m=float(abs(xres).max()),
            max_abs_solver_residual_N=float(abs(a['solver_residual_N']).max())))
        cache[relative]=(target,meta,a['guard'].sum(axis=1),a)
    return cache,pd.DataFrame(costrows),pd.DataFrame(numerics)


def audit_predictions(evaluation,lock,frame,expected,native):
    audit=[];optimizers=[];timings=[];concentrations=[];nested=[];truthframe=frame.set_index(IDENTITY+['method']).sort_index()
    locksha=sha(evaluation/'evaluation_lock.json')
    for bank in range(1,4):
        for relative in expected:
            dest=evaluation/f'bank{bank}/predictions'/relative;meta=read(dest.with_suffix('.json'))
            target,nmeta,guards,raw=native[relative]
            base=dict(bank=bank,preparation_seed=nmeta['preparation_seed'],normal_world=nmeta['normal_world'],friction_world=nmeta['friction_world'],probe=nmeta['probe']['name'])
            if (meta['evaluation_lock_sha256']!=locksha or meta['npz_sha256']!=sha(dest)
                or meta['input_npz_sha256']!=nmeta['npz_sha256'] or meta['input_metadata_sha256']!=sha(Path(lock['data_directory'])/Path(relative).with_suffix('.json'))):
                raise ValueError('Prediction/metadata lock mismatch')
            if meta['bank']!=dict(bank=bank,pi_seed=61000+bank,normal_seed=bank):raise ValueError('Prediction bank mismatch')
            priorfile=evaluation/lock['prior_banks'][str(bank)][f"{base['normal_world']}/{base['friction_world']}"]
            if meta['prior_bank_sha256']!=sha(priorfile):raise ValueError('Prior bank binding mismatch')
            with np.load(dest,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            for k,v in a.items():
                if not np.isfinite(v).all():raise ValueError('Nonfinite model output '+k)
            weights=a['weights']
            if weights.shape!=(36,1024) or np.any(weights<0):raise ValueError('Weights shape/nonnegative')
            near(weights.sum(axis=1),np.ones(36),'Normalized weights',atol=2e-12)
            near(a['prior'],np.full((36,1024),1/1024),'Fixed uniform empirical prior')
            with np.load(priorfile,allow_pickle=False) as priorbank:
                near(a['prior_states'],priorbank['states'],'Frozen public prior particles',atol=0,rtol=0)
            near(a['weight_trace'][:,-1],weights,'Final weight trace')
            end=int(raw['probe_steps'])
            if not np.array_equal(a['observation_steps'],np.arange(0,end+1,10)) or a['weight_trace'].shape!=(36,end//10+1,1024):
                raise ValueError('Causal observation/weight-trace clock mismatch')
            logweight=np.log(a['prior'])+a['log_likelihood'];logweight-=logweight.max(axis=1,keepdims=True)
            checkweight=np.exp(logweight);checkweight/=checkweight.sum(axis=1,keepdims=True)
            near(weights,checkweight,'Posterior from prior and causal likelihood',atol=2e-13)
            ess=1/(weights*weights).sum(axis=1);maxweight=weights.max(axis=1)
            log=np.zeros_like(weights);np.log(weights,out=log,where=weights>0);entropy=-(weights*log).sum(axis=1)
            objective=a['mle_candidate_objectives'];choice=a['mle_selected_start']
            if objective.shape!=(36,2) or not np.array_equal(choice,np.argmin(objective,axis=1)):raise ValueError('Past objective selection')
            near(a['filtered_particles_bounded_mle_start1_forecast'],a['mle_candidate_forecasts'][:,0],'Start1 candidate identity',atol=0,rtol=0)
            near(a['filtered_particles_bounded_mle_two_start_forecast'],a['mle_candidate_forecasts'][np.arange(36),choice],
                'Selected candidate identity',atol=0,rtol=0)
            md=meta['mle_diagnostics']
            if len(md)!=2 or any(len(x)!=36 for x in md):raise ValueError('Optimizer inventory')
            past1=objective[:,0];pastselected=objective[np.arange(36),choice]
            for start in range(2):
                for e,d in enumerate(md[start]):
                    if d['starting_particle']!=(int(np.argmax(weights[e])) if start==0 else 0):raise ValueError('MLE start selection is not the declared causal likelihood maximum/zero')
                    if d['episode']!=e or d['best_objective']>d['initial_objective']+1e-10*max(1,abs(d['initial_objective'])):
                        raise ValueError('Optimizer episode/best objective identity')
                    if d['residual_calls']>512 or d['optimizer_nfev']>120 or d['max_solver_residual_N']>1e-7:
                        raise ValueError('Declared optimizer evaluation/solver budget violated')
                    near(np.array(d['best_objective']),np.array(objective[e,start]),'Stored objective')
                    optimizers.append(dict(**base,episode=e,start=start,**{k:d[k] for k in (
                        'status','solver_success','budget_exhausted','optimizer_nfev','residual_calls','initial_objective','best_objective','seconds','max_solver_residual_N')}))
            methodmse={}
            recorded_metrics=pd.DataFrame(meta['metrics']).set_index(IDENTITY+['method']).sort_index()
            expected_indices=truthframe.index[(truthframe.index.get_level_values('bank')==bank) &
                (truthframe.index.get_level_values('preparation_seed')==base['preparation_seed']) &
                (truthframe.index.get_level_values('normal_world')==base['normal_world']) &
                (truthframe.index.get_level_values('friction_world')==base['friction_world']) &
                (truthframe.index.get_level_values('probe')==base['probe'])]
            csv_metrics=truthframe.loc[expected_indices]
            if not recorded_metrics.index.equals(csv_metrics.index) or set(recorded_metrics)!=set(csv_metrics):raise ValueError('Sidecar/CSV metric inventory mismatch')
            for column in csv_metrics:
                if pd.api.types.is_numeric_dtype(csv_metrics[column]):near(recorded_metrics[column].to_numpy(dtype=float),csv_metrics[column].to_numpy(dtype=float),'Sidecar/CSV '+column)
                elif not np.array_equal(recorded_metrics[column],csv_metrics[column]):raise ValueError('Sidecar/CSV category mismatch')
            for method in METHODS:
                scores=truth_scores(a['filtered_particles_'+method+'_forecast'],target);methodmse[method]=scores['mse']
                index=pd.MultiIndex.from_tuples([tuple(base[k] if k!='episode' else e for k in IDENTITY)+(method,) for e in range(36)],names=IDENTITY+['method'])
                rows=truthframe.loc[index];maximum=0.
                for stat in ('mse','rmse','bias'):
                    for c,name in enumerate(CHANNELS):maximum=max(maximum,near(rows[stat+'_'+name].to_numpy(),scores[stat][:,c],stat+' '+name))
                near(rows.posterior_particle_ess.to_numpy(),ess,'ESS')
                near(rows.posterior_particle_max_weight.to_numpy(),maxweight,'Max weight')
                near(rows.posterior_particle_entropy.to_numpy(),entropy,'Entropy')
                near(rows.native_guard_count.to_numpy(),guards,'CSV guards')
                near(rows.mle_selected_start.to_numpy(),choice,'CSV selected start')
                for start in range(2):
                    near(rows[f'mle_start{start+1}_objective'].to_numpy(),objective[:,start],'CSV objective')
                    for k,dk in [('budget_exhausted','budget_exhausted'),('optimizer_success','solver_success')]:
                        if not np.array_equal(rows[f'mle_start{start+1}_{k}'].to_numpy(),np.array([d[dk] for d in md[start]])):
                            raise ValueError('CSV optimizer flags')
                end=int(raw['probe_steps'])
                for name in ('actuator_work_J','normal_drive_work_discrete_J'):
                    near(rows['probe_signed_'+name].to_numpy(),raw[name][:,:end].sum(axis=1),'CSV signed work')
                    near(rows['probe_absolute_'+name].to_numpy(),abs(raw[name][:,:end]).sum(axis=1),'CSV absolute work')
                audit.append(dict(**base,method=method,episodes=36,max_score_absolute_difference=maximum))
            for e in range(36):
                concentrations.append(dict(**base,episode=e,ess=float(ess[e]),max_weight=float(maxweight[e]),entropy=float(entropy[e])))
                r=dict(**base,episode=e,selected_start=int(choice[e]),past_start1=float(past1[e]),past_selected=float(pastselected[e]),
                    past_improvement=float(past1[e]-pastselected[e]))
                for c,name in enumerate(CHANNELS):
                    r['future_mse_change_'+name]=float(methodmse['bounded_mle_two_start'][e,c]-methodmse['bounded_mle_start1'][e,c])
                nested.append(r)
            pf=meta['particle_filter_plus_forecast_seconds'];mle=meta['mle_two_start_incremental_seconds'];priv=meta['privileged_replay_plus_forecast_seconds']
            timings.append(dict(**base,episodes=36,particle_filter_and_two_weighted_forecasts_seconds=pf,
                two_mle_incremental_seconds=mle,mle_total_with_particle_initialization_seconds=pf+mle,
                privileged_replay_forecast_seconds=priv,optimizer_start1_recorded_fit_seconds=sum(d['seconds'] for d in md[0]),
                optimizer_start2_recorded_fit_seconds=sum(d['seconds'] for d in md[1])))
            for key in ('max_solver_residual_N','privileged_max_solver_residual_N'):
                if float(a[key])>1e-7:raise ValueError('Model implicit solver residual')
    return tuple(pd.DataFrame(x) for x in (audit,optimizers,timings,concentrations,nested))


def plots(out,pooled,nested,concentration):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    directory=out/'figures';directory.mkdir()
    labels=['Prior','Partikel','Priv. Historie','MLE Start1','MLE 2 Starts']
    fig,axes=plt.subplots(2,2,figsize=(10.5,7),constrained_layout=True)
    for ax,c,scale,unit in zip(axes.flat,CHANNELS,[1e6,1e3,1,1],['µm','mm/s','N','N']):
        cell=pooled[(pooled.scope=='global')&(pooled.channel==c)].set_index('method').loc[list(METHODS)]
        values=cell.pooled_rmse.to_numpy()*scale;ax.bar(np.arange(5),values,color=['#98a8b5','#267caa','#969696','#db9d3d','#bd5336'])
        ax.set_xticks(np.arange(5),labels,rotation=20,ha='right');ax.set_ylabel('Gepoolter RMSE ['+unit+']');ax.set_title(c)
        if (values>0).all():ax.set_yscale('log')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Main2: vollständiger Zukunfts-Prognosevergleich\n3 Modellbanken × 4 Historienblöcke; keine Konfidenzintervalle')
    for ext in ('png','pdf'):fig.savefig(directory/('main2_global_errors.'+ext))
    plt.close(fig)
    cell=pooled[(pooled.scope=='composition')&(pooled.channel=='position_m')]
    table=cell.pivot(index=['normal_world','friction_world'],columns='method',values='pooled_rmse').reindex(columns=METHODS)*1e6
    fig,ax=plt.subplots(figsize=(10,5.5),constrained_layout=True)
    from matplotlib.colors import LogNorm
    values=table.to_numpy();norm=LogNorm(vmin=values.min(),vmax=values.max()) if (values>0).all() and values.max()>values.min() else None
    im=ax.imshow(values,aspect='auto',norm=norm,cmap='viridis');ax.set_xticks(range(5),labels)
    ax.set_yticks(range(8),[n+' / '+f for n,f in table.index]);ax.set_title('Alle acht Kompositionen: gepoolter Positions-RMSE')
    for i in range(8):
        for j in range(5):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white' if im.norm(values[i,j])<.6 else 'black')
    fig.colorbar(im,ax=ax,label='µm (logarithmische Farbskala)' if norm else 'µm')
    for ext in ('png','pdf'):fig.savefig(directory/('main2_compositions.'+ext))
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,probe in zip(axes,PROBES):
        for bank,color in zip(range(1,4),['#267caa','#d6852c','#9862aa']):
            z=nested[(nested.probe==probe)&(nested.bank==bank)]
            ax.scatter(z.past_improvement,z.future_mse_change_position_m*1e12,s=6,alpha=.25,color=color,label=f'Bank{bank}',rasterized=True)
        ax.axhline(0,color='black',lw=.7);ax.set_xscale('symlog',linthresh=.01);ax.set_yscale('symlog',linthresh=1.)
        ax.set_title(probe);ax.set_xlabel('Gewinn im vergangenen Zielfunktional');ax.grid(alpha=.15)
    axes[0].set_ylabel('MSE(MLE2) − MSE(MLE1) [µm²]');axes[-1].legend(markerscale=2)
    fig.suptitle('Vergangene Anpassung und zukünftiger Fehler\nAlle 10.368 gepaarten Kontexte; positive y-Werte bedeuten schlechtere Prognose')
    for ext in ('png','pdf'):fig.savefig(directory/('main2_past_vs_future.'+ext))
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.7),constrained_layout=True)
    for ax,probe in zip(axes,PROBES):
        for bank,color in zip(range(1,4),['#267caa','#d6852c','#9862aa']):
            x=concentration[(concentration.probe==probe)&(concentration.bank==bank)].ess
            ax.hist(x,bins=np.geomspace(1,1024,30),histtype='step',label=f'Bank{bank}',color=color)
        ax.set_xscale('log');ax.set_xlim(1,1024);ax.set_title(probe);ax.set_xlabel('Effektive Partikelzahl (ESS)')
    axes[0].set_ylabel('Kontexte');axes[-1].legend();fig.suptitle('Konzentration des P1024-Beobachters; keine Kalibrierungs- oder Konvergenzgarantie')
    for ext in ('png','pdf'):fig.savefig(directory/('main2_particle_ess.'+ext))
    plt.close(fig)


def report(out,pooled,contrast,costs,optimizers,timings,concentration,nested,audit,processes,numerics):
    globaltable=pooled[(pooled.scope=='global')].pivot(index='method',columns='channel',values='pooled_rmse').reindex(METHODS)
    globaltable.position_m*=1e6;globaltable.velocity_m_s*=1e3
    globaltable=globaltable.rename(columns={'position_m':'Position µm','velocity_m_s':'Geschwindigkeit mm/s','normal_N':'Normal N','friction_N':'Reibung N'}).reset_index()
    gc=contrast[(contrast.scope=='global')&(contrast.channel=='position_m')][['contrast','mse_ratio','absolute_mse_difference']].copy()
    gc['absolute_mse_difference']*=1e12;gc=gc.rename(columns={'absolute_mse_difference':'MSE-Differenz µm²'})
    selected=int((nested.selected_start==1).sum());improved=nested.past_improvement>0
    fractions={c:dict(future_worse=int((nested['future_mse_change_'+c]>0).sum()),future_better=int((nested['future_mse_change_'+c]<0).sum()),
        past_improved_future_worse=int((improved&(nested['future_mse_change_'+c]>0)).sum())) for c in CHANNELS}
    budget=int(optimizers.budget_exhausted.sum());failed=int((~optimizers.solver_success).sum())
    computing=dict(particle_total_seconds=float(timings.particle_filter_and_two_weighted_forecasts_seconds.sum()),
        mle_incremental_seconds=float(timings.two_mle_incremental_seconds.sum()),
        mle_including_particle_seconds=float(timings.mle_total_with_particle_initialization_seconds.sum()),
        privileged_seconds=float(timings.privileged_replay_forecast_seconds.sum()),
        summed_bank_current_attempt_seconds=float(sum(p['seconds'] for p in processes.values())),
        summed_prior_attempt_seconds=float(sum(p.get('prior_attempt_seconds',0) for p in processes.values())))
    costsummary=costs.groupby('probe')[['probe_seconds','total_seconds','probe_two_port_absolute_J','total_two_port_absolute_J']].mean().reset_index()
    summary=dict(scope='Held-out descriptive Main2, four history blocks and three previously fitted banks; no inferential intervals',
        native_episodes=3456,model_episode_contexts=10368,episode_method_rows=51840,optimizer_starts=len(optimizers),
        second_start_selected=selected,past_objective_strict_improvement=int(improved.sum()),
        maximum_past_objective_increase=float((-nested.past_improvement).max()),future_change_counts=fractions,
        optimizer_budget_cases=budget,optimizer_nonconverged_starts=failed,
        posterior_ess=dict(min=float(concentration.ess.min()),median=float(concentration.ess.median()),max=float(concentration.ess.max())),
        compute=computing,native_guard_count=int(costs.native_guard_count.sum()),
        max_score_recompute_absolute_difference=float(audit.max_score_absolute_difference.max()),
        max_native_newton_residual_N=float(numerics.max_abs_newton_residual_N.max()),
        max_native_kinematic_residual_m=float(numerics.max_abs_kinematic_residual_m.max()),
        primary_position_contrasts=gc.astype(object).where(pd.notna(gc),None).to_dict('records'))
    write(out/'summary.json',summary)
    lines=['# Main2: unabhängige vollständige Auswertung','',
        'Die Auswertung enthält alle fünf Verfahren, acht Kompositionen und drei Probenarten. '
        'Sie ist eine deskriptive Prüfung neuer Historien; keine p-Werte, Konfidenzintervalle oder nachträgliche Gewinnerauswahl.', '',
        '## Zukunfts-Prognosefehler','',
        'Gepoolter RMSE: erst MSE je Episode mitteln, anschließend die Quadratwurzel. Jede native Episode und Modellbank erhält gleiches Gewicht.', '',markdown(globaltable),'',
        'Geplante Positions-MSE-Kontraste (Quotient der aggregierten MSE, keine Mittelung von Einzelquotienten):','',markdown(gc),'',
        'Alle vier Kanäle und Gruppen sind in pooled_losses.csv und contrasts.csv enthalten. mean_episode_rmse ist dort ausschließlich die separat beschriftete sekundäre Kennzahl. '
        'Ein Nullnenner bleibt undefiniert; keine Äquivalenz- oder globale Optimalitätsaussage folgt aus diesen Quotienten.', '',
        '## Vergangenheitsanpassung und Zukunft','',
        f'Von 10368 gepaarten Modell-/Episodenkontexten wählte das kleinere vergangene Zielfunktional {selected} mal den zweiten festen Start. '
        f'In {int(improved.sum())} Fällen sank das vergangene Zielfunktional strikt; in{fractions["position_m"]["past_improved_future_worse"]} davon verschlechterte sich der zukünftige Positions-MSE. '
        f'Über alle Kontexte wurde er {fractions["position_m"]["future_better"]}mal kleiner und {fractions["position_m"]["future_worse"]}mal größer. '
        'Gleichstände und identische Prognosen bleiben enthalten.', '',
        f'Alle {len(optimizers)} Optimiererstarts sind erhalten: {budget} Budgetfälle und {failed} Starts ohne Erfolgsmeldung. '
        'Diese Flags werden weder entfernt noch mit globalem Optimum gleichgesetzt. Die Box-Unterstützung der kontinuierlichen MLE ist breiter als die empirische Partikelunterstützung; dies ist ein struktureller Unterschied der Verfahren.', '',
        'P1024 war in Development nicht als konvergiert belegt und bleibt eine feste Rechenbudgetapproximation. '
        f'Die End-ESS hat Median {concentration.ess.median():.3f}, Minimum {concentration.ess.min():.3f}, Maximum {concentration.ess.max():.3f}. '
        'Hohe Konzentration bedeutet keine physikalisch korrekte Zustandsschätzung oder kalibrierte Unsicherheit.', '',
        '## Physische und rechnerische Kosten','',
        'Mittel pro physischer nativer Episode; nur 3456 Episoden, nicht einmal je Modellbank/Verfahren gezählt:', '',markdown(costsummary),'',
        'Zwei-Port-Absolutarbeit ist Σ|u·Δx| + Σ|N·Δd|, mit tatsächlicher nativer Verschiebung und bekannter diskreter Normalansteuerung. '
        'Vorzeichenbehaftete Komponenten, Reibungsarbeit und ergänzende N·d_dot·dt-Summen bleiben separat in native_costs.csv. '
        'Die vorangehende ideale 4s-C2-Fixture ist kein kraftbegrenzter Roboter und ihre Arbeit ist hier nicht als native Aktuatorarbeit enthalten.', '',
        f'Gemessene summierte Rechenzeit: Partikelfilter samt Prior-/Posterior-Prognosen {computing["particle_total_seconds"]:.2f}s; '
        f'zusätzliche zwei MLE-Starts samt Prognosen {computing["mle_incremental_seconds"]:.2f}s; '
        f'MLE einschließlich benötigter Partikelinitialisierung {computing["mle_including_particle_seconds"]:.2f}s; '
        f'privilegierte Modellhistorie {computing["privileged_seconds"]:.2f}s. '
        'Batchzeiten sind keine Einzelinstanz-Echtzeitgarantie; die separat nicht gemessenen isolierten MLE-Prognosekosten werden nicht erfunden. '
        f'Banklaufzeiten summieren sich auf {computing["summed_bank_current_attempt_seconds"]:.2f}s aktueller Versuche plus '
        f'{computing["summed_prior_attempt_seconds"]:.2f}s erhaltener früherer Versuche.', '',
        '## Integrität und Aussagegrenzen','',
        f'Alle 288 Prognose-NPZs und 51840 Episode-/Methodenzeilen wurden in vier Kanälen aus gespeicherten Zukunftsprognosen und nativen Zielen unabhängig nachgerechnet. '
        f'Maximale absolute Differenz zu CSV-MSE/RMSE/Bias: {summary["max_score_recompute_absolute_difference"]:.3g}. '
        f'Native Guards: {summary["native_guard_count"]}; Newton-Rest maximal {summary["max_native_newton_residual_N"]:.3g}N; '
        f'kinematischer Rest maximal {summary["max_native_kinematic_residual_m"]:.3g}m. '
        'Gespeicherte Partikel- und MLE-Implizitlöserresiduen entsprechen der festgelegten Grenze. Einzelne zukünftige Partikel-Guardtrajektorien werden nicht gespeichert; deshalb wird keine darüber hinausgehende Guardfreiheit behauptet.', '',
        'Der gebundene Main2-Test untersucht drei zuvor isoliert gefittete Banken und vier neue Historienblöcke mit wiederverwendeten geometrischen Vorzeichenfaktoren. '
        'Privilegierte Modellhistorie ist unter Parameterbias keine Fehleruntergrenze. Der bekannte h/h4-Trajektoriendrift von 0.285–0.881mm begrenzt physikalische Genauigkeitsaussagen: verglichen wird das festgelegte 2,5ms-Simulationssystem. '
        'Das ist weder ein erneuter Main1-Bestätigungstest noch ein Nachweis robotischer Aufgabenverbesserung, allgemeiner Materialübertragung oder einer neuen Bayes-Methode. Der echte G1 bleibt ungenutzt.', '',
        '## Artefakte','',
        '[Protokoll](../../costed_benchmark/main2_protocol.md), [Analyseplan](../../reports/main2_analysis_plan.md), '
        '[Sperrliste](analysis_lock.json), [Prüfsummenprüfung](hash_checks.csv), [Rohprognoseabgleich](forecast_audit.csv), '
        '[gepoolte Fehler](pooled_losses.csv), [Kontraste](contrasts.csv), [MLE-Verläufe](nested_mle.csv), '
        '[physische Kosten](native_costs.csv), [Rechenzeiten](compute_by_batch.csv).','']
    for name in ('global_errors','compositions','past_vs_future','particle_ess'):
        lines += [f'![Main2 {name}](figures/main2_{name}.png)','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    return summary


def run(out):
    out=Path(out).resolve();alock=read(out/'analysis_lock.json');checks=[]
    if read(out/'process.json')['status']!='locked':raise ValueError('Fresh locked analysis only')
    verify_hash_maps([('analysis',alock['sources_sha256'])],checks)
    evaluation=Path(alock['evaluation_directory'])
    # This first call refuses all partial banks before reading any score.
    lock,frame,processes,cfg,expected,maps=read_inputs(evaluation,checks)
    started=time.perf_counter();process=dict(status='running',started_unix=time.time())
    (out/'process.json').write_text(json.dumps(process,indent=2)+'\n')
    try:
        native,costs,numerics=audit_native(Path(lock['data_directory']),cfg,expected)
        audit,optimizers,timings,concentration,nested=audit_predictions(evaluation,lock,frame,expected,native)
        pooled=aggregate(frame);contrast=contrasts(frame)
        for filename,df in [('native_costs',costs),('native_numerics',numerics),('forecast_audit',audit),('optimizer_starts',optimizers),
            ('compute_by_batch',timings),('particle_concentration',concentration),('nested_mle',nested),('pooled_losses',pooled),('contrasts',contrast)]:
            df.to_csv(out/(filename+'.csv'),index=False)
        plots(out,pooled,nested,concentration)
        result=report(out,pooled,contrast,costs,optimizers,timings,concentration,nested,audit,processes,numerics)
        verify_hash_maps(maps+[('analysis_final',alock['sources_sha256'])],checks)
        pd.DataFrame(checks).to_csv(out/'hash_checks.csv',index=False)
        process.update(status='complete',native_episodes=3456,rows=51840,hash_checks=len(checks))
    except BaseException as exc:process.update(status='failed',error=repr(exc));raise
    finally:
        process['seconds']=time.perf_counter()-started
        (out/'process.json').write_text(json.dumps(process,indent=2)+'\n')
        files=[p for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json']
        write(out/'artifact_manifest.json',dict(analysis_lock_sha256=sha(out/'analysis_lock.json'),
            evaluation_lock_sha256=sha(evaluation/'evaluation_lock.json'),outputs_sha256={str(p):sha(p) for p in files}))
    print('MAIN2 ANALYSIS COMPLETE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','run'])
    p.add_argument('--evaluation',type=Path,default=ROOT/'paper_study/results/observer_main2')
    p.add_argument('--out',type=Path,default=ROOT/'paper_study/results/statistics_main2')
    args=p.parse_args();freeze(args.evaluation,args.out) if args.stage=='freeze' else run(args.out)
