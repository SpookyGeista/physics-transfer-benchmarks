"""Compare complete locked timestep and nested-particle controls, descriptively.

No model/runner/statistical-analysis imports and no simulation. Stored future
states share 2.5-ms endpoints; N/Ft share interval means. Matched fitted laws
are synthetic truth, never independent material or calibration validation.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

CHANNELS = ('position_m','velocity_m_s','normal_N','friction_N')
BASE_METHODS = ('prior','posterior','privileged_history')
ALL_METHODS = ('prior','posterior','mle_start1','mle_two_start','privileged_history')
CORE_CONFIG = ('normals','cases','profiles_per_history','sign_assignment','durations','strengths_N',
    'future_amplitudes_N','future_duration','geometry','cam_amplitude_m','sensor_sigma',
    'numerical_floor','noise_stream_key','initial_compression_m','mle','maximum_solver_residual_N')
STRATA = ('bank','history','normal','case','sign','duration','strength','sensor','future')
CELL = (*STRATA,'profile','channel')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def jsonable(value):
    if isinstance(value,dict): return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [jsonable(v) for v in value]
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,Path): return str(value)
    return value


def write(path,value):
    with Path(path).open('x') as f: json.dump(jsonable(value),f,indent=2,allow_nan=False);f.write('\n')


def verify(bindings):
    for path,digest in bindings.items():
        if sha(path)!=digest: raise ValueError('Changed source/input: '+path)


def add_bindings(bindings,more):
    for path,digest in more.items():
        if path in bindings and bindings[path]!=digest:
            raise ValueError('Inconsistent shared source/input hash: '+path)
        bindings[path]=digest


def complete_gate(paths):
    """Read status only until every requested run has completed."""
    for root in paths:
        p=Path(root)/'process.json'
        if not p.exists() or read(p).get('status')!='complete':
            raise ValueError('Every requested input must be complete: '+str(root))


def inventory(root,bindings):
    root=Path(root).resolve()
    process,lock,manifest=[read(root/name) for name in ('process.json','lock.json','manifest.json')]
    locksha=sha(root/'lock.json')
    if process['status']!='complete' or process['lock_sha256']!=locksha or manifest['lock_sha256']!=locksha:
        raise ValueError('Incomplete or inconsistent process/lock/manifest')
    cfg=lock['config'];batches={};ensembles={};preparations={}
    for bank,normal,case in itertools.product(cfg['banks'],cfg['normals'],cfg['cases']):
        model=f'bank{bank}_{normal}_{case}'
        preparations[(bank,normal,case,'prior')]=root/'preparations'/f'{model}_prior.npz'
        for history in cfg['history_seeds']:
            preparations[(bank,normal,case,history)]=root/'preparations'/f'{model}_h{history}.npz'
        signs=[s for s in (-1,1) if s in cfg['sign_assignment']]
        for sign,duration,strength in itertools.product(signs,cfg['durations'],cfg['strengths_N']):
            arm=f's{sign:+d}_d{round(duration*1000)}_a{strength:g}'
            ensembles[(bank,normal,case,sign,duration,strength)]=root/'ensembles'/model/(arm+'.npz')
            for history in cfg['history_seeds']:
                batches[(bank,normal,case,history,sign,duration,strength)]=root/'batches'/model/f'h{history}'/(arm+'.npz')
    expected=set(map(str,[*batches.values(),*ensembles.values(),*preparations.values()]))
    if set(manifest['outputs_sha256'])!=expected:
        raise ValueError('Manifest does not cover exactly the declared factorial files')
    if set(map(str,root.rglob('*.npz')))!=expected:
        raise ValueError('Extra or missing NPZ in result directory')
    if process['batches']!=len(batches) or process['files']!=len(expected):
        raise ValueError('Recorded completion counts disagree with inventory')
    add_bindings(bindings,{str(root/name):sha(root/name) for name in ('process.json','lock.json','manifest.json')})
    add_bindings(bindings,lock['source_sha256']);add_bindings(bindings,manifest['outputs_sha256'])
    return dict(root=root,config=cfg,process=process,lock_sha256=locksha,batches=batches,
                ensembles=ensembles,preparations=preparations)


def validate_pair(main,control,family):
    a,b=main['config'],control['config']
    if a['kind']!='main' or b['kind']!=family:
        raise ValueError('Expected main and correctly labeled control run')
    if b['banks']!=[1] or b['history_seeds']!=[71001]:
        raise ValueError('Fixed control subset is bank 1 / history 71001')
    for key in CORE_CONFIG:
        if a[key]!=b[key]: raise ValueError('Changed non-control factor: '+key)
    prior_a=a['prior_seeds'][a['banks'].index(1)]
    prior_b=b['prior_seeds'][b['banks'].index(1)]
    if prior_a!=prior_b: raise ValueError('Controls require the same public history seed')
    if family=='refinement':
        if b['particles']!=a['particles'] or b['dt']>=a['dt'] or not b.get('mle_enabled',True):
            raise ValueError('Refinement must preserve P/methods and use a finer dt')
        ratio=a['dt']/b['dt']
        if not np.isclose(ratio,round(ratio),rtol=0,atol=1e-12): raise ValueError('Non-nested integrator clock')
    elif b['dt']!=a['dt'] or b['particles']==a['particles'] or b.get('mle_enabled',True):
        raise ValueError('Resolution changes only particle budget and omits MLE')
    if not set(control['batches']).issubset(main['batches']): raise ValueError('Unpaired control cells')


def load_npz(path):
    with np.load(path,allow_pickle=False) as f:
        arrays={k:f[k] for k in f.files if k!='metadata_json'}
        meta=json.loads(str(f['metadata_json'].item()))
    for key,value in arrays.items():
        if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
            raise ValueError('Nonfinite/non-numeric array: '+str(path)+'/'+key)
    return arrays,meta


def exact(a,b,label):
    if not np.array_equal(a,b): raise ValueError('Required exact equality failed: '+label)


def nested_particles(reference,current):
    """P changes preserve every overlapping state and propagated particle."""
    small=min(len(reference['initial_pi']),len(current['initial_pi']))
    exact(reference['normal0'],current['normal0'],'nested prior normal state')
    exact(reference['initial_pi'][:small],current['initial_pi'][:small],'nested initial PI coordinates')
    exact(reference['particle_past'][:small],current['particle_past'][:small],'nested past trajectories')
    exact(reference['particle_future'][:,:small],current['particle_future'][:,:small],'nested future trajectories')
    return small


def validate_arrays(batch,meta,ensemble,ensmeta,cfg):
    N,F,E,M,T,C=batch['predictions'].shape
    P,S,SC=ensemble['particle_past'].shape
    methods=ALL_METHODS if cfg.get('mle_enabled',True) else BASE_METHODS
    if tuple(meta['methods'])!=methods or meta['sensors']!=list(cfg['sensor_sigma']) or meta['futures']!=cfg['future_amplitudes_N']:
        raise ValueError('Method/sensor/future labels differ from locked design')
    if (N,F,M,T,C)!=(len(meta['sensors']),len(meta['futures']),len(methods),160,4) or SC!=3 or P!=cfg['particles']:
        raise ValueError('Unexpected prediction/particle axes')
    expected={'truth_future':(F,E,T,C),'truth_past':(E,S,3),'observed':(N,E,S,3),'weights':(N,E,P),
              'noise_standard':(E,S,3),'effective_sigma':(N,3),'truth_initial_pi':(E,3),'normal0':(3,)}
    for key,shape in expected.items():
        if batch[key].shape!=shape: raise ValueError('Wrong batch array shape: '+key)
    if ensemble['particle_future'].shape!=(F,P,T,C): raise ValueError('Wrong future particle shape')
    if meta['channel_order']!=list(CHANNELS) or meta['force_convention']!='mean over each 2.5ms output block; x/v at block end':
        raise ValueError('State/force clock convention differs')
    exact(ensemble['past_times'],np.arange(1,S+1)*.025,'25 ms past output clock')
    exact(ensemble['future_times'],np.arange(1,T+1)*.0025,'2.5 ms future output clock')
    if S!=round(meta['duration']/.025): raise ValueError('Incomplete past window')
    if np.any(batch['weights']<0) or not np.allclose(batch['weights'].sum(axis=2),1,rtol=0,atol=1e-12):
        raise ValueError('Invalid particle weights')
    if meta['particles']!=P or ensmeta['particles']!=P or meta['dt']!=cfg['dt'] or ensmeta['dt']!=cfg['dt']:
        raise ValueError('Metadata numerical setting mismatch')
    indices=np.flatnonzero(np.asarray(cfg['sign_assignment'])==meta['sign'])
    exact(batch['profile_indices'],indices,'profile/sign pairing')
    if meta['ensemble_sha256']!=sha(meta['ensemble_path']): raise ValueError('Unbound ensemble side link')


def metric_rows(batch,meta,reference,refmeta,family,label,dt,particles,methods):
    """All episode/channel losses; forecast drift is distinct from truth error."""
    prediction=batch['predictions'];truth=batch['truth_future']
    mse=np.mean((prediction-truth[None,:,:,None])**2,axis=-2)
    bias=np.mean(prediction-truth[None,:,:,None],axis=-2)
    ess=1/np.sum(batch['weights']**2,axis=2)
    rows=[]
    for ni,sensor in enumerate(meta['sensors']):
        for fi,future in enumerate(meta['futures']):
            for ei,profile in enumerate(batch['profile_indices']):
                for method in methods:
                    mi=meta['methods'].index(method);ri=refmeta['methods'].index(method)
                    drift=prediction[ni,fi,ei,mi]-reference['predictions'][ni,fi,ei,ri]
                    for ci,channel in enumerate(CHANNELS):
                        rows.append(dict(family=family,run=label,dt=dt,particles=particles,bank=meta['bank'],history=meta['history'],
                            normal=meta['normal'],case=meta['case'],sign=meta['sign'],duration=meta['duration'],strength=meta['strength'],
                            sensor=sensor,future=future,profile=int(profile),method=method,channel=channel,
                            mse=float(mse[ni,fi,ei,mi,ci]),episode_rmse=float(np.sqrt(mse[ni,fi,ei,mi,ci])),
                            bias=float(bias[ni,fi,ei,mi,ci]),ess=float(ess[ni,ei]),
                            forecast_drift_rms=float(np.sqrt(np.mean(drift[:,ci]**2))),
                            forecast_drift_max=float(np.max(abs(drift[:,ci])))))
    return rows


def drift_rows(a,b,context,array,channels=None):
    if a.shape!=b.shape: raise ValueError('Unpaired drift shape: '+array)
    diff=np.asarray(b)-np.asarray(a)
    if channels is None:
        return [dict(**context,array=array,channel='state_vector',rms=float(np.sqrt(np.mean(diff**2))),maximum_abs=float(np.max(abs(diff))))]
    return [dict(**context,array=array,channel=channel,rms=float(np.sqrt(np.mean(diff[...,i]**2))),
                 maximum_abs=float(np.max(abs(diff[...,i])))) for i,channel in enumerate(channels)]


def pooled_tables(frame):
    groups=[('all',()),('sensor',('sensor',)),('composition',('normal','case')),
            ('sensor_arm_future',('sensor','duration','strength','future')),
            ('full_stratum',STRATA)]
    rows=[]
    for scope,fields in groups:
        keys=['family','run','dt','particles','method','channel',*fields]
        for values,g in frame.groupby(keys,sort=True,dropna=False):
            record=dict(zip(keys,values));record.update(scope=scope,episodes=len(g),mse=float(g.mse.mean()),
                pooled_rmse=float(np.sqrt(g.mse.mean())),mean_episode_rmse=float(g.episode_rmse.mean()),
                mean_bias=float(g.bias.mean()),mean_ess=float(g.ess.mean()),median_ess=float(g.ess.median()),
                minimum_ess=float(g.ess.min()),maximum_ess=float(g.ess.max()),
                pooled_forecast_drift=float(np.sqrt(np.mean(g.forecast_drift_rms**2))),
                max_forecast_drift=float(g.forecast_drift_max.max()))
            rows.append(record)
    return pd.DataFrame(rows)


def gain_tables(frame):
    keys=['family','run','dt','particles',*CELL]
    table=frame.pivot(index=keys,columns='method',values='mse')
    rows=[]
    for a,b in itertools.combinations(ALL_METHODS,2):
        if a not in table or b not in table: continue
        valid=table[[a,b]].dropna()
        for idx,r in valid.iterrows():
            rows.append(dict(zip(keys,idx),method_a=a,method_b=b,gain=float(r[a]-r[b])))
    gains=pd.DataFrame(rows);stability=[]
    pairkey=[*CELL,'method_a','method_b']
    for family,group in gains.groupby('family'):
        ordering=frame[frame.family==family][['run','dt','particles']].drop_duplicates()
        ordering=ordering.sort_values('dt' if family=='refinement' else 'particles',ascending=family!='refinement')
        for before,after in itertools.combinations(ordering.run,2):
            left=group[group.run==before].set_index(pairkey)
            right=group[group.run==after].set_index(pairkey)
            if set(left.index)!=set(right.index): raise ValueError('Missing paired gain cells')
            for index in left.index:
                a=float(left.loc[index,'gain']);b=float(right.loc[index,'gain'])
                stability.append(dict(zip(pairkey,index),family=family,before=before,after=after,
                    gain_before=a,gain_after=b,gain_change=b-a,strict_sign_flip=bool(a*b<0),
                    zero_touch_or_departure=bool((a==0)!=(b==0))))
    return gains,pd.DataFrame(stability)


def gain_summary(stability):
    rows=[]
    for scope,fields in [('all',()),('sensor',('sensor',)),('full_stratum',STRATA)]:
        keys=['family','before','after','method_a','method_b','channel',*fields]
        for values,g in stability.groupby(keys,sort=True,dropna=False):
            a,b=float(g.gain_before.mean()),float(g.gain_after.mean())
            rows.append(dict(zip(keys,values),scope=scope,episodes=len(g),mean_gain_before=a,mean_gain_after=b,
                pooled_gain_sign_flip=bool(a*b<0),episode_strict_sign_flips=int(g.strict_sign_flip.sum()),
                episode_zero_touches=int(g.zero_touch_or_departure.sum())))
    return pd.DataFrame(rows)


def compare_family(main,controls,family,bindings):
    if not controls: return [],[],[]
    methods=ALL_METHODS if family=='refinement' else BASE_METHODS
    rows=[];drifts=[];nesting=[]
    subset=controls[0]['batches']
    if any(set(c['batches'])!=set(subset) for c in controls): raise ValueError('Control subsets differ')
    baseline_label=f"h={main['config']['dt']:g}" if family=='refinement' else f"P={main['config']['particles']}"
    seen_preps=set();seen_ensembles=set()
    for key in sorted(subset):
        base,bmeta=load_npz(main['batches'][key])
        bank,normal,case,history,sign,duration,strength=key
        enskey=(bank,normal,case,sign,duration,strength)
        ensemble,emeta=load_npz(main['ensembles'][enskey])
        validate_arrays(base,bmeta,ensemble,emeta,main['config'])
        rows.extend(metric_rows(base,bmeta,base,bmeta,family,baseline_label,main['config']['dt'],main['config']['particles'],methods))
        for control in controls:
            cfg=control['config'];label=f"h={cfg['dt']:g}" if family=='refinement' else f"P={cfg['particles']}"
            batch,meta=load_npz(control['batches'][key]);ce,cemeta=load_npz(control['ensembles'][enskey])
            validate_arrays(batch,meta,ce,cemeta,cfg)
            for field in (*STRATA,'futures','sensors','channel_order','force_convention'):
                if field=='sensor' or field=='future': continue
                if meta[field]!=bmeta[field]: raise ValueError('Unpaired metadata factor: '+field)
            exact(batch['profile_indices'],base['profile_indices'],'profile indices')
            exact(batch['noise_standard'],base['noise_standard'],'same noise realization at physical clock')
            exact(batch['effective_sigma'],base['effective_sigma'],'same likelihood scales')
            context=dict(family=family,run=label,dt=cfg['dt'],particles=cfg['particles'],bank=bank,history=history,
                         normal=normal,case=case,sign=sign,duration=duration,strength=strength)
            if family=='resolution':
                shared=nested_particles(ensemble,ce)
                for field in ('truth_initial_pi','normal0','truth_past','truth_future','observed','truth_boundary_xv','truth_boundary_pi','truth_boundary_normal'):
                    exact(batch[field],base[field],'resolution identical '+field)
                nesting.append(dict(**context,shared_particles=shared,exact_overlap=True))
            for array,channels in [('truth_past',CHANNELS[:3]),('truth_initial_pi',None),('normal0',None)]:
                drifts.extend(drift_rows(base[array],batch[array],context,array,channels))
            for fi,future in enumerate(meta['futures']):
                drifts.extend(drift_rows(base['truth_future'][fi],batch['truth_future'][fi],dict(**context,future=future),'truth_future',CHANNELS))
            for prep_kind in ('prior',history):
                pkey=(bank,normal,case,prep_kind);token=(label,pkey)
                if token not in seen_preps:
                    pa,pam=load_npz(main['preparations'][pkey]);pb,pbm=load_npz(control['preparations'][pkey])
                    overlap=min(len(pa['initial_pi']),len(pb['initial_pi']))
                    if pam['seed']!=pbm['seed']: raise ValueError('Changed preparation seed')
                    for ma,mb in zip(pam['history_rows'][:overlap],pbm['history_rows'][:overlap]):
                        for field in ('seed','index','waypoints_m','knot_times_s'):
                            if ma[field]!=mb[field]: raise ValueError('Continuous preparation curve changed')
                    pc=dict(family=family,run=label,dt=cfg['dt'],particles=cfg['particles'],bank=bank,normal=normal,case=case,
                            preparation=str(prep_kind),shared_profiles=overlap)
                    drifts.extend(drift_rows(pa['initial_pi'][:overlap],pb['initial_pi'][:overlap],pc,'prepared_initial_pi'))
                    drifts.extend(drift_rows(pa['normal0'],pb['normal0'],pc,'prepared_normal0'))
                    if family=='resolution':
                        exact(pa['initial_pi'][:overlap],pb['initial_pi'][:overlap],'nested physical preparation')
                        exact(pa['normal0'],pb['normal0'],'nested prepared normal state')
                    seen_preps.add(token)
            token=(label,enskey)
            if token not in seen_ensembles:
                p=min(len(ce['initial_pi']),len(ensemble['initial_pi']))
                ec={k:v for k,v in context.items() if k!='history'}
                drifts.extend(drift_rows(ensemble['particle_past'][:p],ce['particle_past'][:p],ec,'shared_particle_past',CHANNELS[:3]))
                for fi,future in enumerate(meta['futures']):
                    drifts.extend(drift_rows(ensemble['particle_future'][fi,:p],ce['particle_future'][fi,:p],dict(**ec,future=future),'shared_particle_future',CHANNELS))
                seen_ensembles.add(token)
            rows.extend(metric_rows(batch,meta,base,bmeta,family,label,cfg['dt'],cfg['particles'],methods))
    return rows,drifts,nesting


def plots(frame,pooled,gains,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    colors=dict(prior='#777777',posterior='#0072B2',mle_start1='#E69F00',mle_two_start='#D55E00',privileged_history='#009E73')
    fig,axes=plt.subplots(2,2,figsize=(9.4,7.1))
    for col,sensor in enumerate(('clean','noisy')):
        group=pooled[(pooled.family=='refinement')&(pooled.scope=='sensor')&(pooled.sensor==sensor)&(pooled.channel=='position_m')]
        for method,g in group.groupby('method'):
            g=g.sort_values('dt',ascending=False);axes[0,col].plot(g.dt*1000,g.pooled_rmse*1000,'o-',label=method,color=colors[method])
        axes[0,col].set(xlabel='Integrationsschritt (ms)',ylabel='Gepoolte Positions-RMSE (mm)',title=sensor)
        axes[0,col].invert_xaxis();axes[0,col].grid(alpha=.2)
        selected=gains[(gains.family=='refinement')&(gains.sensor==sensor)&(gains.method_a=='posterior')&(gains.method_b=='mle_start1')&(gains.channel=='position_m')]
        for after,g in selected.groupby('after'):
            axes[1,col].scatter(g.gain_before*1e6,g.gain_after*1e6,s=9,alpha=.35,label=after)
        axes[1,col].axhline(0,color='.4',lw=.7);axes[1,col].axvline(0,color='.4',lw=.7)
        axes[1,col].set(xlabel='Früherer MLE1-Vorteil gegenüber PF (mm²)',ylabel='Späterer MLE1-Vorteil gegenüber PF (mm²)')
        axes[1,col].grid(alpha=.2)
    axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=7)
    fig.suptitle('Zeitschrittkontrolle: identische Programme, neu integrierte Vorgeschichten')
    fig.tight_layout();fig.savefig(out/'timestep_controls.png',dpi=180);fig.savefig(out/'timestep_controls.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9.4,6.8))
    for col,sensor in enumerate(('clean','noisy')):
        group=pooled[(pooled.family=='resolution')&(pooled.scope=='sensor')&(pooled.sensor==sensor)&(pooled.channel=='position_m')]
        for method,g in group.groupby('method'):
            g=g.sort_values('particles');axes[0,col].plot(g.particles,g.pooled_rmse*1000,'o-',label=method,color=colors[method])
        axes[0,col].set(xlabel='Partikelzahl',ylabel='Gepoolte Positions-RMSE (mm)',title=sensor);axes[0,col].grid(alpha=.2)
        e=frame[(frame.family=='resolution')&(frame.sensor==sensor)&(frame.method=='posterior')&(frame.channel=='position_m')]
        e=e.drop_duplicates(['run',*STRATA[:-1],'profile'])
        for p,g in e.groupby('particles'):
            q=np.quantile(g.ess,[0,.25,.5,.75,1])
            axes[1,col].plot([p,p],[q[0],q[-1]],color='#0072B2',alpha=.4)
            axes[1,col].plot([p,p],[q[1],q[3]],color='#0072B2',lw=5)
            axes[1,col].scatter([p],[q[2]],color='black',s=18)
        axes[1,col].set(xlabel='Partikelzahl',ylabel='ESS: Median, Quartile und Bereich');axes[1,col].grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Verschachtelte Partikelkontrolle; Fehler und ESS sind verschiedene Größen')
    fig.tight_layout();fig.savefig(out/'particle_controls.png',dpi=180);fig.savefig(out/'particle_controls.pdf');plt.close(fig)


def run(main_path,refinement,resolution,out):
    paths=[Path(p).resolve() for p in [main_path,*refinement,*resolution]]
    if len(set(paths))!=len(paths): raise ValueError('Duplicate run directory')
    complete_gate(paths)
    out=Path(out).resolve()
    if out.exists(): raise FileExistsError('Fresh output directory required')
    if any(out==p or out.is_relative_to(p) for p in paths): raise ValueError('Comparison outputs must be separate from immutable inputs')
    if not refinement and not resolution: raise ValueError('At least one control run required')
    started=time.perf_counter();bindings={str(Path(__file__).resolve()):sha(__file__)}
    main=inventory(paths[0],bindings)
    refs=[inventory(p,bindings) for p in paths[1:1+len(refinement)]]
    ress=[inventory(p,bindings) for p in paths[1+len(refinement):]]
    for family,controls in (('refinement',refs),('resolution',ress)):
        for control in controls: validate_pair(main,control,family)
        field='dt' if family=='refinement' else 'particles'
        if len({r['config'][field] for r in controls})!=len(controls): raise ValueError('Duplicate control numerical setting')
    verify(bindings)
    out.mkdir(parents=True)
    write(out/'input_lock.json',dict(source_sha256=sha(__file__),inputs_sha256=bindings,
        interpretation='Prespecified descriptive controls; no success threshold or significance test',created_unix=time.time()))
    process=dict(status='running',input_lock_sha256=sha(out/'input_lock.json'),started_unix=time.time(),command=sys.argv)
    write(out/'process.json',process)
    try:
        records=[];drifts=[];nesting=[]
        for family,controls in (('refinement',refs),('resolution',ress)):
            r,d,n=compare_family(main,controls,family,bindings);records.extend(r);drifts.extend(d);nesting.extend(n)
        frame=pd.DataFrame(records);state=pd.DataFrame(drifts)
        pooled=pooled_tables(frame);gains,stability=gain_tables(frame);stable_summary=gain_summary(stability)
        for name,data in [('episode_metrics',frame),('state_drifts',state),('pooled_metrics',pooled),
                          ('paired_gains',gains),('gain_stability',stability),('gain_stability_summary',stable_summary),
                          ('nested_particle_checks',pd.DataFrame(nesting))]:
            data.to_csv(out/(name+'.csv'),index=False)
        global_rows=pooled[(pooled.scope=='sensor')&(pooled.channel=='position_m')]
        summary=dict(status='complete',paired_subset=dict(banks=[1],history_seeds=[71001],compositions=len(main['config']['normals'])*len(main['config']['cases'])),
            episode_method_channel_rows=len(frame),gain_rows=len(gains),gain_stability_rows=len(stability),
            raw_batches_per_control=[dict(path=r['root'],batches=len(r['batches'])) for r in refs+ress],
            all_declared_control_cells_retained=True,nested_checks=len(nesting),
            position_pooled_by_sensor=global_rows.where(pd.notnull(global_rows),None).to_dict('records'),
            paired_gain_global=stable_summary[(stable_summary.scope=='sensor')&(stable_summary.channel=='position_m')].where(pd.notnull(stable_summary),None).to_dict('records'),
            arithmetic='MSE is averaged over time then equal episode factors; pooled RMSE is its square root. No epsilon, p-value or success threshold.',
            physical_scope='Same fitted laws as synthetic truth. Reference drift is numerical drift, not physical material or calibration error.',
            force_clock='State endpoints and interval-mean N/Ft at common 2.5ms; normal/PI preparation re-integrated at each dt.',
            sign_definition='Gain=MSE(method_a)-MSE(method_b), positive favors b; strict flips require opposite nonzero signs; zero touches separate.',
            original_source_or_result_changes=False)
        # Drop structural NaNs introduced by heterogeneous stratum columns.
        summary=json.loads(json.dumps(summary,default=lambda x:jsonable(x)))
        def finite_json(x):
            if isinstance(x,dict):return {k:finite_json(v) for k,v in x.items()}
            if isinstance(x,list):return [finite_json(v) for v in x]
            if isinstance(x,float) and not np.isfinite(x):return None
            return x
        summary=finite_json(summary)
        plots(frame,pooled,stability,out)
        lines=['# Vergleich der Zeit- und Partikelkontrollen','',
            'Alle angeforderten Läufe waren vor der Auswertung vollständig und wurden über Quellen, Lock, Prozess und vollständige Rohdateimanifeste gebunden. Dieser Bericht enthält ausschließlich deskriptive Kontrollen; keine neue Erfolgsschwelle, Signifikanzprüfung oder nachträgliche Methodenwahl.','',
            'Verglichen wird die festgelegte Teilmenge Bank1/Historie71001 mit allen Kompositionen, Vorzeichen, Probeprogrammen, Sensorbedingungen und Zukunftsanregungen. h stammt aus dem Hauptlauf. Zustände werden an gemeinsamen2,5-ms-Endpunkten verglichen; N/Ft sind Mittelwerte der jeweiligen2,5-ms-Blöcke. Sämtliche Materialvorgeschichten wurden beim feineren Integrationsschritt neu integriert.','',
            '| Kontrolle | Einstellung | Sensor | Methode | gepoolte Positions-RMSE (mm) |','|---|---|---|---|---:|']
        for _,r in global_rows.iterrows(): lines.append(f'| {r.family} | {r.run} | {r.sensor} | {r.method} | {r.pooled_rmse*1000:.6g} |')
        lines+=['','`episode_metrics.csv` enthält jeden Kanal und jede Episode einschließlich aller MLE-Ergebnisse. `state_drifts.csv` trennt vorbereitete Normal-/PI-Zustände, vergangene und zukünftige synthetische Wahrheit sowie gemeinsame Partikeltrajektorien. Normalzustandskoordinaten hängen von der Modellfamilie ab und dürfen nicht als ein universeller physikalischer Fehler über Familien zusammengefasst werden.','',
            '`paired_gains.csv` enthält alle Methodenpaare. Ein positiver Gewinn MSE(A)−MSE(B) begünstigt B. `gain_stability.csv` vergleicht jedes Raster-/Budgetpaar auf identischen Episoden; strikte Vorzeichenwechsel und Nullkontakte sind getrennt. Die vollständigen Faktoraufschlüsselungen und gepoolten Vorzeichen stehen in den Summary-CSV-Dateien. Kleine Vorzeichenwechsel erhalten keine nachträglich erfundene Relevanzschwelle.','',
            'Bei den Partikelkontrollen stimmen die überlappenden Anfangskoordinaten und vollständigen gemeinsamen Partikeltrajektorien exakt überein. Unabhängige Wahrheit, Beobachtungen und Rauschströme sind exakt dieselben; nur die endliche Priorbank wird erweitert oder gekürzt. ESS wird getrennt vom Prognosefehler berichtet und belegt weder Posterior-Kalibrierung noch Konvergenz.','',
            'Die Referenz ist jeweils dasselbe gefittete Modell. Diese Kontrolle misst deshalb numerische beziehungsweise endliche-Support-Abhängigkeit, keine Genauigkeit gegenüber einem physischen Material. Anfangszustandsdrift beim Verfeinern wird dokumentiert, nicht durch einen Reset entfernt. Alle Originaldateien bleiben unverändert.','',
            '![Zeitkontrolle](timestep_controls.png)','','![Partikelkontrolle](particle_controls.png)','']
        (out/'REPORT.md').write_text('\n'.join(lines))
        verify(bindings)
        summary.update(seconds=time.perf_counter()-started,unique_source_and_input_hashes=len(bindings),hash_checks_before_and_after=2*len(bindings))
        write(out/'summary.json',summary)
        artifacts={str(p):sha(p) for p in out.iterdir() if p.is_file() and p.name!='process.json'}
        write(out/'manifest.json',dict(input_lock_sha256=process['input_lock_sha256'],outputs_sha256=artifacts))
        (out/'process.json').write_text(json.dumps(dict(process,status='complete',seconds=summary['seconds'],rows=len(frame)),indent=2)+'\n')
        print(json.dumps({k:summary[k] for k in ('status','episode_method_channel_rows','seconds','unique_source_and_input_hashes')},indent=2))
    except BaseException as exc:
        partial={str(p):sha(p) for p in out.iterdir() if p.is_file() and p.name!='process.json'}
        (out/'process.json').write_text(json.dumps(dict(process,status='failed',error=repr(exc),seconds=time.perf_counter()-started,partial_outputs_sha256=partial),indent=2)+'\n')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main',type=Path,required=True)
    parser.add_argument('--refinement',type=Path,nargs='+',default=[])
    parser.add_argument('--resolution',type=Path,nargs='+',default=[])
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();run(a.main,a.refinement,a.resolution,a.out)
