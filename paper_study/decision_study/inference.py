"""Causal scalar particle inference and finite predictive-risk planning.

No reference/preparation imports; inputs stop at the common decision sample.
All later commands are computed from that sample. Fitted coefficients are fixed.
"""
import numpy as np
from scipy.special import logsumexp
from research_next.src.state_id import StateID,CompositeStateID
from paper_study.decision_study.common import REPO,BRANCHES,cam,steps,feedback,branch_commands


def model_paths(nw,fw,cfg):
    base=REPO/'research_next/results/round1/models/state_id'
    return tuple(base/f'{w}/r{cfg["model_seed"]}/{cfg["calibration_noise"]}/blind_selection/model.json'
                 for w in (nw,fw))


def load_model(nw,fw,cfg):
    a,b=model_paths(nw,fw,cfg)
    return CompositeStateID(StateID.load(a).normal_model,StateID.load(b).friction_model)


def prior_path(nw,fw,cfg):
    return REPO/cfg['prior_bank_directory']/f'{nw}_{fw}/r{cfg["model_seed"]}/{cfg["calibration_noise"]}.npz'


def load_prior(nw,fw,cfg):
    with np.load(prior_path(nw,fw,cfg),allow_pickle=False) as bank:
        grid,probability=bank['grid'].copy(),bank['weights'].copy()
    if (grid.shape!=(cfg['particles'],) or probability.shape!=grid.shape or
        not np.isfinite(grid).all() or not np.isfinite(probability).all() or
        np.any(np.diff(grid)<=0) or np.any(probability<=0)):
        raise ValueError('Invalid fixed shared prior bank')
    return grid,probability/probability.sum()


def clone_state(state):
    return {k:np.array(v,copy=True) for k,v in state.items()}


def local_inputs(xv,d,rate,amplitude,geometry):
    g,slope=cam(xv[:,0],amplitude,geometry)
    return d+g,rate+slope*xv[:,1],slope


def motion_step(model,xv,state,u,d,rate,amplitude,geometry,cfg):
    delta,local_rate,slope=local_inputs(xv,d,rate,amplitude,geometry)
    N,Ft,newstate=model.step(delta,local_rate,xv[:,1],state,dt=cfg['dt'])
    vn=xv[:,1]+cfg['dt']*(u-slope*N+Ft)/cfg['mass_kg']
    newxv=np.column_stack((xv[:,0]+cfg['dt']*vn,vn))
    if not all(np.isfinite(v).all() for v in (newxv,N,Ft,*newstate.values())):
        raise FloatingPointError('Nonfinite prediction; no particles removed')
    return newxv,newstate,N,Ft


def sensor_prediction(model,xv,state,amplitude,geometry,cfg):
    """Probe-rate measurement before any future command invocation, no update."""
    delta,rate,_=local_inputs(xv,cfg['initial_compression_m'],0.,amplitude,geometry)
    normal_state={k[len('normal__'):]:v for k,v in state.items() if k.startswith('normal__')}
    N=model.normal_model.step(delta,rate,xv[:,1],normal_state,dt=cfg['dt'])[0]
    return np.column_stack((xv,N))


def normalize_logweight(logweight):
    if np.isnan(logweight).any() or np.isposinf(logweight).any() or np.all(np.isneginf(logweight),axis=-1).any():
        raise FloatingPointError('Invalid likelihood or zero total support')
    return logweight-logsumexp(logweight,axis=-1,keepdims=True)


def posterior(logprior,predicted,observed,sigma):
    """One complete future observation sequence; never difference likelihoods.

    predicted E,P,J,3; observed E,K,J,3; return E,K,P log probabilities.
    """
    if predicted.ndim!=4 or observed.ndim!=4 or predicted.shape[0]!=observed.shape[0] or predicted.shape[2:]!=observed.shape[2:]:
        raise ValueError('Expected matching sensor sequence axes')
    error=(observed[:,:,None,:,:]-predicted[:,None,:,:,:])/sigma
    ll=-.5*np.sum(error*error,axis=(-1,-2))
    return normalize_logweight(logprior[:,None,:]+ll)


def observable_prefix(arrays,cfg):
    common,_,_=steps(cfg)
    times=np.asarray(arrays['observation_steps'])
    mask=times<=common
    result={k:np.array(arrays[k],copy=True) for k in ('amplitude','geometry','excitation_sign')}
    result.update({k:np.array(arrays[k][:,:common],copy=True) for k in ('u','d','d_dot')})
    result.update(observations=np.array(arrays['observations'][:,mask],copy=True),
                  observation_steps=times[mask].copy())
    expected=np.arange(0,common+1,cfg['observation_steps'])
    if not np.array_equal(result['observation_steps'],expected):raise ValueError('Missing common sensor frames')
    n=result['u'].shape[0]
    if result['u'].shape!=(n,common) or result['observations'].shape!=(n,len(expected),3):
        raise ValueError('Incorrect prefix axes')
    if any(result[k].shape!=(n,common) for k in ('d','d_dot')) or any(result[k].shape!=(n,) for k in ('amplitude','geometry','excitation_sign')):
        raise ValueError('Incorrect public command/geometry axes')
    if not all(np.isfinite(x).all() for x in result.values()):raise ValueError('Nonfinite observable prefix')
    if not np.all(result['d']==cfg['initial_compression_m']) or not np.all(result['d_dot']==0.):
        raise ValueError('Common observations must precede any future rate change')
    return result


def filter_prefix(model,grid,prior,inputs,cfg):
    if set(inputs)!={'amplitude','geometry','excitation_sign','u','d','d_dot','observations','observation_steps'}:
        raise ValueError('Prefix input whitelist violated')
    n=inputs['u'].shape[0];p=len(grid)
    xv=np.zeros((n*p,2))
    state=model.initial(np.full(n*p,cfg['initial_compression_m']),np.zeros(n*p),np.zeros(n*p))
    coordinate='friction__mix' if model.friction_family=='maxwell' else 'friction__age'
    state[coordinate]=np.tile(grid,n)
    logweight=np.broadcast_to(np.log(prior),(n,p)).copy()
    amplitude=np.repeat(inputs['amplitude'],p);geometry=np.repeat(inputs['geometry'],p)
    sigma=np.hypot(cfg['sensor_sigma'],cfg['numerical_floor'])
    bytime={int(t):i for i,t in enumerate(inputs['observation_steps'])}
    trace=[np.exp(logweight)];work=np.zeros((n,p));predicted_trace=[]
    for t in range(inputs['u'].shape[1]):
        u,d,rate=[np.repeat(inputs[k][:,t],p) for k in ('u','d','d_dot')]
        before=xv
        xv,state,N,Ft=motion_step(model,xv,state,u,d,rate,amplitude,geometry,cfg)
        work+=((abs(u*before[:,1])+abs(N*rate))*cfg['dt']).reshape(n,p)
        if t+1 in bytime:
            predicted=sensor_prediction(model,xv,state,amplitude,geometry,cfg).reshape(n,p,1,3)
            observed=inputs['observations'][:,bytime[t+1]][:,None,None,:]
            logweight=posterior(logweight,predicted,observed,sigma)[:,0,:]
            trace.append(np.exp(logweight));predicted_trace.append(predicted[:,:,0])
    return dict(xv=xv,state=state,logweight=logweight,weights=np.exp(logweight),
                weight_trace=np.stack(trace,axis=1),prefix_candidate_work_J=work,
                predicted_sensor_trace=np.stack(predicted_trace,axis=2))


def forecast_branch(model,snapshot,inputs,branch,cfg):
    n,p=snapshot['weights'].shape;common,later,horizon=steps(cfg)
    held,bias=feedback(inputs['observations'][:,-1],inputs['amplitude'],inputs['geometry'],cfg)
    commands=branch_commands(branch,held,bias,inputs['excitation_sign'],cfg)
    release=commands['release_step'];extra=release-common
    xv=snapshot['xv'].copy();state=clone_state(snapshot['state'])
    amp=np.repeat(inputs['amplitude'],p);geom=np.repeat(inputs['geometry'],p)
    future=np.empty((n,p,horizon,4));sensors=[];sensor_times=[]
    work=np.zeros((n,p));probe_work=np.zeros((n,p));guards=np.zeros((n,p),dtype=int)
    limits=cfg['guard']
    for j,t in enumerate(range(common,release+horizon)):
        u,d,rate=[np.repeat(commands[k][:,j],p) for k in ('u','d','d_dot')]
        before=xv
        xv,state,N,Ft=motion_step(model,xv,state,u,d,rate,amp,geom,cfg)
        increment=((abs(u*before[:,1])+abs(N*rate))*cfg['dt']).reshape(n,p)
        work+=increment
        if t<release:probe_work+=increment
        else:future[:,:,t-release]=np.column_stack((xv,N,Ft)).reshape(n,p,4)
        if t+1<=release and (t+1)%cfg['observation_steps']==0:
            sensors.append(sensor_prediction(model,xv,state,amp,geom,cfg).reshape(n,p,3))
            sensor_times.append(t+1)
        guarded=(abs(xv[:,0])>limits['max_abs_position_m'])|(abs(xv[:,1])>limits['max_abs_velocity_m_s'])|(N<limits['min_normal_N'])
        guards+=guarded.reshape(n,p)
    return dict(future=future,sensors=np.stack(sensors,axis=2) if sensors else np.empty((n,p,0,3)),
        sensor_steps=np.asarray(sensor_times,dtype=int),probe_work=probe_work,total_remaining_work=work,
        extra_seconds=np.full(n,extra*cfg['dt']),candidate_guard_steps=guards,commands=commands)


def variance_risk(weights,future_x):
    """weights E,K,P; future_x E,P,H. Centering avoids cancellation."""
    mean=np.einsum('ekp,eph->ekh',weights,future_x)
    # Centering each E,H at a fixed candidate keeps variance nonnegative by construction.
    center=future_x[:,:1,:]
    centered=future_x-center
    first=np.einsum('ekp,eph->ekh',weights,centered)
    second=np.einsum('ekp,eph->ekh',weights,centered*centered)
    variance=second-first*first
    tol=128*np.finfo(float).eps*np.maximum(second,1e-30)
    if np.any(variance < -tol):raise FloatingPointError('Substantial negative predictive variance')
    return np.mean(np.maximum(variance,0.),axis=-1),mean


def branch_values(snapshot,branches,cfg):
    p0=snapshot['weights'];logp=snapshot['logweight'];n,p=p0.shape;k=cfg['fantasies']
    rng=np.random.default_rng(cfg['fantasy_seed'])
    # Shared across actions; RNG never accesses a real preparation/episode identity.
    quantiles=rng.random((n,k));drawn=np.sum(quantiles[:,:,None]>np.cumsum(p0,axis=1)[:,None,:],axis=-1)
    drawn=np.minimum(drawn,p-1)
    maxframes=max(b['sensors'].shape[2] for b in branches.values())
    noise=rng.normal(size=(n,k,maxframes,3));sigma=np.hypot(cfg['sensor_sigma'],cfg['numerical_floor'])
    risks=[];informations=[];physical=[];works=[];seconds=[];mcse=[]
    for name in BRANCHES:
        b=branches[name];frames=b['sensors'].shape[2]
        physical_risk,_=variance_risk(p0[:,None,:],b['future'][...,0])
        if frames:
            generated=b['sensors'][np.arange(n)[:,None],drawn]+noise[:,:,:frames]*sigma
            logw=posterior(logp,b['sensors'],generated,sigma);w=np.exp(logw)
            values,_=variance_risk(w,b['future'][...,0])
            # Expected KL on original hypothesis identities, not current-state entropy.
            info=np.sum(w*(logw-logp[:,None,:]),axis=-1)
            if np.any(info < -1e-12):raise FloatingPointError('Negative posterior KL')
            information=np.mean(np.maximum(info,0.),axis=1)
            risk=values.mean(axis=1);error=values.std(axis=1,ddof=1)/np.sqrt(k)
        else:
            risk=physical_risk[:,0];information=np.zeros(n);error=np.zeros(n)
        risks.append(risk);informations.append(information);physical.append(physical_risk[:,0]);mcse.append(error)
        works.append(np.sum(p0*b['probe_work'],axis=1));seconds.append(b['extra_seconds'])
    return dict(risk_m2=np.stack(risks,axis=1),information_nats=np.stack(informations,axis=1),
        physical_risk_m2=np.stack(physical,axis=1),planned_extra_probe_work_J=np.stack(works,axis=1),
        extra_seconds=np.stack(seconds,axis=1),fantasy_risk_mcse_m2=np.stack(mcse,axis=1),
        fantasy_particle_indices=drawn,fantasy_noise=noise)


def select_policies(values,cfg):
    risk=values['risk_m2'];n,a=risk.shape
    if a!=len(BRANCHES) or any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError('Nonfinite planning values')
    rng=np.random.default_rng(cfg['policy_seed']);uniform=rng.random(n)
    records={}
    for budget in cfg['budgets']:
        eligible=(values['extra_seconds']/budget['time_scale_s']<=budget['max_scaled_time']+1e-14)&(
            values['planned_extra_probe_work_J']/budget['work_scale_J']<=budget['max_scaled_work']+1e-14)
        if not eligible[:,0].all():raise ValueError('Release must always be admissible')
        policies={}
        for fixed,name in enumerate(BRANCHES):
            selected=np.where(eligible[:,fixed],fixed,0)
            policies['fixed_'+name]=np.eye(a)[selected]
        probability=eligible/eligible.sum(axis=1,keepdims=True)
        selected=np.sum(uniform[:,None]>np.cumsum(probability,axis=1),axis=1)
        policies['uniform']=np.eye(a)[np.minimum(selected,a-1)]
        policies['uniform_expected']=probability
        for name,score in [('bayes_risk',risk),('information_gain',-values['information_nats']),
                           ('physical_only',values['physical_risk_m2'])]:
            selected=[]
            for e in range(n):
                allowed=np.flatnonzero(eligible[e])
                winner=min(allowed,key=lambda j:(score[e,j],values['extra_seconds'][e,j],
                    values['planned_extra_probe_work_J'][e,j],j))
                selected.append(winner)
            policies[name]=np.eye(a)[selected]
        records[budget['name']]=dict(eligible=eligible,policies=policies)
    return records


def actual_forecast(snapshot,branch,arrays,cfg):
    """Causal later PROBE sensors only; no state/force/future-sensor reads."""
    times=np.asarray(arrays['observation_steps'])
    wanted=branch['sensor_steps'];positions=[]
    for t in wanted:
        hit=np.flatnonzero(times==t)
        if len(hit)!=1:raise ValueError('Missing/duplicate later probe frame')
        positions.append(hit[0])
    logw=snapshot['logweight']
    if len(wanted):
        observed=np.asarray(arrays['observations'])[:,positions][:,None,:,:]
        logw=posterior(logw,branch['sensors'],observed,np.hypot(cfg['sensor_sigma'],cfg['numerical_floor']))[:,0]
    weights=np.exp(logw)
    prediction=np.einsum('ep,epth->eth',weights,branch['future'])
    return prediction,weights
