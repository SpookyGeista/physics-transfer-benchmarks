"""Classical bounded initial-state maximum likelihood, separate development API.

Three unknown PI deflections, fixed fitted coefficients and fresh known normal
preparation. This is an off-grid point estimator, not a Bayesian posterior or a
new architecture. The box prior permits more states than the reachable-history
empirical prior; comparisons must disclose that different structural support.
Only past causal x/v/N and public commands enter the objective. The complete
probe window is retained (a short-window initial-state/MHE formulation).

No task execution or fitting of constitutive parameters occurs on import. This
module is not part of the already frozen particle-resolution diagnostic.
"""
from __future__ import annotations
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from evaluate import PUBLIC_KEYS, PUBLIC_META, clone_state, mechanical_start, forecast_moments

DEFAULTS=dict(max_nfev=60,max_residual_calls=256,max_seconds_per_episode=3.,
    ftol=1e-8,xtol=1e-8,gtol=1e-8,numerical_floor=[1e-6,1e-5,1e-4],max_solver_residual_N=1e-7)


class _BudgetExhausted(RuntimeError):
    pass


def _validate_public(inputs,meta,model):
    expected=set(PUBLIC_KEYS)|{'probe_steps','observations','observation_steps'}
    if set(inputs)!=expected or set(meta)!=set(PUBLIC_META):
        raise ValueError('bounded MLE accepts only the explicit public whitelist')
    if meta['dt']!=model.dt or meta['mass_kg']!=model.mass:
        raise ValueError('model and recorded command timestep/mass must agree')
    steps=np.asarray(inputs['observation_steps'])
    end=inputs['probe_steps'];count,total=inputs['u'].shape
    if count<1 or not 0<=end<total or steps.ndim!=1 or not np.issubdtype(steps.dtype,np.integer):
        raise ValueError('invalid causal window')
    if not len(steps) or steps[0]!=0 or steps[-1]!=end or np.any(np.diff(steps)<=0):
        raise ValueError('unique initial/boundary observations required, no future measurements')
    if inputs['observations'].shape!=(count,len(steps),3):raise ValueError('observation shape mismatch')
    if not all(np.isfinite(value).all() for value in inputs.values()):raise ValueError('nonfinite public input')


def _past_rollout(model,inputs,episode,initial_pi,initial_delta):
    """Pure rollout; observed N never clamps the model or overwrites its state."""
    xv=np.zeros((1,2));state=model.initial(1,initial_delta,np.asarray(initial_pi).reshape(1,3))
    steps=np.asarray(inputs['observation_steps'])
    indices={int(step):i for i,step in enumerate(steps)}
    predicted=[];max_residual=0.
    amp=inputs['amplitude'][episode:episode+1];geom=inputs['geometry'][episode:episode+1]
    for t in range(inputs['probe_steps']):
        commands=[inputs[k][episode:episode+1,t] for k in ('u','d','d_dot')]
        xv,state,_,_,residual=model.step(xv,state,*commands,amp,geom)
        max_residual=max(max_residual,float(abs(residual).max()))
        if t+1 in indices:
            # Boundary N is measured after the future rate command is invoked.
            N=model.normal_force(xv,state,inputs['d'][episode:episode+1,t+1],
                                 inputs['d_dot'][episode:episode+1,t+1],amp,geom)
            predicted.append(np.r_[xv[0],N[0]])
    observations=np.array(predicted).reshape(len(steps)-1,3)
    return observations,xv,state,max_residual


def fit_initial_state(model,inputs,public_meta,prior_states,starting_weights,*,config=None):
    """Fit each episode from the best causally weighted public prior particle.

    prior_states is the shared independently generated (P,3) initial-state bank;
    starting_weights is the causal boundary weight matrix (E,P). These determine
    one deterministic optimizer start only. No future weights/scores are allowed.
    State components q_i=z_i/(mu_i*N0/k_i) have the fixed box [-1,1]. N0 is the
    fitted fresh normal prediction at rate zero, never a boundary measurement.
    A uniform density on this box is the declared prior (constant inside), so
    this is bounded MLE. No empirical-prior KDE term is silently introduced.
    Wall budget is checked between residual calls. One initial and one final
    rollout are mandatory; this is not a hard real-time execution guarantee.
    """
    _validate_public(inputs,public_meta,model)
    cfg={**DEFAULTS,**(config or {})}
    for key in ('max_nfev','max_residual_calls'):
        value=cfg[key]
        if isinstance(value,bool) or int(value)!=value or value<1:raise ValueError(f'invalid {key}')
    if not np.isfinite(cfg['max_seconds_per_episode']) or cfg['max_seconds_per_episode']<=0:
        raise ValueError('positive finite wall budget required')
    if not np.isfinite(cfg['max_solver_residual_N']) or cfg['max_solver_residual_N']<=0:
        raise ValueError('positive finite solver residual tolerance required')
    for key in ('ftol','xtol','gtol'):
        if not np.isfinite(cfg[key]) or cfg[key]<=np.finfo(float).eps:raise ValueError(f'invalid {key}')
    count=inputs['u'].shape[0]
    prior=np.asarray(prior_states,dtype=float);weights=np.asarray(starting_weights,dtype=float)
    if prior.ndim!=2 or prior.shape[1]!=3 or not len(prior) or not np.isfinite(prior).all():
        raise ValueError('finite shared prior states required')
    if weights.shape!=(count,len(prior)) or not np.isfinite(weights).all() or np.any(weights<0) or np.any(weights.sum(axis=1)<=0):
        raise ValueError('causal start-weight dimensions/probabilities invalid')
    bounds=model.pi.alpha*model.prefix_load(public_meta['initial_compression_m'])
    if np.any(bounds<=0):raise ValueError('positive initial normal load and state capacities required')
    normalized=prior/bounds
    if np.max(abs(normalized))>1.+1e-10:raise ValueError('initial prior state lies outside the declared box')
    normal_sigma=[public_meta['sensor_parameters'][key] for key in ('sigma_x_m','sigma_v_m_s','sigma_normal_N')]
    sensor=np.asarray(normal_sigma);floor=np.asarray(cfg['numerical_floor'])
    if sensor.shape!=(3,) or floor.shape!=(3,) or not np.isfinite(sensor).all() or np.any(sensor<0) or not np.isfinite(floor).all() or np.any(floor<=0):
        raise ValueError('invalid likelihood noise scales')
    sigma=np.hypot(sensor,floor)
    start_indices=np.argmax(weights,axis=1)  # fixed first index for ties
    selected=np.empty((count,3));boundary_xv=np.empty((count,2))
    boundary_state={key:np.empty((count,3)) for key in ('normal','pi')}
    diagnostics=[]
    for episode in range(count):
        started=time.perf_counter();deadline=started+cfg['max_seconds_per_episode']
        q0=np.clip(normalized[start_indices[episode]],-1.,1.)
        observed=inputs['observations'][episode,1:]
        best={'cost':np.inf,'q':q0.copy()};calls=0
        def objective(q):
            nonlocal calls
            # The first evaluation is always retained, even if JIT startup uses
            # the wall allowance. Thereafter the fixed budget is enforced.
            if calls and (calls>=cfg['max_residual_calls'] or time.perf_counter()>deadline):
                raise _BudgetExhausted('wall or residual-call budget exhausted')
            predicted,_,_,solver_residual=_past_rollout(model,inputs,episode,np.asarray(q)*bounds,
                                                       public_meta['initial_compression_m'])
            if solver_residual>cfg['max_solver_residual_N']:
                raise FloatingPointError('MLE past-trial solver residual exceeds fixed tolerance')
            residual=((predicted-observed)/sigma).ravel()
            if not np.isfinite(residual).all():raise FloatingPointError('nonfinite MLE trajectory; retain failed trial')
            calls+=1;cost=float(np.dot(residual,residual))
            if cost<best['cost']:best.update(cost=cost,q=np.array(q,copy=True))
            return residual
        initial_residual=objective(q0)
        initial_cost=float(np.dot(initial_residual,initial_residual))
        status='unidentified_no_probe';message='No post-initial sample: start retained without optimization.'
        nfev=0;success=False;budget=False
        if inputs['probe_steps']>0:
            try:
                result=least_squares(objective,q0,bounds=(-np.ones(3),np.ones(3)),method='trf',
                    jac='2-point',x_scale=1.,max_nfev=cfg['max_nfev'],
                    ftol=cfg['ftol'],xtol=cfg['xtol'],gtol=cfg['gtol'])
                nfev=int(result.nfev);success=bool(result.success)
                budget=bool(result.status==0)
                status='converged' if success else 'optimizer_budget_or_nonconvergence'
                message=str(result.message)
            except _BudgetExhausted as exc:
                status='budget_exhausted_best_retained';message=str(exc);budget=True
        q=best['q'];selected[episode]=q*bounds
        _,xv,state,residual=_past_rollout(model,inputs,episode,selected[episode],public_meta['initial_compression_m'])
        if residual>cfg['max_solver_residual_N']:
            raise FloatingPointError('MLE final past-rollout solver residual exceeds fixed tolerance')
        boundary_xv[episode]=xv[0]
        for key in boundary_state:boundary_state[key][episode]=state[key][0]
        diagnostics.append(dict(episode=episode,starting_particle=int(start_indices[episode]),
            status=status,message=message,solver_success=success,budget_exhausted=budget,
            optimizer_nfev=nfev,residual_calls=calls,initial_objective=initial_cost,best_objective=best['cost'],
            seconds=time.perf_counter()-started,max_solver_residual_N=residual,
            normalized_state=q.copy(),initial_normal_N=float(model.prefix_load(public_meta['initial_compression_m']))))
    return dict(initial_pi_state=selected,normalized_state=selected/bounds,
        boundary_xv=boundary_xv,boundary_state=boundary_state,diagnostics=diagnostics,
        config=cfg,effective_sigma=sigma,
        scope='Bounded MLE with uniform box support, one best empirical-posterior start. Fixed law parameters; no calibrated uncertainty.')


def forecast(model,inputs,fitted,*,starts=('filtered_particles',),return_diagnostics=False):
    """Optional deterministic forecast wrapper; no new observations consumed."""
    if not starts or len(set(starts))!=len(starts) or any(s not in ('filtered_particles','measured_start') for s in starts):
        raise ValueError('invalid forecast starts')
    count=len(fitted['boundary_xv']);predictions={};diagnostics={}
    for start in starts:
        xv=mechanical_start(fitted['boundary_xv'][:,None,:],inputs['observations'][:,-1,:2],start)
        mean,_,residual=forecast_moments(model,xv,clone_state(fitted['boundary_state']),inputs,
                                       {'bounded_initial_state_mle':np.ones((count,1))})
        if residual>fitted['config']['max_solver_residual_N']:
            raise FloatingPointError('MLE future-rollout solver residual exceeds fixed tolerance')
        predictions[start+'/bounded_initial_state_mle']=mean['bounded_initial_state_mle']
        diagnostics[start+'_max_solver_residual_N']=residual
    return (predictions,diagnostics) if return_diagnostics else predictions
