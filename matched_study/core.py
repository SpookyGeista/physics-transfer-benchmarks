"""Fixed-law matched-model experiment primitives, without target-data access.

All normal histories start virgin at zero compression, then remain continuous
through the public ramp, hold, friction fixture and coupled experiment. Only
fitted laws from existing calibration banks are loaded. No original file is
modified. The experimental runner is responsible for split/lock authorization.
"""
from pathlib import Path
import os
import sys
import time

os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/physics-transfer-matched-v1-numba')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
from numba import njit
from scipy.optimize import least_squares
from paper_study.costed_benchmark.inference import (
    load_model, _normal_force, _normal_next, _pi_mean, _coupled_step,
)
from paper_study.costed_benchmark.profiles import profile_commands

INITIAL_COMPRESSION = .020
CANONICAL_DT = .0025
OBSERVATION_DT = .025
SOLVER_TOLERANCE_N = 1e-7
MAX_NFEV = 120
MAX_RESIDUAL_CALLS = 512
EVENT_TOLERANCE_M = 1e-14


def _clock(dt):
    if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Positive finite timestep required')
    ratio = CANONICAL_DT / dt
    if not np.isclose(ratio, round(ratio), atol=1e-10, rtol=0):
        raise ValueError('Timestep must refine the canonical 2.5 ms clock')
    return int(round(ratio))


def load(bank, case, normal, dt=CANONICAL_DT):
    """Load noisy PI r61001..3 paired with noisy normal r1..3; no fitting."""
    if isinstance(bank, bool) or bank not in (1, 2, 3):
        raise ValueError('Existing calibration bank must be 1, 2 or 3')
    if case not in ('balanced', 'wide_scales', 'compliant', 'stiff_high_capacity'):
        raise ValueError('Unknown fitted PI family member')
    if normal not in ('maxwell', 'boucwen'):
        raise ValueError('Unknown normal model')
    _clock(dt)
    return load_model(case, normal, 'noisy', pi_seed=61000+int(bank), normal_seed=int(bank),
                      dt=float(dt), mass=2., iterations=40)


def ramp(t):
    """One public C2 ramp: zero to INITIAL_COMPRESSION in 0.5 seconds."""
    t = np.asarray(t, dtype=float)
    phase = np.clip(t/.5, 0., 1.)
    d = INITIAL_COMPRESSION * phase**3 * (10-15*phase+6*phase**2)
    rate = INITIAL_COMPRESSION/.5 * 30*phase**2*(1-phase)**2
    rate = np.where((t > 0) & (t < .5), rate, 0.)
    return d, rate


@njit(cache=True)
def _prepare(displacements, compression, rate, dt, family, parameters, k, alpha):
    count, steps = displacements.shape
    normal = np.zeros(3)
    pi = np.zeros((count, 3))
    maximum_force = np.zeros(count)
    min_load, max_load = np.inf, 0.
    loads = np.zeros(steps)
    for t in range(steps):
        N = _normal_force(compression[t], rate[t], normal, family, parameters)
        loads[t] = N
        min_load, max_load = min(min_load, N), max(max_load, N)
        for i in range(count):
            dx = displacements[i, t]
            Ft = _pi_mean(pi[i], k, alpha, N, dx)
            maximum_force[i] = max(maximum_force[i], abs(Ft))
            for e in range(3):
                cap = alpha[e]*N
                start = min(max(pi[i,e], -cap), cap)
                pi[i,e] = min(max(start+dx, -cap), cap)
        normal = _normal_next(compression[t], rate[t], normal, family, parameters, dt)
    return normal, pi, loads, min_load, max_load, maximum_force


def initial_normal_force(model, normal0):
    """Known carried normal state at x=v=0 and the constant global d."""
    z = np.asarray(normal0, dtype=float)
    if z.shape != (3,) or not np.isfinite(z).all():
        raise ValueError('One finite common normal state of shape (3,) is required')
    return float(_normal_force(INITIAL_COMPRESSION, 0., z, model.family, model.parameters))


def reachable(model, seed, count):
    """Return physically replayed friction states and their common normal0.

    The six seconds consist of a .5 s quintic compression ramp, 1.5 s hold,
    and the existing four-second C2 displacement protocol at flat geometry.
    N(t) keeps relaxing during the fixture. A declared zero-displacement load
    accommodation at its endpoint applies the current N(T) capacity; no time
    passes and the normal history is not advanced or replaced at this boundary.
    """
    stride = _clock(model.dt)
    if isinstance(count, bool) or int(count) != count or count < 1:
        raise ValueError('Positive integer history count required')
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError('Nonnegative integer public history seed required')
    count = int(count)
    profiles = profile_commands(int(seed), np.arange(count), dt=model.dt, duration=4., waypoint_limit=.025)
    time_grid = np.arange(round(6/model.dt)+1)*model.dt
    compression, rate = ramp(time_grid)
    ramp_hold_steps = round(2/model.dt)
    dx = np.zeros((count, len(time_grid)-1))
    dx[:,ramp_hold_steps:] = np.diff(profiles['x'], axis=1)
    normal0, pi, loads, min_load, max_load, max_force = _prepare(
        dx, compression[:-1], rate[:-1], model.dt, model.family, model.parameters,
        model.pi.k, model.pi.alpha)
    N0 = initial_normal_force(model, normal0)
    before = pi.copy()
    # PIModel's zero-displacement update enforces the physically current load;
    # its dt has no effect when v=0. This is not an extra elapsed timestep.
    accommodation_force, pi = model.pi.step(pi, 0., N0, dt=model.dt)
    released = .5*np.sum(model.pi.k*(before**2-pi**2), axis=1)
    return dict(
        normal0=normal0.copy(), initial_pi=pi, initial_xv=np.zeros((count,2)),
        metadata=dict(seed=int(seed), count=count, initial_compression_m=INITIAL_COMPRESSION,
            dt=model.dt, canonical_stride=stride, duration_s=6., ramp_s=.5, hold_s=1.5,
            fixture_s=4., virgin_normal_state=[0.,0.,0.], virgin_pi_state=[0.,0.,0.],
            normal_reset_after_preparation=False, fixture='ideal imposed C2 displacement; flat cam',
            endpoint_accommodation='zero displacement at N(T); no normal update or elapsed time',
            history_rows=profiles['rows']),
        history=dict(time=profiles['time'], x=profiles['x'], interval_velocity=profiles['v'],
            continuous_velocity=profiles['continuous_velocity'], normal_time=time_grid[:-1],
            compression=compression, compression_rate=rate, applied_normal=loads),
        diagnostics=dict(initial_normal_N=N0, minimum_normal_N=float(min_load),
            maximum_normal_N=float(max(max_load,N0)), maximum_abs_friction_N=np.maximum(max_force,abs(accommodation_force)),
            endpoint_capacity_adjustment_max_m=np.max(abs(pi-before),axis=1),
            endpoint_PI_stored_energy_released_J=released,
            endpoint_position_max_m=float(np.max(abs(profiles['x'][:,-1]))),
            endpoint_continuous_velocity_max_m_s=float(np.max(abs(profiles['continuous_velocity'][:,-1])))))


@njit(cache=True)
def _geometry(x, amp, geom):
    z = x/.012
    if geom == 0:
        a = np.tanh(z)
        return amp*a, amp/.012*(1-a*a)
    if geom == 1:
        return amp*np.sin(z), amp/.012*np.cos(z)
    den = 1+z*z
    return amp*z*z/den, 2*amp/.012*z/(den*den)


@njit(cache=True)
def _trajectory(initial_pi, normal0, dt, family, parameters, k, alpha, mass, iterations,
                compression, prep_steps, future_steps, obs_stride, output_stride,
                strength, sign, geom, amp, future_amplitude):
    count = len(initial_pi)
    xv = np.zeros((count,2))
    normal = np.empty((count,3))
    for i in range(count): normal[i] = normal0
    pi = initial_pi.copy()
    past = np.zeros((count,prep_steps//obs_stride,3))
    future = np.zeros((count,future_steps//output_stride,4))
    bxv, bn, bp = xv.copy(), normal.copy(), pi.copy()
    residual_max = np.zeros(count)
    # max u,N,Ft,x,v; min delta; prep signed/absolute work; future signed/abs.
    diagnostic = np.zeros((count,10))
    diagnostic[:,5] = np.inf
    yield_entries = np.zeros((count,2,3),dtype=np.int64)
    plastic_steps = np.zeros((count,2,3),dtype=np.int64)
    capacity_clips = np.zeros((count,2,3),dtype=np.int64)
    previous_yield = np.zeros((count,3),dtype=np.bool_)
    force_sum = np.zeros((count,2))
    delta = np.zeros(count)
    rate = np.zeros(count)
    slope = np.zeros(count)
    commands = np.zeros(count)
    for t in range(prep_steps+future_steps):
        phase = 0 if t < prep_steps else 1
        local_step = t if phase == 0 else t-prep_steps
        phase_steps = prep_steps if phase == 0 else future_steps
        amplitude = strength if phase == 0 else future_amplitude
        u = sign*amplitude*np.sin(2*np.pi*local_step/phase_steps)
        for i in range(count):
            g, slope[i] = _geometry(xv[i,0],amp,geom)
            delta[i], rate[i] = compression+g, slope[i]*xv[i,1]
            commands[i] = u
        nxt, nn, nz, N, Ft, residual = _coupled_step(
            xv,normal,pi,commands,delta,rate,slope,family,parameters,k,alpha,dt,mass,iterations)
        for i in range(count):
            dx = nxt[i,0]-xv[i,0]
            residual_max[i] = max(residual_max[i],abs(residual[i]))
            diagnostic[i,0] = max(diagnostic[i,0],abs(u))
            diagnostic[i,1] = max(diagnostic[i,1],abs(N[i]))
            diagnostic[i,2] = max(diagnostic[i,2],abs(Ft[i]))
            diagnostic[i,3] = max(diagnostic[i,3],abs(nxt[i,0]))
            diagnostic[i,4] = max(diagnostic[i,4],abs(nxt[i,1]))
            diagnostic[i,5] = min(diagnostic[i,5],delta[i])
            diagnostic[i,6+phase*2] += u*dx
            diagnostic[i,7+phase*2] += abs(u*dx)
            for e in range(3):
                cap = alpha[e]*N[i]
                start = min(max(pi[i,e],-cap),cap)
                plastic = abs(start+dx-nz[i,e]) > EVENT_TOLERANCE_M
                plastic_steps[i,phase,e] += int(plastic)
                yield_entries[i,phase,e] += int(plastic and not previous_yield[i,e])
                capacity_clips[i,phase,e] += int(abs(pi[i,e]-start)>EVENT_TOLERANCE_M)
                previous_yield[i,e] = plastic
            if phase == 1:
                force_sum[i,0] += N[i]
                force_sum[i,1] += Ft[i]
        xv, normal, pi = nxt, nn, nz
        if t < prep_steps and (t+1)%obs_stride == 0:
            j = (t+1)//obs_stride-1
            for i in range(count):
                g, s = _geometry(xv[i,0],amp,geom)
                past[i,j,0], past[i,j,1] = xv[i,0],xv[i,1]
                past[i,j,2] = _normal_force(compression+g,s*xv[i,1],normal[i],family,parameters)
        if t+1 == prep_steps:
            bxv, bn, bp = xv.copy(),normal.copy(),pi.copy()
        if t >= prep_steps and (t-prep_steps+1)%output_stride == 0:
            j = (t-prep_steps+1)//output_stride-1
            for i in range(count):
                future[i,j,0], future[i,j,1] = xv[i,0],xv[i,1]
                future[i,j,2], future[i,j,3] = force_sum[i,0]/output_stride,force_sum[i,1]/output_stride
                force_sum[i,0], force_sum[i,1] = 0.,0.
    return past,future,bxv,bn,bp,xv,normal,pi,residual_max,diagnostic,yield_entries,plastic_steps,capacity_clips


def _settings(model, duration=.15, strength=3., sign=1, geom=0, amp=.001,
              future_amplitude=2., future_duration=.4):
    stride = _clock(model.dt)
    if duration not in (.15,.30): raise ValueError('Preparation duration must be .15 or .30 seconds')
    if strength not in (0.,3.,12.): raise ValueError('Preparation strength must be 0, 3 or 12 N')
    if isinstance(sign,bool) or sign not in (-1,1): raise ValueError('Excitation sign must be -1 or +1')
    if isinstance(geom,bool) or geom not in (0,1,2) or int(geom)!=geom: raise ValueError('Unknown geometry')
    if not np.isscalar(amp) or not np.isfinite(amp) or abs(amp)>=INITIAL_COMPRESSION:
        raise ValueError('Finite cam amplitude smaller than global compression required')
    if future_amplitude not in (2.,10.) or future_duration != .4:
        raise ValueError('Future amplitude must be 2 or 10 N over .4 seconds')
    return dict(duration=float(duration),strength=float(strength),sign=int(sign),geom=int(geom),amp=float(amp),
                future_amplitude=float(future_amplitude),future_duration=float(future_duration)),stride


def _simulate(model, initial_pi, normal0, settings, *, past_only=False):
    cfg, stride = _settings(model,**settings)
    z = np.asarray(initial_pi,dtype=float)
    if z.ndim == 1: z = z[None]
    normal0 = np.asarray(normal0,dtype=float)
    if z.ndim!=2 or z.shape[1]!=3 or len(z)<1 or not np.isfinite(z).all():
        raise ValueError('Finite initial PI states with shape (P,3) required')
    N0 = initial_normal_force(model,normal0)
    bounds = model.pi.alpha*N0
    if np.any(bounds<=0) or np.any(abs(z)>bounds+1e-12):
        raise ValueError('Initial states must satisfy the carried-load capacities')
    values = _trajectory(z,normal0,model.dt,model.family,model.parameters,model.pi.k,model.pi.alpha,
        model.mass,model.iterations,INITIAL_COMPRESSION,round(cfg['duration']/model.dt),
        0 if past_only else round(cfg['future_duration']/model.dt),round(OBSERVATION_DT/model.dt),stride,
        cfg['strength'],cfg['sign'],cfg['geom'],cfg['amp'],cfg['future_amplitude'])
    if not all(np.isfinite(a).all() for a in values):
        raise FloatingPointError('Nonfinite trajectory; no candidate dropped')
    if np.max(values[8])>SOLVER_TOLERANCE_N:
        raise FloatingPointError('Coupled solver residual exceeds 1e-7 N')
    return values,cfg


def rollout(model, initial_pi, normal0, *, duration=.15, strength=3., sign=1, geom=0, amp=.001,
            future_amplitude=2., future_duration=.4):
    """Pure command-driven forecast; no observed data enter any dynamics.

    past/truth_past: x/v/instantaneous N at 25 ms times, excluding t=0.
    future: x/v at common 2.5 ms endpoints, N/Ft averaged over each 2.5 ms
    interval. Material states always evolve at model.dt, including refinements.
    Event arrays have shape (P,2,3): prep/future phase and three PI elements.
    """
    settings=dict(duration=duration,strength=strength,sign=sign,geom=geom,amp=amp,
                  future_amplitude=future_amplitude,future_duration=future_duration)
    values,cfg = _simulate(model,initial_pi,normal0,settings)
    past,future,bxv,bn,bp,fxv,fn,fp,residual,d,entries,plastic,clips = values
    P=len(past)
    diagnostics=dict(max_solver_residual_N=float(np.max(residual)),per_particle_max_solver_residual_N=residual,
        maximum_abs_u_N=d[:,0],maximum_abs_normal_N=d[:,1],
        maximum_abs_friction_N=d[:,2],maximum_abs_position_m=d[:,3],maximum_abs_velocity_m_s=d[:,4],
        minimum_compression_m=d[:,5],prep_signed_tangential_work_J=d[:,6],prep_absolute_tangential_work_J=d[:,7],
        future_signed_tangential_work_J=d[:,8],future_absolute_tangential_work_J=d[:,9],
        total_signed_tangential_work_J=d[:,6]+d[:,8],total_absolute_tangential_work_J=d[:,7]+d[:,9],
        prep_signed_normal_work_J=np.zeros(P),prep_absolute_normal_work_J=np.zeros(P),
        future_signed_normal_work_J=np.zeros(P),future_absolute_normal_work_J=np.zeros(P),
        yield_entry_counts=entries,plastic_step_counts=plastic,load_capacity_clip_counts=clips,
        event_tolerance_m=EVENT_TOLERANCE_M,
        event_definition='yield entry: false-to-true plastic-clipping step; load clipping counted separately; phase continuation is not a new entry')
    return dict(past=past,truth_past=past.copy(),future=future,
        past_times=np.arange(1,past.shape[1]+1)*OBSERVATION_DT,
        future_times=np.arange(1,future.shape[1]+1)*CANONICAL_DT,
        boundary_xv=bxv,boundary_normal=bn,boundary_pi=bp,
        final_xv=fxv,final_normal=fn,final_pi=fp,diagnostics=diagnostics,
        settings=cfg,dt=model.dt,initial_normal_N=initial_normal_force(model,normal0))


class _CallBudget(RuntimeError):
    pass


def mle(model, observed, sigma, priorpi, weights, normal0, **commandssettings):
    """Two deterministic bounded initial-state fits; only past samples used.

    Starts: highest supplied causal prior weight, then zero. The caller must
    obtain weights using past information only. Each start receives max_nfev
    120 and at most 512 actual residual evaluations, including mandatory initial
    and final checks. No wall-time stopping rule. Future settings are validated
    but no future step or future observation is evaluated by this function.
    """
    cfg,_ = _settings(model,**commandssettings)
    y,sigma = np.asarray(observed,dtype=float),np.asarray(sigma,dtype=float)
    if y.shape != (round(cfg['duration']/OBSERVATION_DT),3) or not np.isfinite(y).all():
        raise ValueError('Only the complete past observation array (S,3) is accepted')
    if sigma.shape!=(3,) or np.any(sigma<=0) or not np.isfinite(sigma).all():
        raise ValueError('Three strictly positive effective likelihood scales required')
    prior,weight = np.asarray(priorpi,dtype=float),np.asarray(weights,dtype=float)
    N0=initial_normal_force(model,normal0);bound=model.pi.alpha*N0
    if np.any(bound<=0): raise ValueError('Positive initial normal load required')
    if prior.ndim!=2 or prior.shape[1]!=3 or len(prior)<1 or not np.isfinite(prior).all():
        raise ValueError('Finite public prior states required')
    if weight.shape!=(len(prior),) or not np.isfinite(weight).all() or np.any(weight<0) or weight.sum()<=0:
        raise ValueError('Valid causal prior weights required')
    qprior=prior/bound
    if np.any(abs(qprior)>1+1e-10): raise ValueError('Prior outside the known-load state box')
    qstarts=[np.clip(qprior[np.argmax(weight)],-1.,1.),np.zeros(3)]
    candidates=[];costs=[];diagnostics=[]
    for start_id,q0 in enumerate(qstarts):
        calls=0;maximum=0.;best_cost=np.inf;best_q=q0.copy();tic=time.perf_counter()
        def objective(q, final=False):
            nonlocal calls,maximum,best_cost,best_q
            # Reserve the final deterministic verification within the cap.
            if calls >= MAX_RESIDUAL_CALLS-(0 if final else 1):
                raise _CallBudget('Fixed actual-residual-call budget exhausted')
            values,_ = _simulate(model,np.asarray(q)[None]*bound,normal0,cfg,past_only=True)
            maximum=max(maximum,float(np.max(values[8])))
            r=((values[0][0]-y)/sigma).ravel()
            if not np.isfinite(r).all(): raise FloatingPointError('Nonfinite past residual')
            calls+=1;cost=float(r@r)
            if cost<best_cost:
                best_cost,best_q=cost,np.array(q,copy=True)
            return r
        objective(q0)
        success=False;budget=False;nfev=None;status='';message=''
        try:
            answer=least_squares(objective,q0,bounds=(-np.ones(3),np.ones(3)),method='trf',jac='2-point',
                x_scale=1.,max_nfev=MAX_NFEV,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            success=bool(answer.success);nfev=int(answer.nfev);budget=bool(answer.status==0)
            status='converged' if success else 'optimizer_budget_or_nonconvergence'
            message=str(answer.message)
        except _CallBudget as exc:
            budget=True;status='residual_call_budget';message=str(exc)
        objective(best_q.copy(),final=True)
        candidates.append(best_q*bound);costs.append(best_cost)
        diagnostics.append(dict(start=start_id,initial_normal_N=N0,initial_q=q0.tolist(),final_q=best_q.tolist(),
            solver_success=success,budget_exhausted=budget,optimizer_nfev=nfev,residual_calls=calls,
            max_solver_residual_N_all_trials=maximum,status=status,message=message,seconds=time.perf_counter()-tic,
            max_nfev=MAX_NFEV,max_actual_residual_calls=MAX_RESIDUAL_CALLS,wall_budget_used=False))
    costs=np.asarray(costs)
    return dict(initial_candidates=np.asarray(candidates),costs=costs,chosen=int(np.argmin(costs)),
        normalized_candidates=np.asarray(candidates)/bound,
        diagnostics=dict(starts=diagnostics,max_solver_residual_N=max(d['max_solver_residual_N_all_trials'] for d in diagnostics)),
        initial_normal_N=N0,effective_sigma=sigma.copy(),settings=cfg,
        selection='minimum past objective; first start wins exact ties',
        max_solver_residual_N_all_trials=max(d['max_solver_residual_N_all_trials'] for d in diagnostics))


MLE=mle
