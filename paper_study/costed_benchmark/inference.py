"""Fixed-parameter finite-prior inference for coupled three-stop materials.

Only fitted laws, public kinematics, known commands and causal x/v/N samples
enter this module. No reference, true prefix, true force or task state is read.
Particles retain their own mechanical AND normal/friction material trajectory;
observations alter weights only. This is classical Bayes conditioning on a
finite empirical prior, not a calibrated model-uncertainty guarantee.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO/'paper_study/multistate_learning'))
from model import PIModel
sys.path.insert(0, str(REPO/'research_next/src'))
from state_id import StateID
# The frozen local fitter also has a top-level profiles.py. Use this namespace
# to prevent an unrelated cached module selecting the wrong public profile API.
sys.path.insert(0, str(REPO))
from paper_study.costed_benchmark.geometry import cam
from paper_study.costed_benchmark.profiles import profile_commands


@njit(cache=True)
def _normal_force(delta, rate, state, family, p):
    if delta <= 0.:
        return 0.
    if family == 0:
        force = p[0]*delta+p[1]*rate
        for i in range(3):
            force += p[2+i]*(delta-state[i])
    else:
        force = p[0]*delta+p[1]*rate+p[2]*state[0]
    return max(force, 0.)


@njit(cache=True)
def _bw_derivative(z, rate, beta, gamma):
    return rate-beta*abs(rate)*abs(z)*z-gamma*rate*z*z


@njit(cache=True)
def _normal_next(delta, rate, state, family, p, dt):
    new = state.copy()
    if family == 0:
        for i in range(3):
            new[i] += -np.expm1(-dt/p[5+i])*(max(delta, 0.)-state[i])
    elif delta <= 0.:
        new[0] *= np.exp(-dt/.1)  # frozen identified model's declared open rule
    else:
        r = rate/p[3]
        count = max(1, int(np.ceil(abs(r)*dt/.05)))
        h = dt/count
        z = state[0]
        for _ in range(count):
            a = _bw_derivative(z, r, p[4], p[5])
            b = _bw_derivative(z+.5*h*a, r, p[4], p[5])
            c = _bw_derivative(z+.5*h*b, r, p[4], p[5])
            d = _bw_derivative(z+h*c, r, p[4], p[5])
            z += h*(a+2*b+2*c+d)/6
        new[0] = z
    return new


@njit(cache=True)
def _pi_mean(state, k, alpha, normal, dx):
    force = 0.
    direction = 1. if dx > 0 else -1.
    distance = abs(dx)
    for i in range(3):
        bound = alpha[i]*normal
        z = min(max(state[i], -bound), bound)
        if distance == 0.:
            mean = z
        else:
            elastic = min(distance, max(bound-direction*z, 0.))
            fraction = elastic/distance
            mean = fraction*(z+.5*direction*elastic)+(1.-fraction)*direction*bound
        force -= k[i]*mean
    return force


@njit(cache=True)
def _coupled_step(xv, normal_state, pi_state, u, delta, rate, slope,
                  family, parameters, k, alpha, dt, mass, iterations):
    count = len(xv)
    next_xv, next_normal, next_pi = np.empty_like(xv), np.empty_like(normal_state), np.empty_like(pi_state)
    normals, forces, residuals = np.empty(count), np.empty(count), np.empty(count)
    total_capacity = np.sum(k*alpha)
    for j in range(count):
        normal = _normal_force(delta[j], rate[j], normal_state[j], family, parameters)
        next_normal[j] = _normal_next(delta[j], rate[j], normal_state[j], family, parameters, dt)
        free = xv[j,1]+dt*(u[j]-slope[j]*normal)/mass
        capacity = total_capacity*normal
        lo, hi = free-dt*capacity/mass, free+dt*capacity/mass
        for _ in range(iterations):
            middle = .5*(lo+hi)
            force = _pi_mean(pi_state[j], k, alpha, normal, middle*dt)
            if middle-free-dt*force/mass < 0.:
                lo = middle
            else:
                hi = middle
        vn = .5*(lo+hi)
        dx = vn*dt
        force = _pi_mean(pi_state[j], k, alpha, normal, dx)
        for i in range(3):
            bound = normal*alpha[i]
            start = min(max(pi_state[j,i], -bound), bound)
            next_pi[j,i] = min(max(start+dx, -bound), bound)
        next_xv[j,0], next_xv[j,1] = xv[j,0]+dx, vn
        normals[j], forces[j] = normal, force
        residuals[j] = mass*(vn-xv[j,1])/dt-(u[j]-slope[j]*normal+force)
    return next_xv, next_normal, next_pi, normals, forces, residuals


@njit(cache=True)
def _normal_outputs(delta, rate, state, family, parameters):
    out = np.empty(len(delta))
    for j in range(len(delta)):
        out[j] = _normal_force(delta[j], rate[j], state[j], family, parameters)
    return out


@njit(cache=True)
def _prepare_pi(velocity, normal, alpha, dt):
    state = np.zeros((len(velocity), 3))
    for j in range(len(velocity)):
        for t in range(velocity.shape[1]):
            dx = velocity[j,t]*dt
            for i in range(3):
                bound = alpha[i]*normal[j,t]
                z = min(max(state[j,i], -bound), bound)
                state[j,i] = min(max(z+dx, -bound), bound)
    return state


class CoupledPIModel:
    """Frozen PI and normal coefficients; independent explicit material states."""
    def __init__(self, pi_model, normal_model, *, dt=.0025, mass=2., iterations=40):
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(mass) or mass <= 0:
            raise ValueError('positive finite dt/mass required')
        if isinstance(iterations,bool) or int(iterations)!=iterations or iterations<1:
            raise ValueError('positive integer bisection count required')
        self.pi, self.normal_model = pi_model, normal_model
        self.dt, self.mass, self.iterations = float(dt), float(mass), int(iterations)
        if normal_model.world not in ('maxwell','boucwen'):
            raise ValueError('unsupported fitted normal family')
        self.family = 0 if normal_model.world=='maxwell' else 1
        p = normal_model.normal
        self.parameters = np.zeros(8)
        if self.family==0:
            self.parameters[:] = [p['k_eq'], p['c'], *p['k_branches'], *p['tau_branches']]
        else:
            self.parameters[:6] = [p['k'], p['c'], p['F'], p['dy'], p['beta'], p['gamma']]
        # The frozen normal fitter uses NNLS: an unneeded linear spring or
        # dashpot may be exactly zero. Do not alter/reject those valid weights.
        linear=self.parameters[:3] if self.family else self.parameters[:5]
        nonlinear=self.parameters[3:6] if self.family else self.parameters[5:8]
        if not np.isfinite(self.parameters).all() or np.any(linear<0) or np.any(nonlinear<=0):
            raise ValueError('nonnegative linear and positive nonlinear normal coefficients required')
        self.parameters.setflags(write=False)

    def initial(self, count, delta=.02, pi_state=None):
        if isinstance(count,bool) or int(count)!=count or count<1:
            raise ValueError('positive integer count required')
        compression = np.broadcast_to(np.asarray(delta,dtype=float),(count,))
        normal = np.zeros((count,3))
        if self.family==0:
            normal[:]=np.maximum(compression,0.)[:,None]
        else:
            normal[:,0]=np.tanh(np.maximum(compression,0.)/self.parameters[3])
        pi = self.pi.initial(count) if pi_state is None else np.array(pi_state,dtype=float,copy=True)
        if pi.shape!=(count,3) or not np.isfinite(pi).all():
            raise ValueError('finite (count,3) friction state required')
        return dict(normal=normal, pi=pi)

    @staticmethod
    def _vector(value,count,name):
        out=np.broadcast_to(np.asarray(value,dtype=float),(count,)).copy()
        if not np.isfinite(out).all():raise ValueError(f'nonfinite {name}')
        return out

    def local_inputs(self,xv,d,d_dot,amplitude,geometry):
        count=len(xv)
        g,slope=cam(xv[:,0],self._vector(amplitude,count,'amplitude'),geometry)
        delta=self._vector(d,count,'d')+g
        rate=self._vector(d_dot,count,'d_dot')+slope*xv[:,1]
        return delta,rate,slope

    def step(self,xv,state,u,d,d_dot,amplitude,geometry):
        xv=np.asarray(xv,dtype=float)
        if xv.ndim!=2 or xv.shape[1]!=2 or not np.isfinite(xv).all():
            raise ValueError('finite (count,2) mechanical states required')
        count=len(xv)
        if any(np.shape(state[key])!=(count,3) or not np.isfinite(state[key]).all() for key in ('normal','pi')):
            raise ValueError('invalid material states')
        delta,rate,slope=self.local_inputs(xv,d,d_dot,amplitude,geometry)
        result=_coupled_step(xv,state['normal'],state['pi'],self._vector(u,count,'u'),
            delta,rate,slope,self.family,self.parameters,self.pi.k,self.pi.alpha,
            self.dt,self.mass,self.iterations)
        if not all(np.isfinite(a).all() for a in result):
            raise FloatingPointError('nonfinite particle dynamics; no particles silently removed')
        next_xv,normal,pi,N,Ft,residual=result
        return next_xv,dict(normal=normal,pi=pi),N,Ft,residual

    def normal_force(self,xv,state,d,d_dot,amplitude,geometry):
        delta,rate,_=self.local_inputs(np.asarray(xv),d,d_dot,amplitude,geometry)
        return _normal_outputs(delta,rate,state['normal'],self.family,self.parameters)

    def prefix_load(self,delta=.02):
        """Public fresh-normal prediction at rate zero, never boundary measured N."""
        state=self.initial(1,delta)
        return float(self.normal_force(np.zeros((1,2)),state,delta,0.,0.,np.zeros(1,dtype=int))[0])


def model_paths(case,normal_world,calibration_noise,pi_seed=61001,normal_seed=1):
    pi=REPO/f'paper_study/multistate_learning/results/models/{case}/r{pi_seed}/{calibration_noise}/model.json'
    normal=REPO/f'research_next/results/round1/models/state_id/{normal_world}/r{normal_seed}/{calibration_noise}/blind_selection/model.json'
    return pi,normal


def load_model(case,normal_world,calibration_noise='clean',*,pi_seed=61001,normal_seed=1,
               dt=.0025,mass=2.,iterations=40):
    pi_path,normal_path=model_paths(case,normal_world,calibration_noise,pi_seed,normal_seed)
    pi=PIModel.load(pi_path)
    normal=StateID.load(normal_path).normal_model
    return CoupledPIModel(pi,normal,dt=dt,mass=mass,iterations=iterations)


def build_prior(model,*,particles=256,seed=43001,profile_start=0,duration=4.,waypoint_limit=.025):
    """Independent public history draw, propagated only through the fitted PI.

    Uniform probability over histories, not uniform volume in latent space.
    Duplicate reachable states keep their original probability mass.
    """
    if isinstance(particles,bool) or int(particles)!=particles or particles<1:
        raise ValueError('positive integer particle count required')
    commands=profile_commands(seed,np.arange(profile_start,profile_start+particles),
        dt=model.dt,duration=duration,waypoint_limit=waypoint_limit)
    load=model.prefix_load()
    normal=commands['normal_scale']*load
    states=_prepare_pi(commands['v'],normal,model.pi.alpha,model.dt)
    return states,dict(seed=seed,profile_start=profile_start,particles=particles,duration=duration,
        waypoint_limit=waypoint_limit,prefix_load_N=load,
        initial_preparation='known relaxed under the public profile distribution',
        prior='uniform over independently generated public histories; duplicates retain mass')


class ParticleObserver:
    def __init__(self,model,prior_states,initial_xv,*,initial_delta=.02,
                 sensor_sigma=(5e-5,.002,.15),numerical_floor=(1e-6,1e-5,1e-4),prior=None):
        self.model=model
        hypotheses=np.asarray(prior_states,dtype=float)
        initial=np.asarray(initial_xv,dtype=float)
        if hypotheses.ndim!=2 or hypotheses.shape[1]!=3 or not len(hypotheses) or not np.isfinite(hypotheses).all():
            raise ValueError('finite (particles,3) prior states required')
        if initial.ndim!=2 or initial.shape[1]!=2 or not len(initial) or not np.isfinite(initial).all():
            raise ValueError('finite (episodes,2) initial mechanics required')
        self.episodes,self.particles=len(initial),len(hypotheses)
        self.xv=np.repeat(initial[:,None,:],self.particles,axis=1)
        self.state=model.initial(self.episodes*self.particles,initial_delta,
                                 np.tile(hypotheses,(self.episodes,1)))
        sigma,floor=np.asarray(sensor_sigma,dtype=float),np.asarray(numerical_floor,dtype=float)
        if sigma.shape!=(3,) or floor.shape!=(3,) or not np.isfinite(sigma).all() or not np.isfinite(floor).all() or np.any(sigma<0) or np.any(floor<=0):
            raise ValueError('three nonnegative sensor sigmas and positive numerical floors required')
        self.effective_sigma=np.hypot(sigma,floor)
        probability=np.ones(self.particles) if prior is None else np.asarray(prior,dtype=float)
        if probability.shape!=(self.particles,) or not np.isfinite(probability).all() or np.any(probability<=0):
            raise ValueError('strictly positive shared particle probabilities required')
        self.prior=np.broadcast_to(probability/probability.sum(),(self.episodes,self.particles)).copy()
        self.weights=self.prior.copy();self.log_likelihood=np.zeros_like(self.prior)
        self.step_index=0;self.last_observed_step=0;self.observation_count=0
        self.max_solver_residual=0.

    def repeat(self,value):
        return np.repeat(np.broadcast_to(value,(self.episodes,)),self.particles)

    def advance(self,u,d,d_dot,amplitude,geometry):
        args=[self.repeat(a) for a in (u,d,d_dot,amplitude,geometry)]
        xv,state,_,_,residual=self.model.step(self.xv.reshape(-1,2),self.state,*args)
        self.xv=xv.reshape(self.episodes,self.particles,2);self.state=state
        self.max_solver_residual=max(self.max_solver_residual,float(np.max(abs(residual))))
        self.step_index+=1

    def observe(self,x,v,N,*,d,d_dot,amplitude,geometry):
        if self.step_index<=self.last_observed_step:
            raise ValueError('initial or duplicate likelihood sample rejected')
        measured=np.column_stack([np.broadcast_to(value,(self.episodes,)) for value in (x,v,N)])
        if not np.isfinite(measured).all():raise ValueError('nonfinite observation')
        normal=self.model.normal_force(self.xv.reshape(-1,2),self.state,
            *[self.repeat(a) for a in (d,d_dot,amplitude,geometry)]).reshape(self.episodes,self.particles)
        predicted=np.concatenate((self.xv,normal[...,None]),axis=2)
        residual=(measured[:,None,:]-predicted)/self.effective_sigma
        self.log_likelihood-=.5*np.sum(residual*residual,axis=2)
        log_weight=np.log(self.prior)+self.log_likelihood
        if not np.isfinite(log_weight).all():raise FloatingPointError('nonfinite likelihood; retain failed trial')
        log_weight-=log_weight.max(axis=1,keepdims=True)
        probability=np.exp(log_weight)
        self.weights=probability/probability.sum(axis=1,keepdims=True)
        self.last_observed_step=self.step_index;self.observation_count+=1
        return self.weights.copy()

    def snapshot(self):
        return dict(xv=self.xv.copy(),state={k:v.copy() for k,v in self.state.items()},
            weights=self.weights.copy(),prior=self.prior.copy(),log_likelihood=self.log_likelihood.copy(),
            step=self.step_index,observation_count=self.observation_count,
            effective_sigma=self.effective_sigma.copy(),max_solver_residual=self.max_solver_residual)
