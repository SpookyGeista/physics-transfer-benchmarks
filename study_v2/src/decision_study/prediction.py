"""Common causal simulator and identical constant-action MPC for frozen models."""
import numpy as np
from .common import CFG, DT, MASS, cam, features, history_init, history_update
from .models import replace_normal

def predict_forces(model, f, decoupled=False):
    if decoupled:
        # Keep compliance in cam work, but break its current/history load interface.
        fixed = f.copy(); fixed[:, 9] = .5; fixed[:, 10:13] = 0
        n, ft = model.predict_local(fixed)
    else:
        n, ft = model.predict(f)
    bad = (~np.isfinite(n)) | (~np.isfinite(ft)) | (np.abs(n)>1e4) | (np.abs(ft)>1e4)
    return np.clip(np.nan_to_num(n,nan=0,posinf=1e4,neginf=-1e4),0,1e4), np.clip(np.nan_to_num(ft,nan=0,posinf=1e4,neginf=-1e4),-1e4,1e4), bad

def step(model, state, h, previous_n, u, d, rate, amplitude, decoupled=False):
    x,v=state.T; g,slope=cam(x,amplitude); delta=d+g; dr=rate+slope*v
    f=features(delta,dr,v,h,previous_n)
    n,ft,bad=predict_forces(model,f,decoupled)
    force=u-slope*n+ft; nv=v+DT*force/MASS; nx=x+DT*nv
    bad |= (~np.isfinite(nx))|(~np.isfinite(nv))|(np.abs(nx)>2)|(np.abs(nv)>20)
    state=np.column_stack([np.clip(np.nan_to_num(nx,nan=0,posinf=2,neginf=-2),-2,2),np.clip(np.nan_to_num(nv,nan=0,posinf=20,neginf=-20),-20,20)])
    return state,history_update(h,delta,v,n),n,ft,bad

def stage_cost(state,u,target):
    return (state[:,0]-target)**2+.002*state[:,1]**2+.000001*u**2

def plan(model,state,h,normal,d,rate,amplitude,decoupled=False):
    c=CFG['controller'];forces=np.linspace(-c['force_limit'],c['force_limit'],c['candidates']);batch=len(state);K=len(forces)
    s=np.repeat(state,K,axis=0);hh=np.repeat(h,K,axis=0);nn=np.repeat(normal,K);amp=np.repeat(amplitude,K);u=np.tile(forces,batch);cost=np.zeros(batch*K)
    for t in range(c['horizon']):
        s,hh,nn,ft,bad=step(model,s,hh,nn,u,np.repeat(d[:,t],K),np.repeat(rate[:,t],K),amp,decoupled)
        cost+=DT*stage_cost(s,u,c['target'])+1e6*bad
    return forces[np.argmin(cost.reshape(batch,K),axis=1)]

def rollout(model,commands,amplitude,initial_state,initial_normal,noise,seed,decoupled=False):
    state=initial_state.copy().astype(float); n=initial_normal.copy();g,_=cam(state[:,0],amplitude);delta=commands['d'][:,0]+g
    # One common noisy sensor initialization; no future state/force corrections.
    if noise=='noisy':
        rng=np.random.default_rng(830000+seed);state[:,0]+=rng.normal(0,CFG['noise']['delta'],len(n));state[:,1]+=rng.normal(0,CFG['noise']['velocity'],len(n));n+=rng.normal(0,CFG['noise']['normal'],len(n))
        g,_=cam(state[:,0],amplitude);delta=commands['d'][:,0]+g
    h=history_init(delta,state[:,1],n); rec={k:[] for k in ['state','normal','friction','guard']}
    for t in range(commands['u'].shape[1]):
        state,h,n,ft,bad=step(model,state,h,n,commands['u'][:,t],commands['d'][:,t],commands['rate'][:,t],amplitude,decoupled)
        for key,value in [('state',state),('normal',n),('friction',ft),('guard',bad)]:rec[key].append(value.copy())
    return {k:np.stack(v,axis=1) for k,v in rec.items()}
