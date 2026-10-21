import json
from pathlib import Path
import numpy as np
import torch
from .common import CFG, DT, MASS, features, history_update
from .models import Classical, Neural, freeze

def load_model(path):
    path=Path(path)
    if (path/'classical.json').exists():
        item=json.loads((path/'classical.json').read_text()); return Classical(**item)
    item=torch.load(path/'model.pt',map_location='cpu',weights_only=True)
    model=Neural(item['kind']); model.load_state_dict(item['state']); return freeze(model)

def step(model,x,v,h,d,rate,u,slope,decoupled=False,dt=DT):
    effective_slope=np.zeros_like(slope) if decoupled else slope
    delta=d+effective_slope*x; drate=rate+effective_slope*v
    f=features(delta,drate,v,h,np.zeros_like(x))
    n,ft=model.predict(f)
    force=u-slope*n+ft
    vn=v+dt*force/MASS; xn=x+dt*vn
    hn=history_update(h,delta,v,dt)
    return xn,vn,hn,n,ft,delta

def rollout(model,cmd,slope,x0=None,v0=None,noise=False,decoupled=False):
    n,steps=cmd['u'].shape; x=np.zeros(n) if x0 is None else x0.copy(); v=np.zeros(n) if v0 is None else v0.copy()
    h=np.zeros((n,3,2)); h[:,:,0]=cmd['d'][:,0,None]+slope[:,None]*x[:,None]; h[:,:,1]=v[:,None]
    rng=np.random.default_rng(92013)
    if noise:
        x+=rng.normal(0,CFG['noise']['position_m'],n);v+=rng.normal(0,CFG['noise']['velocity_m_s'],n)
        h[:,:,0]+=rng.normal(0,CFG['noise']['compression_m'],(n,3));h[:,:,1]+=rng.normal(0,CFG['noise']['velocity_m_s'],(n,3))
    res={k:[] for k in ['state','next_state','normal','friction','compression']}
    for t in range(steps):
        res['state'].append(np.column_stack([x,v]))
        x,v,h,norm,ft,delta=step(model,x,v,h,cmd['d'][:,t],cmd['rate'][:,t],cmd['u'][:,t],slope,decoupled)
        for key,value in [('next_state',np.column_stack([x,v])),('normal',norm),('friction',ft),('compression',delta)]:res[key].append(value.copy())
    return {k:np.stack(v,axis=1) for k,v in res.items()}

def metrics(pred,truth):
    diff=pred['next_state']-truth['next_state']
    force=np.stack([pred['normal']-truth['normal'],pred['friction']-truth['friction']],-1)
    return dict(position_rmse=np.sqrt(np.mean(diff[:,:,0]**2,axis=1)),velocity_rmse=np.sqrt(np.mean(diff[:,:,1]**2,axis=1)),
        force_rmse=np.sqrt(np.mean(force**2,axis=(1,2))),normal_rmse=np.sqrt(np.mean(force[:,:,0]**2,axis=1)),
        friction_rmse=np.sqrt(np.mean(force[:,:,1]**2,axis=1)),contact_accuracy=np.mean((pred['normal']>.1)==(truth['normal']>.1),axis=1),
        stick_accuracy=np.mean((np.abs(pred['state'][:,:,1])<.01)==(np.abs(truth['state'][:,:,1])<.01),axis=1),
        positive_friction_work=np.sum(np.maximum(0,pred['friction']*pred['state'][:,:,1])*DT,axis=1),
        max_compression=np.max(pred['compression'],axis=1), unstable=np.any(~np.isfinite(diff),axis=(1,2))|np.any(np.abs(pred['next_state'][:,:,0])>2,axis=1))

def choose_action(model,x,v,h,dseq,rseq,slope,decoupled=False):
    cfg=CFG['controller']; n=len(x); k=cfg['candidates']; u=np.tile(np.linspace(-cfg['force_limit'],cfg['force_limit'],k),n)
    xp=np.repeat(x,k);vp=np.repeat(v,k);hp=np.repeat(h,k,axis=0);sp=np.repeat(slope,k)
    for t in range(cfg['horizon_steps']):
        xp,vp,hp,_,_,_=step(model,xp,vp,hp,np.repeat(dseq[:,t],k),np.repeat(rseq[:,t],k),u,sp,decoupled)
    # Fixed stopping-aware target velocity and terminal position cost.
    cost=30*(xp-cfg['target'])**2+2*(vp-np.clip(3*(cfg['target']-xp),-.35,.35))**2+.0002*u*u
    cost=np.nan_to_num(cost,nan=1e12,posinf=1e12,neginf=1e12).reshape(n,k)
    return u.reshape(n,k)[np.arange(n),np.argmin(cost,axis=1)]
