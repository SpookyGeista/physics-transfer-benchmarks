"""Uniform causal NumPy state interface for trained models and MPC copies."""
from pathlib import Path
import numpy as np
import torch
from recurrent import Recurrent


class RecurrentOnline:
    def __init__(self, path):
        checkpoint=torch.load(Path(path),map_location='cpu',weights_only=True)
        self.model=Recurrent(checkpoint['hidden'])
        self.model.load_state_dict(checkpoint['state']);self.model.eval()

    @staticmethod
    def tensor(x):return torch.as_tensor(np.asarray(x),dtype=torch.float32)

    def initial(self,delta,v,normal_sensor,rate=None):
        delta,v,n=np.broadcast_arrays(delta,v,normal_sensor)
        rate=np.zeros_like(delta) if rate is None else np.broadcast_to(rate,delta.shape)
        xn=self.tensor(np.column_stack([delta/.02,rate/.1]))
        xf=self.tensor(np.column_stack([v/.5,np.abs(v)/.5,n/20]))
        with torch.inference_mode():
            return dict(normal=self.model.normal_head.init(xn)[0].numpy().copy(),
                        friction=self.model.friction_head.init(xf)[0].numpy().copy())

    def step(self,delta,rate,v,state,normal_override=None,dt=.0025):
        if abs(dt-.0025)>1e-12:raise ValueError('Discrete GRU was trained at dt=0.0025 only')
        xn=self.tensor(np.column_stack([delta/.02,rate/.1]))
        with torch.inference_mode():
            n,hn=self.model.normal_step(xn,self.tensor(state['normal'])[None])
            nf=n.numpy() if normal_override is None else np.asarray(normal_override)
            xf=self.tensor(np.column_stack([v/.5,np.abs(v)/.5,nf/20]))
            ft,hf=self.model.friction_step(xf,self.tensor(state['friction'])[None])
        return nf.copy(),ft.numpy().copy(),dict(normal=hn[0].numpy().copy(),friction=hf[0].numpy().copy())


class Nominal:
    def initial(self,delta,v,normal_sensor):return {}
    def step(self,delta,rate,v,state,normal_override=None,dt=.0025):
        n=np.maximum(500*np.asarray(delta),0) if normal_override is None else np.asarray(normal_override)
        return n,-.35*n*np.tanh(np.asarray(v)/.04),{}


class FeatureOnline:
    """Adapt the explicit measured-load mode only for observed history/step 0."""
    def __init__(self,path):
        from feature_baselines import FeatureBaseline
        self.model=FeatureBaseline.load(path)
    def initial(self,delta,v,normal_sensor):return self.model.initial(delta,v,normal_sensor)
    def step(self,delta,rate,v,state,normal_override=None,dt=.0025):
        if normal_override is None:return self.model.step(delta,rate,v,state,dt=dt)
        _,ft,newstate=self.model.conditional_step(delta,rate,v,normal_override,state,dt=dt)
        return np.asarray(normal_override).copy(),ft,newstate


class Composed:
    """Known mechanical wiring, independently trained N and Ft components."""
    def __init__(self,normal,friction):self.normal,self.friction=normal,friction
    def initial(self,delta,v,normal_sensor):
        ns=self.normal.initial(delta,v,normal_sensor);fs=self.friction.initial(delta,v,normal_sensor)
        return {**{'N__'+k:x for k,x in ns.items()},**{'F__'+k:x for k,x in fs.items()}}
    def step(self,delta,rate,v,state,normal_override=None,dt=.0025):
        ns={k[3:]:x for k,x in state.items() if k.startswith('N__')};fs={k[3:]:x for k,x in state.items() if k.startswith('F__')}
        n,_,ns=self.normal.step(delta,rate,v,ns,normal_override=normal_override,dt=dt)
        _,ft,fs=self.friction.step(delta,rate,v,fs,normal_override=n,dt=dt)
        return n,ft,{**{'N__'+k:x for k,x in ns.items()},**{'F__'+k:x for k,x in fs.items()}}


class SensorPersistent:
    """Keep identified friction dynamics but hold last observed N in forecast."""
    def __init__(self,model):self.model=model
    def initial(self,delta,v,normal_sensor):
        return {**self.model.initial(delta,v,normal_sensor),'observed_load':np.asarray(normal_sensor).copy()}
    def step(self,delta,rate,v,state,normal_override=None,dt=.0025):
        load=state['observed_load'] if normal_override is None else np.asarray(normal_override)
        base={k:x for k,x in state.items() if k!='observed_load'}
        n,ft,new=self.model.step(delta,rate,v,base,normal_override=load,dt=dt)
        return n,ft,{**new,'observed_load':load.copy()}


def clone_state(state,repeats=1):
    return {key:np.repeat(np.asarray(value),repeats,axis=0).copy() for key,value in state.items()}


def motion_step(model,position_velocity,state,u,d,rate,amplitude,dt=.0025,normal_override=None):
    x,v=np.asarray(position_velocity).T
    z=np.tanh(x/.04);slope=amplitude*(1-z*z)/.04
    delta=d+amplitude*z;dr=rate+slope*v
    n,ft,newstate=model.step(delta,dr,v,state,normal_override=normal_override,dt=dt)
    bad=(~np.isfinite(n))|(~np.isfinite(ft))|(np.abs(n)>1e4)|(np.abs(ft)>1e4)
    n=np.clip(np.nan_to_num(n,nan=0,posinf=1e4,neginf=0),0,1e4)
    ft=np.clip(np.nan_to_num(ft,nan=0,posinf=1e4,neginf=-1e4),-1e4,1e4)
    vn=v+dt*(u-slope*n+ft)/2.;xn=x+dt*vn
    bad|=(~np.isfinite(xn))|(~np.isfinite(vn))|(np.abs(xn)>2)|(np.abs(vn)>20)
    result=np.column_stack([np.clip(np.nan_to_num(xn,nan=0,posinf=2,neginf=-2),-2,2),np.clip(np.nan_to_num(vn,nan=0,posinf=20,neginf=-20),-20,20)])
    return result,newstate,n,ft,bad
