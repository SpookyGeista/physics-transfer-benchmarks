"""Causal local recurrent material predictors; no reference-law imports."""
import numpy as np
import torch
from torch import nn

class Head(nn.Module):
    def __init__(self, inputs, hidden=12):
        super().__init__();self.hidden=hidden
        self.initial=nn.Linear(inputs,hidden)
        self.cell=nn.GRU(inputs,hidden,batch_first=True)
        self.readout=nn.Sequential(nn.Linear(inputs+hidden,24),nn.Tanh(),nn.Linear(24,1))
        nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias)
    def init(self,x):return torch.tanh(self.initial(x)).unsqueeze(0)
    def sequence(self,x):
        h0=self.init(x[:,0]);future,_=self.cell(x,h0)
        past=torch.cat([h0.transpose(0,1),future[:,:-1]],1)
        return self.readout(torch.cat([x,past],-1)).squeeze(-1)
    def step(self,x,h):
        if h is None:h=self.init(x)
        force=self.readout(torch.cat([x,h[0]],-1)).squeeze(-1)
        _,next_h=self.cell(x[:,None],h)
        return force,next_h

class Recurrent(nn.Module):
    def __init__(self,hidden=12):
        super().__init__();self.hidden=hidden;self.normal_head=Head(2,hidden);self.friction_head=Head(3,hidden);self.viscous=nn.Parameter(torch.tensor(-2.))
    def normal_sequence(self,x):return torch.relu(10*x[...,0]+20*self.normal_head.sequence(x))*(x[...,0]>0)
    def friction_sequence(self,x):
        mu=nn.functional.softplus(self.friction_head.sequence(x)-.87)
        return -20*torch.relu(x[...,2])*mu*torch.tanh(x[...,0]*.5/.04)-nn.functional.softplus(self.viscous)*x[...,0]*.5
    def normal_step(self,x,h=None):
        raw,h=self.normal_head.step(x,h);return torch.relu(10*x[:,0]+20*raw)*(x[:,0]>0),h
    def friction_step(self,x,h=None):
        raw,h=self.friction_head.step(x,h);ft=-20*torch.relu(x[:,2])*nn.functional.softplus(raw-.87)*torch.tanh(x[:,0]*.5/.04)-nn.functional.softplus(self.viscous)*x[:,0]*.5
        return ft,h
    def metadata(self):return {'hidden':self.hidden,'parameters':sum(p.numel() for p in self.parameters()),'history':'causal GRU, output before state update','known_dt':.0025,'reference_parameters_used':False}

def inputs(data,mode):
    if mode=='normal':return np.stack([data['delta']/.02,data['rate']/.1],-1).astype(np.float32)
    return np.stack([data['state'][...,1]/.5,np.abs(data['state'][...,1])/.5,data['normal']/20],-1).astype(np.float32)
