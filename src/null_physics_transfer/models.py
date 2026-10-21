"""Predictors use declared geometry, unit scales and measured calibration only."""
import numpy as np
import torch
from torch import nn
from .common import VREG

NI=[0,1,3,4,5]; FI=[2,6,7,8,9]

def mlp(inp,width,out):
    return nn.Sequential(nn.Linear(inp,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,out))

class Neural(nn.Module):
    def __init__(self,kind):
        super().__init__(); self.kind=kind
        if kind in ('modular','no_history'):
            self.normal_net=mlp(5,24,1); self.friction_net=mlp(5,24,1)
        elif kind=='moe':
            self.experts=nn.ModuleList([mlp(10,20,16),mlp(10,20,16)])
            self.gate=mlp(10,12,2); self.head=nn.Linear(16,2)
        else:
            self.net=mlp(10,32,2)

    def outputs(self,f):
        if self.kind=='no_history':
            f=f.clone(); f[:,3:9]=0
        if self.kind in ('modular','no_history'):
            return torch.cat([self.normal_net(f[:,NI]),self.friction_net(f[:,FI])],-1)
        if self.kind=='moe':
            # PRISM-WM-inspired state-space adaptation: gate and stabilized
            # Gram-Schmidt on latent experts. No policy/reward/latent encoder reproduction.
            a,b=[e(f) for e in self.experts]
            a=nn.functional.normalize(a,dim=-1,eps=1e-6)
            b=nn.functional.normalize(b-(a*b).sum(-1,keepdim=True)*a,dim=-1,eps=1e-6)
            w=torch.softmax(self.gate(f),-1)
            return self.head(w[:,0,None]*a+w[:,1,None]*b)
        return self.net(f)

    def normal(self,f):
        f=f.clone(); f[:,9]=0 # target normal force is NEVER input to this head
        n=torch.relu(10*f[:,0]+20*self.outputs(f)[:,0])
        return n*(f[:,0]>0)

    def friction(self,f):
        # Local dissipative regularized friction law; speed regularization known.
        mu=nn.functional.softplus(self.outputs(f)[:,1]-.87)
        return -20*f[:,9]*mu*torch.tanh(.5*f[:,2]/VREG)

    @torch.no_grad()
    def predict(self,f):
        t=torch.as_tensor(f,dtype=torch.float32)
        n=self.normal(t); t=t.clone(); t[:,9]=n/20
        ft=self.friction(t)
        return n.numpy(),ft.numpy()

class Classical:
    def __init__(self,kind='nominal',params=None):
        self.kind=kind; self.params=params or {'k':500.,'c':0.,'mu':.35}
    def predict(self,f):
        d=f[:,0]*.02; rate=f[:,1]*.1; v=f[:,2]*.5
        if self.kind in ('nominal','identified'):
            n=np.maximum(0,self.params['k']*d+self.params['c']*rate)*(d>0)
            ft=-self.params['mu']*n*np.tanh(v/VREG)
        else:
            n=np.maximum(0,normal_basis(f)@np.array(self.params['normal']))*(d>0)
            f=f.copy();f[:,9]=n/20
            mu=np.maximum(0,friction_basis(f)@np.array(self.params['friction']))
            ft=-n*mu*np.tanh(v/VREG)
        return n,ft

def normal_basis(f):
    d=f[:,0]
    return np.column_stack([d,f[:,1],d**3,f[:,3:6],np.tanh(f[:,3:6]),d**2])

def friction_basis(f):
    v=f[:,2]*.5; hv=v[:,None]-.5*f[:,6:9]
    return np.column_stack([np.ones(len(f)), np.exp(-(v/.16)**2), 1/(1+(v/.12)**2),
        np.abs(hv),np.tanh(f[:,9]*20/15),np.abs(v),
        v/(np.tanh(v/VREG)+np.where(v>=0,1e-5,-1e-5))/np.maximum(f[:,9]*20,1)])

def fit_classical(kind,f,n,ft,mask_n,mask_f):
    from scipy.optimize import nnls
    if kind=='identified':
        coef=nnls(np.column_stack([f[mask_n,0]*.02,f[mask_n,1]*.1]),n[mask_n])[0]
        reg=-n[mask_f]*np.tanh(f[mask_f,2]*.5/VREG)
        mu=max(0,float(np.dot(reg,ft[mask_f])/np.dot(reg,reg)))
        return Classical(kind,dict(k=float(coef[0]),c=float(coef[1]),mu=mu))
    # Broad analytic dictionary: two velocity shapes, multiple relaxation scales,
    # pressure dependence and nonlinear spring; identified only from calibration.
    cn=nnls(normal_basis(f[mask_n]),n[mask_n])[0]
    use=mask_f & (np.abs(f[:,2])>.015)
    y=-ft[use]/(n[use]*np.tanh(f[use,2]*.5/VREG))
    cf=nnls(friction_basis(f[use]),y)[0]
    return Classical('analytic',dict(normal=cn.tolist(),friction=cf.tolist()))

def freeze(model):
    if isinstance(model,nn.Module):
        model.eval()
        for p in model.parameters(): p.requires_grad_(False)
    return model
