import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
CFG=json.loads((ROOT/'configs/experiment.json').read_text());DT=CFG['dt'];MASS=CFG['mass'];VREG=CFG['vreg'];TAUS=np.array(CFG['history_taus'])
NI=[0,1,3,4,5];FI=[2,6,7,8,9,10,11,12]

def dump(path,data):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2,allow_nan=False))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def history_init(delta,v,n):return np.repeat(np.column_stack([delta,v,n])[:,None,:],3,axis=1)
def history_update(h,d,v,n,dt=DT):return h+(1-np.exp(-dt/TAUS))[None,:,None]*(np.column_stack([np.maximum(d,0),v,n])[:,None,:]-h)
def features(d,r,v,h,n):
 return np.column_stack([d/.02,r/.1,v/.5,(d[:,None]-h[:,:,0])/.02,(v[:,None]-h[:,:,1])/.5,n/20,(n[:,None]-h[:,:,2])/20]).astype(np.float32)
def cam(x,amplitude):
 z=np.tanh(x/CFG['cam_length']);return amplitude*z,amplitude*(1-z*z)/CFG['cam_length']
def profiles(split,seed,count=None,steps=None):
 count=count or CFG['profiles_'+split];steps=steps or CFG['steps'];t=np.arange(steps+1)*DT;T=CFG['steps']*DT
 rows=[];arrays={k:[] for k in ['delta','rate','v','x']}
 for i in range(count):
  rng=np.random.default_rng(CFG['source_seeds'][split]+seed*1000+i)
  f1,f2=rng.uniform(.35,1.2,2);p=rng.uniform(0,2*np.pi);amp=rng.uniform(.004,.008)
  delta=.02+amp*(.7*np.sin(2*np.pi*f1*t+p)+.3*np.sin(2*np.pi*f2*t))
  rate=amp*(.7*2*np.pi*f1*np.cos(2*np.pi*f1*t+p)+.3*2*np.pi*f2*np.cos(2*np.pi*f2*t))
  # Smooth starts/stops; varied reversals, no reference parameters in stimulus.
  env=np.sin(np.pi*np.clip(t/T,0,1))**2
  v=rng.uniform(.10,.28)*env*(.75*np.sin(2*np.pi*rng.uniform(.4,1.2)*t)+.25*np.sin(2*np.pi*rng.uniform(1.2,2.4)*t))
  x=np.zeros_like(v);x[1:]=np.cumsum(v[1:])*DT
  for key,value in [('delta',delta),('rate',rate),('v',v),('x',x)]:arrays[key].append(value)
  rows.append(dict(id=f'{split}-r{seed}-p{i}',profile=i,replicate=seed,stimulus_seed=CFG['source_seeds'][split]+seed*1000+i))
 return {k:np.array(v) for k,v in arrays.items()},rows

def force_commands(seed,count,steps):
 t=np.arange(steps+CFG['controller']['horizon'])*DT;rng=np.random.default_rng(CFG['source_seeds']['control']+seed)
 d=np.array([.02+.0015*np.sin(2*np.pi*rng.uniform(.4,.9)*t+rng.uniform(0,6.28)) for _ in range(count)])
 # derivative computed from the known command, never from reference force
 rate=np.gradient(d,DT,axis=1)
 return d,rate
