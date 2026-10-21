"""Separated calibration and exact-local-replay composition using existing CPU PhysX."""
import sys,argparse,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from decision_study.common import CFG,DT,MASS,cam,profiles,features,history_init,history_update,dump,digest
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['calibration','replay'],required=True);p.add_argument('--seeds',nargs='+',type=int,default=CFG['seeds']);p.add_argument('--worlds',nargs='+',default=CFG['worlds']);p.add_argument('--out',default='results/data');p.add_argument('--count',type=int);p.add_argument('--steps',type=int);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import numpy as np
import torch
from decision_study.reference import Reference
from decision_study.isaac_rig import IsaacRig
nprof=a.count or CFG['profiles_train'];steps=a.steps or CFG['steps'];nmax=nprof*(len(CFG['cam_amplitudes']) if a.stage=='replay' else 1)
rig=IsaacRig(nmax,a.device);out=Path(a.out)

def save(world,seed,name,arrays,rows,tic,extra=None):
 folder=out/world/f'r{seed}';folder.mkdir(parents=True,exist_ok=True)
 path=folder/f'{name}.npz';np.savez_compressed(path,**arrays)
 dump(path.with_suffix('.json'),dict(world=world,replicate=seed,split=name,rows=rows,dt=DT,steps=steps,seconds=time.time()-tic,backend='Isaac Lab / native CPU PhysX',**(extra or {})))
 print('SAVED',world,seed,name,len(rows),steps,f'{time.time()-tic:.2f}s',flush=True)

def block(n):return {k:[] for k in ['features','normal','friction','state','next_state','delta','rate','u','d','force']}
def append(rec,**items):
 for key,value in items.items():rec[key].append(np.array(value,copy=True))
def stack(rec):return {k:np.stack(v,axis=1) for k,v in rec.items()}
def advance(force):
 f=np.zeros(nmax);f[:len(force)]=force;return rig.advance(f)[:len(force)]

def isolated(world,pr,rows):
 n=len(rows);T=steps;data=[];metas=[]
 normal_load=None
 for mode in ['B','A']:
  rig.reset();d0=pr['delta'][:,0] if mode=='B' else np.full(n,.02);ref=Reference(world,n,d0);rec=block(n);h=None
  for t in range(T):
   state=rig.state()[:n];x,v=state.T
   delta=pr['delta'][:,t] if mode=='B' else np.full(n,.02);rate=pr['rate'][:,t] if mode=='B' else np.zeros(n)
   normal,ft=ref.force(delta,rate,v,normal_override=None if mode=='B' else normal_load[:,t],friction_enabled=mode=='A')
   if h is None:h=history_init(delta,v,normal)
   # B: stationary slider, material compression is imposed independently.
   # A: ideal known tangential actuator tracks profile using measured friction.
   u=np.zeros(n) if mode=='B' else MASS*(pr['v'][:,t+1]-v)/DT-ft
   force=u+ft;ns=advance(force)
   append(rec,features=features(delta,rate,v,h,normal),normal=normal,friction=ft,state=state,next_state=ns,delta=delta,rate=rate,u=u,d=delta,force=force)
   h=history_update(h,delta,v,normal)
  dat=stack(rec)
  if mode=='B':normal_load=dat['normal'].copy()
  data.append(dat);metas.extend([{**r,'mode':mode,'id':r['id']+'-'+mode,'amplitude':0.} for r in rows])
 return {k:np.concatenate([r[k] for r in data]) for k in data[0]},metas

def composed(world,pr,rows,amplitudes):
 idx=np.tile(np.arange(len(rows)),len(amplitudes));amp=np.repeat(amplitudes,len(rows));n=len(idx)
 rig.reset();ref=Reference(world,n,pr['delta'][idx,0]);h=None;rec=block(n);dcmd=[];rcmd=[]
 for t in range(steps):
  state=rig.state()[:n];x,v=state.T
  desired_x=pr['x'][idx,t];desired_v=pr['v'][idx,t];gdes,sdes=cam(desired_x,amp)
  d=pr['delta'][idx,t]-gdes;rate_command=pr['rate'][idx,t]-sdes*desired_v
  gx,slope=cam(x,amp);delta=d+gx;rate=rate_command+slope*v
  normal,ft=ref.force(delta,rate,v)
  if h is None:h=history_init(delta,v,normal)
  # Reference-oracle stimulus construction ONCE; frozen log used by all models.
  u=MASS*(pr['v'][idx,t+1]-v)/DT+slope*normal-ft
  force=u-slope*normal+ft;ns=advance(force)
  append(rec,features=features(delta,rate,v,h,normal),normal=normal,friction=ft,state=state,next_state=ns,delta=delta,rate=rate,u=u,d=d,force=force)
  rcmd.append(rate_command);h=history_update(h,delta,v,normal)
 data=stack(rec);data['compression_rate']=data['rate'];data['rate']=np.stack(rcmd,axis=1)
 metas=[{**rows[j],'mode':'AB','amplitude':float(amp[k]),'id':f'{rows[j]["id"]}-cam{amp[k]:+.3f}'} for k,j in enumerate(idx)]
 return data,metas

if a.stage=='replay':
 lock=json.loads(Path('results/protocol_lock.json').read_text())
 for path,sha in lock['files'].items():assert digest(path)==sha,f'Changed frozen file {path}'
for world in a.worlds:
 for seed in a.seeds:
  if a.stage=='calibration':
   for split in ['train','val']:
    tic=time.time();pr,rows=profiles(split,seed,count=min(nprof,CFG['profiles_'+split]),steps=steps)
    data,meta=isolated(world,pr,rows);save(world,seed,split,data,meta,tic,{'cascade':'B measured normal-force history imposed as independent load in A','oracle_target_data':False})
   tic=time.time();pr,rows=profiles('privileged',seed,count=min(nprof,CFG['profiles_privileged']),steps=steps);data,meta=composed(world,pr,rows,[.004]);save(world,seed,'privileged',data,meta,tic,{'restricted_to':'privileged model','oracle_stimulus':True})
  else:
   tic=time.time();pr,rows=profiles('train',seed,count=nprof,steps=steps);data,meta=composed(world,pr,rows,CFG['cam_amplitudes']);save(world,seed,'replay',data,meta,tic,{'oracle_stimulus':True,'local_profiles_reused':True,'novelty_axis':'mechanical arrangement only','calibration_sha256':digest(out/world/f'r{seed}'/'train.npz')})
print('GENERATION_COMPLETE',flush=True)
app.close(wait_for_replicator=False,skip_cleanup=True)
