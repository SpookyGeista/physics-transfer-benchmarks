"""Fresh free-force native PhysX tasks with common model-predictive controller."""
import sys,argparse,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from decision_study.common import CFG,ROOT,DT,cam,force_commands,features,history_init,history_update,dump,digest
from decision_study.locking import verify_lock
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--worlds',nargs='+',default=CFG['worlds']);p.add_argument('--seeds',type=int,nargs='+',default=CFG['seeds']);p.add_argument('--noise',nargs='+',default=CFG['noise_conditions']);p.add_argument('--methods',nargs='+',default=CFG['methods']);AppLauncher.add_app_launcher_args(p);a=p.parse_args();lockhash=verify_lock();app=AppLauncher(a).app
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from decision_study.common import NI,FI
from decision_study.reference import Reference
from decision_study.models import load_model
from decision_study.prediction import plan,stage_cost
from decision_study.isaac_rig import IsaacRig
torch.set_num_threads(1);c=CFG['controller'];amplitude=np.repeat(CFG['cam_amplitudes'],c['per_condition']);n=len(amplitude);rig=IsaacRig(n,a.device);out=ROOT/'results/control';out.mkdir(exist_ok=True)
for world in a.worlds:
 for seed in a.seeds:
  d,rate=force_commands(seed,c['per_condition'],c['steps']);d=np.tile(d,(len(CFG['cam_amplitudes']),1));rate=np.tile(rate,(len(CFG['cam_amplitudes']),1))
  for noise in a.noise:
   # Full observation sequences drawn independently of model trajectories, paired across methods.
   rng=np.random.default_rng(940000+seed);measurement={k:rng.normal(0,s,(n,c['steps'])) if noise=='noisy' else np.zeros((n,c['steps'])) for k,s in CFG['noise'].items()}
   for method in a.methods:
    path=out/f'{world}_r{seed}_{noise}_{method}.npz';md=path.with_suffix('.json')
    if path.exists() and md.exists():assert json.loads(md.read_text())['lock_sha256']==lockhash;print('SKIP',path.name,flush=True);continue
    model=load_model(ROOT/f'results/models/{world}/r{seed}/b24/{noise}/{"modular" if method=="decoupled" else method}');rig.reset();ref=Reference(world,n,d[:,0]);h=None;true_h=None;u=np.zeros(n);cost=np.zeros(n);tic=time.perf_counter();rec={k:[] for k in ['state','next_state','features','normal','friction','u','d','rate']}
    for t in range(c['steps']):
     state=rig.state();x,v=state.T;g,slope=cam(x,amplitude);delta=d[:,t]+g;dr=rate[:,t]+slope*v;normal,ft=ref.force(delta,dr,v)
     observed=state.copy();observed[:,0]+=measurement['delta'][:,t];observed[:,1]+=measurement['velocity'][:,t];od=delta+measurement['delta'][:,t];on=normal+measurement['normal'][:,t]
     if h is None:h=history_init(od,observed[:,1],on)
     if true_h is None:true_h=history_init(delta,v,normal)
     f=features(delta,dr,v,true_h,normal)
     if t%c['replan']==0:
      u=plan(model,observed,h,on,d[:,t:t+c['horizon']],rate[:,t:t+c['horizon']],amplitude,method=='decoupled')
     ns=rig.advance(u-slope*normal+ft);cost+=DT*stage_cost(ns,u,c['target'])
     for key,value in [('state',state),('next_state',ns),('features',f),('normal',normal),('friction',ft),('u',u),('d',d[:,t]),('rate',rate[:,t])]:rec[key].append(value.copy())
     h=history_update(h,od,observed[:,1],on);true_h=history_update(true_h,delta,v,normal)
    arrays={k:np.stack(v,axis=1) for k,v in rec.items()};arrays['amplitude']=amplitude;arrays['cost']=cost
    terminal=arrays['next_state'][:,-round(.1/DT):];meanx=np.mean(abs(terminal[:,:,0]-c['target']),axis=1);meanv=np.mean(abs(terminal[:,:,1]),axis=1)
    arrays['success']=(meanx<c['position_tolerance'])&(meanv<c['velocity_tolerance']);np.savez_compressed(path,**arrays)
    rows=[dict(world=world,seed=seed,noise=noise,method=method,episode=i,amplitude=float(amplitude[i]),waveform=i%c['per_condition'],cost=float(cost[i]),success=bool(arrays['success'][i]),terminal_position_error=float(meanx[i]),terminal_velocity=float(meanv[i]),tracking_rmse=float(np.sqrt(np.mean((arrays['next_state'][i,:,0]-c['target'])**2)))) for i in range(n)]
    dump(md,dict(lock_sha256=lockhash,seconds=time.perf_counter()-tic,rows=rows,backend='native CPU PhysX',continuous_noisy_feedback=noise=='noisy',observed_fields=['position','velocity','compression','normal force'],force_driven=True,oracle_stimulus=False));print('DONE',path.name,f'{time.perf_counter()-tic:.2f}s',flush=True)
rows=[row for f in sorted(out.glob('*.json')) for row in json.loads(f.read_text())['rows']];pd.DataFrame(rows).to_csv(ROOT/'results/control_metrics.csv',index=False)
print('CONTROL_COMPLETE',flush=True);app.close(wait_for_replicator=False,skip_cleanup=True)
