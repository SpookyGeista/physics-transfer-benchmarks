"""Locked primary evaluation; no reference module is imported by any predictor."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from decision_study.common import ROOT,CFG,dump,digest
from decision_study.models import load_model
from decision_study.prediction import rollout
from decision_study.locking import verify_lock
p=argparse.ArgumentParser();p.add_argument('--worlds',nargs='+',default=CFG['worlds']);p.add_argument('--seeds',nargs='+',type=int,default=CFG['seeds']);p.add_argument('--budgets',nargs='+',type=int,default=CFG['budgets']);p.add_argument('--noise',nargs='+',default=CFG['noise_conditions']);p.add_argument('--methods',nargs='+',default=CFG['methods']);a=p.parse_args()
lockhash=verify_lock();torch.set_num_threads(1);output=ROOT/'results/evaluation';output.mkdir(exist_ok=True);traces=ROOT/'results/traces';traces.mkdir(exist_ok=True)
for world in a.worlds:
 for seed in a.seeds:
  path=ROOT/f'results/data/{world}/r{seed}/replay.npz';data=dict(np.load(path));meta=json.loads(path.with_suffix('.json').read_text())['rows'];amp=np.array([r['amplitude'] for r in meta]);ref=data['next_state'];commands={k:data[k] for k in ['u','d','rate']}
  for budget in a.budgets:
   for noise in a.noise:
    for method in a.methods:
     dest=output/f'{world}_r{seed}_b{budget}_{noise}_{method}.csv';md=dest.with_suffix('.json')
     if dest.exists() and md.exists():
      previous=json.loads(md.read_text());assert previous['lock_sha256']==lockhash and previous['data_sha256']==digest(path);print('SKIP',dest.name,flush=True);continue
     model=load_model(ROOT/f'results/models/{world}/r{seed}/b{budget}/{noise}/{"modular" if method=="decoupled" else method}');tic=time.perf_counter()
     pred=rollout(model,commands,amp,data['state'][:,0],data['normal'][:,0],noise,seed,method=='decoupled');elapsed=time.perf_counter()-tic
     conditional_features=data['features'].reshape(-1,13).copy()
     if method=='decoupled':conditional_features[:,9]=.5;conditional_features[:,10:13]=0
     fn,fft=model.predict_local(conditional_features);fn=fn.reshape(data['normal'].shape);fft=fft.reshape(fn.shape)
     rows=[]
     for i,row in enumerate(meta):
      error=pred['state'][i]-ref[i]
      rows.append(dict(world=world,seed=seed,budget=budget,noise=noise,method=method,**row,position_rmse=float(np.sqrt(np.mean(error[:,0]**2))),velocity_rmse=float(np.sqrt(np.mean(error[:,1]**2))),final_error=float(abs(error[-1,0])),normal_rmse=float(np.sqrt(np.mean((pred['normal'][i]-data['normal'][i])**2))),friction_rmse=float(np.sqrt(np.mean((pred['friction'][i]-data['friction'][i])**2))),conditional_normal_rmse=float(np.sqrt(np.mean((fn[i]-data['normal'][i])**2))),conditional_friction_rmse=float(np.sqrt(np.mean((fft[i]-data['friction'][i])**2))),guard_steps=int(pred['guard'][i].sum()),seconds_per_episode=elapsed/len(meta)))
     pd.DataFrame(rows).to_csv(dest,index=False);dump(md,dict(lock_sha256=lockhash,data_sha256=digest(path),seconds=elapsed,rows=len(rows),conditional_measurement='true local states and current N; diagnostic only',initialization='common state and current N sensor, zero previous history departures; no future corrections'))
     if seed==0 and budget==24 and noise=='clean':np.savez_compressed(traces/f'{world}_{method}.npz',**pred,reference=ref,amplitude=amp)
     print('DONE',dest.name,f'{elapsed:.2f}s',flush=True)
files=sorted(output.glob('*.csv'));pd.concat([pd.read_csv(f) for f in files],ignore_index=True).to_csv(ROOT/'results/prediction_metrics.csv',index=False)
print('EVALUATION_COMPLETE',flush=True)
