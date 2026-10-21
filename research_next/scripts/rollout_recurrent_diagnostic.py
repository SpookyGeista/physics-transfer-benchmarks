"""Exploratory reuse of inspected v2 replay data, never a new confirmation."""
import argparse, hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent/'study_v2/src')]
import numpy as np
import pandas as pd
import torch
from recurrent import Recurrent
from decision_study.common import cam, DT, MASS

p=argparse.ArgumentParser();p.add_argument('--model-root',type=Path,default=ROOT/'results/recurrent_diagnostic');p.add_argument('--out',type=Path,default=ROOT/'results/recurrent_development_rollout');a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
metrics=[]
for world in ['maxwell','boucwen']:
    dest=a.out/f'{world}.npz'
    assert not dest.exists(),'choose new output directory'
    path=ROOT.parent/f'study_v2/supplement_exact/results/data/{world}/r0/replay.npz'
    data=dict(np.load(path));rows=json.loads(path.with_suffix('.json').read_text())['rows']
    checkpoint=torch.load(a.model_root/world/'model.pt',map_location='cpu',weights_only=True)
    model=Recurrent(checkpoint['hidden']);model.load_state_dict(checkpoint['state']);model.eval()
    state=data['state'][:,0].copy();amp=np.array([r['amplitude'] for r in rows]);hn=hf=None
    rec={k:[] for k in ['state','normal','friction','guard']};tic=time.perf_counter()
    with torch.inference_mode():
        for t in range(data['u'].shape[1]):
            x,v=state.T;g,slope=cam(x,amp);delta=data['d'][:,t]+g;rate=data['rate'][:,t]+slope*v
            ni=torch.tensor(np.column_stack([delta/.02,rate/.1]),dtype=torch.float32)
            n,hn=model.normal_step(ni,hn)
            fi=torch.tensor(np.column_stack([v/.5,np.abs(v)/.5,n.numpy()/20]),dtype=torch.float32)
            if t==0:
                initial_f=fi.clone();initial_f[:,2]=torch.tensor(data['normal'][:,0]/20,dtype=torch.float32)
                hf=model.friction_head.init(initial_f)
            ft,hf=model.friction_step(fi,hf);n=n.numpy().astype(float);ft=ft.numpy().astype(float)
            guard=(~np.isfinite(n))|(~np.isfinite(ft))|(np.abs(n)>1e4)|(np.abs(ft)>1e4)
            n=np.clip(np.nan_to_num(n,nan=0,posinf=1e4,neginf=0),0,1e4);ft=np.clip(np.nan_to_num(ft,nan=0,posinf=1e4,neginf=-1e4),-1e4,1e4)
            vn=v+DT*(data['u'][:,t]-slope*n+ft)/MASS;xn=x+DT*vn
            guard|=(~np.isfinite(xn))|(~np.isfinite(vn))|(np.abs(xn)>2)|(np.abs(vn)>20)
            state=np.column_stack([np.clip(np.nan_to_num(xn,nan=0,posinf=2,neginf=-2),-2,2),np.clip(np.nan_to_num(vn,nan=0,posinf=20,neginf=-20),-20,20)])
            for k,value in [('state',state),('normal',n),('friction',ft),('guard',guard)]:rec[k].append(value.copy())
    rec={k:np.stack(v,1) for k,v in rec.items()}
    np.savez_compressed(dest,**rec,reference=data['next_state'])
    for i,row in enumerate(rows):
        metrics.append(dict(world=world,**row,position_rmse=float(np.mean((rec['state'][i,:,0]-data['next_state'][i,:,0])**2)**.5),normal_rmse=float(np.mean((rec['normal'][i]-data['normal'][i])**2)**.5),friction_rmse=float(np.mean((rec['friction'][i]-data['friction'][i])**2)**.5),guard_steps=int(rec['guard'][i].sum())))
    print(world,'mean x RMSE mm',np.mean([r['position_rmse'] for r in metrics if r['world']==world])*1000,'seconds',time.perf_counter()-tic,flush=True)
pd.DataFrame(metrics).to_csv(a.out/'metrics.csv',index=False)
(a.out/'provenance.json').write_text(json.dumps(dict(scope='exploratory old inspected v2 r0 replay; not confirmation',model_root=str(a.model_root),initialization='one common initial N measurement initializes friction latent; future N predicted; normal state has no force-label input',source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'src/recurrent.py']}),indent=2))
