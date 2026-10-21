import argparse,json,sys,time,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from null_physics_transfer.common import CFG,digest,dump
from null_physics_transfer.predict import load_model,rollout,metrics
p=argparse.ArgumentParser();p.add_argument('--data',default='results/data');p.add_argument('--models',default='results/models');p.add_argument('--out',default='results/evaluation');p.add_argument('--split',default='test');a=p.parse_args()
torch.set_num_threads(2);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
if a.split=='test':
    lock=Path('results/protocol_lock.json')
    assert lock.exists(),'Freeze protocol and checkpoints before test evaluation.'
    locked=json.loads(lock.read_text())
    for path,sha in locked['files'].items(): assert digest(path)==sha,f'Changed frozen file: {path}'
rows=[]
for world in CFG['worlds']:
    dpath=Path(a.data)/f'{world}_{a.split}.npz';data=np.load(dpath);meta=json.loads(dpath.with_suffix('.json').read_text());cases=meta['rows'];slope=np.array([r['slope'] for r in cases])
    cmd={k:data[k] for k in ['u','d','rate']}
    for mpath in sorted((Path(a.models)/world).glob('b*/s*/*/metadata.json')):
        md=json.loads(mpath.read_text());model=load_model(mpath.parent)
        methods=[md['method']]+(['decoupled'] if md['method']=='modular' else [])
        for method in methods:
            for noisy in [False,True]:
                tic=time.time();pred=rollout(model,cmd,slope,x0=data['state'][:,0,0],v0=data['state'][:,0,1],noise=noisy,decoupled=method=='decoupled');elapsed=time.time()-tic
                vals=metrics(pred,data)
                for i,c in enumerate(cases):
                    row=dict(world=world,budget=md['budget'],seed=md['seed'],method=method,noise=int(noisy),episode=c['id'],condition=c['condition'],seconds=elapsed/len(cases))
                    row.update({k:float(v[i]) for k,v in vals.items()});rows.append(row)
                if md['budget']==max(CFG['budgets']) and md['seed']==0 and not noisy:
                    np.savez_compressed(out/f'{world}_{method}_trajectories.npz',**pred)
                print('EVAL',world,md['budget'],md['seed'],method,'noise',noisy,'x',float(vals['position_rmse'].mean()),flush=True)
with (out/'prediction.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
dump(out/'manifest.json',dict(split=a.split,rows=len(rows),data={str(p):digest(p) for p in Path(a.data).glob(f'*_{a.split}.npz')}))
