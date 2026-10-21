import argparse, copy, json, sys, time, hashlib, resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from null_physics_transfer.common import CFG, dump, digest
from null_physics_transfer.models import Neural, Classical, fit_classical, freeze
p=argparse.ArgumentParser(); p.add_argument('--data',default='results/data'); p.add_argument('--out',default='results/models')
p.add_argument('--seeds',type=int,nargs='+',default=CFG['training_seeds']); p.add_argument('--budgets',type=int,nargs='+',default=CFG['budgets']); p.add_argument('--epochs',type=int,default=CFG['epochs']); p.add_argument('--worlds',nargs='+',default=CFG['worlds']); p.add_argument('--methods',nargs='+',default=['nominal','identified','analytic','joint','moe','modular','no_history','privileged']); a=p.parse_args()
torch.set_num_threads(2)

def dataset(world,split,budget=None,seed=0):
    path=Path(a.data)/f'{world}_{split}.npz'; data=np.load(path)
    meta=json.loads(path.with_suffix('.json').read_text()); rows=meta['rows']; ids=list(range(len(rows)))
    if budget is not None:
        rng=np.random.default_rng(seed+170)
        ids=[]
        for mode in ['A','B']:
            choices=np.array([i for i,r in enumerate(rows) if r['mode']==mode])
            ids.extend(rng.permutation(choices)[:budget])
    f=data['features'][ids].reshape(-1,10); n=data['normal'][ids].ravel(); ft=data['friction'][ids].ravel()
    mn=np.repeat([rows[i]['mode']!='A' for i in ids],data['features'].shape[1]); mf=np.repeat([rows[i]['mode']!='B' for i in ids],data['features'].shape[1])
    return f,n,ft,mn,mf,[rows[i]['id'] for i in ids],digest(path)

for world in a.worlds:
    val=dataset(world,'val'); vf,vn,vft,vmn,vmf=[torch.tensor(x) for x in val[:5]]
    for budget in a.budgets:
        for seed in a.seeds:
            data=dataset(world,'train',budget,seed)
            for method in a.methods:
                dest=Path(a.out)/world/f'b{budget}'/f's{seed}'/method; dest.mkdir(parents=True,exist_ok=True)
                if (dest/'metadata.json').exists():
                    print('SKIP completed',dest,flush=True); continue
                start=time.time(); torch.manual_seed(seed); np.random.seed(seed)
                f,n,ft,mn,mf,ids,sha=data
                sources=[sha]
                if method=='privileged':
                    extra=dataset(world,'privileged'); f,n,ft,mn,mf=[np.concatenate([x,y]) for x,y in zip([f,n,ft,mn,mf],extra[:5])]; ids=ids+extra[5]; sources.append(extra[6])
                curves=[]
                if method in ['nominal','identified','analytic']:
                    model=Classical() if method=='nominal' else fit_classical(method,f,n,ft,mn,mf)
                    dump(dest/'classical.json',dict(kind=model.kind,params=model.params)); params=len(model.params) if method!='analytic' else sum(map(len,model.params.values()))
                else:
                    model=Neural('joint' if method=='privileged' else method)
                    opt=torch.optim.Adam(model.parameters(),lr=CFG['learning_rate'],weight_decay=1e-6)
                    f,n,ft,mn,mf=[torch.tensor(x) for x in [f,n,ft,mn,mf]]
                    idx_n=torch.where(mn)[0]; idx_f=torch.where(mf)[0]
                    best=float('inf'); best_state=None; best_epoch=0
                    for epoch in range(a.epochs):
                        bn=idx_n[torch.randint(len(idx_n),(CFG['batch_size']//2,))]; bf=idx_f[torch.randint(len(idx_f),(CFG['batch_size']//2,))]
                        loss=((model.normal(f[bn])-n[bn])/20).square().mean()+((model.friction(f[bf])-ft[bf])/10).square().mean()
                        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
                        if epoch%50==0 or epoch==a.epochs-1:
                            with torch.no_grad():
                                vl=(((model.normal(vf[vmn])-vn[vmn])/20).square().mean()+((model.friction(vf[vmf])-vft[vmf])/10).square().mean()).item()
                            curves.append(dict(update=epoch,train_loss=float(loss.item()),val_loss=vl))
                            if vl<best: best=vl;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
                    model.load_state_dict(best_state); freeze(model); torch.save(dict(kind=model.kind,state=model.state_dict()),dest/'model.pt')
                    params=sum(p.numel() for p in model.parameters())
                    dump(dest/'learning.json',curves)
                elapsed=time.time()-start
                dump(dest/'metadata.json',dict(world=world,budget=budget,seed=seed,method=method,parameters=params,seconds=elapsed,
                    training_ids=ids,data_sha256=sources,validation_sha256=val[6],epochs=a.epochs if curves else 0,best_epoch=best_epoch if curves else None,
                    frozen=True,test_data_used=False,device='cpu',max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                print(f'DONE {world} b{budget} s{seed} {method} {elapsed:.1f}s '+(f'val={best:.6g}' if curves else ''),flush=True)
