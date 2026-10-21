"""Bounded local-only GRU development run; no new test access."""
import argparse,json,sys,time,copy,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from recurrent import Recurrent,inputs
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'study_v2'
p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--updates',type=int,default=1500);p.add_argument('--hidden',type=int,default=12);p.add_argument('--worlds',nargs='+',default=['maxwell','boucwen']);p.add_argument('--seed',type=int,default=0);p.add_argument('--out',type=Path,default=ROOT/'results/recurrent_diagnostic');p.add_argument('--data-root',type=Path,default=OLD/'results/data');p.add_argument('--noise',choices=['clean','noisy'],default='clean');a=p.parse_args();torch.set_num_threads(2)
if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(.25)

def data(world,split):
    path=a.data_root/f'{world}/r{a.seed}/{split}.npz';arrays=dict(np.load(path));rows=json.loads(path.with_suffix('.json').read_text())['rows'];normal=np.array([r['mode']=='B' for r in rows]);friction=np.array([r['mode']=='A' for r in rows])
    if a.noise=='noisy':
        rng=np.random.default_rng(7200000+a.seed*100+{'train':1,'val':2}[split])
        arrays['delta']+=rng.normal(0,.00005,arrays['delta'].shape)
        arrays['rate']+=rng.normal(0,.001,arrays['rate'].shape)
        arrays['state'][...,1]+=rng.normal(0,.002,arrays['state'][...,1].shape)
        arrays['normal']+=rng.normal(0,.15,arrays['normal'].shape)
        arrays['friction']+=rng.normal(0,.10,arrays['friction'].shape)
    return [torch.as_tensor(x,device=a.device) for x in [inputs(arrays,'normal')[normal],arrays['normal'][normal].astype(np.float32),inputs(arrays,'friction')[friction],arrays['friction'][friction].astype(np.float32)]]
for world in a.worlds:
    dest=a.out/world;dest.mkdir(parents=True,exist_ok=True);assert not (dest/'model.pt').exists(),'use fresh output, never silently overwrite'
    source_files=[Path(__file__),ROOT/'src/recurrent.py',*[a.data_root/f'{world}/r{a.seed}/{split}.{ext}' for split in ['train','val'] for ext in ['npz','json']]]
    used_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    torch.manual_seed(a.seed);model=Recurrent(a.hidden).to(a.device);optimizer=torch.optim.Adam(model.parameters(),lr=.002,weight_decay=1e-6);tr=data(world,'train');va=data(world,'val');best=float('inf');beststate=None;curves=[];tic=time.perf_counter()
    for update in range(a.updates):
        model.train();loss=((model.normal_sequence(tr[0])-tr[1])/20).square().mean()+((model.friction_sequence(tr[2])-tr[3])/10).square().mean();assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if update%50==0 or update==a.updates-1:
            model.eval()
            with torch.no_grad():
                nr=((model.normal_sequence(va[0])-va[1])**2).mean().sqrt().item();fr=((model.friction_sequence(va[2])-va[3])**2).mean().sqrt().item();vl=(nr/20)**2+(fr/10)**2
            if vl<best:best=vl;beststate=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()});best_update=update
            curves.append(dict(update=update,train_loss=loss.item(),val_loss=vl,normal_rmse=nr,friction_rmse=fr,elapsed=time.perf_counter()-tic))
            (dest/'learning.json').write_text(json.dumps(curves,indent=2));torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if a.device=='cuda' else None,update=update,args=vars(a)|{'out':str(a.out),'data_root':str(a.data_root)}),dest/'continuation.pt')
            print(world,update,'local val N/Ft',nr,fr,'seconds',round(time.perf_counter()-tic,1),flush=True)
    assert used_hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},'inputs changed during fitting'
    model.load_state_dict(beststate);torch.save(dict(state=beststate,hidden=a.hidden),dest/'model.pt');(dest/'metadata.json').write_text(json.dumps(dict(**model.metadata(),world=world,seed=a.seed,noise=a.noise,selected_update=best_update,seconds=time.perf_counter()-tic,updates=a.updates,device=a.device,data_root=str(a.data_root),scope='isolated train/validation only; role determined by enclosing locked experiment',source_sha256=used_hashes,model_sha256=hashlib.sha256((dest/'model.pt').read_bytes()).hexdigest()),indent=2))
