import argparse,json,sys,time,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from null_physics_transfer.common import CFG,DT,design,commands,history_update,dump,digest
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--models',default='results/models');p.add_argument('--out',default='results/control');p.add_argument('--count',type=int,default=3);p.add_argument('--seeds',type=int,nargs='+',default=CFG['training_seeds']);AppLauncher.add_app_launcher_args(p);a=p.parse_args()
app=AppLauncher(a).app
import numpy as np
import torch
from null_physics_transfer.reference import Reference
from null_physics_transfer.predict import load_model,choose_action
from null_physics_transfer.isaac_rig import IsaacRig

torch.set_num_threads(2);cases=design('control',a.count);n=len(cases);slope=np.array([r['slope'] for r in cases]);cfg=CFG['controller'];cmd=commands(cases,cfg['duration_steps']+cfg['horizon_steps']);rig=IsaacRig(n,a.device);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);records=[]
lock=json.loads(Path('results/protocol_lock.json').read_text())
for path,sha in lock['files'].items():assert digest(path)==sha,f'Changed frozen file {path}'
for world in CFG['worlds']:
    for seed in a.seeds:
        for method in CFG['methods']:
            path=Path(a.models)/world/f'b{max(CFG["budgets"])}'/f's{seed}'/('modular' if method=='decoupled' else method)
            result_path=out/f'{world}_{method}_s{seed}.json'
            if result_path.exists():records.extend(json.loads(result_path.read_text()));continue
            model=load_model(path);rig.reset();ref=Reference(world,n,cmd['d'][:,0]);hist=np.zeros((n,3,2));hist[:,:,0]=cmd['d'][:,0,None];trajectory=[];forces=[];tic=time.time();positive=np.zeros(n);maxdelta=np.zeros(n);maxx=np.zeros(n)
            for t in range(cfg['duration_steps']):
                st=rig.state();x,v=st.T
                if t%cfg['replan_steps']==0:
                    u=choose_action(model,x,v,hist,cmd['d'][:,t:t+cfg['horizon_steps']],cmd['rate'][:,t:t+cfg['horizon_steps']],slope,method=='decoupled')
                delta=cmd['d'][:,t]+slope*x;rate=cmd['rate'][:,t]+slope*v
                normal,ft=ref.observe_force(delta,rate,v);force=u-slope*normal+ft
                nxt=rig.advance(force);trajectory.append(nxt);forces.append(np.column_stack([u,normal,ft]));hist=history_update(hist,delta,v)
                positive+=np.maximum(0,ft*v)*DT;maxdelta=np.maximum(maxdelta,delta);maxx=np.maximum(maxx,np.abs(nxt[:,0]))
            last=rig.state();elapsed=time.time()-tic;success=(np.abs(last[:,0]-cfg['target'])<cfg['position_tolerance'])&(np.abs(last[:,1])<cfg['velocity_tolerance'])
            batch=[]
            for i,c in enumerate(cases):
                batch.append(dict(world=world,seed=seed,method=method,episode=c['id'],condition=c['condition'],success=int(success[i]),position_error=float(abs(last[i,0]-cfg['target'])),final_velocity=float(last[i,1]),seconds=elapsed/n,max_compression=float(maxdelta[i]),positive_friction_work=float(positive[i]),unstable=int(maxx[i]>2 or not np.isfinite(last[i]).all())))
            dump(result_path,batch);records.extend(batch)
            if seed==0:np.savez_compressed(out/f'{world}_{method}_trajectory.npz',state=np.stack(trajectory,axis=1),forces=np.stack(forces,axis=1),d=cmd['d'][:,:cfg['duration_steps']],slope=slope)
            print(f'CONTROL {world} s{seed} {method}: {success.sum()}/{n}, {elapsed:.2f}s',flush=True)
with (out/'control.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
app.close(wait_for_replicator=False, skip_cleanup=True)
