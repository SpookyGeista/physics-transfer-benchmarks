"""Native exploratory learned-model task probe; existing preflight conditions."""
import argparse,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'study_v2/src')]
import numpy as np
from controller_preflight import conditions,commands,preparation,FORCES,HORIZON,REPLAN,WARM_STEPS,STEPS,DT,stage_cost,observe_force,arrival_metrics
from decision_study.common import cam
from decision_study.reference import Reference
from online import RecurrentOnline,Nominal,clone_state,motion_step


def plan(model,state,history,normal,d,rate,target,amplitude):
    n=len(state);k=len(FORCES);s=np.repeat(state,k,axis=0);h=clone_state(history,k)
    amp=np.repeat(amplitude,k);u=np.tile(FORCES,n);cost=np.zeros(n*k);bad_all=np.zeros(n*k,bool)
    for t in range(d.shape[1]):
        s,h,_,_,bad=motion_step(model,s,h,u,np.repeat(d[:,t],k),np.repeat(rate[:,t],k),amp,
                               normal_override=np.repeat(normal,k) if t==0 else None)
        cost+=DT*stage_cost(s,u,np.repeat(target[:,t],k))+1e6*bad;bad_all|=bad
    cost=cost.reshape(n,k)
    return FORCES[np.argmin(cost,axis=1)],cost,bad_all.reshape(n,k)


def load(method,world):
    if method=='nominal':return Nominal(),[]
    if method=='recurrent':
        p=ROOT/f'results/recurrent_development_6000/{world}/model.pt'
        return RecurrentOnline(p),[p]
    if method in ['state_id','state_reset']:
        from state_id import StateID
        p=ROOT/f'results/state_id_development/{world}/r0/clean/model.json'
        return StateID.load(p),[p]
    if method=='blind_state_id':
        from state_id import CompositeStateID
        p=ROOT/f'results/state_id_development/{world}/r0/clean/blind_selection/model.json'
        return CompositeStateID.load(p),[p]
    if method in ['old_flex','path_ridge']:
        from online import FeatureOnline
        p=ROOT/f'results/feature_development/{world}/r0/clean/{method}/model.json'
        return FeatureOnline(p),[p]
    if method=='sensor_state_id':
        from online import SensorPersistent
        model,files=load('blind_state_id',world)
        return SensorPersistent(model),files
    raise ValueError(method)


def run(args):
    from isaaclab.app import AppLauncher
    app=AppLauncher(args).app
    try:
        import torch
        from decision_study.isaac_rig import IsaacRig
        torch.set_num_threads(2)
        rows=conditions();n=len(rows);amp=np.array([r['amplitude'] for r in rows]);sign=np.array([r['preparation_sign'] for r in rows]);d,rate,target=commands(rows)
        rig=IsaacRig(n,'cpu');out=ROOT/'results/model_development';out.mkdir(exist_ok=True)
        for world in args.worlds:
            for method in args.methods:
                dest=out/f'{world}_{method}.npz';metadata=dest.with_suffix('.json')
                model,files=load(method,world)
                hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/online.py',ROOT/'scripts/controller_preflight.py',*files]}
                if dest.exists() and metadata.exists():
                    old=json.loads(metadata.read_text());assert old['source_sha256']==hashes,'source/model changed; preserve old result';print('SKIP',dest.name,flush=True);continue
                assert not dest.exists() and not metadata.exists(),'partial result retained'
                rig.reset();ref=Reference(world,n,.02);initial_n=observe_force(ref,np.full(n,.02),np.zeros(n),np.zeros(n))[0]
                h=model.initial(np.full(n,.02),np.zeros(n),initial_n)
                warm={key:[] for key in ['delta','rate','normal','pred_normal']};tic=time.perf_counter()
                for t in range(WARM_STEPS):
                    dd,rr=preparation(t*DT,sign);nn,ff=ref.force(dd,rr,np.zeros(n));pred_n,_,_=model.step(dd,rr,np.zeros(n),h)
                    _,_,h=model.step(dd,rr,np.zeros(n),h,normal_override=nn)
                    ss=rig.advance(np.zeros(n));assert np.max(np.abs(ss))<1e-9
                    for key,value in [('delta',dd),('rate',rr),('normal',nn),('pred_normal',pred_n)]:warm[key].append(value.copy())
                if method=='state_reset':
                    nn=observe_force(ref,np.full(n,.02),np.zeros(n),np.zeros(n))[0]
                    h=model.initial(np.full(n,.02),np.zeros(n),nn)
                warm_seconds=time.perf_counter()-tic
                rec={key:[] for key in ['state','next_state','normal','friction','u','delta','compression_rate','net_force','stage_cost']};plans=[];guard=[];latency=[];u=np.zeros(n)
                for t in range(STEPS):
                    ss=rig.state();g,slope=cam(ss[:,0],amp);dd=d[:,t]+g;rr=rate[:,t]+slope*ss[:,1]
                    if t%REPLAN==0:
                        measured=observe_force(ref,dd,rr,ss[:,1])[0];pstart=time.perf_counter()
                        u,cost,bad=plan(model,ss,h,measured,d[:,t:t+HORIZON],rate[:,t:t+HORIZON],target[:,t:t+HORIZON],amp)
                        latency.append(time.perf_counter()-pstart);plans.append(cost);guard.append(bad)
                    nn,ff=ref.force(dd,rr,ss[:,1]);_,_,h=model.step(dd,rr,ss[:,1],h,normal_override=nn)
                    net=u-slope*nn+ff;ns=rig.advance(net);assert np.isfinite(ns).all() and np.max(np.abs(ns))<20
                    for key,value in [('state',ss),('next_state',ns),('normal',nn),('friction',ff),('u',u),('delta',dd),('compression_rate',rr),('net_force',net),('stage_cost',stage_cost(ns,u,target[:,t]))]:rec[key].append(value.copy())
                arrays={k:np.stack(v,1) for k,v in rec.items()};arrays.update({'warm_'+k:np.stack(v,1) for k,v in warm.items()})
                arrays.update(d=d[:,:STEPS],rate=rate[:,:STEPS],target=target[:,:STEPS],amplitude=amp,candidate_cost=np.stack(plans,1),candidate_guard=np.stack(guard,1),plan_seconds=latency)
                arrays['integral_cost']=DT*np.sum(arrays['stage_cost'],axis=1);arrays['arrival_s'],arrays['settling_s']=arrival_metrics(arrays['next_state'],arrays['target'])
                np.savez_compressed(dest,**arrays)
                metadata.write_text(json.dumps(dict(world=world,method=method,rows=rows,exploratory=True,source_sha256=hashes,seconds=time.perf_counter()-tic,warm_seconds=warm_seconds,first_planner_step='all models use current normal sensor; future normals predicted',real_history_update='measured current local kinematics and N; normal hidden state never uses own labels',npz_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()),indent=2))
                print('DONE',world,method,'cost',arrays['integral_cost'].mean(),'guard candidates',sum(map(np.sum,guard)),'seconds',time.perf_counter()-tic,flush=True)
    finally:app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__=='__main__':
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser();p.add_argument('--worlds',nargs='+',default=['maxwell','boucwen']);p.add_argument('--methods',nargs='+',default=['state_id','state_reset','recurrent']);AppLauncher.add_app_launcher_args(p);a=p.parse_args()
    if a.device!='cpu':p.error('Native CPU only')
    run(a)
