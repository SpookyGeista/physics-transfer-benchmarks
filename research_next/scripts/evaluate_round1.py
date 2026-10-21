"""Locked open predictions and native controller evaluations, all rows retained."""
import argparse,copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'study_v2/src')]
import numpy as np
from round1_lock import OUT,verify_model_lock,verify_test_lock,digest
from round1_models import load
from online import clone_state,motion_step
from run_model_development import plan
from controller_preflight import stage_cost
from decision_study.common import cam

def save(dest,arrays,metadata):
    dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists() and not dest.with_suffix('.json').exists()
    with dest.open('xb') as f:np.savez_compressed(f,**arrays)
    metadata['npz_sha256']=digest(dest)
    with dest.with_suffix('.json').open('x') as f:json.dump(metadata,f,indent=2)

def done(dest,lock,testsha):
    if dest.exists() and dest.with_suffix('.json').exists():
        meta=json.loads(dest.with_suffix('.json').read_text())
        assert meta['model_lock_sha256']==lock and meta['test_sha256']==testsha and meta['npz_sha256']==digest(dest)
        return True
    if dest.exists() or dest.with_suffix('.json').exists():raise RuntimeError(f'Partial output retained, requires review: {dest}')
    return False

def load_test(path,lock):
    metadata=json.loads(path.with_suffix('.json').read_text());h=digest(path)
    if metadata['model_lock_sha256']!=lock or metadata['npz_sha256']!=h:raise RuntimeError('Test provenance/content changed')
    for source,expected in metadata['source_sha256'].items():
        if digest(source)!=expected:raise RuntimeError(f'Test source changed: {source}')
    return dict(np.load(path)),metadata,h

def initial_history(model,data,cfg):
    n=len(data['warm_delta']);v=np.zeros(n)
    h=model.initial(data['warm_delta'][:,0],v,data['warm_normal'][:,0])
    for t in range(data['warm_delta'].shape[1]):
        _,_,h=model.step(data['warm_delta'][:,t],data['warm_rate'][:,t],v,h,normal_override=data['warm_normal'][:,t],dt=cfg['dt'])
    return h

def run_predictions(args,cfg,lock):
    import torch
    torch.set_num_threads(2);start=time.perf_counter()
    for nw,fw in cfg['compositions']:
        for seed in args.seeds:
            path=OUT/f'test/{nw}_{fw}/r{seed}.npz';data,meta,testsha=load_test(path,lock)
            if meta['model_lock_sha256']!=lock:raise RuntimeError('Test belongs to different locked models')
            for noise in cfg['noise']:
                for method in args.methods:
                    dest=OUT/f'predictions/{nw}_{fw}/r{seed}/{noise}/{method}.npz'
                    if done(dest,lock,testsha):continue
                    if time.perf_counter()-start>cfg['resources']['native_evaluation_wall_seconds_limit']:raise TimeoutError('Evaluation budget reached')
                    tic=time.perf_counter();model=load(method,nw,fw,seed,noise);h=initial_history(model,data,cfg);s=data['state'][:,0].copy()
                    if method=='state_reset':h=model.initial(data['delta'][:,0],s[:,1],data['normal'][:,0])
                    amp=data['amplitude'];rec={k:[] for k in ['state','normal','friction','guard']}
                    for t in range(data['u'].shape[1]):
                        s,h,nn,ff,bad=motion_step(model,s,h,data['u'][:,t],data['d'][:,t],data['rate'][:,t],amp,dt=cfg['dt'],normal_override=data['normal'][:,0] if t==0 else None)
                        for key,value in [('state',s),('normal',nn),('friction',ff),('guard',bad)]:rec[key].append(value.copy())
                    arrays={k:np.stack(v,1) for k,v in rec.items()}
                    save(dest,arrays,dict(normal_world=nw,friction_world=fw,seed=seed,noise=noise,method=method,rows=meta['rows'],model_lock_sha256=lock,test_sha256=testsha,seconds=time.perf_counter()-tic))
                    print('PREDICTION',nw,fw,seed,noise,method,round(time.perf_counter()-tic,2),flush=True)

def run_controllers(args,cfg,lock):
    from isaaclab.app import AppLauncher
    app=AppLauncher(args).app
    try:
        import torch
        from decision_study.isaac_rig import IsaacRig
        from round1_reference import CompositeReference
        from round1_stimuli import commands,preparation
        torch.set_num_threads(2);count=cfg['profiles_per_replicate']['evaluation'];rig=IsaacRig(count,'cpu',dt=cfg['dt']);start=time.perf_counter()
        for nw,fw in cfg['compositions']:
            for seed in args.seeds:
                path=OUT/f'test/{nw}_{fw}/r{seed}.npz';test,meta,testsha=load_test(path,lock)
                if meta['model_lock_sha256']!=lock:raise RuntimeError('Test/model lock mismatch')
                rows=meta['rows'];d,rate,target=commands(rows,cfg);amp=test['amplitude'];steps=round(cfg['motion_seconds']/cfg['dt']);warmsteps=round(cfg['warmup_seconds']/cfg['dt'])
                for noise in cfg['noise']:
                    for method in args.methods:
                        dest=OUT/f'controllers/{nw}_{fw}/r{seed}/{noise}/{method}.npz'
                        if done(dest,lock,testsha):continue
                        if time.perf_counter()-start>cfg['resources']['native_evaluation_wall_seconds_limit']:raise TimeoutError('Native evaluation budget reached')
                        tic=time.perf_counter();rig.reset();ref=CompositeReference(nw,fw,count,.02);model=load(method,nw,fw,seed,noise)
                        h=model.initial(test['warm_delta'][:,0],np.zeros(count),test['warm_normal'][:,0])
                        for t in range(warmsteps):
                            dd,rr=preparation(t*cfg['dt'],rows)
                            nn,ff=ref.force(dd,rr,np.zeros(count),dt=cfg['dt'])
                            if np.max(np.abs(nn-test['warm_normal'][:,t]))>1e-9:raise AssertionError('Controller preparation differs from frozen replay')
                            _,_,h=model.step(dd,rr,np.zeros(count),h,normal_override=nn,dt=cfg['dt'])
                            if np.max(np.abs(rig.advance(np.zeros(count))))>1e-9:raise AssertionError('Clamp drift')
                        if method=='state_reset':
                            nn=ref.observe(np.full(count,.02),rate[:,0],np.zeros(count),dt=cfg['dt'])[0]
                            h=model.initial(np.full(count,.02),np.zeros(count),nn)
                        rec={k:[] for k in ['state','next_state','normal','friction','u','delta','compression_rate','net_force','stage_cost']};plans=[];guards=[];latency=[];u=np.zeros(count)
                        for t in range(steps):
                            s=rig.state();g,slope=cam(s[:,0],amp);dd=d[:,t]+g;rr=rate[:,t]+slope*s[:,1]
                            if t%cfg['controller']['replan_steps']==0:
                                observed=ref.observe(dd,rr,s[:,1],dt=cfg['dt'])[0];pstart=time.perf_counter();end=t+cfg['controller']['horizon_steps']
                                u,cost,bad=plan(model,s,h,observed,d[:,t:end],rate[:,t:end],target[:,t:end],amp)
                                latency.append(time.perf_counter()-pstart);plans.append(cost);guards.append(bad)
                            nn,ff=ref.force(dd,rr,s[:,1],dt=cfg['dt']);_,_,h=model.step(dd,rr,s[:,1],h,normal_override=nn,dt=cfg['dt'])
                            net=u-slope*nn+ff;ns=rig.advance(net)
                            if not np.isfinite(ns).all() or np.max(np.abs(ns))>20:
                                failure=dest.with_name(dest.stem+'_failed.npz');failure.parent.mkdir(parents=True,exist_ok=True)
                                preserved={k:np.stack(v,1) for k,v in rec.items() if v}
                                with failure.open('xb') as f:np.savez_compressed(f,**preserved,failed_state=ns,failed_step=t)
                                failure.with_suffix('.json').write_text(json.dumps(dict(model_lock_sha256=lock,test_sha256=testsha,worlds=[nw,fw],seed=seed,noise=noise,method=method,failure='nonfinite/out-of-bounds native state',step=t,rows=rows),indent=2))
                                raise FloatingPointError(f'Native reference failure; partial raw trajectory preserved: {failure}')
                            for key,value in [('state',s),('next_state',ns),('normal',nn),('friction',ff),('u',u),('delta',dd),('compression_rate',rr),('net_force',net),('stage_cost',stage_cost(ns,u,target[:,t]))]:rec[key].append(value.copy())
                        arrays={k:np.stack(v,1) for k,v in rec.items()};arrays.update(d=d[:,:steps],rate=rate[:,:steps],target=target[:,:steps],amplitude=amp,candidate_cost=np.stack(plans,1),candidate_guard=np.stack(guards,1),plan_seconds=np.array(latency))
                        save(dest,arrays,dict(normal_world=nw,friction_world=fw,seed=seed,noise=noise,method=method,rows=rows,model_lock_sha256=lock,test_sha256=testsha,seconds=time.perf_counter()-tic,preparation='full known observed prefix; ideal clamp',observer='real kinematics/current N, updated once; normal model has no own force labels'))
                        print('CONTROLLER',nw,fw,seed,noise,method,round(time.perf_counter()-tic,2),flush=True)
        assert verify_model_lock()==lock
    finally:app.close(wait_for_replicator=False,skip_cleanup=True)

def metrics(cfg,lock):
    import pandas as pd
    for stage in ['prediction','controller']:
        records=[]
        for nw,fw in cfg['compositions']:
            for seed in cfg['replicates']:
                testpath=OUT/f'test/{nw}_{fw}/r{seed}.npz'
                if not testpath.exists():continue
                test,_,testsha=load_test(testpath,lock)
                for noise in cfg['noise']:
                    for method in cfg['methods']:
                        path=OUT/f'{stage}s/{nw}_{fw}/r{seed}/{noise}/{method}.npz'
                        if not path.exists():continue
                        if not done(path,lock,testsha):raise RuntimeError('Missing complete artifact')
                        a=dict(np.load(path));m=json.loads(path.with_suffix('.json').read_text())
                        for i,row in enumerate(m['rows']):
                            base=dict(seed=seed,normal_world=nw,friction_world=fw,noise=noise,method=method,episode=i)
                            if stage=='prediction':
                                base.update(position_rmse=float(np.mean((a['state'][i,:,0]-test['next_state'][i,:,0])**2)**.5),normal_rmse=float(np.mean((a['normal'][i]-test['normal'][i])**2)**.5),friction_rmse=float(np.mean((a['friction'][i]-test['friction'][i])**2)**.5),guard_steps=int(a['guard'][i].sum()))
                            else:
                                error=a['next_state'][i,:,0]-a['target'][i]
                                base.update(position_squared_integral=float(cfg['dt']*np.sum(error**2)),integral_cost=float(cfg['dt']*np.sum(a['stage_cost'][i])),force_squared_integral=float(cfg['dt']*np.sum(a['u'][i]**2)),guard_steps=int(a['candidate_guard'][i].any(axis=1).sum()),plan_seconds=float(a['plan_seconds'].sum()/len(m['rows'])))
                                for deadline in cfg['checkpoints_seconds']:
                                    end=round(deadline/cfg['dt']);tail=round(.1/cfg['dt']);goal=a['target'][i,end-1];s=a['next_state'][i,end-tail:end]
                                    base[f'success_{round(deadline*100):03d}']=int(np.mean(np.abs(s[:,0]-goal))<.004 and np.mean(np.abs(s[:,1]))<.04)
                            records.append(base)
        frame=pd.DataFrame(records)
        status=dict(stage=stage,model_lock_sha256=lock,rows=len(records),expected_rows=7168,complete=len(records)==7168)
        (OUT/f'{stage}_metrics_status.json').write_text(json.dumps(status,indent=2))
        from analyze_round1 import validate_metrics
        frame=validate_metrics(frame,stage,cfg)
        frame.to_csv(OUT/f'{stage}_metrics.csv',index=False)
        print('METRICS',stage,len(records),flush=True)

def main():
    preliminary=argparse.ArgumentParser(add_help=False);preliminary.add_argument('stage',choices=['prediction','controller','metrics']);stage,_=preliminary.parse_known_args()
    p=argparse.ArgumentParser(parents=[preliminary]);p.add_argument('--seeds',nargs='+',type=int);p.add_argument('--methods',nargs='+')
    if stage.stage=='controller':
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(p)
    a=p.parse_args();cfg=json.loads((ROOT/'configs/round1.json').read_text());verify_test_lock();lock=verify_model_lock();a.seeds=cfg['replicates'] if a.seeds is None else a.seeds;a.methods=cfg['methods'] if a.methods is None else a.methods
    if not set(a.seeds)<=set(cfg['replicates']) or not set(a.methods)<=set(cfg['methods']):p.error('Only locked conditions permitted')
    if a.stage=='prediction':run_predictions(a,cfg,lock)
    elif a.stage=='controller':
        if a.device!='cpu':p.error('Native CPU only')
        run_controllers(a,cfg,lock)
    else:metrics(cfg,lock)
    assert verify_model_lock()==lock

if __name__=='__main__':main()
