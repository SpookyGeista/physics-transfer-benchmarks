"""Native three-branch development; importing this module never starts Isaac.

Only main imports IsaacRig after AppLauncher. Reference/preparation objects are
truth-side only and never imported by inference.py. All branches share exactly
one public feedback/noise history before t=50ms, audited bitwise after generation.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/paper-study-numba')
import sys
sys.dont_write_bytecode=True
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
import argparse
import json
import signal
import time
import numpy as np
from paper_study.decision_study.common import (HERE,BRANCHES,cam,steps,load_config,feedback,
    force_before_decision,branch_commands,sha,jsonable,write_json,fresh_output,verify_hashes)
sys.path.insert(0,str(REPO/'paper_study/src'))
from preparation_v2 import referencepair_indices,truth_preparation,truth_snapshot


def factors(seed,cfg):
    if seed not in cfg['preparation_seeds']:raise ValueError('Use the two declared development seeds')
    return [dict(episode=e,preparation_seed=seed,preparation_index=p,geometry=g,
                 amplitude=cfg['cam_amplitude_m'],excitation_sign=s,
                 episode_id=f'decision-p{seed}-h{p}-g{g}-s{s:+d}')
            for e,(p,g,s) in enumerate((p,g,s) for p in range(cfg['profiles_per_seed'])
                for g in cfg['geometry_ids'] for s in cfg['excitation_signs'])]


FIELDS=('state','next_state','normal','friction','u','d','d_dot','delta','compression_rate','net_force',
        'phase','guard','momentum_residual_N','kinematic_residual_m',
        'actuator_work_J','normal_drive_work_J','friction_work_J','absolute_work_J')


def prefix_audit(arrays,cfg):
    common,_,_=steps(cfg)
    anchor=arrays['release'];mismatches=[];digests={}
    for branch in BRANCHES:
        data=arrays[branch];digests[branch]={}
        for key in FIELDS:
            a,b=anchor[key][:,:common],data[key][:,:common]
            if not np.array_equal(a,b):mismatches.append(f'{branch}/{key}')
            import hashlib
            digests[branch][key]=hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest()
        for key in ('observations','observation_steps'):
            aa=anchor['observation_steps']<=common;bb=data['observation_steps']<=common
            a=anchor[key][:,aa] if key=='observations' else anchor[key][aa]
            b=data[key][:,bb] if key=='observations' else data[key][bb]
            if not np.array_equal(a,b):mismatches.append(f'{branch}/{key}')
    return dict(bitwise_identical=not mismatches,mismatches=mismatches,prefix_microsteps=common,
                common_sensor_steps=anchor['observation_steps'].tolist(),array_sha256=digests)


def run_triplet(rig,nw,fw,seed,cfg,destination=None,*,factor_rows=None,deadline=None):
    """One joint native rig (3*E bodies), returned/written separate branch arrays."""
    rows=factors(seed,cfg) if factor_rows is None else factor_rows
    n=len(rows);dt=cfg['dt'];common,later,future=steps(cfg)
    release={name:common if name=='release' else later for name in BRANCHES}
    total={name:end+future for name,end in release.items()}
    amp=np.asarray([r['amplitude'] for r in rows]);geom=np.asarray([r['geometry'] for r in rows],dtype=int)
    signs=np.asarray([r['excitation_sign'] for r in rows]);indices=np.asarray([r['preparation_index'] for r in rows])
    base=referencepair_indices(fw,indices,seed,'dev',normal_world=nw)
    refs={name:base.clone() for name in BRANCHES}
    preparation=truth_preparation(base);initial_truth=truth_snapshot(base)
    sigma=np.asarray(cfg['sensor_sigma'])
    worlds=('maxwell','boucwen')
    keys=[[cfg['sensor_seed'],seed,worlds.index(nw),worlds.index(fw),r['preparation_index'],r['geometry'],
           int(r['excitation_sign']>0)] for r in rows]
    noise=np.stack([np.random.default_rng(np.random.SeedSequence(k)).normal(size=(later+1,3)) for k in keys])
    records={name:{k:[] for k in FIELDS} for name in BRANCHES}
    observations={name:[] for name in BRANCHES};commands={};held={name:np.zeros(n) for name in BRANCHES}
    rig.reset();error=None;started=time.perf_counter()
    try:
        for t in range(max(total.values())):
            if deadline is not None and time.perf_counter()>deadline:raise TimeoutError('Native budget exhausted')
            allstate=rig.state()
            if allstate.shape!=(3*n,2) or not np.isfinite(allstate).all():raise FloatingPointError('Invalid native state')
            current=allstate.reshape(3,n,2);net=np.zeros((3,n));pending={}
            for b,name in enumerate(BRANCHES):
                if t>=total[name]:continue
                xv=current[b];g,slope=cam(xv[:,0],amp,geom)
                if t<=release[name] and t%cfg['observation_steps']==0:
                    # Measurement ALWAYS under old/probe rate, including release boundaries.
                    N0=refs[name].observe(cfg['initial_compression_m']+g,slope*xv[:,1],xv[:,1],dt=dt)[0]
                    observed=np.column_stack((xv,N0))+noise[:,t]*sigma
                    observations[name].append((t,observed.copy()))
                    if t<=common:
                        held[name],bias=feedback(observed,amp,geom,cfg)
                        if t==common:commands[name]=branch_commands(name,held[name],bias,signs,cfg)
                if t<common:
                    u=force_before_decision(t,held[name],signs,cfg)
                    d=np.full(n,cfg['initial_compression_m']);rate=np.zeros(n)
                else:
                    c=commands[name];j=t-common
                    u,d,rate=[c[k][:,j] for k in ('u','d','d_dot')]
                delta=d+g;localrate=rate+slope*xv[:,1]
                N,Ft=refs[name].force(delta,localrate,xv[:,1],dt=dt)
                net[b]=u-slope*N+Ft
                pending[name]=(b,xv.copy(),N,Ft,u,d,rate,delta,localrate)
            finite=np.isfinite(net).all()
            next_all=rig.advance(net.ravel()).reshape(3,n,2) if finite else current.copy()
            for name,(b,xv,N,Ft,u,d,rate,delta,localrate) in pending.items():
                nv=next_all[b];limits=cfg['guard']
                guard=(~np.isfinite(nv).all(axis=1))|(~np.isfinite(net[b]))|(abs(nv[:,0])>limits['max_abs_position_m'])|(abs(nv[:,1])>limits['max_abs_velocity_m_s'])|(N<limits['min_normal_N'])
                data=dict(state=xv,next_state=nv,normal=N,friction=Ft,u=u,d=d,d_dot=rate,delta=delta,
                    compression_rate=localrate,net_force=net[b],phase=np.full(n,0 if t<common else 1 if t<release[name] else 2),guard=guard,
                    momentum_residual_N=cfg['mass_kg']*(nv[:,1]-xv[:,1])/dt-net[b],
                    kinematic_residual_m=nv[:,0]-xv[:,0]-dt*nv[:,1],actuator_work_J=u*xv[:,1]*dt,
                    normal_drive_work_J=N*rate*dt,friction_work_J=Ft*xv[:,1]*dt,
                    absolute_work_J=(abs(u*xv[:,1])+abs(N*rate))*dt)
                for key in FIELDS:records[name][key].append(np.array(data[key],copy=True))
            if not finite or not np.isfinite(next_all).all():raise FloatingPointError('Nonfinite native force/state retained')
    except BaseException as exc:error=exc
    arrays={}
    for name in BRANCHES:
        data={k:np.stack(v,axis=1) if v else np.empty((n,0,2) if k in ('state','next_state') else (n,0))
              for k,v in records[name].items()}
        data.update(observation_steps=np.asarray([t for t,_ in observations[name]],dtype=int),
            observations=np.stack([y for _,y in observations[name]],axis=1) if observations[name] else np.empty((n,0,3)),
            amplitude=amp,geometry=geom,excitation_sign=signs,common_step=np.array(common),release_step=np.array(release[name]),
            preparation_seed=np.full(n,seed),preparation_index=indices,episode=np.arange(n))
        arrays[name]=data
    audit=prefix_audit(arrays,cfg)
    if not audit['bitwise_identical'] and error is None:error=RuntimeError('Common native prefix not bitwise identical')
    complete=error is None and all(arrays[name]['u'].shape==(n,total[name]) for name in BRANCHES)
    if not complete and error is None:error=RuntimeError('Incomplete native branch horizon')
    metadata=dict(status='complete' if complete else 'failed',error=None if error is None else repr(error),
        normal_world=nw,friction_world=fw,preparation_seed=seed,rows=rows,dt=dt,mass_kg=cfg['mass_kg'],
        sensor_sigma=sigma,noise_stream_keys=keys,common_seconds=cfg['common_seconds'],future_seconds=cfg['future_seconds'],
        seconds=time.perf_counter()-started,prefix_audit=audit,
        observation_scope='All stored observations are probe-side before any future rate invocation; no future sensors are stored.',
        branch_scope='Independent bodies; after a branch endpoint its unrecorded free body cannot affect other branches (collisions disabled).',
        fixture_scope='Same public reachable scalar recipe distribution under imposed v/N; not force-limited native preparation.',
        backend='Isaac/CPU PhysX; synthetic rig only in explicitly labelled tests/smoke')
    if destination is not None:
        destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
        write_json(destination/'preparation_audit.json',dict(preparation=preparation,initial_truth=initial_truth))
        for name,data in arrays.items():
            path=destination/(name+'.npz')
            with path.open('xb') as f:np.savez_compressed(f,**data)
            write_json(path.with_suffix('.json'),dict(**metadata,branch=name,npz_sha256=sha(path),
                expected_steps=total[name],recorded_steps=data['u'].shape[1],guards=int(data['guard'].sum()),
                reference_updates=refs[name].updates,
                preparation_audit_sha256=sha(destination/'preparation_audit.json')))
        write_json(destination/'triplet_audit.json',metadata)
    if error is not None:raise error
    return arrays,metadata


def generation_sources(config_path):
    cfg=load_config(config_path)
    paths=sorted(HERE.glob('*.py'))+sorted((HERE/'tests').glob('*.py'))+[
        Path(config_path).resolve(),HERE/'protocol.md',
        REPO/'paper_study/costed_benchmark/geometry.py',REPO/'paper_study/src/preparation_v2.py',
        REPO/'paper_study/src/preparation.py',REPO/'study_v2/src/decision_study/reference.py',
        REPO/'study_v2/src/decision_study/isaac_rig.py',REPO/'study_v2/src/decision_study/common.py',
        REPO/'study_v2/configs/experiment.json',
        REPO/'research_next/src/state_id.py',REPO/cfg['prior_bank_directory']/'prior_lock.json']
    for nw,fw in cfg['compositions']:
        for w in (nw,fw):
            paths.append(REPO/f'research_next/results/round1/models/state_id/{w}/r{cfg["model_seed"]}/{cfg["calibration_noise"]}/blind_selection/model.json')
        paths.append(REPO/cfg['prior_bank_directory']/f'{nw}_{fw}/r{cfg["model_seed"]}/{cfg["calibration_noise"]}.npz')
    return list(dict.fromkeys(paths))


def validate_legacy_contract(cfg):
    # IsaacRig imports MASS from this legacy module, not our config.
    from decision_study import common as legacy
    if legacy.MASS!=cfg['mass_kg'] or legacy.DT!=cfg['dt'] or legacy.VREG!=.04:
        raise ValueError('Legacy IsaacRig mass/time/velocity-regularization contract mismatch')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'config.json');parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true',help='One seed, maxwell/boucwen; all36 factors and all3 branches')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    if args.device!='cpu':parser.error('Only CPU PhysX is permitted')
    cfg=load_config(args.config);out=fresh_output(args.out)
    sourcehash={str(p):sha(p) for p in generation_sources(args.config)}
    write_json(out/'generation_lock.json',dict(config=cfg,sources_sha256=sourcehash,smoke=args.smoke,
        role='Exploratory branch generation locked before native execution'))
    process=dict(status='running',pid=os.getpid(),command=sys.argv,started_epoch=time.time())
    (out/'process.json').write_text(json.dumps(process,indent=2)+'\n')
    app=None;started=time.perf_counter();batches=0
    def stopped(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    try:
        app=AppLauncher(args).app
        signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
        validate_legacy_contract(cfg)
        from decision_study.isaac_rig import IsaacRig
        rig=IsaacRig(3*36,device='cpu',dt=cfg['dt'])
        seeds=cfg['preparation_seeds'][:1] if args.smoke else cfg['preparation_seeds']
        compositions=[['maxwell','boucwen']] if args.smoke else cfg['compositions']
        for seed in seeds:
            for nw,fw in compositions:
                run_triplet(rig,nw,fw,seed,cfg,out/f'p{seed}/{nw}_{fw}',deadline=started+cfg['native_wall_seconds'])
                batches+=1;print(f'DECISION NATIVE p{seed} {nw}/{fw}: three branches, 36 contexts, prefix bitwise equal',flush=True)
        verify_hashes(sourcehash);process.update(status='complete',batches=batches)
    except BaseException as exc:
        process.update(status='failed',error=repr(exc));raise
    finally:
        process.update(seconds=time.perf_counter()-started,batches=batches)
        (out/'process.json').write_text(json.dumps(process,indent=2)+'\n')
        if app is not None:app.close()


if __name__=='__main__':main()
