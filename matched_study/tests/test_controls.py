"""Synthetic controls only: never draw or inspect a reserved history."""
from pathlib import Path
import copy
import json
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from matched_study import compare_controls as c


def ensemble(p):
    z=np.arange(p*3,dtype=float).reshape(p,3)*1e-5
    past=np.broadcast_to(z[:,None,:],(p,6,3)).copy()
    future=np.zeros((2,p,160,4));future[:,:,:,:3]=z[None,:,None,:]
    return dict(initial_pi=z,normal0=np.array([.02,.02,.02]),particle_past=past,particle_future=future,
                past_times=np.arange(1,7)*.025,future_times=np.arange(1,161)*.0025)


def test_nested_support_checks_entire_shared_trajectories():
    assert c.nested_particles(ensemble(4),ensemble(7))==4
    broken=ensemble(7);broken['particle_future'][1,0,-1,0]+=1e-9
    with pytest.raises(ValueError,match='nested future'):
        c.nested_particles(ensemble(4),broken)
    broken=ensemble(3);broken['normal0'][0]+=1e-12
    with pytest.raises(ValueError,match='normal state'):
        c.nested_particles(ensemble(4),broken)


def rows_for_gains():
    rows=[]
    for label,dt,losses in [('h',.0025,[10,2,1,3,0]),('h2',.00125,[10,1,2,4,0]),('h4',.000625,[10,1,1,4,0])]:
        for profile in (0,1):
            for method,mse in zip(c.ALL_METHODS,losses):
                rows.append(dict(family='refinement',run=label,dt=dt,particles=4,bank=1,history=71001,
                    normal='maxwell',case='balanced',sign=-1,duration=.15,strength=0.,sensor='clean',future=2.,
                    profile=profile,method=method,channel='position_m',mse=float(mse),episode_rmse=np.sqrt(mse),
                    bias=0.,ess=2.,forecast_drift_rms=0.,forecast_drift_max=0.))
    return pd.DataFrame(rows)


def test_all_pair_gains_distinguish_strict_flip_and_zero_touch():
    gains,stable=c.gain_tables(rows_for_gains())
    assert len(gains)==3*2*10 and len(stable)==3*2*10
    q=stable[(stable.method_a=='posterior')&(stable.method_b=='mle_start1')]
    assert q[(q.before=='h')&(q.after=='h2')].strict_sign_flip.all()
    assert q[(q.before=='h2')&(q.after=='h4')].zero_touch_or_departure.all()
    assert not q[(q.before=='h2')&(q.after=='h4')].strict_sign_flip.any()
    assert len(c.gain_summary(stable))>0


def test_pooled_rmse_is_not_mean_episode_rmse():
    frame=rows_for_gains()
    selected=(frame.run=='h')&(frame.method=='posterior')
    frame.loc[selected,'mse']=[1.,9.]
    frame.loc[selected,'episode_rmse']=[1.,3.]
    pooled=c.pooled_tables(frame)
    row=pooled[(pooled.scope=='all')&(pooled.run=='h')&(pooled.method=='posterior')].iloc[0]
    assert row.pooled_rmse==pytest.approx(np.sqrt(5))
    assert row.mean_episode_rmse==2.


def test_pending_gate_reads_no_arrays_or_creates_output(tmp_path):
    data=tmp_path/'pending';data.mkdir()
    (data/'process.json').write_text(json.dumps({'status':'running'}))
    with pytest.raises(ValueError,match='must be complete'):
        c.run(data,[],[],tmp_path/'out')
    assert not (tmp_path/'out').exists()


def config(kind,p):
    return dict(kind=kind,banks=[1],normals=['maxwell'],cases=['balanced'],history_seeds=[71001],
        prior_seeds=[72001],profiles_per_history=1,sign_assignment=[-1],durations=[.15],strengths_N=[0.],
        future_amplitudes_N=[2.,10.],future_duration=.4,geometry=0,cam_amplitude_m=.001,
        sensor_sigma={'clean':[0.,0.,0.],'noisy':[5e-5,.002,.15]},numerical_floor=[1e-6,1e-5,1e-4],
        noise_stream_key=73001,initial_compression_m=.02,mle={'max_nfev':120},maximum_solver_residual_N=1e-7,
        particles=p,dt=.0025,mle_enabled=kind!='resolution')


def synthetic_run(root,kind,p,source):
    """Simple arrays, not output of any material or preparation generator."""
    root.mkdir();cfg=config(kind,p)
    c.write(root/'lock.json',dict(config=cfg,source_sha256={str(source):c.sha(source)}))
    model='bank1_maxwell_balanced';arm='s-1_d150_a0'
    ep=root/'ensembles'/model/(arm+'.npz');ep.parent.mkdir(parents=True)
    ens=ensemble(p)
    prior=root/'preparations'/f'{model}_prior.npz';prior.parent.mkdir()
    prepmeta=dict(seed=72001,history_rows=[dict(seed=72001,index=i,waypoints_m=[0.,.01,0.],knot_times_s=[0.,2.,4.]) for i in range(p)])
    np.savez_compressed(prior,initial_pi=ens['initial_pi'],normal0=ens['normal0'],initial_xv=np.zeros((p,2)),metadata_json=np.array(json.dumps(prepmeta)))
    history=root/'preparations'/f'{model}_h71001.npz'
    hmeta=dict(seed=71001,history_rows=[dict(seed=71001,index=0,waypoints_m=[0.,-.01,0.],knot_times_s=[0.,2.,4.])])
    np.savez_compressed(history,initial_pi=np.zeros((1,3)),normal0=ens['normal0'],initial_xv=np.zeros((1,2)),metadata_json=np.array(json.dumps(hmeta)))
    emeta=dict(particles=p,dt=.0025)
    np.savez_compressed(ep,**ens,metadata_json=np.array(json.dumps(emeta)))
    methods=list(c.ALL_METHODS if kind!='resolution' else c.BASE_METHODS)
    meta=dict(bank=1,normal='maxwell',case='balanced',history=71001,sign=-1,duration=.15,strength=0.,
        sensors=['clean','noisy'],futures=[2.,10.],methods=methods,particles=p,dt=.0025,
        channel_order=list(c.CHANNELS),force_convention='mean over each 2.5ms output block; x/v at block end',
        ensemble_path=str(ep),ensemble_sha256=c.sha(ep))
    pred=np.zeros((2,2,1,len(methods),160,4))
    for i,method in enumerate(methods):
        if method in ('prior','posterior'):
            pred[:,:,:,i]=ens['particle_future'].mean(axis=1)[None,:,None]
        elif method!='privileged_history': pred[:,:,:,i]=.0001
    batch=dict(predictions=pred,truth_future=np.zeros((2,1,160,4)),truth_past=np.zeros((1,6,3)),observed=np.zeros((2,1,6,3)),
        weights=np.full((2,1,p),1/p),noise_standard=np.zeros((1,6,3)),effective_sigma=np.ones((2,3)),
        truth_initial_pi=np.zeros((1,3)),normal0=ens['normal0'],profile_indices=np.array([0]),
        truth_boundary_xv=np.zeros((1,2)),truth_boundary_pi=np.zeros((1,3)),truth_boundary_normal=ens['normal0'][None])
    bp=root/'batches'/model/'h71001'/(arm+'.npz');bp.parent.mkdir(parents=True)
    np.savez_compressed(bp,**batch,metadata_json=np.array(json.dumps(meta)))
    outputs={str(f):c.sha(f) for f in (prior,history,ep,bp)}
    h=c.sha(root/'lock.json')
    c.write(root/'manifest.json',dict(lock_sha256=h,outputs_sha256=outputs))
    c.write(root/'process.json',dict(status='complete',lock_sha256=h,batches=1,files=4))
    return root


def test_complete_synthetic_particle_comparison_and_hashes(tmp_path):
    source=tmp_path/'source.txt';source.write_text('immutable synthetic source')
    main=synthetic_run(tmp_path/'main','main',4,source)
    smaller=synthetic_run(tmp_path/'p2','resolution',2,source)
    larger=synthetic_run(tmp_path/'p6','resolution',6,source)
    out=tmp_path/'comparison'
    c.run(main,[],[smaller,larger],out)
    result=c.read(out/'summary.json')
    assert result['status']=='complete' and result['nested_checks']==2
    assert result['episode_method_channel_rows']==3*2*2*1*3*4
    process=c.read(out/'process.json');assert process['status']=='complete'
    c.verify(c.read(out/'manifest.json')['outputs_sha256'])
    for name in ('particle_controls.png','particle_controls.pdf','timestep_controls.png','timestep_controls.pdf'):
        assert (out/name).stat().st_size>1000
    with pytest.raises(FileExistsError): c.run(main,[],[smaller,larger],out)
    source.write_text('changed')
    with pytest.raises(ValueError,match='Changed source'):
        c.run(main,[],[smaller,larger],tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()


def test_inventory_rejects_extra_or_missing_raw_file(tmp_path):
    source=tmp_path/'source.txt';source.write_text('fixed')
    main=synthetic_run(tmp_path/'main','main',4,source)
    np.savez(main/'extra.npz',x=np.zeros(1))
    with pytest.raises(ValueError,match='Extra or missing'):
        c.inventory(main,{})


def test_control_rejects_changed_physical_program():
    a=dict(config=config('main',4),batches={('x',):'x'})
    b=dict(config=config('resolution',2),batches={('x',):'x'})
    c.validate_pair(a,b,'resolution')
    broken=copy.deepcopy(b);broken['config']['cam_amplitude_m']=.002
    with pytest.raises(ValueError,match='non-control'):
        c.validate_pair(a,broken,'resolution')
