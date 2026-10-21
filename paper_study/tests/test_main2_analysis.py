import importlib.util
from pathlib import Path
import itertools,json
import numpy as np
import pandas as pd
import pytest
PATH=Path(__file__).resolve().parents[1]/'scripts/analyze_main2.py'
spec=importlib.util.spec_from_file_location('main2_analysis',PATH);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def small_frame():
    rows=[]
    for method in m.METHODS:
        for e in range(2):
            values=[1.,100.] if method=='prior_propagation' else [2.,100.]
            row=dict(bank=1,preparation_seed=53201,normal_world='maxwell',friction_world='balanced',probe='hold_150',episode=e,method=method)
            for c in m.CHANNELS:row.update({f'mse_{c}':values[e],f'rmse_{c}':np.sqrt(values[e]),f'bias_{c}':0.})
            rows.append(row)
    return pd.DataFrame(rows)


def test_raw_scores_and_pooled_rmse_not_mean_rmse():
    target=np.zeros((2,3,4));pred=target.copy();pred[0,:,0]=1;pred[1,:,0]=3
    scores=m.truth_scores(pred,target)
    assert np.array_equal(scores['mse'][:,0],[1,9])
    f=small_frame();f['mse_position_m']=np.tile([1.,9.],5);f['rmse_position_m']=np.tile([1.,3.],5)
    a=m.aggregate(f);r=a[(a.scope=='global')&(a.method=='prior_propagation')&(a.channel=='position_m')].iloc[0]
    assert r.pooled_rmse==np.sqrt(5) and r.mean_episode_rmse==2


def test_ratio_is_after_pooled_mse_not_mean_episode_ratio():
    x=m.contrasts(small_frame());r=x[(x.scope=='global')&(x.contrast=='assimilation/prior')&(x.channel=='position_m')].iloc[0]
    assert r.mse_ratio==102/101
    assert r.absolute_mse_difference==.5
    assert r.mse_ratio!=1.5


def test_zero_denominator_is_undefined_and_unpaired_fails():
    f=small_frame()
    for c in m.CHANNELS:f.loc[f.method=='prior_propagation','mse_'+c]=0.
    r=m.contrasts(f);x=r[(r.scope=='global')&(r.contrast=='assimilation/prior')]
    assert not x.ratio_defined.any() and x.mse_ratio.isna().all()
    with pytest.raises(ValueError,match='Unpaired'):m.contrasts(f.iloc[1:])


def test_full_inventory_rejects_false_completeness_and_nonfinite():
    rows=[]
    for b,s,n,c,p,e,method in itertools.product(range(1,4),m.SEEDS,m.WORLDS,m.CASES,m.PROBES,range(36),m.METHODS):
        r=dict(bank=b,preparation_seed=s,normal_world=n,friction_world=c,probe=p,episode=e,method=method,
            calibration_noise='noisy',sensor='noisy',forecast_start='filtered_particles',future_steps=160,
            pi_model_seed=61000+b,normal_model_seed=b,profile_index=e//6,geometry=(e%6)//2,excitation_sign=2*(e%2)-1,
            episode_id=f'test-p{s}-h{e//6}-g{(e%6)//2}-s{(2*(e%2)-1):+d}')
        for channel in m.CHANNELS:r.update({f'mse_{channel}':1.,f'rmse_{channel}':1.,f'bias_{channel}':0.})
        rows.append(r)
    frame=pd.DataFrame(rows);m.validate_inventory(frame)
    bad=frame.copy();bad.loc[0,'method']='undeclared'
    with pytest.raises(ValueError,match='Cartesian'):m.validate_inventory(bad)
    bad=frame.copy();bad.loc[0,'rmse_position_m']=-1
    with pytest.raises(ValueError,match='Invalid score'):m.validate_inventory(bad)
    bad=frame.copy();bad.loc[0,'normal_model_seed']=99
    with pytest.raises(ValueError,match='bank identity'):m.validate_inventory(bad)


def test_nonfinite_forecast_is_never_dropped():
    a=np.zeros((1,2,4));a[0,0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):m.truth_scores(a,np.zeros_like(a))


def test_native_cost_count_once_uses_actual_displacement(tmp_path):
    directory=tmp_path/'p53201/maxwell_balanced/hold_150';directory.mkdir(parents=True)
    e,t,dt=60,220,.0025
    # Constant velocity, exact force balance; all36 episode costs must appear only once.
    x=np.arange(t)*dt*.01;state=np.broadcast_to(np.column_stack((x,np.full(t,.01))),(36,t,2)).copy()
    nxt=state.copy();nxt[...,0]+=dt*.01
    d=np.where(np.arange(t)*dt>=e*dt,.02+.002*np.sin(2*np.pi*(np.arange(t)*dt-e*dt)/.4),.02)
    dn=np.where(np.arange(1,t+1)*dt>=e*dt,.02+.002*np.sin(2*np.pi*(np.arange(1,t+1)*dt-e*dt)/.4),.02)
    a=dict(state=state,next_state=nxt,normal=np.full((36,t),5.),friction=np.full((36,t),-3.),u=np.full((36,t),3.),
        d=np.broadcast_to(d,(36,t)),d_dot=np.zeros((36,t)),net_force=np.zeros((36,t)),slope=np.zeros((36,t)),guard=np.zeros((36,t),int),
        probe_steps=np.array(e),normal_updates=np.broadcast_to(np.arange(1,t+1),(36,t)),
        observations=np.zeros((36,t//10+1,3)),observation_steps=np.arange(0,t+1,10),solver_residual_N=np.zeros((36,t)))
    dx=nxt[...,0]-state[...,0];a.update(actuator_work_J=a['u']*dx,normal_drive_work_discrete_J=5*(dn[None,:]-a['d']),
        normal_drive_work_zoh_J=np.zeros((36,t)),friction_applied_work_J=a['friction']*dx,
        native_momentum_residual_N=np.zeros((36,t)),native_position_residual_m=dx-dt*.01)
    file=directory/'noisy.npz';np.savez_compressed(file,**a)
    meta=dict(npz_sha256=m.sha(file),status='complete',split='test',preparation_seed=53201,normal_world='maxwell',
        friction_world='balanced',probe=dict(name='hold_150',duration_s=.15),recorded_steps=t)
    file.with_suffix('.json').write_text(json.dumps(meta))
    _,costs,_=m.audit_native(tmp_path,dict(dt=dt),['p53201/maxwell_balanced/hold_150/noisy.npz'])
    assert len(costs)==36
    assert np.allclose(costs.probe_actuator_absolute_J,3*.01*.15)
    assert costs.probe_normal_drive_absolute_J.max()==0.


def test_plotter_exports_all_four_figures_without_optional_table_package(tmp_path):
    rows=[];nested=[];concentration=[]
    for bank,n,c,probe in itertools.product(range(1,4),m.WORLDS,m.CASES,m.PROBES):
        for methodindex,method in enumerate(m.METHODS):
            row=dict(bank=bank,normal_world=n,friction_world=c,probe=probe,preparation_seed=53201,episode=0,method=method)
            for ch in m.CHANNELS:row.update({f'mse_{ch}':1e-8*(1+methodindex),f'rmse_{ch}':np.sqrt(1e-8*(1+methodindex)),f'bias_{ch}':0.})
            rows.append(row)
        nested.append(dict(bank=bank,probe=probe,past_improvement=.1*bank,future_mse_change_position_m=(-1)**bank*1e-8))
        concentration.append(dict(bank=bank,probe=probe,ess=1+30*bank))
    m.plots(tmp_path,m.aggregate(pd.DataFrame(rows)),pd.DataFrame(nested),pd.DataFrame(concentration))
    assert len(list((tmp_path/'figures').glob('*.png')))==4
    assert len(list((tmp_path/'figures').glob('*.pdf')))==4
