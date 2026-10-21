"""Synthetic contracts only; never reads Main1 or starts Isaac."""
from pathlib import Path
import copy
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from paper_study.decision_study.common import BRANCHES,load_config,steps,branch_commands,feedback
from paper_study.decision_study.generate import run_triplet,factors,prefix_audit
from paper_study.decision_study.inference import (load_model,load_prior,observable_prefix,filter_prefix,
    forecast_branch,branch_values,select_policies,actual_forecast,posterior,variance_risk,sensor_prediction)


class SyntheticRig:
    """Float64 semi-implicit point mass, explicitly NOT a native backend."""
    def __init__(self,n,dt=.0025):self.n=n;self.dt=dt;self.reset()
    def reset(self):self.xv=np.zeros((self.n,2))
    def state(self):return self.xv.copy()
    def advance(self,force):
        self.xv[:,1]+=np.asarray(force)*self.dt/2.
        self.xv[:,0]+=self.dt*self.xv[:,1]
        return self.state()


@pytest.fixture(scope='module')
def cfg():return load_config()


@pytest.fixture(scope='module')
def raw(cfg):
    c=copy.deepcopy(cfg);c['sensor_sigma']=[0.,0.,0.]
    rows=factors(34001,c)[:2]
    arrays,metadata=run_triplet(SyntheticRig(3*len(rows)),'maxwell','boucwen',34001,c,factor_rows=rows)
    return arrays,metadata,c


def test_common_prefix_bitwise_equal_and_correct_event_order(raw):
    arrays,metadata,c=raw
    assert prefix_audit(arrays,c)['bitwise_identical']
    assert metadata['prefix_audit']['bitwise_identical']
    common,later,future=steps(c)
    for branch in BRANCHES:
        a=arrays[branch];release=common if branch=='release' else later
        assert a['u'].shape[1]==release+future
        np.testing.assert_array_equal(a['observation_steps'],np.arange(0,release+1,10))
        assert np.max(abs(a['momentum_residual_N']))<1e-12
        assert np.max(abs(a['kinematic_residual_m']))<1e-15
        assert np.isfinite(a['absolute_work_J']).all()
    # Release observation is under old rate, while first future force includes new rate.
    boundary=arrays['release']['observations'][:,-1,2]
    np.testing.assert_array_equal(boundary,arrays['hold100']['observations'][:,2,2])
    assert np.max(abs(boundary-arrays['release']['normal'][:,common]))>.1
    assert np.all(arrays['release']['d_dot'][:,common]>0)


def test_future_commands_are_frozen_from_common_measurement(raw):
    arrays,_,cfg=raw;common,_,_=steps(cfg)
    a=arrays['release'];held,bias=feedback(a['observations'][:,-1],a['amplitude'],a['geometry'],cfg)
    for branch in BRANCHES:
        commands=branch_commands(branch,held,bias,a['excitation_sign'],cfg)
        for key in ('u','d','d_dot'):
            np.testing.assert_array_equal(commands[key],arrays[branch][key][:,common:])
    assert not np.array_equal(arrays['hold100']['u'][:,common],arrays['continue100']['u'][:,common])


def test_prefix_firewall_excludes_future_truth_and_future_commands(raw):
    arrays,_,cfg=raw;a=arrays['release'];common,_,_=steps(cfg)
    expected=observable_prefix(a,cfg)
    modified={k:v.copy() for k,v in a.items()}
    modified['normal'][:]=np.nan;modified['state'][:]=np.nan;modified['friction'][:]=np.nan
    for key in ('u','d','d_dot'):modified[key][:,common:]=np.nan
    modified['observation_steps']=np.r_[modified['observation_steps'],common+10]
    modified['observations']=np.concatenate((modified['observations'],np.full((len(a['u']),1,3),np.nan)),axis=1)
    actual=observable_prefix(modified,cfg)
    assert set(actual)=={'amplitude','geometry','excitation_sign','u','d','d_dot','observations','observation_steps'}
    for key in expected:np.testing.assert_array_equal(expected[key],actual[key])


def test_likelihood_scores_each_sensor_once():
    lp=np.log([[.5,.5]])
    predicted=np.array([[[[0.,0.,0.]],[[1.,0.,0.]]]])
    observed=np.zeros((1,1,1,3))
    p=np.exp(posterior(lp,predicted,observed,np.ones(3)))[0,0]
    np.testing.assert_allclose(p[1]/p[0],np.exp(-.5),rtol=1e-14)
    duplicated=np.repeat(predicted,2,axis=2)
    twice=np.exp(posterior(lp,duplicated,np.zeros((1,1,2,3)),np.ones(3)))[0,0]
    np.testing.assert_allclose(twice[1]/twice[0],np.exp(-1.),rtol=1e-14)


def test_variance_and_info_distinguish_physical_collapse_from_information(cfg):
    p=np.array([[.5,.5]]);snapshot=dict(weights=p,logweight=np.log(p))
    branches={}
    for b in BRANCHES:
        x=np.array([[-1.,1.]]) if b!='hold100' else np.zeros((1,2))
        future=np.zeros((1,2,4,4));future[:,:,:,0]=x[:,:,None]
        sensors=np.zeros((1,2,0 if b=='release' else 1,3))
        if b=='continue100':sensors[0,:,0,0]=[-1.,1.]
        branches[b]=dict(future=future,sensors=sensors,probe_work=np.zeros((1,2)),
            extra_seconds=np.array([0. if b=='release' else .1]))
    values=branch_values(snapshot,branches,cfg)
    np.testing.assert_allclose(values['risk_m2'][0,0],1.)
    np.testing.assert_allclose(values['risk_m2'][0,1:],0.,atol=1e-12)
    np.testing.assert_allclose(values['information_nats'][0,:2],0.,atol=1e-12)
    np.testing.assert_allclose(values['information_nats'][0,2],np.log(2),rtol=1e-12)
    assert values['fantasy_risk_mcse_m2'][0,0]==0.


def test_budget_and_uniform_rules_are_complete_and_reproducible(cfg):
    values=dict(risk_m2=np.array([[3.,2.,1.]]),information_nats=np.array([[0.,.1,.4]]),
        physical_risk_m2=np.array([[3.,2.,1.]]),planned_extra_probe_work_J=np.array([[0.,.0005,.006]]),
        extra_seconds=np.array([[0.,.1,.1]]))
    result=select_policies(values,cfg)
    for name,r in result.items():
        assert r['eligible'][0,0]
        assert len(r['policies'])==8
        for p in r['policies'].values():
            np.testing.assert_allclose(p.sum(axis=1),1.)
            assert np.all(p[~r['eligible']]==0)
    for p in result['B0']['policies'].values():np.testing.assert_array_equal(p,[[1,0,0]])
    np.testing.assert_array_equal(result['B1']['policies']['bayes_risk'],[[0,1,0]])
    np.testing.assert_array_equal(result['B3']['policies']['bayes_risk'],[[0,0,1]])
    np.testing.assert_array_equal(result['B3']['policies']['uniform_expected'],[[1/3]*3])
    again=select_policies(values,cfg)
    np.testing.assert_array_equal(result['B3']['policies']['uniform'],again['B3']['policies']['uniform'])


def test_model_batch_axes_no_mutation_and_future_sensor_firewall(raw):
    arrays,_,cfg=raw;model=load_model('maxwell','boucwen',cfg);grid,prior=load_prior('maxwell','boucwen',cfg)
    # A smaller synthetic grid retains its own normalized fixed prior for axis testing only.
    indices=np.arange(0,len(grid),20);grid=grid[indices];prior=prior[indices];prior/=prior.sum()
    public=observable_prefix(arrays['release'],cfg);s=filter_prefix(model,grid,prior,public,cfg)
    original={k:v.copy() for k,v in s['state'].items()}
    sensor_prediction(model,s['xv'],s['state'],np.repeat(public['amplitude'],len(grid)),
                      np.repeat(public['geometry'],len(grid)),cfg)
    for key in original:np.testing.assert_array_equal(s['state'][key],original[key])
    single={k:v[:1] if k!='observation_steps' else v for k,v in public.items()}
    ss=filter_prefix(model,grid,prior,single,cfg)
    np.testing.assert_allclose(s['weights'][:1],ss['weights'],atol=1e-14)
    np.testing.assert_allclose(s['xv'].reshape(2,len(grid),2)[:1],ss['xv'].reshape(1,len(grid),2),atol=1e-14)
    b=forecast_branch(model,s,public,'hold100',cfg)
    prediction,weights=actual_forecast(s,b,arrays['hold100'],cfg)
    changed={k:v.copy() for k,v in arrays['hold100'].items()}
    changed['observations']=np.concatenate((changed['observations'],np.full((2,1,3),np.nan)),axis=1)
    changed['observation_steps']=np.r_[changed['observation_steps'],70]
    changed['friction'][:]=np.nan;changed['next_state'][:]=np.nan
    other,other_weights=actual_forecast(s,b,changed,cfg)
    np.testing.assert_array_equal(prediction,other);np.testing.assert_array_equal(weights,other_weights)
    assert prediction.shape==(2,100,4)


def test_reference_seeds_and_declared_factor_product(cfg):
    rows=factors(34001,cfg)
    assert len(rows)==36
    assert len({(r['preparation_index'],r['geometry'],r['excitation_sign']) for r in rows})==36
    with pytest.raises(ValueError):factors(51201,cfg)


def test_legacy_mass_config_is_frozen_and_checked(cfg,monkeypatch):
    from paper_study.decision_study.generate import generation_sources,validate_legacy_contract
    from paper_study.decision_study.common import HERE,REPO
    from decision_study import common as legacy
    assert REPO/'study_v2/configs/experiment.json' in generation_sources(HERE/'config.json')
    validate_legacy_contract(cfg)
    monkeypatch.setattr(legacy,'MASS',3.)
    with pytest.raises(ValueError,match='contract mismatch'):validate_legacy_contract(cfg)


def test_synthetic_full_evaluation_roundtrip_and_factor_rejection(cfg,tmp_path,monkeypatch):
    """One full36-context synthetic triplet, no Isaac or reserved histories."""
    import json
    from paper_study.decision_study import common,evaluate
    source_root=common.HERE
    data=tmp_path/'synthetic_data';dest=data/'p34001/maxwell_boucwen'
    run_triplet(SyntheticRig(108),'maxwell','boucwen',34001,cfg,dest)
    for branch in BRANCHES:
        p=dest/(branch+'.json');meta=json.loads(p.read_text())
        meta['backend']='Synthetic Float64 test fixture, NOT Isaac'
        p.write_text(json.dumps(meta))
    (data/'process.json').write_text(json.dumps({'status':'complete','role':'synthetic test'}))
    (data/'generation_lock.json').write_text(json.dumps(dict(config=cfg,smoke=True,
        sources_sha256={str(source_root/'generate.py'):common.sha(source_root/'generate.py')})))
    valid=json.loads((dest/'release.json').read_text())
    evaluate.validate_native_branch(dest/'release.npz',valid,cfg)
    wrong=copy.deepcopy(valid);wrong['rows'][0]['geometry']=2
    with pytest.raises(ValueError,match='factor order/product'):evaluate.validate_native_branch(dest/'release.npz',wrong,cfg)
    # Only the unit-test process allows its /tmp fixture; CLI has no override.
    monkeypatch.setattr(common,'HERE',tmp_path);monkeypatch.setattr(evaluate,'HERE',tmp_path)
    monkeypatch.setattr(evaluate,'source_paths',lambda:[source_root/'evaluate.py',source_root/'inference.py'])
    out=evaluate.freeze(data,tmp_path/'synthetic_evaluation')
    evaluate.run(out)
    status=json.loads((out/'process.json').read_text())
    assert status['status']=='complete'
    assert status['policy_rows']==36*4*8 and status['branch_rows']==36*3
    import pandas as pd
    policy=pd.read_csv(out/'policy_episodes.csv')
    assert policy.shape[0]==1152
    b0=policy[policy.budget=='B0']
    assert (b0.probability_release==1).all()
    assert b0.position_mse_m2.groupby(b0.episode).nunique().max()==1
    assert np.isfinite(policy.position_mse_m2).all()
    manifest=json.loads((out/'artifact_manifest.json').read_text())
    common.verify_hashes(manifest['outputs_sha256'])
