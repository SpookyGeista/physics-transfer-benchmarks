"""Independent causal/formula checks; synthetic CPU bodies, no native results."""
from pathlib import Path
import sys
import copy
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paper_study.decision_study.common import load_config,BRANCHES,steps,branch_commands
from paper_study.decision_study.generate import run_triplet,generation_sources
from paper_study.decision_study.inference import (
    observable_prefix,load_model,load_prior,filter_prefix,forecast_branch,
    posterior,variance_risk,branch_values,select_policies,actual_forecast)
from paper_study.decision_study.evaluate import branch_scores,episode_rows


class SyntheticRig:
    def __init__(self,n,dt,mass):self.n=n;self.dt=dt;self.mass=mass;self.reset()
    def reset(self):self.xv=np.zeros((self.n,2))
    def state(self):return self.xv.copy()
    def advance(self,force):
        assert np.asarray(force).shape==(self.n,)
        self.xv[:,1]+=self.dt*np.asarray(force)/self.mass
        self.xv[:,0]+=self.dt*self.xv[:,1]
        return self.state()


@pytest.fixture(scope='module')
def triplet():
    cfg=load_config()
    return cfg,*run_triplet(SyntheticRig(108,cfg['dt'],cfg['mass_kg']),
        'maxwell','boucwen',34001,cfg)


def test_all_108_synthetic_bodies_share_exact_prefix_and_old_rate_boundary(triplet):
    cfg,arrays,meta=triplet;common,later,horizon=steps(cfg)
    assert meta['prefix_audit']['bitwise_identical']
    anchor=arrays['release']
    for name in BRANCHES:
        a=arrays[name]
        assert len(a['state'])==36
        np.testing.assert_array_equal(a['state'][:,:common],anchor['state'][:,:common])
        np.testing.assert_array_equal(a['next_state'][:,:common],anchor['next_state'][:,:common])
        np.testing.assert_array_equal(a['normal'][:,:common],anchor['normal'][:,:common])
        np.testing.assert_array_equal(a['u'][:,:common],anchor['u'][:,:common])
        hit=np.flatnonzero(a['observation_steps']==common)
        np.testing.assert_array_equal(a['observations'][:,hit],anchor['observations'][:,[2]])
        assert np.all(a['d_dot'][:,:common]==0.)
        end=int(a['release_step'])
        assert a['observation_steps'][-1]==end
        assert np.all(a['d_dot'][:,end]>0.)
    assert np.all(arrays['hold100']['d_dot'][:,common:later]==0.)
    assert np.all(arrays['continue100']['d_dot'][:,common:later]==0.)


def test_private_and_postdecision_poison_cannot_change_public_prefix(triplet):
    cfg,arrays,_=triplet;common,_,_=steps(cfg)
    original=observable_prefix(arrays['release'],cfg)
    poisoned={k:np.array(v,copy=True) for k,v in arrays['release'].items()}
    for k in ('state','next_state','normal','friction','preparation_seed','preparation_index'):
        poisoned[k]=np.full_like(poisoned[k],999)
    for k in ('u','d','d_dot'):poisoned[k][:,common:]=np.nan
    # Add invalid future measurements; the public cutoff must still ignore them.
    poisoned['observation_steps']=np.r_[poisoned['observation_steps'],common+1]
    poisoned['observations']=np.concatenate((poisoned['observations'],np.full((36,1,3),np.nan)),axis=1)
    again=observable_prefix(poisoned,cfg)
    for k in original:np.testing.assert_array_equal(original[k],again[k])


def test_later_commands_precommitted_and_sensor_forecast_does_not_advance_snapshot(triplet):
    cfg,arrays,_=triplet
    public=observable_prefix(arrays['release'],cfg)
    # One context suffices to check mechanics and commands without expanding
    # the full 36x201 particle development evaluation.
    public={k:(v[:1].copy() if k!='observation_steps' else v.copy()) for k,v in public.items()}
    model=load_model('maxwell','boucwen',cfg);grid,prior=load_prior('maxwell','boucwen',cfg)
    snapshot=filter_prefix(model,grid,prior,public,cfg)
    saved=copy.deepcopy(snapshot)
    b=forecast_branch(model,snapshot,public,'continue100',cfg)
    for k in ('xv','weights','logweight'):np.testing.assert_array_equal(snapshot[k],saved[k])
    for k in snapshot['state']:np.testing.assert_array_equal(snapshot['state'][k],saved['state'][k])
    assert b['sensor_steps'].tolist()==[30,40,50,60]
    sensors={k:arrays['continue100'][k][:1].copy() if k=='observations' else arrays['continue100'][k].copy()
             for k in ('observations','observation_steps')}
    first=actual_forecast(snapshot,b,sensors,cfg)
    sensors['observations']=np.concatenate((sensors['observations'],np.full((1,1,3),np.nan)),axis=1)
    sensors['observation_steps']=np.r_[sensors['observation_steps'],61]
    repeated=actual_forecast(snapshot,b,sensors,cfg)
    for a,z in zip(first,repeated):np.testing.assert_array_equal(a,z)


def test_sequence_likelihood_matches_direct_bayes_formula():
    prior=np.array([[.2,.3,.5]])
    prediction=np.array([[[[0.,1.,2.],[1.,2.,3.]],[[1.,2.,3.],[2.,3.,4.]],[[2.,3.,4.],[3.,4.,5.]]]])
    observation=np.array([[[[.4,1.5,2.5],[1.2,2.1,3.3]],[[1.8,2.8,3.9],[2.9,3.7,5.2]]]])
    sigma=np.array([.5,.8,1.2])
    actual=np.exp(posterior(np.log(prior),prediction,observation,sigma))
    expected=np.empty((1,2,3))
    for k in range(2):
        for p in range(3):
            expected[0,k,p]=prior[0,p]*np.prod(np.exp(-.5*((observation[0,k]-prediction[0,p])/sigma)**2))
        expected[0,k]/=expected[0,k].sum()
    np.testing.assert_allclose(actual,expected,rtol=1e-14)


def test_variance_risk_is_expected_task_squared_error():
    weights=np.array([[[.2,.3,.5],[.7,.2,.1]]]);future=np.array([[[1.,2.],[2.,4.],[-1.,3.]]])
    risk,mean=variance_risk(weights,future)
    expected=np.sum(weights[:,:,:,None]*(future[:,None]-mean[:,:,None])**2,axis=2).mean(axis=2)
    np.testing.assert_allclose(risk,expected,rtol=1e-14)


def test_mixture_eig_and_risk_distinguish_information_from_physical_reset():
    cfg=load_config();p=np.array([[.25,.75]])
    snapshot={'weights':p,'logweight':np.log(p)}
    def branch(x,s):
        future=np.zeros((1,2,2,4));future[...,0]=np.asarray(x)[None,:,None]
        sensors=np.zeros((1,2,1,3));sensors[0,:,0,0]=s
        return dict(future=future,sensors=sensors,probe_work=np.zeros((1,2)),extra_seconds=np.array([.1]))
    branches={'release':branch([0,2],[0,0]),'hold100':branch([0,2],[-1,1]),'continue100':branch([1,1],[0,0])}
    branches['release']['sensors']=np.empty((1,2,0,3));branches['release']['extra_seconds'][:]=0
    values=branch_values(snapshot,branches,cfg)
    np.testing.assert_allclose(values['risk_m2'][0],[.75,0.,0.],atol=1e-12)
    np.testing.assert_allclose(values['physical_risk_m2'][0],[.75,.75,0.],atol=1e-12)
    expected=-np.log(p[0,values['fantasy_particle_indices'][0]]).mean()
    np.testing.assert_allclose(values['information_nats'][0],[0.,expected,0.],atol=1e-12)
    choices=select_policies(values,cfg)['B3']['policies']
    assert np.argmax(choices['information_gain'])==1
    assert np.argmax(choices['physical_only'])==2


def test_scaled_work_budgets_and_fallback_are_shared_by_policies():
    cfg=load_config()
    values=dict(risk_m2=np.array([[3.,2.,1.]]),physical_risk_m2=np.array([[3.,2.,1.]]),
        information_nats=np.array([[0.,1.,2.]]),extra_seconds=np.array([[0.,.1,.1]]),
        planned_extra_probe_work_J=np.array([[0.,.002,.01]]))
    results=select_policies(values,cfg)
    assert results['B0']['eligible'].tolist()==[[True,False,False]]
    assert results['B1']['eligible'].tolist()==[[True,False,False]]
    assert results['B2']['eligible'].tolist()==[[True,True,False]]
    assert results['B3']['eligible'].tolist()==[[True,True,True]]
    for budget,item in results.items():
        for weights in item['policies'].values():
            np.testing.assert_allclose(weights.sum(axis=1),1.)
            assert np.all(weights[~item['eligible']]==0.)
    assert np.argmax(results['B2']['policies']['fixed_continue100'])==0
    assert np.argmax(results['B2']['policies']['bayes_risk'])==1


def test_randomized_policy_loss_is_mixture_of_losses_not_loss_of_mixture(triplet):
    cfg,arrays,meta=triplet
    # Deliberately unequal per-action losses, with otherwise minimal scored data.
    n=36;scores={name:{'position_mse_m2':np.full(n,(i+1.)**2),'extra_probe_work_J':np.zeros(n)} for i,name in enumerate(BRANCHES)}
    values={'risk_m2':np.tile([1.,4.,9.],(n,1)),'information_nats':np.zeros((n,3)),
        'planned_extra_probe_work_J':np.zeros((n,3)),'fantasy_risk_mcse_m2':np.zeros((n,3)),
        'extra_seconds':np.zeros((n,3)),'physical_risk_m2':np.tile([1.,4.,9.],(n,1))}
    branches={name:{'candidate_guard_steps':np.zeros((n,2),dtype=int)} for name in BRANCHES}
    rows,_=episode_rows(meta,scores,select_policies(values,cfg),values,branches,cfg)
    expected=[r for r in rows if r['method']=='uniform_expected']
    assert expected and all(np.isclose(r['position_mse_m2'],14/3) for r in expected)


def test_generation_hashes_actual_native_mass_and_reference_configuration():
    root=Path(__file__).resolve().parents[2]
    captured={p.resolve() for p in generation_sources(root/'paper_study/decision_study/config.json')}
    assert (root/'study_v2/configs/experiment.json').resolve() in captured
