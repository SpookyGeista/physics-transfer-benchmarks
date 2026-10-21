"""Independent pretest contracts. Synthetic laws only; no reserved histories drawn."""
import copy
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from paper_study.decision_test import common,generate,planning,lock
from paper_study.decision_study import generate as basegen,inference as baseinf,evaluate as baseeval


class SyntheticReference:
    def __init__(self,n,events,clone_id=-1):
        self.n=n;self.events=events;self.clone_id=clone_id;self.updates=0
    def clone(self):
        identity=sum(event[0]=='clone' for event in self.events)
        self.events.append(('clone',identity))
        return SyntheticReference(self.n,self.events,identity)
    def observe(self,d,r,v,dt=.0025):
        self.events.append(('observe',self.clone_id,self.updates,np.array(r).copy()))
        n=500*np.asarray(d)+7*np.asarray(r)
        return n,-.3*n*np.tanh(np.asarray(v)/.04)
    def force(self,d,r,v,dt=.0025):
        self.events.append(('force',self.clone_id,self.updates,np.array(r).copy()))
        self.updates+=1
        n=500*np.asarray(d)+7*np.asarray(r)
        return n,-.3*n*np.tanh(np.asarray(v)/.04)


class SyntheticRig:
    def __init__(self,n,dt,mass):self.dt=dt;self.mass=mass;self.xv=np.zeros((n,2))
    def reset(self):self.xv[:]=0.
    def state(self):return self.xv.copy()
    def advance(self,force):
        self.xv[:,1]+=self.dt*np.asarray(force)/self.mass
        self.xv[:,0]+=self.dt*self.xv[:,1]
        return self.xv.copy()


def test_exact_native_code_adapter_preserves_physics_and_old_rate_boundary(tmp_path,monkeypatch):
    cfg=common.load_config();events=[];source=basegen.run_triplet
    globals_before=dict(source.__globals__)
    code_before=source.__code__
    # Replace truth exporters only in a local synthetic function namespace.
    # No real Reference or preparation plan is called. The unchanged loop's
    # public sensor-noise RNG acts on this synthetic law only; no reserved
    # physical history or corresponding test outcome is generated.
    synthetic=common.copied_function(source,{'truth_preparation':lambda _: {'synthetic':True},
        'truth_snapshot':lambda _: {'synthetic':True}})
    def reference_stub(world,indices,seed,split,**kwargs):
        events.append(('sentinel_factory',world,seed,split,kwargs))
        assert split=='test' and kwargs['normal_world']=='maxwell'
        return SyntheticReference(len(indices),events)
    monkeypatch.setattr(generate,'original',SimpleNamespace(factors=basegen.factors,run_triplet=synthetic,referencepair_indices=reference_stub))
    def guard(*args):events.append(('guard',));return {'config':cfg}
    monkeypatch.setattr(generate,'verify_test_lock',guard)
    authorization=tmp_path/'synthetic-lock';authorization.write_text('sentinel, not an actual test authorization')
    rig=SyntheticRig(108,cfg['dt'],cfg['mass_kg'])
    arrays,meta=generate.run_triplet(rig,'maxwell','boucwen',55201,cfg,
        config_path='synthetic-config',test_lock=authorization)
    assert events[0][0]=='guard'
    factory_index=next(i for i,e in enumerate(events) if e[0]=='sentinel_factory')
    assert events[factory_index-1][0]=='guard' and events[-1][0]=='guard'
    assert meta['prefix_audit']['bitwise_identical'] and meta['split']=='test'
    for branch,total in [('release',120),('hold100',160),('continue100',160)]:
        data=arrays[branch]
        assert data['u'].shape==(36,total)
        assert np.max(abs(data['momentum_residual_N']))<1e-11
        assert np.max(abs(data['kinematic_residual_m']))<1e-14
        np.testing.assert_array_equal(data['d_dot'][:,:int(data['release_step'])],0.)
    # At every release, the sensor event occurs before the force update and
    # uses probe-side rate; the force sees the subsequently invoked future rate.
    expected_jump=.002*2*np.pi/.25
    for clone_id,release in [(0,20),(1,60),(2,60)]:
        selected=[(i,e) for i,e in enumerate(events) if e[0] in ('observe','force') and e[1]==clone_id and e[2]==release]
        assert [e[0] for _,e in selected]==['observe','force']
        np.testing.assert_allclose(selected[1][1][3]-selected[0][1][3],expected_jump,atol=1e-14,rtol=0)
    assert source.__code__ is code_before
    assert source.__globals__.keys()==globals_before.keys()
    assert all(source.__globals__[k] is v for k,v in globals_before.items())


def independent_candidates():
    rng=np.random.default_rng(8876);e,p,h=2,5,7
    weights=rng.uniform(.1,1.,(e,p));weights/=weights.sum(axis=1,keepdims=True)
    snapshot={'weights':weights,'logweight':np.log(weights)}
    branches={}
    for name,frames in zip(common.BRANCHES,[0,4,4]):
        branches[name]=dict(future=rng.normal(0,.004,(e,p,h,4)),sensors=rng.normal(0,.02,(e,p,frames,3)),
            probe_work=rng.uniform(0,.01,(e,p)) if frames else np.zeros((e,p)),extra_seconds=np.full(e,.1 if frames else 0.))
    return snapshot,branches


@pytest.mark.parametrize('chunk',[1,32,64])
def test_chunk_rng_and_risk_against_direct_probability_calculation(chunk):
    cfg=common.load_config();cfg['fantasies']=65;cfg['posterior_chunk_size']=chunk
    snapshot,branches=independent_candidates();saved=copy.deepcopy((snapshot,branches))
    result=planning.branch_values(snapshot,branches,cfg)
    p0=snapshot['weights'];e,p=p0.shape;k=cfg['fantasies']
    rng=np.random.default_rng(cfg['fantasy_seed'])
    quantiles=rng.random((e,k));indices=(quantiles[:,:,None]>p0.cumsum(axis=1)[:,None]).sum(axis=2).clip(max=p-1)
    noise=rng.normal(size=(e,k,4,3));sigma=np.hypot(cfg['sensor_sigma'],cfg['numerical_floor'])
    np.testing.assert_array_equal(result['fantasy_particle_indices'],indices)
    np.testing.assert_array_equal(result['fantasy_noise'],noise)
    for a,name in enumerate(common.BRANCHES):
        branch=branches[name];x=branch['future'][...,0];frames=branch['sensors'].shape[2]
        if frames:
            observed=branch['sensors'][np.arange(e)[:,None],indices]+noise[:,:,:frames]*sigma
            residual=(observed[:,:,None]-branch['sensors'][:,None])/sigma
            unnormalized=np.log(p0)[:,None]-.5*np.sum(residual**2,axis=(-1,-2))
            shifted=unnormalized-unnormalized.max(axis=2,keepdims=True)
            probability=np.exp(shifted);probability/=probability.sum(axis=2,keepdims=True)
            predicted_mean=np.sum(probability[:,:,:,None]*x[:,None],axis=2)
            variance=np.sum(probability[:,:,:,None]*(x[:,None]-predicted_mean[:,:,None])**2,axis=2)
            expected=variance.mean(axis=(1,2))
        else:
            mean=(p0[:,:,None]*x).sum(axis=1)
            expected=(p0[:,:,None]*(x-mean[:,None])**2).sum(axis=1).mean(axis=1)
        np.testing.assert_allclose(result['risk_m2'][:,a],expected,atol=1e-16,rtol=1e-12)
        np.testing.assert_allclose(result['planned_extra_probe_work_J'][:,a],(p0*branch['probe_work']).sum(axis=1),atol=1e-15)
    baseline=baseinf.branch_values(*saved,cfg)
    for key in result:np.testing.assert_array_equal(result[key],baseline[key])
    for key in snapshot:np.testing.assert_array_equal(snapshot[key],saved[0][key])
    for name in branches:
        for key in branches[name]:np.testing.assert_array_equal(branches[name][key],saved[1][name][key])


def test_union_covers_all_banks_priors_test_adapters_and_review_test():
    cfg=common.load_config(final=True);paths=lock.source_paths(common.HERE/'config.json')
    expected={Path(__file__).resolve(),common.HERE/'common.py',common.HERE/'planning.py',common.HERE/'generate.py',common.HERE/'evaluate.py',common.HERE/'lock.py'}
    assert expected<=set(paths)
    for bank in range(1,9):
        local=common.bank_config(cfg,bank)
        for n,f in cfg['compositions']:
            assert set(baseinf.model_paths(n,f,local))<=set(paths)
            assert baseinf.prior_path(n,f,local) in paths
    assert sum(p.name=='model.json' for p in paths)==16
    assert sum(p.suffix=='.npz' for p in paths)==32


def test_expected_uniform_loss_averages_losses_not_opposing_trajectories():
    cfg=common.load_config();cfg['budgets']=cfg['budgets'][:1]
    scores={b:dict(position_mse_m2=np.array([v]),extra_probe_work_J=np.zeros(1)) for b,v in zip(common.BRANCHES,[1.,1.,9.])}
    metadata=dict(preparation_seed=0,normal_world='synthetic',friction_world='synthetic',rows=[dict(preparation_index=0,geometry=0,excitation_sign=1)])
    policies={'B0':{'policies':{'uniform_expected':np.array([[.5,.5,0.]])}}}
    values=dict(risk_m2=np.zeros((1,3)),information_nats=np.zeros((1,3)),planned_extra_probe_work_J=np.zeros((1,3)),fantasy_risk_mcse_m2=np.zeros((1,3)))
    branches={b:dict(candidate_guard_steps=np.zeros((1,1),dtype=int)) for b in common.BRANCHES}
    rows,_=baseeval.episode_rows(metadata,scores,policies,values,branches,cfg)
    assert rows[0]['position_mse_m2']==1. # +1/-1 forecasts would falsely cancel if averaged before loss.


@pytest.mark.parametrize('override',[None,'/tmp/decision-test-review-explicit-cache'])
def test_package_sets_cache_before_numba_import_and_preserves_explicit_override(override):
    environment=dict(os.environ)
    if override is None:environment.pop('NUMBA_CACHE_DIR',None)
    else:environment['NUMBA_CACHE_DIR']=override
    expected=override or '/tmp/physics-transfer-decision-test-v1-qualified-numba'
    script=('import os; import paper_study.decision_test; import numba; '
            f'assert os.environ["NUMBA_CACHE_DIR"]=={expected!r}; '
            'assert numba.config.CACHE_DIR==os.environ["NUMBA_CACHE_DIR"]')
    subprocess.run([sys.executable,'-c',script],check=True,env=environment,cwd=common.REPO,capture_output=True,text=True)
