"""Synthetic/development core checks only; no main dataset is generated."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from matched_study import core


@pytest.fixture(scope='module')
def material():
    return core.load(1,'balanced','maxwell')


@pytest.fixture(scope='module')
def prepared(material):
    return core.reachable(material,991010,3)


def test_ramp_is_continuous_common_curve():
    times=np.array([0.,.25,.5,1.,2.,6.])
    d,r=core.ramp(times)
    assert core.INITIAL_COMPRESSION==.020
    np.testing.assert_array_equal(d[[0,2,3,4,5]],[0.,.02,.02,.02,.02])
    np.testing.assert_array_equal(r[[0,2,3,4,5]],0.)
    assert d[1]==.01 and r[1]>0


def test_reachable_repeat_and_common_normal(material,prepared):
    again=core.reachable(material,991010,3)
    other=core.reachable(material,991011,2)
    for key in ('normal0','initial_pi','initial_xv'):
        np.testing.assert_array_equal(prepared[key],again[key])
    np.testing.assert_array_equal(prepared['normal0'],other['normal0'])
    np.testing.assert_array_equal(prepared['initial_xv'],0.)
    np.testing.assert_array_equal(prepared['history']['x'][:,[0,-1]],0.)
    np.testing.assert_array_equal(prepared['history']['continuous_velocity'][:,[0,-1]],0.)
    assert prepared['metadata']['normal_reset_after_preparation'] is False
    # Carried Maxwell state after finite relaxation differs from a fresh reset.
    assert np.max(abs(prepared['normal0']-.02))>0
    assert prepared['diagnostics']['minimum_normal_N']==0
    assert prepared['diagnostics']['initial_normal_N']>0
    bound=material.pi.alpha*core.initial_normal_force(material,prepared['normal0'])
    assert np.max(abs(prepared['initial_pi'])-bound)<=1e-15
    assert np.min(prepared['diagnostics']['endpoint_PI_stored_energy_released_J'])>=-1e-18


@pytest.mark.parametrize('world',['maxwell','boucwen'])
def test_preparation_uses_same_continuous_curves_at_refinements(world):
    preps=[core.reachable(core.load(1,'balanced',world,dt=.0025/f),991013,2) for f in (1,2,4)]
    for i,f in enumerate((1,2,4)):
        np.testing.assert_allclose(preps[i]['history']['x'][:,::f],preps[0]['history']['x'],rtol=0,atol=1e-14)
        np.testing.assert_allclose(preps[i]['history']['compression'][::f],preps[0]['history']['compression'],rtol=0,atol=1e-15)
        np.testing.assert_allclose(preps[i]['history']['compression_rate'][::f],preps[0]['history']['compression_rate'],rtol=0,atol=1e-14)
    # Shared continuous forcing need not produce identical discrete histories.
    d12=np.linalg.norm(preps[0]['normal0']-preps[1]['normal0'])
    d24=np.linalg.norm(preps[1]['normal0']-preps[2]['normal0'])
    assert d24<=d12*.8+1e-12


def test_rollout_exact_self_prediction_and_no_mutation(material,prepared):
    z=prepared['initial_pi'].copy();normal=prepared['normal0'].copy()
    a=core.rollout(material,z,normal)
    b=core.rollout(material,z.copy(),normal.copy())
    for key in ('past','future','boundary_xv','boundary_pi','boundary_normal'):
        np.testing.assert_array_equal(a[key],b[key])
    np.testing.assert_array_equal(z,prepared['initial_pi'])
    np.testing.assert_array_equal(normal,prepared['normal0'])
    assert a['past'].shape==(3,6,3) and a['future'].shape==(3,160,4)
    np.testing.assert_array_equal(a['truth_past'],a['past'])
    assert a['past_times'][-1]==pytest.approx(.15)
    assert a['future_times'][-1]==pytest.approx(.4)
    assert a['diagnostics']['max_solver_residual_N']<=1e-7
    assert np.min(a['diagnostics']['minimum_compression_m'])>.018
    for k in ('prep_signed_normal_work_J','prep_absolute_normal_work_J','future_signed_normal_work_J','future_absolute_normal_work_J'):
        np.testing.assert_array_equal(a['diagnostics'][k],0.)


def test_future_does_not_change_probe_or_boundary(material,prepared):
    a=core.rollout(material,prepared['initial_pi'],prepared['normal0'],future_amplitude=2)
    b=core.rollout(material,prepared['initial_pi'],prepared['normal0'],future_amplitude=10)
    for key in ('past','boundary_xv','boundary_pi','boundary_normal'):
        np.testing.assert_array_equal(a[key],b[key])
    assert np.max(abs(a['future']-b['future']))>0
    np.testing.assert_array_equal(a['diagnostics']['prep_absolute_tangential_work_J'],b['diagnostics']['prep_absolute_tangential_work_J'])


@pytest.mark.parametrize('factor',[1,2,4])
def test_fast_rollout_matches_public_model_steps_and_force_alignment(factor):
    m=core.load(1,'balanced','boucwen',dt=.0025/factor)
    p=core.reachable(m,991019,1)
    a=core.rollout(m,p['initial_pi'],p['normal0'],strength=12,geom=1,future_amplitude=10)
    xv=np.zeros((1,2));state={'normal':p['normal0'][None].copy(),'pi':p['initial_pi'].copy()}
    past=[];future=[];sum_forces=np.zeros(2);signed=0.;absolute=0.
    prep_steps=round(.15/m.dt);steps=prep_steps+round(.4/m.dt)
    for t in range(steps):
        before=xv.copy()
        local=t if t<prep_steps else t-prep_steps
        phase_steps=prep_steps if t<prep_steps else round(.4/m.dt)
        strength=12 if t<prep_steps else 10
        u=strength*np.sin(2*np.pi*local/phase_steps)
        xv,state,N,Ft,residual=m.step(xv,state,u,core.INITIAL_COMPRESSION,0.,.001,np.array([1]))
        assert np.max(abs(residual))<=1e-7
        if t<prep_steps:
            work=u*(xv[0,0]-before[0,0]);signed+=work;absolute+=abs(work)
            if (t+1)%round(.025/m.dt)==0:
                normal=m.normal_force(xv,state,.02,0.,.001,np.array([1]))
                past.append(np.r_[xv[0],normal[0]])
        else:
            sum_forces+=np.r_[N[0],Ft[0]]
            if (local+1)%factor==0:
                future.append(np.r_[xv[0],sum_forces/factor]);sum_forces[:]=0.
    np.testing.assert_allclose(a['past'][0],past,atol=2e-12,rtol=2e-12)
    np.testing.assert_allclose(a['future'][0],future,atol=2e-12,rtol=2e-12)
    np.testing.assert_allclose(a['diagnostics']['prep_signed_tangential_work_J'][0],signed,atol=1e-13)
    np.testing.assert_allclose(a['diagnostics']['prep_absolute_tangential_work_J'][0],absolute,atol=1e-13)
    assert np.sum(a['diagnostics']['plastic_step_counts'])>0
    assert a['diagnostics']['yield_entry_counts'].shape==(1,2,3)


def test_mle_past_only_and_exact_zero_tie(material,prepared,monkeypatch):
    prior=np.zeros((1,3))
    observed=core.rollout(material,prior,prepared['normal0'])['past'][0]
    original=core._simulate
    calls=[]
    def monitored(*args,**kwargs):
        assert kwargs.get('past_only') is True
        result=original(*args,**kwargs)
        assert result[0][1].shape[1]==0
        calls.append(1)
        return result
    monkeypatch.setattr(core,'_simulate',monitored)
    a=core.mle(material,observed,np.array([5e-5,.002,.15]),prior,np.ones(1),prepared['normal0'],future_amplitude=2)
    b=core.MLE(material,observed,np.array([5e-5,.002,.15]),prior,np.ones(1),prepared['normal0'],future_amplitude=10)
    assert len(calls)>0
    for result in (a,b):
        np.testing.assert_array_equal(result['costs'],0.)
        np.testing.assert_array_equal(result['initial_candidates'],0.)
        assert result['chosen']==0
        assert len(result['diagnostics']['starts'])==2
        for diagnostic in result['diagnostics']['starts']:
            assert diagnostic['residual_calls']<=512
            assert diagnostic['optimizer_nfev']<=120
            assert diagnostic['max_solver_residual_N_all_trials']<=1e-7
            assert diagnostic['wall_budget_used'] is False


def test_mle_objective_reconstructs_from_returned_candidates(material,prepared):
    target=core.rollout(material,prepared['initial_pi'][0],prepared['normal0'],duration=.15,strength=3)['past'][0]
    sigma=np.array([5e-5,.002,.15]);observed=target.copy();observed[-1,0]+=2e-5
    result=core.mle(material,observed,sigma,prepared['initial_pi'],np.array([1.,0.,0.]),prepared['normal0'])
    replay=core.rollout(material,result['initial_candidates'],prepared['normal0'])['past']
    objective=np.sum(((replay-observed)/sigma)**2,axis=(1,2))
    np.testing.assert_allclose(objective,result['costs'],atol=1e-11,rtol=1e-10)
    assert result['chosen']==int(np.argmin(result['costs']))


def test_reject_future_observations_and_invalid_state(material,prepared):
    with pytest.raises(ValueError,match='past observation'):
        core.mle(material,np.zeros((7,3)),np.ones(3),prepared['initial_pi'],np.ones(3),prepared['normal0'])
    with pytest.raises(TypeError):
        core.mle(material,np.zeros((6,3)),np.ones(3),prepared['initial_pi'],np.ones(3),prepared['normal0'],future_truth=np.zeros(1))
    with pytest.raises(ValueError,match='capacities'):
        core.rollout(material,np.ones((1,3)),prepared['normal0'])
    with pytest.raises(ValueError,match='common normal'):
        core.rollout(material,prepared['initial_pi'],np.zeros((3,3)))


def test_solver_failure_is_loud(material,prepared):
    bad=copy.copy(material);bad.iterations=1
    with pytest.raises(FloatingPointError,match='solver residual'):
        core.rollout(bad,prepared['initial_pi'],prepared['normal0'])
