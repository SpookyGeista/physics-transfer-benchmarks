"""Independent synthetic checks; no native data, fitting of laws or source edits."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'costed_benchmark'))
import bounded_mle as bm
from test_bounded_mle import synthetic


def test_exact_past_objective_and_boundary_state_are_replayed_from_selected_initial_state():
    model, data, meta, bank, weights = synthetic(probe_steps=20)
    before={k:np.array(v,copy=True) for k,v in data.items()}
    bank_before=bank.copy();weights_before=weights.copy()
    result=bm.fit_initial_state(model,data,meta,bank,weights,config={'max_nfev':5})
    y,xv,state,_=bm._past_rollout(model,data,0,result['initial_pi_state'][0],meta['initial_compression_m'])
    residual=(y-data['observations'][0,1:])/result['effective_sigma']
    np.testing.assert_allclose(np.sum(residual**2),result['diagnostics'][0]['best_objective'],rtol=1e-14)
    np.testing.assert_array_equal(xv,result['boundary_xv'])
    for key in state:np.testing.assert_array_equal(state[key],result['boundary_state'][key])
    for key in before:np.testing.assert_array_equal(before[key],data[key])
    np.testing.assert_array_equal(bank_before,bank)
    np.testing.assert_array_equal(weights_before,weights)


def test_box_faces_valid_and_outside_prior_rejected():
    model,data,meta,bank,weights=synthetic(probe_steps=0)
    bounds=model.pi.alpha*model.prefix_load()
    bank=np.array([[-1.,1.,-1.],[1.,-1.,1.]])*bounds
    result=bm.fit_initial_state(model,data,meta,bank,weights)
    np.testing.assert_array_equal(result['normalized_state'],[[-1.,1.,-1.]])
    with pytest.raises(ValueError,match='outside'):
        bm.fit_initial_state(model,data,meta,bank*1.01,weights)


def test_noisy_normal_is_likelihood_only_and_boundary_uses_invoked_rate():
    model,data,meta,bank,weights=synthetic(probe_steps=20)
    q=np.zeros(3)
    original=bm._past_rollout(model,data,0,q,.02)
    changed={**data,'observations':data['observations']*10}
    repeated=bm._past_rollout(model,changed,0,q,.02)
    for a,b in zip(original[:2],repeated[:2]):np.testing.assert_array_equal(a,b)
    for key in original[2]:np.testing.assert_array_equal(original[2][key],repeated[2][key])
    changed={**data,'d_dot':data['d_dot'].copy()}
    changed['d_dot'][:,data['probe_steps']]=.05
    boundary=bm._past_rollout(model,changed,0,q,.02)
    np.testing.assert_array_equal(original[1],boundary[1])
    for key in original[2]:np.testing.assert_array_equal(original[2][key],boundary[2][key])
    np.testing.assert_array_equal(original[0][:-1],boundary[0][:-1])
    assert boundary[0][-1,2]!=original[0][-1,2]


def test_identical_episodes_same_solution_without_unintended_cross_episode_state():
    model,data,meta,bank,weights=synthetic(probe_steps=20)
    repeated={k:(np.repeat(v,2,axis=0) if isinstance(v,np.ndarray) and k!='observation_steps' else v) for k,v in data.items()}
    result=bm.fit_initial_state(model,repeated,meta,bank,np.repeat(weights,2,axis=0),config={'max_nfev':3})
    for key in ('initial_pi_state','boundary_xv'):
        np.testing.assert_array_equal(result[key][0],result[key][1])
    for value in result['boundary_state'].values():np.testing.assert_array_equal(value[0],value[1])


def test_nonfinite_trial_fails_loudly_instead_of_returning_earlier_success(monkeypatch):
    model,data,meta,bank,weights=synthetic(probe_steps=20)
    original=bm._past_rollout
    calls=0
    def nonfinite_trial(*args,**kwargs):
        nonlocal calls
        calls+=1
        result=original(*args,**kwargs)
        if calls>1:result[0][:]=np.nan
        return result
    monkeypatch.setattr(bm,'_past_rollout',nonfinite_trial)
    with pytest.raises(FloatingPointError):
        bm.fit_initial_state(model,data,meta,bank,weights)


def test_past_solver_error_must_fail_loudly():
    model,data,meta,bank,weights=synthetic(probe_steps=20)
    model.iterations=1
    with pytest.raises((FloatingPointError,ValueError),match='solver|residual|iteration'):
        bm.fit_initial_state(model,data,meta,bank,weights,config={'max_nfev':2})


def test_future_solver_error_must_fail_loudly():
    model,data,meta,bank,weights=synthetic(probe_steps=0)
    fitted=bm.fit_initial_state(model,data,meta,bank,weights)
    model.iterations=1
    with pytest.raises((FloatingPointError,ValueError),match='solver|residual|iteration'):
        bm.forecast(model,data,fitted)
