"""Synthetic CPU tests for the new continuous-state baseline, no native data."""
from pathlib import Path
import sys
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from inference import load_model
from bounded_mle import fit_initial_state,forecast


def synthetic(probe_steps=40,normal_world='maxwell'):
    model=load_model('balanced',normal_world,'noisy')
    total=probe_steps+8
    u=6.*np.sin(np.arange(total)*.13)[None,:]
    obs_steps=np.array(sorted(set([0,*range(10,probe_steps+1,10),probe_steps])),dtype=int)
    data=dict(u=u,d=np.full((1,total),.02),d_dot=np.zeros((1,total)),
        amplitude=np.array([.003]),geometry=np.array([0]),probe_steps=probe_steps,
        observation_steps=obs_steps)
    bound=model.pi.alpha*model.prefix_load()
    state=model.initial(1,pi_state=(np.array([.1,-.2,.3])*bound)[None,:])
    xv=np.zeros((1,2));samples=[]
    for t in range(probe_steps+1):
        if t in obs_steps:
            N=model.normal_force(xv,state,.02,0.,data['amplitude'],data['geometry'])
            samples.append(np.column_stack((xv,N)))
        if t<probe_steps:xv,state,_,_,_=model.step(xv,state,u[:,t],.02,0.,data['amplitude'],data['geometry'])
    data['observations']=np.stack(samples,axis=1)
    meta=dict(dt=model.dt,mass_kg=model.mass,initial_compression_m=.02,
              sensor_parameters=dict(sigma_x_m=0.,sigma_v_m_s=0.,sigma_normal_N=0.))
    bank=np.array([[0.,0.,0.],[.3,.3,.3]])*bound
    return model,data,meta,bank,np.array([[.9,.1]])


@pytest.mark.parametrize('normal_world',['maxwell','boucwen'])
def test_bounded_causal_fit_improves_past_objective_and_keeps_weights(normal_world):
    model,data,meta,bank,weights=synthetic(normal_world=normal_world)
    k=model.pi.k.copy();normal=model.parameters.copy();prior=bank.copy()
    result=fit_initial_state(model,data,meta,bank,weights,config={'max_seconds_per_episode':5.})
    info=result['diagnostics'][0]
    assert info['best_objective']<info['initial_objective']*.99
    assert np.max(abs(result['normalized_state']))<=1.
    assert info['residual_calls']<=256
    np.testing.assert_array_equal(model.pi.k,k)
    np.testing.assert_array_equal(model.parameters,normal)
    np.testing.assert_array_equal(bank,prior)
    prediction=forecast(model,data,result)['filtered_particles/bounded_initial_state_mle']
    assert prediction.shape==(1,8,4) and np.isfinite(prediction).all()


def test_future_commands_do_not_enter_past_state_objective():
    model,data,meta,bank,weights=synthetic(probe_steps=20)
    original=fit_initial_state(model,data,meta,bank,weights,config={'max_nfev':6})
    # First future d_dot is the already-invoked boundary sample convention;
    # future u and commands strictly after that boundary cannot alter this fit.
    changed={k:np.array(v,copy=True) if isinstance(v,np.ndarray) else v for k,v in data.items()}
    end=data['probe_steps']
    changed['u'][:,end:]=99.
    changed['d'][:,end+1:]=.5
    changed['d_dot'][:,end+1:]=3.
    repeated=fit_initial_state(model,changed,meta,bank,weights,config={'max_nfev':6})
    np.testing.assert_array_equal(original['initial_pi_state'],repeated['initial_pi_state'])
    np.testing.assert_array_equal(original['boundary_xv'],repeated['boundary_xv'])


def test_no_probe_retains_declared_start_and_does_not_use_initial_N():
    model,data,meta,bank,weights=synthetic(probe_steps=0)
    data['observations'][0,0,2]=1000.
    result=fit_initial_state(model,data,meta,bank,weights)
    assert result['diagnostics'][0]['status']=='unidentified_no_probe'
    np.testing.assert_array_equal(result['initial_pi_state'],bank[[0]])
    assert result['diagnostics'][0]['initial_normal_N']==model.prefix_load()


def test_fixed_call_budget_retains_best_finite_state():
    model,data,meta,bank,weights=synthetic(probe_steps=20)
    result=fit_initial_state(model,data,meta,bank,weights,config={'max_residual_calls':1})
    info=result['diagnostics'][0]
    assert info['budget_exhausted'] and info['residual_calls']==1
    assert info['best_objective']==info['initial_objective']
    np.testing.assert_array_equal(result['initial_pi_state'],bank[[0]])


def test_truth_metadata_or_future_observations_rejected():
    model,data,meta,bank,weights=synthetic(probe_steps=20)
    with pytest.raises(ValueError,match='whitelist'):
        fit_initial_state(model,{**data,'true_state':np.zeros(3)},meta,bank,weights)
    with pytest.raises(ValueError,match='whitelist'):
        fit_initial_state(model,data,{**meta,'prepared_reference_audit':{}},bank,weights)
    bad={**data,'observation_steps':np.r_[data['observation_steps'],data['probe_steps']+1]}
    with pytest.raises(ValueError,match='future'):
        fit_initial_state(model,bad,meta,bank,weights)


def test_inaccurate_solver_fails_loud_in_past_and_future():
    model,data,meta,bank,weights=synthetic(probe_steps=20)
    result=fit_initial_state(model,data,meta,bank,weights,config={'max_nfev':3})
    model.iterations=1
    with pytest.raises(FloatingPointError,match='past-trial solver'):
        fit_initial_state(model,data,meta,bank,weights)
    with pytest.raises(FloatingPointError,match='future-rollout solver'):
        forecast(model,data,result)


def test_invalid_solver_tolerance_rejected():
    model,data,meta,bank,weights=synthetic(probe_steps=0)
    with pytest.raises(ValueError,match='solver residual tolerance'):
        fit_initial_state(model,data,meta,bank,weights,config={'max_solver_residual_N':0.})
