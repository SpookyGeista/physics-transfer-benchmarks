"""CPU synthetic/public-bank checks; no actual hidden or reserved profiles."""
from pathlib import Path
import importlib.util,sys
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pi_timestep_runner',HERE/'run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)


def test_common_clock_force_mean_and_mechanical_endpoints():
    values=np.arange(2*8*4,dtype=float).reshape(2,8,4)
    got=run.common_grid(values,2)
    np.testing.assert_array_equal(got[:,:,:2],values[:,1::2,:2])
    np.testing.assert_array_equal(got[:,:,2:],values.reshape(2,4,2,4)[:,:,:,2:].mean(axis=2))
    with pytest.raises(ValueError):run.common_grid(values,3)


@pytest.mark.parametrize('normal,case',[('maxwell','balanced'),('boucwen','wide_scales')])
def test_public_preparation_refines_both_clocks_with_same_normalized_state(normal,case):
    canonical=None;initial=None
    for divisor in (1,2,4):
        model=run.load_model(case,normal,'noisy',dt=.0025/divisor)
        states,_=run.build_prior(model,particles=8,seed=43001)
        q,applied,error=run.canonical_prior(model,states,canonical)
        assert error<1e-10
        if canonical is not None:np.testing.assert_array_equal(q,canonical)
        canonical=q
        actual=model.initial(1)['normal']
        if initial is not None:np.testing.assert_array_equal(actual,initial)
        initial=actual
        np.testing.assert_allclose(applied,states,rtol=0,atol=1e-12)
    with pytest.raises(ValueError,match='precheck'):
        run.canonical_prior(model,states,canonical+.001)


def test_mle_fixed_work_scaled_wall_and_past_choice_before_futures(monkeypatch):
    events=[];budgets=[];counter=[0]
    def fit(*args,**kwargs):
        i=counter[0];counter[0]+=1;events.append('fit');budgets.append(kwargs['config'])
        return dict(index=i,diagnostics=[dict(best_objective=v) for v in ([3.,1.] if i==0 else [1.,2.])],initial_pi_state=np.full((2,3),i))
    def forecast(model,inputs,candidate,**kwargs):
        events.append('future')
        return {'filtered_particles/bounded_initial_state_mle':np.full((2,4,4),candidate['index'])},{}
    monkeypatch.setattr(run,'fit_initial_state',fit);monkeypatch.setattr(run,'forecast',forecast)
    cfg=run.read(HERE/'config.json')
    pred,detail,_,_=run.mle_two(None,None,None,np.zeros((4,3)),np.ones((2,4)),cfg,4)
    assert events==['fit','fit','future','future']
    assert all(b['max_nfev']==120 and b['max_residual_calls']==512 and b['max_seconds_per_episode']==24. for b in budgets)
    np.testing.assert_array_equal(detail['selected_start'],[1,0])
    np.testing.assert_array_equal(pred['bounded_mle_start1'],0)
    np.testing.assert_array_equal(pred['bounded_mle_two_start'][:,0,0],[1,0])


def test_public_subset_does_not_change_sensor_clock_or_allow_actual_history():
    inputs=dict(u=np.zeros((36,4)),d=np.full((36,4),.02),d_dot=np.zeros((36,4)),amplitude=np.full(36,.003),geometry=np.zeros(36,dtype=int),
        observations=np.zeros((36,3,3)),observation_steps=np.array([0,1,2]),probe_steps=2)
    selected=run.subset(inputs,np.array([0,7,14,21,28,35]))
    assert selected['u'].shape==(6,4)
    np.testing.assert_array_equal(selected['observation_steps'],[0,1,2])
    assert selected['probe_steps']==2


def test_later_mle_failure_retains_completed_start_diagnostics(monkeypatch):
    calls=[0]
    diagnostic=[dict(best_objective=1.,budget_exhausted=True,solver_success=False)]
    def fit(*args,**kwargs):
        calls[0]+=1
        if calls[0]==2:raise FloatingPointError('synthetic later start failure')
        return dict(diagnostics=diagnostic,initial_pi_state=np.zeros((1,3)))
    monkeypatch.setattr(run,'fit_initial_state',fit)
    with pytest.raises(FloatingPointError) as exc:
        run.mle_two(None,None,None,np.zeros((1,3)),np.ones((1,1)),run.read(HERE/'config.json'),1)
    assert exc.value.completed_mle_diagnostics==[diagnostic]
