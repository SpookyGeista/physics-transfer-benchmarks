from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pilot import contrast,replay,load_model
from bounded_mle import _past_rollout


def test_equal_objective_does_not_mean_equal_curve():
    past=np.array([[[[-1.,0.,0.]],[[1.,0.,0.]]]])
    y=np.zeros((1,1,3));d=contrast(past,y,y,np.ones(3))
    assert d['stable_gap'][0]==0
    assert d['distance'][0]==2
    assert d['linear_term'][0]==-4 and d['distance_squared'][0]==4


def test_exact_noise_decomposition_nontrivial():
    rng=np.random.default_rng(811)
    past=rng.normal(size=(8,2,7,3));clean=rng.normal(size=(8,7,3));obs=clean+rng.normal(size=clean.shape)*.2
    d=contrast(past,obs,clean,np.array([.1,.2,.3]))
    np.testing.assert_allclose(d['stable_gap'],d['clean_gap']+d['noise_term'],atol=1e-10,rtol=1e-12)
    assert np.max(d['gap_identity_error'])<1e-12
    assert np.max(d['noise_identity_error'])<1e-12


def test_identical_candidates_have_zero_diagnostic():
    rng=np.random.default_rng(121);x=rng.normal(size=(5,4,3));obs=rng.normal(size=x.shape)
    d=contrast(np.stack([x,x],axis=1),obs,obs+.01,np.ones(3))
    for key in ('distance','stable_gap','noise_term','clean_gap'):assert np.all(d[key]==0)


def test_swap_changes_gap_sign_but_not_distance():
    rng=np.random.default_rng(321);x=rng.normal(size=(6,2,3,3));obs=rng.normal(size=(6,3,3))
    a=contrast(x,obs,obs,np.ones(3));b=contrast(x[:,::-1],obs,obs,np.ones(3))
    np.testing.assert_allclose(a['distance'],b['distance'])
    np.testing.assert_allclose(a['stable_gap'],-b['stable_gap'],atol=1e-13)


def test_batched_replay_matches_scalar_production_and_clock():
    model=load_model('balanced','boucwen','noisy',pi_seed=61001,normal_seed=1)
    count=3;total=40;end=20;t=np.arange(total)*model.dt
    inputs=dict(u=np.broadcast_to(2*np.sin(20*t),(count,total)).copy(),d=np.full((count,total),.02),
        d_dot=np.zeros((count,total)),amplitude=np.array([.003,-.003,.002]),geometry=np.array([0,1,2]),
        probe_steps=end,observation_steps=np.array([0,10,20]),observations=np.zeros((count,3,3)))
    inputs['d_dot'][:,end:]=.003  # boundary force uses next-rate convention
    scale=model.pi.alpha*model.prefix_load(.02)
    initial=np.array([[[.2,-.4,.1],[-.1,.3,-.2]]]*count)*scale
    past,future,boundary,solver=replay(model,inputs,initial,.02)
    assert past.shape==(3,2,2,3) and future.shape==(3,2,20,4)
    assert solver<1e-7
    for e in range(count):
        for c in range(2):
            p,xv,state,_=_past_rollout(model,inputs,e,initial[e,c],.02)
            np.testing.assert_allclose(past[e,c],p,rtol=1e-11,atol=1e-12)
            np.testing.assert_allclose(boundary['xv'][e,c],xv[0],atol=1e-12)
            for key in ('normal','pi'):np.testing.assert_allclose(boundary[key][e,c],state[key][0],atol=1e-12)


def test_pair_count_validation():
    with pytest.raises(ValueError,match='Exactly two'):
        replay(None,{},np.zeros((3,3,3)),.02)
