from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run

def test_likelihood_exact_and_finite():
    p=np.array([[[0.,0.,0.]],[[1.,0.,0.]]]);y=np.array([[[0.,0.,0.]]])
    w=run.weights(p,y,np.ones(3))
    np.testing.assert_allclose(w,[[1/(1+np.exp(-.5)),1/(1+np.exp(.5))]])
    np.testing.assert_array_equal(run.weights(p,y,np.full(3,1e-6)),[[1.,0.]])

def test_centered_variance_and_bias_identity():
    q=np.random.default_rng(5).normal(size=(2,7,4,4))
    w=np.array([[.1,.2,.1,.1,.1,.1,.3],[.2,.1,.1,.1,.1,.2,.2]])
    mu,var=run.moments(q,w)
    y=np.random.default_rng(6).normal(size=(2,2,4,4))
    total=np.einsum('ep,feptc->fetc',w,(q[:,None]-y[:,:,None])**2)
    np.testing.assert_allclose(total,var+(mu-y)**2,rtol=1e-14,atol=1e-14)

def test_noise_stream_pairing_and_prefix():
    cfg=run.plan('main')
    a=run.noise_draw(cfg,71001,[0,2],-1,6);b=run.noise_draw(cfg,71001,[0,2],-1,12)
    np.testing.assert_array_equal(a,b[:,:6])
    assert not np.array_equal(a,run.noise_draw(cfg,71002,[0,2],-1,6))
    assert not np.array_equal(a,run.noise_draw(cfg,71001,[0,2],1,6))

def test_declared_design_and_disjoint_seeds():
    c=run.plan('main');d=run.plan('development')
    assert set(c['history_seeds']).isdisjoint(c['prior_seeds'])
    assert set(c['history_seeds']).isdisjoint(d['history_seeds'])
    assert c['initial_compression_m']==.02
    n=np.prod([len(c[k]) for k in ['banks','normals','cases','history_seeds','durations','strengths_N','future_amplitudes_N','sensor_sigma']])*c['profiles_per_history']
    assert n==4608
    assert run.plan('resolution',particles=1024)['mle_enabled'] is False
