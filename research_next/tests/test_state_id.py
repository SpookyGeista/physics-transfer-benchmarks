"""Known synthetic coefficients verify causality, serialization and identification."""
import ast
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from state_id import StateID, CompositeStateID, fit_state_id, fit_blind_state_id


def known(world):
    if world == 'maxwell':
        return StateID(world, dict(k_eq=640., c=3.7, k_branches=[240., 90., 40.],
                                 tau_branches=[.015, .09, .65]),
                       dict(mu_c=.25, delta_mu=.18, vs=.14, power=1.7, mixing=.55,
                            tau0=.075, tau_load=.0005, load_offset=7., load_exponent=-.08,
                            load_reference=20., viscous=.45))
    return StateID(world, dict(k=710., c=3., F=5., dy=.014, beta=.9, gamma=.4),
                   dict(mu0=.42, a=.021, b=.014, slip=.012, theta0=.09, vfloor=.0035, viscous=.4))


@pytest.mark.parametrize('world', ['maxwell', 'boucwen'])
def test_state_is_causal_owned_and_normal_override_is_local(world, tmp_path):
    model = known(world); d = np.array([.017, .023]); v = np.array([.12, -.13])
    state = model.initial(d, v, np.array([10., 20.])); before = {k: x.copy() for k, x in state.items()}
    n, ft, after = model.step(d, np.array([.02, -.015]), v, state)
    for k in state:
        assert np.array_equal(state[k], before[k])
        assert not np.shares_memory(after[k], state[k])
    n2, ft2, _ = model.step(d, np.array([.02, -.015]), v, state, normal_override=np.array([3., 5.]))
    assert np.array_equal(n2, [3., 5.]) and not np.array_equal(ft, ft2)
    assert np.all(n >= 0) and np.all(ft*v <= 0)
    opened, drag, _ = model.step(-d, np.zeros(2), v, state, normal_override=50.)
    assert np.all(opened == 0) and np.all(drag == 0)
    model.save(tmp_path/'model.json'); loaded = StateID.load(tmp_path/'model.json')
    nl, fl, _ = loaded.step(d, np.array([.02, -.015]), v, loaded.initial(d, v, n))
    assert np.array_equal(n, nl) and np.array_equal(ft, fl)


def test_normal_force_precedes_history_update():
    model = known('maxwell'); state = model.initial([.02], [0.], [0.]); d = .024
    n, _, after = model.step([d], [0.], [0.], state, dt=.01)
    expected = model.normal['k_eq']*d + sum(model.normal['k_branches'])*.004
    assert np.isclose(n[0], expected)
    next_n, _, _ = model.step([d], [0.], [0.], after, dt=.01)
    assert next_n[0] < n[0]


def test_boucwen_history_survives_hold():
    model = known('boucwen'); state = model.initial([.02, .02], [0., 0.], [0., 0.])
    for t in np.arange(0, 1, .005):
        d=.02+np.array([1., -1.])*.007*np.sin(np.pi*t)**2
        r=np.array([1., -1.])*.007*np.pi*np.sin(2*np.pi*t)
        _, _, state=model.step(d, r, np.zeros(2), state, dt=.005)
    held=state['z'].copy()
    for _ in range(200):
        n, _, state=model.step(np.full(2, .02), np.zeros(2), np.zeros(2), state, dt=.005)
    assert np.array_equal(held, state['z']) and abs(n[0]-n[1]) > .5


def synthetic_data(world, seed, noise=0.):
    rng=np.random.default_rng(seed); n=5; T=300; dt=.005; t=np.arange(T)*dt
    phase=rng.uniform(-2, 2, (n, 1)); freq=rng.uniform(.3, 1.3, (n, 1))
    delta=.021+.007*np.sin(2*np.pi*freq*t+phase)
    rate=.007*2*np.pi*freq*np.cos(2*np.pi*freq*t+phase)
    velocity=.22*np.sin(2*np.pi*.8*t+phase)*np.sin(np.pi*t/(T*dt))**2
    load=14+5*np.sin(2*np.pi*.6*t+phase)
    model=known(world); bn=model.initial(delta[:,0], np.zeros(n), np.zeros(n))
    af=model.initial(np.full(n,.02), velocity[:,0], load[:,0]); normal=[]; friction=[]
    for j in range(T):
        nv, _, bn=model.step(delta[:,j], rate[:,j], np.zeros(n), bn, dt=dt)
        _, fv, af=model.step(np.full(n,.02), np.zeros(n), velocity[:,j], af, normal_override=load[:,j], dt=dt)
        normal.append(nv);friction.append(fv)
    normal=np.stack(normal,1);friction=np.stack(friction,1)
    normal+=rng.normal(0,noise,normal.shape);friction+=rng.normal(0,noise,friction.shape)
    return dict(normal=dict(delta=delta, rate=rate, target=normal),
                friction=dict(velocity=velocity, normal=load, target=friction))


@pytest.mark.parametrize('world', ['maxwell', 'boucwen'])
@pytest.mark.parametrize('noise', [0., .02])
def test_fit_clean_and_noisy_synthetic_sequences(world, noise):
    train=synthetic_data(world, 22, noise);val=synthetic_data(world, 33, noise)
    model=fit_state_id(world, train, val, dt=.005, max_seconds=8., restarts=1)
    assert model.metadata['family_known'] and model.metadata['stronger_structural_prior']
    assert not model.metadata['reference_parameters_used']
    assert model.metadata['normal_validation_rmse'] < .12
    assert model.metadata['friction_validation_rmse'] < .08


def test_production_has_no_reference_import():
    path=Path(__file__).resolve().parents[1]/'src/state_id.py'
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            assert 'reference' not in (node.module or '')
        if isinstance(node, ast.Import):
            assert all('reference' not in item.name for item in node.names)


def test_composite_mpc_clones_and_checkpoint_roundtrip(tmp_path):
    model=CompositeStateID(known('maxwell'), known('boucwen'))
    d=np.array([.02,.023]);v=np.array([.1,-.15]);rate=np.array([.01,-.02])
    state=model.initial(d,v,np.ones(2));before={k:a.copy() for k,a in state.items()}
    normal,friction,new=model.step(d,rate,v,state)
    copies={k:np.repeat(a,7,axis=0) for k,a in state.items()}
    nc,fc,_=model.step(np.repeat(d,7),np.repeat(rate,7),np.repeat(v,7),copies)
    assert np.allclose(nc,np.repeat(normal,7)) and np.allclose(fc,np.repeat(friction,7))
    for k in state:
        assert np.array_equal(before[k],state[k]) and not np.shares_memory(new[k],state[k])
    model.save(tmp_path/'model.json');restored=StateID.load(tmp_path/'model.json')
    nr,fr,_=restored.step(d,rate,v,restored.initial(d,v,np.ones(2)))
    assert isinstance(restored,CompositeStateID)
    assert np.array_equal(nr,normal) and np.array_equal(fr,friction)


def test_blind_selection_can_choose_different_local_families():
    train=synthetic_data('maxwell',22);val=synthetic_data('maxwell',33)
    train['friction']=synthetic_data('boucwen',22)['friction']
    val['friction']=synthetic_data('boucwen',33)['friction']
    model=fit_blind_state_id(train,val,dt=.005,max_seconds=12.,restarts=1)
    assert model.normal_family=='maxwell' and model.friction_family=='boucwen'
    assert not model.metadata['true_family_label_used']
    assert set(model.metadata['all_validation_rmse'])=={'maxwell','boucwen'}
