"""Implementation checks before freezing; no main calibration/test scores."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import PIModel
import fit as fitter
from profiles import profiles


def test_learning_files_have_no_truth_imports():
    for name in ('model.py', 'fit.py'):
        tree = ast.parse((HERE/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or '']
            else:
                continue
            assert not any(any(word in module for word in ('reference', 'generate', 'multistate_friction'))
                           for module in modules)


def test_independent_online_equations_against_truth_and_pure_state(tmp_path):
    # Truth is available only in this numerical implementation test, not Fitter.
    path = HERE.parent/'alternatives/multistate_friction.py'
    spec = importlib.util.spec_from_file_location('test_only_stop_truth', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    truth = module.MultiStateFriction('balanced', n=3)
    p = module.PARAMETER_SETS['balanced']
    model = PIModel(p.stiffness, p.friction_capacity)
    state = model.initial(3)
    rng = np.random.default_rng(18531)
    for _ in range(60):
        v, N = rng.uniform(-.1,.1,3), rng.uniform(0.,25.,3)
        old = state.copy()
        actual, new = model.step(state, v, N)
        expected = truth.force(v, N)
        np.testing.assert_array_equal(state, old)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(new, truth.audit()['deflection'], atol=1e-15, rtol=0)
        state = new
    target = tmp_path/'model.json'; model.save(target)
    restored = PIModel.load(target)
    np.testing.assert_array_equal(restored.k, model.k)
    np.testing.assert_array_equal(restored.mu, model.mu)


def test_batch_basis_matches_causal_steps_and_gauge():
    model = PIModel([300.,500.,400.], [.03,.30,.10])
    assert np.all(np.diff(model.alpha)>=0)
    rng = np.random.default_rng(4831)
    v, N = rng.uniform(-.2,.2,(2,100)), rng.uniform(0,20,(2,100))
    state = model.initial(2)
    online=[]
    for t in range(100):
        f, state = model.step(state, v[:,t], N[:,t]); online.append(f)
    np.testing.assert_allclose(model.predict(v,N), np.array(online).T, atol=1e-12, rtol=0)


def test_profiles_reproducible_independent_and_known_start():
    cfg=json.loads((HERE/'config.json').read_text())
    a, rows=profiles(cfg,123,'train')
    b, _=profiles(cfg,123,'train')
    c, _=profiles(cfg,123,'val')
    np.testing.assert_array_equal(a['v'],b['v'])
    assert a['v'].shape==(12,1200) and c['v'].shape==(6,1200)
    assert not np.array_equal(a['v'][:6],c['v'])
    np.testing.assert_array_equal(a['x'][:,0],0.)
    np.testing.assert_array_equal(a['x'][:,-1],0.)
    np.testing.assert_array_equal(a['v'][:,-200:],0.)
    assert set(np.unique(a['N']))=={10.,15.,20.}


def test_fitter_measurement_schema_rejects_latent_labels(tmp_path,monkeypatch):
    monkeypatch.setattr(fitter,'ROOT',tmp_path)
    path=tmp_path/'forbidden.npz'
    np.savez(path,v=np.zeros((1,1)),N=np.ones((1,1)),Ft=np.zeros((1,1)),dt=.0025,
             latent_state=np.zeros((1,3)))
    manifest={'files':{'forbidden.npz':fitter.digest(path)}}
    with pytest.raises(ValueError,match='forbidden'):
        fitter.load_measured(path,manifest)


def test_optimizer_wiring_on_separate_synthetic_sensor_data():
    cfg=json.loads((HERE/'config.json').read_text())['fit']
    cfg=dict(cfg,multistarts=1,max_nfev_per_start=5,max_wall_seconds=20.)
    # Not one of the four truth cards; only verifies conditional LS/API wiring.
    k=np.array([300.,350.,500.])
    alpha=np.asarray(cfg['nominal_start_yield_at_15N_m'])/15.
    source=PIModel(k,k*alpha)
    t=np.arange(400)*.0025
    v=np.tile(.2*np.sin(4*np.pi*t),(2,1)); N=np.array([10.,20.])[:,None]*np.ones_like(v)
    measured=source.predict(v,N)
    fitted,diagnostic=fitter.fit_measured(v,N,measured,.0025,cfg,123)
    assert diagnostic['training_rmse_N']<1e-10
    assert np.all(fitted.k>0) and np.all(fitted.mu>0)
    assert diagnostic['training_logparameter_jacobian_rank']==6
