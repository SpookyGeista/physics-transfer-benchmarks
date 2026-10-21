"""Scientific interface checks independent of actual hidden material equations."""
import ast
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from decision_study.common import features, history_init
from decision_study.models import Classical, Neural, fit_classical, flex_basis, freeze, load_model, parameter_count, replace_normal


def fixture():
    rng = np.random.default_rng(923)
    d = rng.uniform(.008, .03, 128); v = rng.uniform(-.3, .3, 128); n = rng.uniform(4, 22, 128)
    h = history_init(d*.8, v*.7, n*.9)
    return features(d, rng.uniform(-.04, .04, 128), v, h, n)


def test_no_normal_measurement_or_history_leakage():
    f = torch.as_tensor(fixture())
    alt = f.clone(); alt[:, 9:13] += torch.randn_like(alt[:, 9:13])*100
    for kind in ('modular', 'joint', 'moe', 'no_history'):
        model = freeze(Neural(kind))
        assert not model.training and all(not p.requires_grad for p in model.parameters())
        assert torch.equal(model.normal(f), model.normal(alt))


def test_load_interface_replacement_preserves_past_force():
    f = fixture(); before = f[:, 9, None]-f[:, 10:13]
    g = replace_normal(f, np.full(len(f), 40.))
    assert np.allclose(g[:, 9], 2.)
    assert np.allclose(g[:, 9, None]-g[:, 10:13], before, atol=1e-6)
    for model in (Classical(), Neural('modular')):
        n, ft = model.predict(f)
        ln, lft = model.predict_local(replace_normal(f, n))
        assert np.allclose(n, ln) and np.allclose(ft, lft)


def test_dissipative_force_and_open_contact():
    f = fixture()
    for model in (Classical(), Neural('modular'), Neural('joint'), Neural('moe')):
        n, ft = model.predict(f)
        assert np.all(n >= 0) and np.all(ft*f[:, 2] <= 1e-7)
        opened = f.copy(); opened[:, 0] = -.1
        n, ft = model.predict(opened)
        assert np.all(n == 0) and np.all(ft == 0)


def test_analytic_systematic_basis_and_fit():
    f = fixture(); mask = np.ones(len(f), dtype=bool)
    assert flex_basis(f, 'normal').shape[1] == 56
    assert flex_basis(f, 'friction').shape[1] == 53
    nominal = Classical(); n, ft = nominal.predict_local(f)
    for kind in ('identified', 'analytic_v1', 'analytic_flex'):
        model = fit_classical(kind, f, n, ft, mask, mask, (f, n, ft, mask, mask))
        pn, pft = model.predict_local(f)
        assert np.sqrt(np.mean((pn-n)**2)) < .01
        assert np.sqrt(np.mean((pft-ft)**2)) < .01
    assert parameter_count(model) == 109


def test_no_hidden_reference_import():
    root = Path(__file__).resolve().parents[1]
    for path in (root/'src/decision_study/models.py', root/'scripts/train.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert 'reference' not in (node.module or '')
            if isinstance(node, ast.Import):
                assert all('reference' not in item.name for item in node.names)


def test_neural_checkpoint_roundtrip(tmp_path):
    model = freeze(Neural('modular'))
    torch.save(dict(kind=model.kind, state=model.state_dict()), tmp_path/'model.pt')
    restored = load_model(tmp_path)
    assert np.array_equal(model.predict(fixture())[0], restored.predict(fixture())[0])


def test_noisy_measurement_history_is_causal():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('v2_train', root/'scripts/train.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    shape = (2, 12)
    arrays = dict(delta=np.full(shape, .02), rate=np.zeros(shape), normal=np.full(shape, 10.),
                  friction=np.ones(shape), state=np.zeros((*shape, 2)))
    f, _, _ = module.measured_arrays(arrays, 'noisy', 0, 'train')
    changed = {k: v.copy() for k, v in arrays.items()}; changed['normal'][:, 8:] += 50
    alternate, _, _ = module.measured_arrays(changed, 'noisy', 0, 'train')
    assert np.array_equal(f[:, :8], alternate[:, :8])
    assert np.allclose(f[:, 0, 3:9], 0) and np.allclose(f[:, 0, 10:13], 0)


def test_training_checkpoint_resume_and_mismatch(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('v2_train_resume', root/'scripts/train.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    torch.set_num_threads(2)
    f = fixture(); n, ft = Classical().predict_local(f); mask = np.ones(len(f), dtype=bool)
    data = (f, n, ft, mask, mask)
    first, _, _, _ = module.train_one('modular', data, data, 2, tmp_path, 'fixed', 0)
    resumed, _, _, _ = module.train_one('modular', data, data, 2, tmp_path, 'fixed', 0)
    assert np.array_equal(first.predict(f)[0], resumed.predict(f)[0])
    with pytest.raises(AssertionError, match='stale interrupted run'):
        module.train_one('modular', data, data, 2, tmp_path, 'changed', 0)


def test_no_history_control_ignores_all_histories():
    f = torch.as_tensor(fixture()); alternate = f.clone()
    alternate[:, 3:9] += 3; alternate[:, 10:13] -= 7
    model = Neural('no_history')
    assert torch.equal(model.normal(f), model.normal(alternate))
    assert torch.equal(model.friction(f), model.friction(alternate))


def test_noisy_privileged_data_uses_local_compression_rate():
    """AB logs contain two distinct rates; measured features must use material rate."""
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('v2_train_local_rate', root/'scripts/train.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    shape = (2, 12)
    arrays = dict(delta=np.full(shape, .02), rate=np.full(shape, -.017),
                  compression_rate=np.full(shape, .031), normal=np.full(shape, 10.),
                  friction=np.ones(shape), state=np.zeros((*shape, 2)))
    actual, normal, friction = module.measured_arrays(arrays, 'noisy', 2, 'privileged')
    # The otherwise identical isolated-form input carries the same local rate in
    # its sole rate channel, so paired noise must produce identical features.
    isolated_form = {key: value.copy() for key, value in arrays.items() if key != 'compression_rate'}
    isolated_form['rate'][:] = arrays['compression_rate']
    expected, expected_normal, expected_friction = module.measured_arrays(
        isolated_form, 'noisy', 2, 'privileged')
    assert np.array_equal(actual, expected)
    assert np.array_equal(normal, expected_normal)
    assert np.array_equal(friction, expected_friction)
    # Changing only the actuator command derivative cannot alter a local label.
    arrays['rate'][:] = 1.5
    changed, _, _ = module.measured_arrays(arrays, 'noisy', 2, 'privileged')
    assert np.array_equal(actual, changed)
