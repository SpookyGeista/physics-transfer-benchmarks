"""Scientific invariants for online feature controls and independent calibration."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from feature_baselines import (DT, TAUS, FeatureBaseline, advance_state, basis,
                               feature_sequence, feature_step, fit_head, initial_state)


def sample(count=3, steps=40):
    rng = np.random.default_rng(7101)
    d = .02 + .002*rng.normal(size=(count, steps))
    v = .1*rng.normal(size=(count, steps))
    return dict(delta=d, rate=.02*rng.normal(size=d.shape),
                state=np.stack([np.zeros_like(v), v], -1),
                normal=12 + rng.normal(size=d.shape))


def constant_model(kind):
    nf, ff = feature_step(kind, [.02], [0], [.1], [12], initial_state(kind, [.02], [.1], [12]))
    fits = {}
    for head, values, intercept in [("normal", nf, .5), ("friction", ff, .4)]:
        size = basis(values, head).shape[1]
        fit = dict(mean=np.zeros(size).tolist(), scale=np.ones(size).tolist(), coef=np.zeros(size).tolist())
        fit["coef"][0] = intercept
        fits[head] = fit
    return FeatureBaseline(kind, **fits)


@pytest.mark.parametrize("kind", TAUS)
def test_old_diagnostic_feature_parity(kind):
    # Extract the unchanged prior function only: no training or old-output writes.
    source = (ROOT/"diagnostics/run_local_diagnostics.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "build_features")
    namespace = dict(np=np, DT=DT, TAUS_LONG=TAUS["path_ridge"])
    exec(compile(ast.Module(body=[node], type_ignores=[]), "prior_diagnostic", "exec"), namespace)
    data = sample()
    if kind == "old_flex":
        # Original v2 observed-feature reconstruction, with independent indexing.
        sys.path.insert(0, str(ROOT.parent/"study_v2/src"))
        from decision_study.common import features, history_init, history_update
        d, r, v, n = data["delta"], data["rate"], data["state"][..., 1], data["normal"]
        h = history_init(d[:, 0], v[:, 0], n[:, 0]); old = []
        for t in range(d.shape[1]):
            old.append(features(d[:, t], r[:, t], v[:, t], h, n[:, t]))
            h = history_update(h, d[:, t], v[:, t], n[:, t])
        data["features"] = np.stack(old, 1)
    expected = namespace["build_features"](data, "base3" if kind == "old_flex" else "bank9_path")
    actual = feature_sequence(data, kind)
    for x, y in zip(actual, expected):
        np.testing.assert_allclose(x, y, rtol=0, atol=1e-6)


@pytest.mark.parametrize("kind", TAUS)
def test_normal_excludes_force_labels_and_future_is_causal(kind):
    data = sample()
    expected_n, expected_f = feature_sequence(data, kind)
    altered = copy.deepcopy(data); altered["normal"] += 50
    np.testing.assert_array_equal(expected_n, feature_sequence(altered, kind)[0])
    altered = copy.deepcopy(data)
    for key in altered:
        altered[key][:, 20:] += 2
    nf, ff = feature_sequence(altered, kind)
    np.testing.assert_array_equal(nf[:, :20], expected_n[:, :20])
    np.testing.assert_array_equal(ff[:, :20], expected_f[:, :20])


@pytest.mark.parametrize("kind,parameters", [("old_flex", 109), ("path_ridge", 1409)])
def test_online_coupling_prediction_and_state_ownership(kind, parameters):
    model = constant_model(kind)
    state = model.initial([.02, .02], [.2, -.2], [100, 100])
    before = copy.deepcopy(state)
    n, ft, new = model.step([.02, .02], [.01, -.01], [.2, -.2], state)
    np.testing.assert_allclose(n, 10)
    np.testing.assert_allclose(ft, -10*.4*np.tanh(np.array([.2, -.2])/.04))
    assert np.all(ft*np.array([.2, -.2]) <= 0)
    assert model.parameter_count() == parameters
    for key in state:
        np.testing.assert_array_equal(state[key], before[key])
        assert not np.shares_memory(state[key], new[key])
    weight = 1 - np.exp(-DT/np.array(TAUS[kind]))
    np.testing.assert_allclose(new["history"][..., 2], np.broadcast_to(100 + weight*(10-100), new["history"][..., 2].shape))
    _, imposed_ft, imposed_state = model.conditional_step([.02, .02], [0, 0], [.2, -.2], [30, 30], state)
    np.testing.assert_allclose(imposed_ft, ft*3)
    assert not np.array_equal(new["history"][..., 2], imposed_state["history"][..., 2])


@pytest.mark.parametrize("kind", TAUS)
def test_contact_open_and_initial_load_does_not_enter_normal(kind):
    model = constant_model(kind)
    left = model.initial([.02], [.1], [5]); right = model.initial([.02], [.1], [500])
    for _ in range(3):
        n1, _, left = model.step([.018], [-.01], [.1], left)
        n2, _, right = model.step([.018], [-.01], [.1], right)
        np.testing.assert_array_equal(n1, n2)
    n, ft, _ = model.step([0], [0], [.2], left)
    np.testing.assert_array_equal(n, [0]); np.testing.assert_array_equal(ft, [0])


def test_readout_basis_matches_original_v2():
    sys.path.insert(0, str(ROOT.parent/"study_v2/src"))
    from decision_study.models import flex_basis
    rng = np.random.default_rng(43)
    original = rng.normal(size=(17, 13)).astype(np.float32)
    for head, selected in [("normal", [0, 1, 3, 4, 5]), ("friction", [2, 6, 7, 8, 9, 10, 11, 12])]:
        np.testing.assert_array_equal(basis(original[:, selected], head), flex_basis(original, head))


def test_serialization_and_branch_replay(tmp_path):
    model = constant_model("path_ridge")
    path = tmp_path/"model.json"; model.save(path)
    loaded = FeatureBaseline.load(path)
    with pytest.raises(FileExistsError):
        loaded.save(path)
    state = model.initial([.02], [0], [12])
    expected = model.step([.025], [.04], [.2], state)
    got = loaded.step([.025], [.04], [.2], state)
    for a, b in zip(expected[:2], got[:2]):
        np.testing.assert_array_equal(a, b)
    for key in expected[2]:
        np.testing.assert_array_equal(expected[2][key], got[2][key])


def test_ridge_fits_force_at_zero_speed_without_label_division():
    rng = np.random.default_rng(71)
    x = rng.normal(size=(80, 8)).astype(np.float32)
    factor = np.linspace(-12, 12, len(x)); factor[::4] = 0
    target = .35*factor
    fitted = fit_head("old_flex", "friction", x, target, x, target, factor, factor, block_size=19)
    pred = factor*np.maximum(((basis(x, "friction")-fitted["mean"])/fitted["scale"])@fitted["coef"], 0)
    assert np.sqrt(np.mean((pred-target)**2)) < 1e-8
    assert np.all(np.isfinite(pred)) and len(fitted["candidates"]) == 5


def test_calibration_noise_matches_state_id_exactly(tmp_path):
    source = (ROOT/"scripts/fit_state_id_diagnostic.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "load_data")
    namespace = dict(np=np, Path=Path, json=json,
                     sha=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest())
    exec(compile(ast.Module(body=[node], type_ignores=[]), "state_id_noise", "exec"), namespace)
    spec = importlib.util.spec_from_file_location("feature_fitting", ROOT/"scripts/fit_feature_baselines.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    data = sample(4, 2400)
    data["normal"][2:] = data["normal"][:2]
    data["friction"] = np.full((4, 2400), -3.)
    rows = [dict(mode=mode, profile=i % 2, id=f"train-{i}") for i, mode in enumerate(["B", "B", "A", "A"])]
    folder = tmp_path/"maxwell/r0"; folder.mkdir(parents=True)
    np.savez(folder/"train.npz", **data)
    (folder/"train.json").write_text(json.dumps(dict(rows=rows, steps=2400, dt=DT, role="exploratory_development_only")))
    expected, _ = namespace["load_data"](tmp_path, "maxwell", 0, "train", "noisy")
    actual, _, index, _ = module.load_data(tmp_path, "maxwell", 0, "train", "noisy")
    for head, mode in [("normal", "B"), ("friction", "A")]:
        for key, field in [("delta", "delta"), ("rate", "rate")]:
            np.testing.assert_array_equal(actual[field][index[mode]], expected[head][key])
    np.testing.assert_array_equal(actual["normal"][index["B"]], expected["normal"]["target"])
    np.testing.assert_array_equal(actual["normal"][index["A"]], expected["friction"]["normal"])
    np.testing.assert_array_equal(actual["friction"][index["A"]], expected["friction"]["target"])
    np.testing.assert_array_equal(actual["state"][index["A"], :, 1], expected["friction"]["velocity"])
