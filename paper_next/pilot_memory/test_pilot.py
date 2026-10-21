"""Independent small checks for causal isolation and physical conditioning."""
import os
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/pilot-memory-numba")
import ast
from pathlib import Path
import numpy as np
import pytest
from observer import candidate_bank, posterior_weights, summarize_posterior
from reference_fixture import run_fixture

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def model(world):
    return REPO / f"research_next/results/round1/models/state_id/{world}/r1/clean/blind_selection/model.json"


def test_no_reference_import_in_observer():
    tree = ast.parse((HERE / "observer.py").read_text())
    imports = [getattr(n, "module", "") or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any("reference" in name for name in imports)


def test_empty_force_observations_equal_blinded_prior():
    force = np.empty((11, 0))
    observed = posterior_weights(force, np.empty((3, 0)), .05)
    blind = posterior_weights(force, None, .05, batch_size=3)
    np.testing.assert_array_equal(observed, blind)
    np.testing.assert_array_equal(blind, np.full((3, 11), 1 / 11))


def test_grid_posterior_uses_likelihood_not_state_labels():
    force = np.array([[0., 0.], [1., 1.], [2., 2.]])
    w = posterior_weights(force, np.array([[1., 1.]]), .01)
    assert np.argmax(w) == 1 and w[0, 1] > .999
    b = posterior_weights(force, None, .01, batch_size=1)
    np.testing.assert_array_equal(b, np.full((1, 3), 1 / 3))


@pytest.mark.parametrize("world", ["maxwell", "boucwen"])
def test_probe_outputs_before_update_and_future_after_probe(world):
    grid = np.array([.1, .5, .9]) if world == "maxwell" else np.array([.03, .3, 3.])
    v = np.full(4, .08); n = np.full(4, 15.)
    bank = candidate_bank(model(world), grid, v, n, v[:1], n[:1], [0, 1, 4])
    np.testing.assert_array_equal(bank["current"][0], grid)
    np.testing.assert_array_equal(bank["future"][0, :, 0], bank["force"][:, 0])
    np.testing.assert_array_equal(bank["future"][1, :, 0], bank["force"][:, 1])
    # Altering only known future commands must not alter probe inference inputs.
    alternative = candidate_bank(model(world), grid, v, n, -v[:1], n[:1], [0, 1, 4])
    np.testing.assert_array_equal(bank["force"], alternative["force"])
    np.testing.assert_array_equal(bank["current"], alternative["current"])
    assert not np.array_equal(bank["future"], alternative["future"])


@pytest.mark.parametrize("world", ["maxwell", "boucwen"])
@pytest.mark.parametrize("probe_kind", ["rest", "slip"])
def test_no_force_information_at_rest_and_exact_contraction(world, probe_kind):
    initial = np.linspace(.1, .9, 7) if world == "maxwell" else np.geomspace(.03, 3., 7)
    v = np.full(40, .08 if probe_kind == "slip" else 0.); n = np.full(40, 15.)
    truth = run_fixture(world, initial, v, n, v[:2], n[:2], [0, 1, 40])
    np.testing.assert_array_equal(truth["at_rest_force"], np.zeros(len(initial)))
    ratio = np.ptp(truth["current"], axis=1) / np.ptp(initial)
    np.testing.assert_allclose(ratio, truth["analytic_contraction"], rtol=1e-11)
    contracts = world == "maxwell" or probe_kind == "slip"
    if contracts:
        assert ratio[-1] < 1
    else:
        np.testing.assert_allclose(ratio, 1., atol=1e-13)
    # Physical contraction occurs even for an observer that keeps its full prior.
    weights = posterior_weights(np.zeros((len(initial), 40)), None, .1, batch_size=1)
    a = summarize_posterior(weights, initial, truth["current"][0], truth["future"][0])
    b = summarize_posterior(weights, initial, truth["current"][-1], truth["future"][-1])
    np.testing.assert_array_equal(a["initial_mean"], b["initial_mean"])
    np.testing.assert_array_equal(a["entropy"], b["entropy"])
    if contracts:
        assert b["current_width90"][0] < a["current_width90"][0]
    else:
        np.testing.assert_allclose(b["current_width90"], a["current_width90"], atol=1e-13)


@pytest.mark.parametrize("world", ["maxwell", "boucwen"])
def test_resting_force_measurements_do_not_update_state_posterior(world):
    grid = np.linspace(.1, .9, 11) if world == "maxwell" else np.geomspace(.03, 3., 11)
    bank = candidate_bank(model(world), grid, np.zeros(4), np.full(4, 15.),
                          np.array([.08]), np.array([15.]), [0, 4])
    np.testing.assert_array_equal(bank["force"], np.zeros((11, 4)))
    y = np.random.default_rng(3).normal(0, .2, size=(10, 4))
    sensed = posterior_weights(bank["force"], y, .2)
    blind = posterior_weights(bank["force"], None, .2, batch_size=10)
    np.testing.assert_array_equal(sensed, blind)
