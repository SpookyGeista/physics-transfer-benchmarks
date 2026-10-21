"""Small CPU checks of observer causality, likelihood and frozen-law interface."""
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper_study" / "src"))
from observer import GridObserver, initial_state_grid


class ToyMaterial:
    """A dissipative latent-drag law with exactly inspectable timing."""
    def initial(self, delta, v, normal_sensor):
        return {"normal__relaxed": np.column_stack((delta, delta)),
                "friction__mix": np.ones(len(delta))}

    def step(self, delta, rate, v, state, normal_override=None, dt=.0025):
        n = 500*np.maximum(delta, 0) if normal_override is None else np.asarray(normal_override)
        ft = -(1+state["friction__mix"])*n*np.tanh(v/.04)
        new = {"normal__relaxed": state["normal__relaxed"] + dt*rate[:, None],
               "friction__mix": state["friction__mix"]*np.exp(-dt)}
        return n.copy(), ft, new


def make(**kwargs):
    args = dict(model=ToyMaterial(), grid=np.array([.1, .5, .9]),
                initial_xv=[0., .04], initial_d=.02, initial_N=15.,
                sigma_x=.001, sigma_v=.01)
    args.update(kwargs)
    return GridObserver(**args)


def test_pre_state_force_and_semiimplicit_mechanics():
    ob = make()
    old_v = .04
    ft = -(1+ob.grid)*15*np.tanh(old_v/.04)
    ob.advance(2., .02, 0., 0.)
    expected_v = old_v+.0025*(2+ft)/2
    np.testing.assert_allclose(ob.grid_xv[0, :, 1], expected_v)
    np.testing.assert_allclose(ob.grid_xv[0, :, 0], .0025*expected_v)
    np.testing.assert_allclose(ob.state["friction__mix"], ob.grid*np.exp(-.0025))


def test_one_independent_measurement_each_frame_no_differencing():
    ob = make(prior=[.2, .3, .5])
    expected_loglik = np.zeros((1, 3))
    for frame in range(3):
        ob.advance(4., .02, .001, .005, steps=10)
        observed = ob.grid_xv[0, 1]+[.0002*(frame+1), -.003]
        expected_loglik -= .5*np.sum(((observed-ob.grid_xv)/ob.effective_sigma)**2, axis=2)
        ob.observe(*observed, N=15.+frame)
        np.testing.assert_allclose(ob.log_likelihood, expected_loglik)
        expected = np.log(ob.prior)+expected_loglik
        expected = np.exp(expected-expected.max(axis=1, keepdims=True))
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(ob.weights, expected)
    assert ob.observation_count == 3 and ob.step_index == 30
    before = ob.snapshot()
    with pytest.raises(ValueError, match="duplicate"):
        ob.observe(0., 0., N=15.)
    np.testing.assert_array_equal(ob.weights, before["weights"])


def test_new_force_measurement_is_only_a_future_input():
    a, b = make(), make()
    for ob in (a, b):
        ob.advance(2., .02, 0., .005, steps=10)
    a.observe(.001, .02, 8.)
    b.observe(.001, .02, 20.)
    np.testing.assert_array_equal(a.weights, b.weights)
    np.testing.assert_array_equal(a.grid_xv, b.grid_xv)
    for key in a.state:
        np.testing.assert_array_equal(a.state[key], b.state[key])
    a.advance(2., .02, 0., .005)
    b.advance(2., .02, 0., .005)
    assert not np.array_equal(a.grid_xv, b.grid_xv)


def test_blind_gets_same_physical_inputs_and_keeps_prior():
    sensed, blind = make(), make(assimilate=False)
    for _ in range(3):
        for ob in (sensed, blind):
            ob.advance(4., .02, .002, -.005, steps=10)
            ob.observe(.002, .035, 14.)
        np.testing.assert_array_equal(sensed.grid_xv, blind.grid_xv)
        for key in sensed.state:
            np.testing.assert_array_equal(sensed.state[key], blind.state[key])
        np.testing.assert_array_equal(blind.weights, blind.prior)
    assert not np.allclose(sensed.weights, sensed.prior)


def test_batch_axes_match_independent_observers():
    initial = np.array([[.002, .04], [-.003, -.06]])
    batch = make(initial_xv=initial, initial_d=[.019, .024], initial_N=[12., 18.])
    singles = [make(initial_xv=initial[e], initial_d=[.019, .024][e], initial_N=[12., 18.][e])
               for e in range(2)]
    for _ in range(3):
        batch.advance([2., -3.], [.022, .024], [.002, -.003], [.005, -.005], steps=4)
        batch.observe([.001, -.002], [.03, -.04], [13., 17.])
        for e, ob in enumerate(singles):
            ob.advance([2., -3.][e], [.022, .024][e], [.002, -.003][e], [.005, -.005][e], steps=4)
            ob.observe([.001, -.002][e], [.03, -.04][e], [13., 17.][e])
            np.testing.assert_array_equal(batch.grid_xv[e], ob.grid_xv[0])
            np.testing.assert_array_equal(batch.weights[e], ob.weights[0])
            for key in batch.state:
                np.testing.assert_array_equal(batch.state[key][e*3:(e+1)*3], ob.state[key])


def test_snapshot_forecast_and_future_cannot_mutate_current_inference():
    ob = make()
    ob.advance(2., .02, 0., .005, steps=10)
    ob.observe(.001, .02, 15.)
    before = ob.snapshot()
    forecast = ob.forecast_start()
    np.testing.assert_array_equal(forecast["xv"], np.tile([.001, .02], (3, 1)))
    assert "normal_override" not in forecast and "last_observed_N" not in forecast
    forecast["state"]["friction__mix"][:] = 0
    forecast["weights"][:] = 0
    forecast["xv"][:] = 10
    np.testing.assert_array_equal(ob.weights, before["weights"])
    for key in ob.state:
        np.testing.assert_array_equal(ob.state[key], before["state"][key])
    ob.advance(-8., .04, -.1, -.005, steps=10)
    ob.observe(-.01, -.3, 12.)
    assert before["step"] == 10
    assert not np.array_equal(before["grid_xv"], ob.grid_xv)


def test_zero_noise_has_explicit_positive_floor_and_start_no_information():
    ob = make(sigma_x=0., sigma_v=0., sigma_N=0.)
    np.testing.assert_array_equal(ob.weights, ob.prior)
    np.testing.assert_array_equal(ob.effective_sigma, [1e-6, 1e-5])
    with pytest.raises(ValueError, match="initial"):
        ob.observe(0., .04, 15.)
    ob.advance(2., .02, 0., 0.)
    ob.observe(*ob.grid_xv[0, 1], N=15.)
    assert np.isfinite(ob.weights).all() and ob.weights.sum() == pytest.approx(1.)
    assert ob.weights[0, 1] > .999


def test_declared_prior_initialization_does_not_replace_normal_history():
    ob = make(initial_xv=[[0., 0.], [0., 0.]], initial_d=[.018, .025])
    np.testing.assert_array_equal(ob.state["normal__relaxed"],
                                  np.repeat([[.018, .018], [.025, .025]], 3, axis=0))
    np.testing.assert_array_equal(ob.state["friction__mix"], np.tile(ob.grid, 2))
    np.testing.assert_array_equal(initial_state_grid("maxwell")[[0, -1]], [0., 1.])
    np.testing.assert_array_equal(initial_state_grid("boucwen")[[0, -1]], [.02, 8.])


@pytest.mark.parametrize("normal_world,friction_world", [
    ("maxwell", "maxwell"), ("maxwell", "boucwen"),
    ("boucwen", "maxwell"), ("boucwen", "boucwen")])
def test_frozen_composition_matches_standard_motion_step(normal_world, friction_world):
    sys.path.insert(0, str(REPO / "research_next" / "src"))
    from state_id import CompositeStateID
    from online import motion_step
    def load(world):
        path = REPO / "research_next" / "results" / "round1" / "models" / "state_id" / world / "r1" / "clean" / "blind_selection" / "model.json"
        return CompositeStateID.load(path)
    model = CompositeStateID(load(normal_world).normal_model, load(friction_world).friction_model)
    ob = GridObserver(model, initial_state_grid(model.friction_family, 5),
                      [[0., .01], [.001, -.01]], [.02, .021], [15., 16.])
    start = ob.snapshot()
    xv, state = start["grid_xv"].reshape(-1, 2), start["state"]
    for _ in range(3):
        xv, state, n, ft, bad = motion_step(model, xv, state, np.repeat([2., -2.], 5),
                                            np.repeat([.02, .022], 5), .002,
                                            np.repeat([.005, -.005], 5),
                                            normal_override=np.repeat([15., 16.], 5))
        assert not bad.any()
        assert np.all(ft * start["grid_xv"].reshape(-1, 2)[:, 1] <= 1e-12) if _ == 0 else True
        ob.advance([2., -2.], [.02, .022], .002, [.005, -.005])
        np.testing.assert_array_equal(ob.grid_xv.reshape(-1, 2), xv)
        for key in state:
            np.testing.assert_array_equal(ob.state[key], state[key])


def test_bad_arguments_rejected_without_hidden_fallback():
    for kwargs in (dict(numerical_x=0.), dict(sigma_N=-1.), dict(grid=[.5, .1]),
                   dict(grid=[-.1, 1.]), dict(prior=[0., 0., 0.])):
        with pytest.raises(ValueError):
            make(**kwargs)
    ob = make()
    with pytest.raises(ValueError):
        ob.advance(0., .02, 0., 0., steps=1.5)
    with pytest.raises(ValueError):
        ob.advance([1., 2.], .02, 0., 0.)
    ob.advance(0., .02, 0., 0.)
    with pytest.raises(ValueError, match="current observation"):
        ob.forecast_start()
