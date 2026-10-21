"""Synthetic-only firewall/forecast-start checks; no native evaluation run."""
from pathlib import Path
import copy
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
import evaluate_native_v2 as evaluation
from models import load_model, motion_step


def fixture(probe_steps=10):
    count, total = 2, probe_steps+20
    steps = np.array(sorted(set(range(0, total+1, 10)) | {probe_steps, total}))
    d = np.full((count, total), .02)
    rate = np.zeros_like(d); rate[:, probe_steps:] = .05
    obs = np.empty((count, len(steps), 3))
    obs[...] = [.001, .03, 15.]
    obs[1, :, :2] *= -1
    arrays = dict(u=np.repeat([[2.], [-2.]], total, axis=1), d=d, d_dot=rate,
                  amplitude=np.array([.005, -.005]), observations=obs,
                  observation_steps=steps, probe_steps=np.array(probe_steps),
                  state=np.zeros((count, total, 2)), next_state=np.zeros((count, total, 2)),
                  normal=np.ones((count, total))*15, friction=np.zeros((count, total)),
                  guard=np.zeros((count, total), dtype=bool),
                  true_hidden_state=np.ones((count, total))*12345,
                  preparation_index=np.arange(count))
    metadata = dict(dt=.0025, sensor_parameters=dict(sigma_x_m=5e-5, sigma_v_m_s=.002, sigma_normal_N=.15),
                    normal_world="boucwen", friction_world="maxwell", sensor="noisy", preparation_seed=31011,
                    probe=dict(name="synthetic", duration_s=probe_steps*.0025),
                    initial_truth_audit=dict(forbidden=[12345, 54321]),
                    factor_rows=[dict(recipe="not_observable") for _ in range(count)],
                    prepared_reference_audit=dict(commands={
                        "delta": np.full((count, 5), .02).tolist(), "rate": np.zeros((count, 5)).tolist(),
                        "v": np.full((count, 5), .04).tolist(), "normal": np.full((count, 5), 15.).tolist()}))
    cfg = dict(grid_size=7, prior=dict(maxwell=dict(bounds=[0., 1.], spacing="linear"),
                                     boucwen=dict(bounds=[.02, 8.], spacing="log")),
               numerical_floor=dict(x_m=1e-6, v_m_s=1e-5, normal_N=1e-4))
    return arrays, metadata, cfg


@pytest.fixture(scope="module")
def model():
    return load_model("boucwen", "maxwell", 1, "clean")


@pytest.mark.parametrize("kind", evaluation.FILTERS)
def test_future_truth_and_hidden_labels_do_not_enter_forecast(model, kind):
    arrays, meta, cfg = fixture()
    changed = {k: np.array(v, copy=True) for k, v in arrays.items()}
    for key in ("state", "next_state", "normal", "friction", "true_hidden_state", "preparation_index"):
        changed[key][...] = 123456
    changed["observations"][:, arrays["observation_steps"] > int(arrays["probe_steps"])] = 987654
    wrong_meta = copy.deepcopy(meta)
    wrong_meta["initial_truth_audit"] = {"forbidden": [-1e100]}
    wrong_meta["prepared_reference_audit"] = {"forbidden": "not exposed"}
    wrong_meta["factor_rows"] = [{"recipe": "different"}]
    wrong_meta["preparation_seed"] = 59999
    public, other = evaluation.observable_inputs(arrays), evaluation.observable_inputs(changed)
    assert set(public) == {"u", "d", "d_dot", "amplitude", "observations", "observation_steps", "probe_steps"}
    for key in public:
        np.testing.assert_array_equal(public[key], other[key])
    first, detail = evaluation.infer_and_forecast(model, public, meta, cfg, kind)
    second, second_detail = evaluation.infer_and_forecast(model, other, wrong_meta, cfg, kind)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    for key in detail:
        np.testing.assert_array_equal(detail[key], second_detail[key])


def test_each_start_uses_same_particles_for_prior_and_posterior(model, monkeypatch):
    arrays, meta, cfg = fixture()
    inputs = evaluation.observable_inputs(arrays)
    original = evaluation.propagate_forecast
    calls = []
    def capture(model, xv, state, inputs, dt, grid_size=1):
        result = original(model, xv, state, inputs, dt, grid_size)
        calls.append((xv.copy(), copy.deepcopy(state), result.copy()))
        return result
    monkeypatch.setattr(evaluation, "propagate_forecast", capture)
    predictions, detail = evaluation.infer_and_forecast(model, inputs, meta, cfg)
    assert len(calls) == 2  # one tensor per start, not a new trajectory per estimator
    for condition, (xv, state, particles) in zip(evaluation.STARTS, calls):
        np.testing.assert_array_equal(xv.reshape(2, 7, 2), detail[condition+"_initial_xv"])
        for estimator, weight in (("prior_propagation", detail["prior"]), ("trajectory_assimilation", detail["weights"])):
            expected = np.einsum("eg,egtk->etk", weight, particles)
            np.testing.assert_array_equal(predictions[f"joint_sensor_likelihood/{condition}/{estimator}"], expected)
    np.testing.assert_array_equal(calls[0][0].reshape(2, 7, 2), detail["filtered_grid_xv"])
    np.testing.assert_array_equal(calls[1][0], np.repeat(inputs["observations"][:, -1, :2], 7, axis=0))
    assert not np.allclose(calls[0][0], calls[1][0])
    for key in calls[0][1]:
        np.testing.assert_array_equal(calls[0][1][key], calls[1][1][key])


def test_measured_start_ablation_reproduces_frozen_v1(model):
    # Same public synthetic inputs isolate only the start condition. The old
    # evaluator is read-only and no native data/old outputs are opened.
    from evaluate_native import infer_and_forecast as original_infer
    arrays, meta, cfg = fixture()
    inputs = evaluation.observable_inputs(arrays)
    for kind in evaluation.FILTERS:
        old, _ = original_infer(model, inputs, meta, cfg, kind)
        new, _ = evaluation.infer_and_forecast(model, inputs, meta, cfg, kind)
        for estimator in ("prior_propagation", "trajectory_assimilation"):
            np.testing.assert_array_equal(old[f"{kind}/{estimator}"], new[f"{kind}/measured_start/{estimator}"])


def test_zero_probe_has_no_likelihood_gain_and_preserves_known_initial_mechanics(model):
    arrays, meta, cfg = fixture(probe_steps=0)
    inputs = evaluation.observable_inputs(arrays)
    pred, detail = evaluation.infer_and_forecast(model, inputs, meta, cfg)
    np.testing.assert_array_equal(detail["weights"], detail["prior"])
    np.testing.assert_array_equal(detail["filtered_particles_initial_xv"], np.zeros((2, 7, 2)))
    assert np.any(detail["measured_start_initial_xv"] != 0)
    for start in evaluation.STARTS:
        np.testing.assert_array_equal(pred[f"joint_sensor_likelihood/{start}/prior_propagation"],
                                      pred[f"joint_sensor_likelihood/{start}/trajectory_assimilation"])


def test_boundary_normal_likelihood_uses_boundary_rate(model, monkeypatch):
    arrays, meta, cfg = fixture()
    rates = []
    original = evaluation.JointGridObserver
    class RecordingJoint(original):
        def observe(self, *args, **kwargs):
            rates.append(kwargs["d_dot"].copy())
            return super().observe(*args, **kwargs)
    monkeypatch.setattr(evaluation, "JointGridObserver", RecordingJoint)
    evaluation.infer_and_forecast(model, evaluation.observable_inputs(arrays), meta, cfg)
    assert len(rates) == 1
    np.testing.assert_array_equal(rates[-1], [.05, .05])
    assert not np.array_equal(rates[-1], arrays["d_dot"][:, int(arrays["probe_steps"])-1])


def test_privileged_start_carries_own_modeled_mechanics_not_truth(model):
    arrays, meta, cfg = fixture()
    public = evaluation.observable_inputs(arrays)
    pred, detail = evaluation.privileged_forecasts(model, public, meta, cfg)
    commands = {k: np.asarray(v) for k, v in meta["prepared_reference_audit"]["commands"].items()}
    fm = model.friction_model
    fs = fm.initial(commands["delta"][:, 0], commands["v"][:, 0], commands["normal"][:, 0])
    for t in range(4):
        _, _, fs = fm.step(commands["delta"][:, t], commands["rate"][:, t], commands["v"][:, t],
                           fs, normal_override=commands["normal"][:, t], dt=meta["dt"])
    state = model.initial(np.full(2, .02), np.zeros(2), public["observations"][:, 0, 2])
    state.update({"friction__"+k: v for k, v in fs.items()})
    xv = np.zeros((2, 2))
    for t in range(public["probe_steps"]):
        xv, state, _, _ = motion_step(model, xv, state, *(public[k][:, t] for k in ("u", "d", "d_dot")),
                                      public["amplitude"], dt=meta["dt"])
    np.testing.assert_array_equal(detail["privileged_filtered_initial_xv"], xv)
    assert not np.array_equal(xv, public["observations"][:, -1, :2])
    wrong_meta = copy.deepcopy(meta)
    wrong_meta["initial_truth_audit"] = {"forbidden": 1e100}
    other, _ = evaluation.privileged_forecasts(model, public, wrong_meta, cfg)
    for key in pred:
        np.testing.assert_array_equal(pred[key], other[key])


def test_bank_reads_only_fixed_grid_weights_and_rejects_episode_specific_prior(model, tmp_path):
    arrays, meta, cfg = fixture()
    grid = np.linspace(0., 1., cfg["grid_size"])
    weights = np.arange(1., cfg["grid_size"]+1)
    path = tmp_path/"prior.npz"
    # Reading this sentinel array with allow_pickle=False would fail. The
    # explicit grid/weight reader must never access training-history payloads.
    np.savez(path, grid=grid, weights=weights, sampled_initial_state=np.array([{"forbidden": True}], dtype=object))
    loaded_grid, loaded_weight = evaluation.load_prior_bank(path, cfg)
    np.testing.assert_array_equal(loaded_grid, grid)
    np.testing.assert_array_equal(loaded_weight, weights)
    public = evaluation.observable_inputs(arrays)
    _, detail = evaluation.infer_and_forecast(model, public, meta, cfg, grid=loaded_grid, prior=loaded_weight)
    np.testing.assert_allclose(detail["prior"], np.tile(weights/weights.sum(), (2, 1)))
    with pytest.raises(ValueError, match="shared"):
        evaluation.infer_and_forecast(model, public, meta, cfg, grid=grid, prior=np.tile(weights, (2, 1)))
    cfg["prior_bank_directory"] = str(tmp_path)
    assert evaluation.prior_path(cfg, "boucwen", "maxwell", 1, "clean") == tmp_path/"boucwen_maxwell/r1/clean.npz"


def test_csv_factors_and_metadata_seed_are_separate_from_inference(model):
    arrays, meta, cfg = fixture()
    predictions, _ = evaluation.infer_and_forecast(model, evaluation.observable_inputs(arrays), meta, cfg)
    rows = evaluation.score_forecasts(predictions, arrays, meta, 1, "clean")
    assert len(rows) == 8
    assert {row["preparation_seed"] for row in rows} == {31011}
    assert {row["forecast_start"] for row in rows} == set(evaluation.STARTS)
    for row in rows:
        assert row["method"] == "/".join([row["filter_kind"], row["forecast_start"], row["estimator"]])


def test_invalid_public_inputs_and_factor_names_fail(model):
    arrays, meta, cfg = fixture()
    arrays["observation_steps"][1] = 0
    with pytest.raises(ValueError, match="unique"):
        evaluation.observable_inputs(arrays)
    with pytest.raises(ValueError):
        evaluation.configured_starts({"forecast_starts": ["true_state"]})
    with pytest.raises(ValueError):
        evaluation.mechanical_start(np.zeros((1, 1, 2)), np.zeros((1, 2)), "true_state")
