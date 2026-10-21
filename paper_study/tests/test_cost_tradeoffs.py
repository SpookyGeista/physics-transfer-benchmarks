"""Small deterministic checks of costs, dominance, units and cluster counts."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import analyze_cost_tradeoffs as analysis


def arrays(probe=2):
    state = np.array([[[0., 1.], [.1, -1.], [0., 2.]]])
    next_state = np.array([[[.1, 1.], [0., -1.], [.2, 2.]]])
    return dict(state=state, next_state=next_state, normal=np.array([[3., 4., 5.]]),
                friction=-state[..., 1], u=np.array([[2., 2., -1.]]),
                d_dot=np.array([[.1, -.2, 0.]]), probe_steps=np.array(probe))


def test_work_counts_both_ports_before_absolute_sum_and_keeps_sign():
    result = analysis.physical_costs(arrays(), .1)
    assert result["total_absolute_actuator_work_J"][0] == pytest.approx(.71)
    assert result["probe_absolute_actuator_work_J"][0] == pytest.approx(.51)
    assert result["total_compression_absolute_work_J"][0] == pytest.approx(.11)
    assert result["total_signed_actuator_work_J"][0] == pytest.approx(-.25)
    assert result["probe_friction_work_J"][0] == pytest.approx(-.2)
    assert result["probe_path_length_m"][0] == pytest.approx(.2)


def test_zero_probe_retained_with_zero_probe_cost_but_nonzero_future():
    result = analysis.physical_costs(arrays(0), .1)
    assert result["probe_time_s"][0] == 0
    assert result["probe_absolute_actuator_work_J"][0] == 0
    assert result["total_absolute_actuator_work_J"][0] == pytest.approx(.71)


def test_three_objective_dominance_ties_and_tradeoffs():
    points = [[1, 1, 1], [2, 1, 1], [.5, 2, 2], [1, 1, 1]]
    np.testing.assert_array_equal(analysis.nondominated(points), [True, False, True, True])
    np.testing.assert_array_equal(analysis.nondominated([[1, 1, 4], [2, 2, 1]]), [True, True])
    with pytest.raises(ValueError): analysis.nondominated([[np.nan, 1]])


def test_replica_counts_do_not_make_single_bank_bootstrappable():
    frame = pd.DataFrame([dict(model_seed=1, preparation_seed=31001, error=i%6)
                          for i in range(3072)])
    intervals, status = analysis.crossed_bootstrap(frame, ["error"])
    assert intervals == {} and not status["interval_available"]
    assert status["model_banks"] == status["preparation_seed_blocks"] == 1


def test_two_way_cluster_bootstrap_preserves_paired_columns():
    frame = pd.DataFrame([dict(model_seed=b, preparation_seed=p, error=b+p+r, gain=.25)
                          for b in range(8) for p in range(8) for r in range(4)])
    intervals, status = analysis.crossed_bootstrap(frame, ["error", "gain"], draws=256)
    assert status["interval_available"] and status["resamples"] == 256
    assert intervals["error"][0] < frame.error.mean() < intervals["error"][1]
    np.testing.assert_allclose(intervals["gain"], [.25, .25], atol=1e-15)
    incomplete = frame[~((frame.model_seed == 0)&(frame.preparation_seed == 0))]
    with pytest.raises(ValueError, match="all bank/preparation"):
        analysis.crossed_bootstrap(incomplete, ["error"])


def metric_frame():
    rows = []
    for estimator in ("prior_propagation", "trajectory_assimilation"):
        for probe in ("none", "gentle_150"):
            for episode in range(2):
                mse = float((episode+1)**2)*(1 if estimator == "prior_propagation" else .5)
                rows.append(dict(preparation_seed=31001, normal_world="maxwell", friction_world="boucwen",
                    sensor="noisy", calibration_noise="clean", filter_kind="joint_sensor_likelihood",
                    forecast_start="filtered_particles", estimator=estimator, probe=probe, episode=episode,
                    preparation_index=episode, model_seed=1, future_position_mse_m2=mse,
                    future_position_rmse_m=np.sqrt(mse), future_friction_mse_N2=mse,
                    probe_time_s=0 if probe == "none" else .15,
                    total_absolute_actuator_work_J=1 if probe == "none" else 2,
                    probe_absolute_actuator_work_J=0 if probe == "none" else 1,
                    native_guard_count=int(episode == 1)))
    return pd.DataFrame(rows)


def test_summary_uses_sqrt_mean_mse_and_retains_guards():
    summary, status = analysis.summaries(metric_frame(), 128)
    row = summary[(summary.probe == "none")&(summary.estimator == "prior_propagation")].iloc[0]
    assert row.future_position_mse_m2 == 2.5
    assert row.pooled_future_position_rmse_m == pytest.approx(np.sqrt(2.5))
    assert row.mean_episode_position_rmse_m == 1.5
    assert row.native_guard_steps == 1
    assert not summary.interval_available.any()
    assert set(summary.probe) == {"none", "gentle_150"}


def test_paired_gain_requires_identical_physical_costs():
    frame = metric_frame()
    paired, summary = analysis.paired_gains(frame, 128)
    assert np.all(paired.prior_minus_assimilated_future_position_mse_m2 > 0)
    assert not summary.interval_available.any()
    frame.loc[frame.estimator == "trajectory_assimilation", "total_absolute_actuator_work_J"] += 1
    with pytest.raises(AssertionError):
        analysis.paired_gains(frame, 128)


def test_duplicate_prediction_identity_rejected_before_loading_outputs(tmp_path):
    frame = metric_frame()
    duplicated = pd.concat((frame, frame.iloc[:1]), ignore_index=True)
    duplicated.to_csv(tmp_path/"episodes.csv", index=False)
    with pytest.raises(ValueError, match="duplicate prediction"):
        analysis.verify_and_join_metrics(tmp_path, tmp_path, {}, pd.DataFrame(), {})
