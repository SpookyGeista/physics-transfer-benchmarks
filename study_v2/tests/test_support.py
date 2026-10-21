"""Checks for calibration-only distances and recorded exact-profile correspondence."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from decision_study.common import features, history_init, NI, FI


spec = importlib.util.spec_from_file_location('v2_support', Path(__file__).resolve().parents[1]/'scripts/support.py')
support = importlib.util.module_from_spec(spec); spec.loader.exec_module(support)


def recorded_fixture():
    t = np.arange(7)
    f, d, r, v, n, ft, rows = [], [], [], [], [], [], []
    for mode in ('B', 'A'):
        for profile in range(2):
            delta = .02 + (.001*(profile+1)*t/7 if mode == 'B' else np.zeros(7))
            velocity = .01*(profile+1)*t if mode == 'A' else np.zeros(7)
            rate = np.full(7, .01*(profile+1)) if mode == 'B' else np.zeros(7)
            normal = 10+profile+.1*t
            friction = -normal*.3*np.tanh(velocity/.04) if mode == 'A' else np.zeros(7)
            h = history_init(delta, velocity, normal)
            f.append(features(delta, rate, velocity, h, normal)); d.append(delta); r.append(rate)
            v.append(velocity); n.append(normal); ft.append(friction)
            rows.append(dict(id=f'train-r0-p{profile}-{mode}', profile=profile, mode=mode, amplitude=0.))
    train = dict(features=np.array(f), delta=np.array(d), rate=np.array(r), normal=np.array(n), friction=np.array(ft),
                 state=np.stack([np.zeros_like(v), v], -1))
    replay = {key: [] for key in ('features', 'delta', 'compression_rate', 'normal', 'friction', 'state')}
    replay_rows = []
    amplitudes = [-.006, .006]
    for amplitude in amplitudes:
        for profile in range(2):
            rf = train['features'][profile].copy(); rf[:, FI] = train['features'][profile+2][:, FI]
            replay['features'].append(rf); replay['delta'].append(train['delta'][profile])
            replay['compression_rate'].append(train['rate'][profile]); replay['normal'].append(train['normal'][profile])
            replay['friction'].append(train['friction'][profile+2]); replay['state'].append(train['state'][profile+2])
            replay_rows.append(dict(id=f'train-r0-p{profile}-cam{amplitude:+.3f}', mode='AB', profile=profile, amplitude=amplitude))
    return train, rows, {key: np.array(value) for key, value in replay.items()}, replay_rows, amplitudes


def test_exact_match_checks_measured_force_and_history():
    train, rows, replay, replay_rows, amplitudes = recorded_fixture()
    audits = support.audit_exact(train, rows, replay, replay_rows, amplitudes)
    assert len(audits) == 4 and all(row['exact_match'] for row in audits)
    replay['normal'][0, 3] += .001
    assert not support.audit_exact(train, rows, replay, replay_rows, amplitudes)[0]['exact_match']
    replay['normal'][0, 3] -= .001
    replay['features'][1, 4, 11] += .001
    assert not support.audit_exact(train, rows, replay, replay_rows, amplitudes)[1]['exact_match']


def test_metadata_order_and_profile_grid_are_required():
    train, rows, replay, replay_rows, amplitudes = recorded_fixture()
    replay_rows[0], replay_rows[1] = replay_rows[1], replay_rows[0]
    with pytest.raises(AssertionError, match='metadata order'):
        support.audit_exact(train, rows, replay, replay_rows, amplitudes)


def test_distance_threshold_uses_independent_validation_and_all_train_points():
    train, rows, replay, _, _ = recorded_fixture()
    val = {key: value.copy() for key, value in train.items()}
    val['features'] += .125
    val_rows = [{**row, 'id': row['id'].replace('train-', 'val-')} for row in rows]
    points, info = support.fit_arrays(train, rows, val, val_rows, [1, 2], .995)
    assert points['normal_b1'].shape == (7, 5)
    assert points['normal_b2'].shape == (14, 5)
    for key in info:
        expected = max(1e-6, float(np.quantile(points[key+'_val_distances'], .995, method='linear')))
        assert info[key]['threshold'] == expected
    diagnostics = support.episode_support(replay, points['normal_b2'], 1e-6, NI)
    assert all(row['coverage'] == 1 for row in diagnostics)
    with pytest.raises(AssertionError):
        support.fit_arrays(train, rows, val, rows, [2], .995)


def test_fit_forbids_replay_and_evaluation_requires_lock(tmp_path):
    with pytest.raises(ValueError, match='only isolated'):
        support.load_data(tmp_path, 'replay', fitting=True)
    with pytest.raises(AssertionError, match='freeze the protocol'):
        support.verify_lock(tmp_path/'missing.json', tmp_path)


def test_joint_controller_coverage_requires_both_heads():
    distances = np.array([[0., .2, .4], [.1, .3, .5]])
    summary = support.distance_summary(distances, .25)
    assert summary[0]['coverage'] == 2/3
    assert summary[1]['coverage'] == 1/3
    opposite = distances[:, ::-1]
    joint = np.mean((distances <= .25) & (opposite <= .25), axis=1)
    assert joint[0] == 1/3 and joint[1] == 0
