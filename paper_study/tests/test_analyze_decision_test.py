"""Pure synthetic audit contracts. No reserved material, predictions or scores."""
import importlib.util
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/analyze_decision_test.py'
spec = importlib.util.spec_from_file_location('independent_decision_test_analysis', PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def config():
    return json.loads(audit.CONFIG.read_text())


def physical():
    return json.loads((audit.ROOT / 'decision_test/config.json').read_text())


def raw_target():
    release, total = 2, 5
    state = np.arange(20, dtype=float).reshape(2, 5, 2) / 10
    normal = np.arange(10, dtype=float).reshape(2, 5)
    return dict(release_step=np.array(release), common_step=np.array(1),
        next_state=state, normal=normal, friction=-2 * normal,
        absolute_work_J=np.arange(10, dtype=float).reshape(2, 5) / 100,
        guard=np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=int))


def test_four_channel_timing_and_work_are_independent():
    raw = raw_target()
    target = np.concatenate((raw['next_state'][:, 2:], raw['normal'][:, 2:, None], raw['friction'][:, 2:, None]), axis=2)
    score = audit.branch_metrics(target + np.array([1., 2., 3., 4.]), raw, .0025)
    for key, value in zip(audit.CHANNELS, [1., 4., 9., 16.]):
        np.testing.assert_allclose(score[key], value)
    np.testing.assert_allclose(score['probe_seconds'], .005)
    np.testing.assert_allclose(score['probe_absolute_work_J'], [.01, .11])
    np.testing.assert_allclose(score['extra_probe_work_J'], [.01, .06])
    np.testing.assert_array_equal(score['native_guard_count'], [1, 0])
    with pytest.raises(ValueError, match='shape'):
        audit.branch_metrics(target[:, :-1], raw, .0025)


def test_uniform_expected_is_mean_loss_not_mean_trajectory():
    raw = raw_target()
    target = np.concatenate((raw['next_state'][:, 2:], raw['normal'][:, 2:, None], raw['friction'][:, 2:, None]), axis=2)
    left = target.copy(); right = target.copy()
    left[..., 0] += 2.; right[..., 0] -= 2.
    losses = {'left': audit.branch_metrics(left, raw, .0025), 'right': audit.branch_metrics(right, raw, .0025)}
    result = audit.weighted_metrics(losses, np.full((2, 2), .5), ['left', 'right'])
    np.testing.assert_allclose(result['position_mse_m2'], 4.)
    assert np.mean(((left + right) / 2 - target)**2) < 1e-28


def test_policy_feasibility_ties_fallback_and_seeded_uniform():
    cfg = config(); p = physical()
    plan = dict(risk_m2=np.array([[0., 0., 0.], [5., 1., 1.]]),
        information_nats=np.array([[0., 0., 0.], [0., 1., 2.]]),
        physical_risk_m2=np.array([[0., 0., 0.], [5., 1., 0.]]),
        extra_seconds=np.array([[0., .1, .1], [0., .1, .1]]),
        planned_extra_probe_work_J=np.array([[0., .0005, .006], [0., .0005, .006]]))
    first = audit.action_probabilities(plan, p, cfg['methods'])
    second = audit.action_probabilities(plan, p, cfg['methods'])
    for budget in cfg['budgets']:
        for method in cfg['methods']:
            np.testing.assert_array_equal(first[budget]['probabilities'][method], second[budget]['probabilities'][method])
            np.testing.assert_array_equal(first[budget]['probabilities'][method].sum(axis=1), 1.)
            assert not first[budget]['probabilities'][method][~first[budget]['eligible']].any()
    np.testing.assert_array_equal(first['B0']['probabilities']['bayes_risk'], [[1, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(first['B1']['probabilities']['fixed_continue100'], [[1, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(first['B3']['probabilities']['bayes_risk'], [[1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(first['B3']['probabilities']['information_gain'][1], [0, 0, 1])
    np.testing.assert_array_equal(first['B3']['probabilities']['uniform_expected'], np.full((2, 3), 1 / 3))


def test_keyed_comparison_detects_missing_duplicate_wrong_values():
    cfg = config()
    a = pd.DataFrame({'id': [1, 2], 'value': [0., 1.]})
    assert audit.compare_tables(a, a.iloc[::-1], ['id'], ['value'], cfg)['value'] == 0.
    with pytest.raises(ValueError, match='row identities'):
        audit.compare_tables(a, a.iloc[:1], ['id'], ['value'], cfg)
    with pytest.raises(ValueError, match='row identities'):
        audit.compare_tables(a, pd.concat([a, a.iloc[:1]]), ['id'], ['value'], cfg)
    b = a.copy(); b.loc[0, 'value'] = .01
    with pytest.raises(ValueError, match='mismatch'):
        audit.compare_tables(a, b, ['id'], ['value'], cfg)


def test_exact_cartesian_inventory_rejects_one_missing_policy():
    cfg = dict(config(), model_banks=[1], history_seeds=[55201], worlds=['maxwell'])
    values = itertools.product(cfg['model_banks'], cfg['history_seeds'], cfg['worlds'], cfg['worlds'], range(36), cfg['budgets'], cfg['methods'])
    frame = pd.DataFrame(values, columns=audit.KEY + ['budget', 'method'])
    audit.validate_grid(frame, cfg)
    with pytest.raises(ValueError, match='Cartesian'):
        audit.validate_grid(frame.iloc[:-1], cfg)


def test_mse_first_and_declared_contrasts_without_population_inference():
    rows = []
    for bank, history, method in itertools.product((1, 2), (55201, 55202), ('fixed_continue100', 'bayes_risk')):
        value = (1. if bank == 1 else 9.) + (2. if method == 'bayes_risk' else 0.)
        rows.append(dict(model_seed=bank, preparation_seed=history, normal_world='maxwell', friction_world='boucwen',
            budget='B3', method=method, **{key: value for key in audit.CHANNELS + audit.COSTS},
            real_extra_work_budget_violation_probability=0., planning_candidate_guard_count=0., planned_risk_m2=value))
    summary, contrasts = audit.summarize(pd.DataFrame(rows))
    reference = summary['pooled'].set_index('method').loc['fixed_continue100']
    assert reference.position_mse_m2 == 5.
    assert reference.pooled_position_rmse_um == np.sqrt(5.) * 1e6
    assert reference.mean_episode_position_rmse_um == 2e6
    delta = contrasts['pooled'].set_index('method')
    assert delta.loc['bayes_risk', 'position_mse_m2_difference_vs_fixed_continue100'] == 2.
    assert all(not col.startswith(('ci_', 'p_value')) and not col.endswith(('_ci_low', '_ci_high')) for col in delta.columns)


def test_native_geometry_ports_and_guards_on_synthetic_zero_position():
    cfg = physical(); rows = audit.expected_rows(55201)
    geometry = np.array([r['geometry'] for r in rows])
    slope = np.where(geometry == 2, 0., .25)[:, None]
    state = np.zeros((36, 120, 2), dtype=np.float32)
    state[..., 1] = .125  # exactly representable; no native/material history generated
    normal = np.full((36, 120), 10.)
    u = np.ones((36, 120)); ft = np.full_like(u, -1.)
    local_rate = np.broadcast_to(slope * .125, u.shape).copy()
    raw = dict(state=state, next_state=state.copy(), normal=normal, friction=ft,
        u=u, d=np.full_like(u, .02), d_dot=np.zeros_like(u), delta=np.full_like(u, .02),
        compression_rate=local_rate, net_force=u-slope*normal+ft,
        actuator_work_J=u*.125*.0025, normal_drive_work_J=np.zeros_like(u),
        friction_work_J=ft*.125*.0025, absolute_work_J=abs(u*.125)*.0025,
        guard=np.zeros_like(u, dtype=bool), common_step=np.array(20), release_step=np.array(20),
        observation_steps=np.array([0, 10, 20]))
    for key in ('preparation_seed', 'preparation_index', 'geometry', 'amplitude', 'excitation_sign', 'episode'):
        raw[key] = np.array([r[key] for r in rows])
    maxima = {}
    audit.check_native(raw, dict(rows=rows, guards=0), cfg, 'release', maxima)
    assert all(v == 0. for v in maxima.values())
    raw['compression_rate'][0, 0] += .001
    with pytest.raises(ValueError, match='compression_rate'):
        audit.check_native(raw, dict(rows=rows, guards=0), cfg, 'release', {})


def test_freeze_never_opens_reserved_results(tmp_path, monkeypatch):
    cfg = config()
    path = tmp_path / 'config.json'; path.write_text(json.dumps(cfg))
    script = tmp_path / 'known.py'; script.write_text('synthetic source')
    monkeypatch.setattr(audit, 'CONFIG', path)
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    monkeypatch.setattr(audit, 'parent_authorization', lambda _: {'config': dict(preparation_seeds=cfg['history_seeds'], model_seeds=cfg['model_banks'])})
    monkeypatch.setattr(audit, 'source_snapshot', lambda: {str(script): audit.sha(script)})
    original_read = audit.read
    def restricted_read(p):
        if Path(p) != path:
            pytest.fail('Freeze accessed a file beyond its known config/source: ' + str(p))
        return original_read(p)
    monkeypatch.setattr(audit, 'read', restricted_read)
    out = tmp_path / 'results/statistics'
    audit.freeze(out)
    locked = json.loads((out / 'analysis_lock.json').read_text())
    assert locked['reserved_scores_read'] is False
    with pytest.raises(ValueError, match='fresh'):
        audit.freeze(out)
