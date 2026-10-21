"""Prospectively locked independent row-level audit; never runs a simulation."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/decision-test-audit-mpl')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
import argparse
import hashlib
import itertools
import json
import time
import numpy as np
import pandas as pd

CONFIG = ROOT / 'configs/decision_test_analysis.json'
PLAN = ROOT / 'reports/decision_test_analysis_plan.md'
TESTS = ROOT / 'tests/test_analyze_decision_test.py'
KEY = ['model_seed', 'preparation_seed', 'normal_world', 'friction_world', 'episode']
CHANNELS = ['position_mse_m2', 'velocity_mse', 'normal_mse_N2', 'friction_mse_N2']
COSTS = ['probe_seconds', 'absolute_work_J', 'probe_absolute_work_J', 'extra_probe_work_J', 'native_guard_count']
GROUPS = {'pooled': ['budget', 'method'], 'by_bank': ['model_seed', 'budget', 'method'],
          'by_history': ['preparation_seed', 'budget', 'method'],
          'by_composition': ['normal_world', 'friction_world', 'budget', 'method'],
          'by_bank_history': ['model_seed', 'preparation_seed', 'budget', 'method']}
read = lambda p: json.loads(Path(p).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify(mapping):
    for path, digest in mapping.items():
        if sha(path) != digest:
            raise ValueError('Hash mismatch: ' + str(path))


def source_snapshot():
    return {str(p): sha(p) for p in (Path(__file__).resolve(), CONFIG, PLAN, TESTS)}


def parent_authorization(cfg):
    path = REPO / cfg['parent_authorization']
    if sha(path) != cfg['parent_authorization_sha256']:
        raise ValueError('Wrong prospective parent authorization')
    parent = read(path)
    if parent['study_id'] != cfg['parent_study_id'] or parent['status'] != 'locked':
        raise ValueError('Parent study is not locked')
    verify(parent['sources_sha256'])
    verify(parent['review_sha256'])
    return parent


def freeze(out):
    """No native, forecast, plan-value or result-table access in this function."""
    cfg = read(CONFIG)
    parent = parent_authorization(cfg)
    if parent['config']['preparation_seeds'] != cfg['history_seeds'] or parent['config']['model_seeds'] != cfg['model_banks']:
        raise ValueError('Analysis design differs from the parent factorial')
    out = Path(out).resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'results') or out == ROOT / 'results':
        raise ValueError('A fresh named analysis directory under results is required')
    sources = source_snapshot()
    out.mkdir(parents=True)
    write_json(out / 'analysis_lock.json', dict(study_id=cfg['study_id'], config=cfg,
        created_unix=time.time(), sources_sha256=sources, reserved_scores_read=False,
        parent_authorization_sha256=cfg['parent_authorization_sha256'],
        scope='Independent descriptive audit; all policies and budgets; no success or population test'))
    print('ANALYSIS LOCKED', sha(out / 'analysis_lock.json'), flush=True)


def verify_analysis(out):
    lock = read(Path(out) / 'analysis_lock.json')
    if lock['config'] != read(CONFIG) or lock['sources_sha256'] != source_snapshot() or lock['reserved_scores_read'] is not False:
        raise ValueError('Independent pre-score analysis lock changed')
    parent_authorization(lock['config'])
    return lock


def expected_cells(cfg):
    return [(s, n, f) for s in cfg['history_seeds'] for n in cfg['worlds'] for f in cfg['worlds']]


def expected_rows(seed):
    return [dict(episode=e, preparation_seed=seed, preparation_index=p, geometry=g,
                 amplitude=.003, excitation_sign=s, episode_id=f'decision-p{seed}-h{p}-g{g}-s{s:+d}')
            for e, (p, g, s) in enumerate(itertools.product(range(6), range(3), (-1, 1)))]


def action_probabilities(plan, physical, methods):
    """Independent finite-policy reconstruction; consumes no native target."""
    n = len(plan['risk_m2'])
    uniform = np.random.default_rng(physical['policy_seed']).random(n)
    result = {}
    for budget in physical['budgets']:
        eligible = ((plan['extra_seconds'] / budget['time_scale_s'] <= budget['max_scaled_time'] + 1e-14) &
                    (plan['planned_extra_probe_work_J'] / budget['work_scale_J'] <= budget['max_scaled_work'] + 1e-14))
        if not eligible[:, 0].all():
            raise ValueError('Release must always be eligible')
        probabilities = {m: np.zeros((n, 3)) for m in methods}
        for e in range(n):
            allowed = np.flatnonzero(eligible[e])
            for a, name in enumerate(('release', 'hold100', 'continue100')):
                probabilities['fixed_' + name][e, a if eligible[e, a] else 0] = 1.
            p = eligible[e].astype(float) / len(allowed)
            probabilities['uniform_expected'][e] = p
            chosen = min(int(np.count_nonzero(uniform[e] > np.cumsum(p))), 2)
            probabilities['uniform'][e, chosen] = 1.
            for method, values in [('bayes_risk', plan['risk_m2']),
                                   ('information_gain', -plan['information_nats']),
                                   ('physical_only', plan['physical_risk_m2'])]:
                chosen = sorted(allowed, key=lambda a: (values[e, a], plan['extra_seconds'][e, a],
                                plan['planned_extra_probe_work_J'][e, a], a))[0]
                probabilities[method][e, chosen] = 1.
        result[budget['name']] = dict(eligible=eligible, probabilities=probabilities)
    return result


def branch_metrics(prediction, raw, dt):
    release, common = int(raw['release_step']), int(raw['common_step'])
    target = np.concatenate((raw['next_state'][:, release:], raw['normal'][:, release:, None],
                             raw['friction'][:, release:, None]), axis=2)
    if target.shape != prediction.shape or not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('Finite complete forecast/target shape required')
    mse = np.mean(np.square(prediction - target), axis=1)
    return dict(zip(CHANNELS, mse.T)) | dict(
        probe_seconds=np.full(len(target), release * dt),
        absolute_work_J=raw['absolute_work_J'].sum(axis=1),
        probe_absolute_work_J=raw['absolute_work_J'][:, :release].sum(axis=1),
        extra_probe_work_J=raw['absolute_work_J'][:, common:release].sum(axis=1),
        native_guard_count=raw['guard'].sum(axis=1))


def weighted_metrics(branches, weights, names):
    """Average branch losses, never branch trajectories."""
    return {key: sum(weights[:, i] * branches[name][key] for i, name in enumerate(names))
            for key in branches[names[0]]}


def compare_tables(actual, reported, keys, columns, cfg):
    a, b = actual.set_index(keys).sort_index(), reported.set_index(keys).sort_index()
    if not a.index.is_unique or not b.index.is_unique or not a.index.equals(b.index):
        raise ValueError('Missing, duplicate or unequal row identities')
    maximum = {}
    for column in columns:
        x, y = a[column].to_numpy(float), b[column].to_numpy(float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite result field: ' + column)
        maximum[column] = float(np.max(abs(x - y)))
        if not np.allclose(x, y, rtol=cfg['float_comparison_rtol'], atol=cfg['float_comparison_atol']):
            raise ValueError('Raw/result-table mismatch: ' + column + ' ' + str(maximum[column]))
    return maximum


def validate_grid(frame, cfg, branch=False):
    columns = KEY + (['branch'] if branch else ['budget', 'method'])
    levels = [cfg['model_banks'], cfg['history_seeds'], cfg['worlds'], cfg['worlds'], range(36)]
    levels += [cfg['branches']] if branch else [cfg['budgets'], cfg['methods']]
    expected = pd.MultiIndex.from_product(levels, names=columns).sort_values()
    actual = pd.MultiIndex.from_frame(frame[columns]).sort_values()
    if not actual.is_unique or not actual.equals(expected):
        raise ValueError('Exact complete Cartesian row inventory required')


def check_native(raw, metadata, physical, branch, maxima):
    release = 20 if branch == 'release' else 60
    steps = release + 100
    if raw['u'].shape != (36, steps) or raw['next_state'].shape != (36, steps, 2):
        raise ValueError('Native axes are incomplete')
    if int(raw['release_step']) != release or int(raw['common_step']) != 20 or not np.array_equal(raw['observation_steps'], np.arange(0, release + 1, 10)):
        raise ValueError('Wrong native/probe clocks')
    for key in ('preparation_seed', 'preparation_index', 'geometry', 'amplitude', 'excitation_sign', 'episode'):
        if not np.array_equal(raw[key], [r[key] for r in metadata['rows']]):
            raise ValueError('Native factor mismatch: ' + key)
    x, v = raw['state'][..., 0].astype(float), raw['state'][..., 1]
    z, amp, geom = x / .012, raw['amplitude'][:, None], raw['geometry'][:, None]
    g = np.where(geom == 0, amp * np.tanh(z), np.where(geom == 1, amp * np.sin(z), amp * z*z / (1 + z*z)))
    slope = np.where(geom == 0, amp / .012 * (1 - np.tanh(z)**2),
        np.where(geom == 1, amp / .012 * np.cos(z), amp / .012 * 2*z / (1 + z*z)**2))
    dt = physical['dt']
    pairs = {'delta': (raw['delta'], raw['d'] + g),
        'compression_rate': (raw['compression_rate'], raw['d_dot'] + slope * v),
        'net_force': (raw['net_force'], raw['u'] - slope * raw['normal'] + raw['friction']),
        'actuator_work_J': (raw['actuator_work_J'], raw['u'] * v * dt),
        'normal_drive_work_J': (raw['normal_drive_work_J'], raw['normal'] * raw['d_dot'] * dt),
        'friction_work_J': (raw['friction_work_J'], raw['friction'] * v * dt),
        'absolute_work_J': (raw['absolute_work_J'], (abs(raw['u'] * v) + abs(raw['normal'] * raw['d_dot'])) * dt)}
    for name, (a, b) in pairs.items():
        maximum = float(np.max(abs(a - b)))
        maxima[name] = max(maxima.get(name, 0.), maximum)
        if not np.isfinite(a).all() or not np.allclose(a, b, rtol=2e-11, atol=2e-13):
            raise ValueError('Native physical coupling/power inconsistency: ' + name)
    bounds = physical['guard']
    guard = (~np.isfinite(raw['next_state']).all(axis=2)) | (~np.isfinite(raw['net_force'])) | (
        abs(raw['next_state'][..., 0]) > bounds['max_abs_position_m']) | (
        abs(raw['next_state'][..., 1]) > bounds['max_abs_velocity_m_s']) | (raw['normal'] < bounds['min_normal_N'])
    if not np.array_equal(guard, raw['guard']) or int(guard.sum()) != metadata['guards']:
        raise ValueError('Native guard accounting mismatch')


def verify_complete_inputs(out, cfg):
    """Only called after our analysis lock exists and root reports completion."""
    parent = parent_authorization(cfg)
    evaluation, native = REPO / cfg['evaluation_directory'], REPO / cfg['native_directory']
    status = read(evaluation / 'process.json')
    if status['status'] != 'complete' or status['policy_rows'] != 294912 or status['branch_rows'] != 27648:
        raise ValueError('Wait for all eight banks and complete original aggregation')
    elock_path = evaluation / 'evaluation_lock.json'
    elock, digest = read(elock_path), sha(elock_path)
    if elock['test_lock_sha256'] != cfg['parent_authorization_sha256'] or elock['config'] != parent['config']:
        raise ValueError('Evaluation/parent source binding differs')
    if Path(elock['data_directory']) != native or status['evaluation_lock_sha256'] != digest:
        raise ValueError('Wrong native data or aggregation binding')
    expected_triplets = [dict(seed=s, normal=n, friction=f) for s, n, f in expected_cells(cfg)]
    if elock['triplets'] != expected_triplets:
        raise ValueError('Incomplete evaluation triplet inventory')
    bindings = dict(elock['inputs_sha256'])
    verify(bindings)
    analysis_manifest_path = evaluation / 'analysis/artifact_manifest.json'
    am = read(analysis_manifest_path)
    if status['analysis_manifest_sha256'] != sha(analysis_manifest_path) or am['evaluation_lock_sha256'] != digest:
        raise ValueError('Wrong original aggregate manifest')
    verify(am['inputs_sha256']); verify(am['outputs_sha256'])
    bindings.update(am['inputs_sha256']); bindings.update(am['outputs_sha256'])
    bindings.update({str(p): sha(p) for p in (elock_path, evaluation / 'process.json', analysis_manifest_path)})
    native_status = read(native / 'process.json')
    if (native_status['status'], native_status['triplets'], native_status['contexts'], native_status['native_branches']) != ('complete', 32, 1152, 3456):
        raise ValueError('Incomplete native process')
    expected = {native / f'p{s}/{n}_{f}/{b}.npz' for s, n, f in expected_cells(cfg) for b in cfg['branches']}
    if set(native.rglob('*.npz')) != expected:
        raise ValueError('Native branch file inventory differs')
    attempts = {}
    retained = []
    for bank in cfg['model_banks']:
        bankdir = evaluation / f'banks/r{bank}'
        if (bankdir / '.active_worker').exists():
            raise ValueError('A bank worker remains active')
        pointer = read(bankdir / 'process.json')
        if pointer['status'] != 'complete' or pointer['evaluation_lock_sha256'] != digest:
            raise ValueError('Incomplete or incorrectly bound bank')
        if pointer['attempt_index'] != len(pointer['attempts']):
            raise ValueError('Attempt history is not complete')
        expected_attempts = {bankdir / 'attempts' / f'attempt_{i:03d}' for i in range(1, pointer['attempt_index'] + 1)}
        if set((bankdir / 'attempts').iterdir()) != expected_attempts:
            raise ValueError('Unexpected or omitted attempt directory')
        for i, binding in enumerate(pointer['attempts'], 1):
            attempt = bankdir / 'attempts' / f'attempt_{i:03d}'
            if {str(p) for p in attempt.rglob('*') if p.is_file()} != set(binding['outputs_sha256']):
                raise ValueError('An attempt artifact is absent from its retained manifest')
            verify(binding['outputs_sha256']); bindings.update(binding['outputs_sha256'])
            ap = read(attempt / 'process.json')
            if ap['evaluation_lock_sha256'] != digest or ap['attempt_index'] != i:
                raise ValueError('Attempt identity mismatch')
            retained.append(dict(bank=bank, attempt=i, status=ap['status'], seconds=ap['seconds'], error=ap.get('error')))
        selected = Path(pointer['selected_attempt'])
        if selected != bankdir / 'attempts' / f'attempt_{pointer["attempt_index"]:03d}':
            raise ValueError('Only the final complete attempt may supply scores')
        marker = read(selected / 'planning_complete.json')
        if marker['plans'] != 32 or marker['future_targets_entered_scoring'] is not False or marker['later_sensors_entered_assimilation'] is not False:
            raise ValueError('Full pre-assimilation/scoring plan marker is missing')
        plans = {selected / f'p{s}/{n}_{f}/planned.npz' for s, n, f in expected_cells(cfg)}
        if {str(p) for p in plans} != set(marker['plans_sha256']):
            raise ValueError('Missing or extra cell plan binding')
        verify(marker['plans_sha256'])
        attempts[bank] = selected
    verify(bindings)
    return parent['config'], attempts, bindings, retained


def summarize(frame):
    metrics = [*CHANNELS, *COSTS, 'real_extra_work_budget_violation_probability',
               'planning_candidate_guard_count', 'planned_risk_m2']
    summaries, contrasts = {}, {}
    for name, keys in GROUPS.items():
        group = frame.groupby(keys, sort=True)
        table = group[metrics].mean().reset_index()
        table['pooled_position_rmse_um'] = 1e6 * np.sqrt(table.position_mse_m2)
        rms = frame.assign(mean_episode_position_rmse_um=1e6 * np.sqrt(frame.position_mse_m2))
        table = table.merge(rms.groupby(keys).mean_episode_position_rmse_um.mean().reset_index(), on=keys, validate='one_to_one')
        summaries[name] = table
        join = [key for key in keys if key != 'method']
        reference = table[table.method == 'fixed_continue100'][join + metrics]
        merged = table.merge(reference, on=join, suffixes=('', '_reference'), validate='many_to_one')
        contrast = merged[keys].copy()
        for metric in metrics:
            contrast[metric + '_difference_vs_fixed_continue100'] = merged[metric] - merged[metric + '_reference']
        contrasts[name] = contrast
    return summaries, contrasts


def figures(out, summaries, contrasts, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    short = {'fixed_release': 'Release', 'fixed_hold100': 'Hold', 'fixed_continue100': 'Continue',
        'uniform': 'Uniform draw', 'uniform_expected': 'Uniform mean', 'bayes_risk': 'Bayes risk',
        'information_gain': 'Information', 'physical_only': 'Physical only'}
    pooled = summaries['pooled']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for budget, axis in zip(cfg['budgets'], axes.flat):
        values = pooled[pooled.budget == budget].set_index('method').loc[cfg['methods']]
        axis.barh(range(8), 1e12 * values.position_mse_m2, color='#426a8b')
        axis.set(yticks=range(8), yticklabels=[short[m] for m in cfg['methods']],
                 xlabel='Pooled future position MSE (µm²)', title=budget)
        axis.invert_yaxis()
    fig.savefig(out / 'position_mse_all_budgets.png', dpi=180); plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for method in cfg['methods']:
        values = pooled[pooled.method == method].sort_values('budget')
        axis.plot(1000 * values.extra_probe_work_J, 1e12 * values.position_mse_m2, 'o-', label=short[method])
    axis.set(xlabel='Mean realized additional probe work (mJ)', ylabel='Pooled future position MSE (µm²)')
    axis.legend(fontsize=8)
    fig.savefig(out / 'work_and_forecast_loss.png', dpi=180); plt.close(fig)
    values = pooled.pivot(index='budget', columns='method', values='real_extra_work_budget_violation_probability').loc[cfg['budgets'], cfg['methods']]
    fig, axis = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    im = axis.imshow(values, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    axis.set(xticks=range(8), xticklabels=[short[m] for m in cfg['methods']], yticks=range(4), yticklabels=cfg['budgets'])
    axis.tick_params(axis='x', rotation=25)
    fig.colorbar(im, ax=axis, label='Realized additional-work violation probability')
    fig.savefig(out / 'work_budget_violations.png', dpi=180); plt.close(fig)
    delta = contrasts['by_bank_history']
    delta = delta[delta.method == 'bayes_risk']
    metric = 'position_mse_m2_difference_vs_fixed_continue100'
    limit = max(float(abs(delta[metric]).max() * 1e12), 1e-30)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for budget, axis in zip(cfg['budgets'], axes.flat):
        values = delta[delta.budget == budget].pivot(index='model_seed', columns='preparation_seed', values=metric)
        im = axis.imshow(values * 1e12, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
        axis.set(title=budget, xlabel='History block', ylabel='Fitted bank',
                 xticks=range(8), xticklabels=range(1, 9), yticks=range(8), yticklabels=range(1, 9))
    fig.colorbar(im, ax=list(axes.flat), label='Bayes-risk − fixed-continue position MSE (µm²); positive is worse')
    fig.savefig(out / 'bank_history_contrasts.png', dpi=180); plt.close(fig)


def run(out, report_prefix):
    out, report_prefix = Path(out).resolve(), Path(report_prefix).resolve()
    analysis = verify_analysis(out)
    cfg = analysis['config']
    if (out / 'process.json').exists():
        raise FileExistsError('No automatic overwrite/retry of an analysis attempt')
    if any(report_prefix.with_suffix(ext).exists() for ext in ('.md', '.json')):
        raise FileExistsError('Use fresh final review report paths')
    started = time.perf_counter()
    process = dict(status='running', pid=os.getpid(), started_unix=time.time(),
                   analysis_lock_sha256=sha(out / 'analysis_lock.json'), command=sys.argv)
    write_json(out / 'process.json', process)
    branch_records, policy_records, checks, native_cache = [], [], [], {}
    physical_maxima = {}
    try:
        physical, attempts, bindings, retained = verify_complete_inputs(out, cfg)
        native, evaluation = REPO / cfg['native_directory'], REPO / cfg['evaluation_directory']
        native_guard_total = 0
        # Native raw values are checked once; the same unchanged physical data serves every bank.
        for seed, nw, fw in expected_cells(cfg):
            relative = Path(f'p{seed}/{nw}_{fw}')
            current = {}
            for branch in cfg['branches']:
                path = native / relative / (branch + '.npz')
                meta = read(path.with_suffix('.json'))
                if (meta['rows'] != expected_rows(seed) or meta['status'] != 'complete'
                        or meta['test_lock_sha256'] != cfg['parent_authorization_sha256']
                        or meta['dt'] != .0025 or meta['mass_kg'] != 2. or meta['npz_sha256'] != sha(path)):
                    raise ValueError('Native metadata/factor/clock mismatch')
                with np.load(path, allow_pickle=False) as z:
                    raw = {k: z[k].copy() for k in z.files}
                check_native(raw, meta, physical, branch, physical_maxima)
                native_guard_total += int(raw['guard'].sum())
                current[branch] = raw
            for branch in cfg['branches']:
                for key in ('state', 'next_state', 'normal', 'friction', 'u', 'd', 'd_dot', 'net_force', 'absolute_work_J', 'guard'):
                    if not np.array_equal(current[branch][key][:, :20], current['release'][key][:, :20]):
                        raise ValueError('Common physical prefix differs across branch bodies')
                if not np.array_equal(current[branch]['observations'][:, :3], current['release']['observations']):
                    raise ValueError('Common sensors differ across branches')
            native_cache[(seed, nw, fw)] = current
        for bank in cfg['model_banks']:
            attempt = attempts[bank]
            reported_branch = pd.read_csv(attempt / 'branch_episodes.csv')
            reported_policy = pd.read_csv(attempt / 'policy_episodes.csv')
            localcfg = dict(cfg, model_banks=[bank])
            validate_grid(reported_branch, localcfg, branch=True); validate_grid(reported_policy, localcfg)
            bank_branches, bank_policies = [], []
            for seed, nw, fw in expected_cells(cfg):
                if time.perf_counter() - started > cfg['max_wall_seconds']:
                    raise TimeoutError('Independent analysis CPU wall guard')
                relative = Path(f'p{seed}/{nw}_{fw}'); cell = attempt / relative
                pm = read(cell / 'planned.json')
                if (pm['model_seed'] != bank or pm['particles'] != 201 or pm['fantasies'] != 1024
                        or pm['status'] != 'planned_before_later_sensor_assimilation_or_target_scoring'
                        or pm['npz_sha256'] != sha(cell / 'planned.npz')):
                    raise ValueError('Incorrect persisted plan metadata')
                with np.load(cell / 'planned.npz', allow_pickle=False) as z:
                    plan = {k: z[k].copy() for k in z.files}
                with np.load(cell / 'forecasts.npz', allow_pickle=False) as z:
                    predictions = {k: z[k].copy() for k in z.files}
                for weights in [plan['decision_weights'], *(predictions[b + '_weights'] for b in cfg['branches'])]:
                    if weights.shape != (36, 201) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1., rtol=0., atol=cfg['posterior_probability_atol']):
                        raise ValueError('Invalid posterior/decision weights')
                values = action_probabilities(plan, physical, cfg['methods'])
                metrics, candidate_guards = {}, {}
                selection = reported_branch[(reported_branch.preparation_seed == seed) &
                    (reported_branch.normal_world == nw) & (reported_branch.friction_world == fw)]
                for a, branch in enumerate(cfg['branches']):
                    metrics[branch] = branch_metrics(predictions[branch + '_forecast'], native_cache[(seed, nw, fw)][branch], physical['dt'])
                    rows = selection[selection.branch == branch].sort_values('episode')
                    guard = rows.planning_candidate_guard_count.to_numpy(float)
                    if guard.shape != (36,) or not np.isfinite(guard).all() or np.any(guard < 0) or np.any(guard != np.round(guard)):
                        raise ValueError('Invalid frozen candidate guard count')
                    candidate_guards[branch] = guard
                    for e in range(36):
                        identity = dict(model_seed=bank, preparation_seed=seed, normal_world=nw, friction_world=fw, episode=e)
                        record = dict(**identity, branch=branch, **{k: float(v[e]) for k, v in metrics[branch].items()},
                            planned_risk_m2=float(plan['risk_m2'][e, a]), planned_information_nats=float(plan['information_nats'][e, a]),
                            planned_extra_work_J=float(plan['planned_extra_probe_work_J'][e, a]), fantasy_mcse_m2=float(plan['fantasy_risk_mcse_m2'][e, a]),
                            planning_candidate_guard_count=float(guard[e]))
                        bank_branches.append(record)
                for budget in physical['budgets']:
                    name = budget['name']
                    if not np.array_equal(plan[name + '_eligible'], values[name]['eligible']):
                        raise ValueError('Stored/reconstructed eligibility differs')
                    over = np.stack([metrics[b]['extra_probe_work_J'] / budget['work_scale_J'] > budget['max_scaled_work'] + 1e-14 for b in cfg['branches']], axis=1)
                    for method in cfg['methods']:
                        probability = values[name]['probabilities'][method]
                        if not np.array_equal(probability, plan[name + '__' + method]):
                            raise ValueError('Stored/reconstructed policy probability differs')
                        weighted = weighted_metrics(metrics, probability, cfg['branches'])
                        for e in range(36):
                            identity = dict(model_seed=bank, preparation_seed=seed, normal_world=nw, friction_world=fw, episode=e)
                            record = dict(**identity, budget=name, method=method,
                                **{k: float(v[e]) for k, v in weighted.items()},
                                probability_release=float(probability[e, 0]), probability_hold100=float(probability[e, 1]), probability_continue100=float(probability[e, 2]),
                                chosen_branch=cfg['branches'][int(np.argmax(probability[e]))] if np.count_nonzero(probability[e]) == 1 else 'probability_mixture',
                                real_extra_work_budget_violation_probability=float(probability[e] @ over[e]),
                                planned_risk_m2=float(probability[e] @ plan['risk_m2'][e]),
                                planning_candidate_guard_count=float(sum(probability[e, a] * candidate_guards[b][e] for a, b in enumerate(cfg['branches']))))
                            bank_policies.append(record)
                # Every stored per-cell table is independently checked as well as bank totals.
                cell_branch, cell_policy = pd.DataFrame(bank_branches[-108:]), pd.DataFrame(bank_policies[-1152:])
                checks.append(compare_tables(cell_branch, pd.read_csv(cell / 'branch_episodes.csv'), KEY + ['branch'],
                    CHANNELS + COSTS + ['planned_risk_m2', 'planned_information_nats', 'planned_extra_work_J', 'fantasy_mcse_m2'], cfg))
                checks.append(compare_tables(cell_policy, pd.read_csv(cell / 'policy_episodes.csv'), KEY + ['budget', 'method'],
                    CHANNELS + COSTS + ['real_extra_work_budget_violation_probability', 'planning_candidate_guard_count', 'planned_risk_m2'], cfg))
            a, b = pd.DataFrame(bank_branches), pd.DataFrame(bank_policies)
            checks.append(compare_tables(a, reported_branch, KEY + ['branch'], CHANNELS + COSTS, cfg))
            checks.append(compare_tables(b, reported_policy, KEY + ['budget', 'method'], CHANNELS + COSTS +
                ['probability_release', 'probability_hold100', 'probability_continue100', 'planning_candidate_guard_count', 'real_extra_work_budget_violation_probability'], cfg))
            if not b.set_index(KEY + ['budget', 'method']).chosen_branch.sort_index().equals(reported_policy.set_index(KEY + ['budget', 'method']).chosen_branch.sort_index()):
                raise ValueError('Chosen-branch labels disagree with plan probabilities')
            branch_records.extend(bank_branches); policy_records.extend(bank_policies)
            print('INDEPENDENT AUDIT bank', bank, 'complete', flush=True)
        branches, frame = pd.DataFrame(branch_records), pd.DataFrame(policy_records)
        validate_grid(branches, cfg, branch=True); validate_grid(frame, cfg)
        checks.append(compare_tables(branches, pd.read_csv(evaluation / 'analysis/branch_episodes.csv'), KEY + ['branch'], CHANNELS + COSTS, cfg))
        checks.append(compare_tables(frame, pd.read_csv(evaluation / 'analysis/policy_episodes.csv'), KEY + ['budget', 'method'], CHANNELS + COSTS +
            ['real_extra_work_budget_violation_probability', 'planning_candidate_guard_count'], cfg))
        summaries, contrasts = summarize(frame)
        for name, table in summaries.items():
            reported = pd.read_csv(evaluation / 'analysis' / (name + '.csv'))
            common_columns = [c for c in table if c in reported and c not in GROUPS[name]]
            checks.append(compare_tables(table, reported, GROUPS[name], common_columns, cfg))
            table.to_csv(out / (name + '.csv'), index=False)
            contrasts[name].to_csv(out / ('contrasts_' + name + '.csv'), index=False)
        branches.to_csv(out / 'branch_episodes_recomputed.csv', index=False)
        frame.to_csv(out / 'policy_episodes_recomputed.csv', index=False)
        figures(out, summaries, contrasts, cfg)
        maxima = {key: max(v.get(key, 0.) for v in checks) for key in set().union(*(c.keys() for c in checks))}
        result = dict(status='complete', analysis_lock_sha256=sha(out / 'analysis_lock.json'),
            parent_authorization_sha256=cfg['parent_authorization_sha256'], input_hashes_verified=len(bindings),
            expected_branch_rows=27648, actual_branch_rows=len(branches), expected_policy_rows=294912, actual_policy_rows=len(frame),
            plan_count=256, history_blocks=8, model_banks=8, physical_contexts=1152,
            raw_csv_max_absolute_errors=maxima, native_physical_max_absolute_errors=physical_maxima,
            native_guard_count=native_guard_total, maximum_candidate_guard_count=float(branches.planning_candidate_guard_count.max()),
            retained_attempts=retained, pooled=summaries['pooled'].to_dict('records'),
            pooled_contrasts=contrasts['pooled'].to_dict('records'), no_population_confidence_intervals=True,
            no_positive_success_gate=True, candidate_guard_scope='Frozen reported candidate counts; policy weighting checked, no independent candidate rollout',
            seconds=time.perf_counter() - started, command=sys.argv)
        verify(bindings); verify_analysis(out)
        write_json(out / 'input_binding.json', dict(inputs_sha256=bindings))
        write_json(out / 'audit.json', result)
        report_prefix.parent.mkdir(parents=True, exist_ok=True)
        write_json(report_prefix.with_suffix('.json'), result)
        write_report(report_prefix.with_suffix('.md'), out, result, summaries, cfg)
        process.update(status='complete', policy_rows=len(frame), branch_rows=len(branches))
    except BaseException as exc:
        process.update(status='failed', error=repr(exc), complete_policy_rows=len(policy_records), complete_branch_rows=len(branch_records))
        raise
    finally:
        process['seconds'] = time.perf_counter() - started
        temporary = out / 'process.final.json'
        write_json(temporary, process); temporary.replace(out / 'process.json')
        files = [p for p in out.iterdir() if p.is_file() and p.name != 'artifact_manifest.json']
        reports = [report_prefix.with_suffix(ext) for ext in ('.json', '.md') if report_prefix.with_suffix(ext).exists()]
        write_json(out / 'artifact_manifest.json', dict(analysis_lock_sha256=sha(out / 'analysis_lock.json'),
            outputs_sha256={str(p): sha(p) for p in files + reports}))


def write_report(path, out, result, summaries, cfg):
    lines = ['# Independent held-out decision audit', '',
        f"All **{result['actual_policy_rows']:,} policy rows and {result['actual_branch_rows']:,} forecast-branch rows** were independently recomputed. "
        f"All {result['input_hashes_verified']:,} recorded input/artifact hashes verify. "
        'The complete eight-bank, eight-history-block crossed design is retained. Primary numbers below are pooled MSE; no CI or positive-success criterion is used.', '',
        '| Budget | Method | Position MSE (m²) | sqrt pooled MSE (µm) | Extra work (mJ) | Violation probability |',
        '|---|---|---:|---:|---:|---:|']
    for budget in cfg['budgets']:
        values = summaries['pooled'][summaries['pooled'].budget == budget].set_index('method').loc[cfg['methods']]
        for method, r in values.iterrows():
            lines.append(f'| {budget} | {method} | {r.position_mse_m2:.6g} | {r.pooled_position_rmse_um:.3f} | {1000*r.extra_probe_work_J:.4f} | {r.real_extra_work_budget_violation_probability:.4f} |')
    lines += ['', f"Native guard events: {result['native_guard_count']}. Maximum frozen candidate guard count per forecast branch: {result['maximum_candidate_guard_count']:.0f}. "
        'The latter counts are imported from the immutable branch records; their policy weighting is checked, but candidate rollouts were not independently regenerated.', '',
        'Every forecast channel, native clock/factor, common physical/sensor prefix, work term, eligibility and policy probability was checked. '
        'Loss-weighted randomized policies are distinguished from averaged trajectories. '
        'All per-cell, per-bank and aggregate score tables were compared. The JSON records actual numerical discrepancies, retained attempts and resource costs.', '',
        'Interpretation is limited to the tested simulator, known constitutive equation prior, public fixture recipe population and finite P201/K1024 approximation. '
        'The crossed rows are not independent samples. Expected model work limits do not guarantee realized feasibility. '
        'Previously observed timestep sensitivity remains material; small effects here do not establish a general or hardware-level advantage. '
        'No real G1 was used. No model, policy, original source or original result was modified during this audit.', '',
        'All groups and every method versus the fixed reference are published in the statistics directory. Positive contrast values mean larger loss/cost. '
        'No strongest-looking reference or favorable budget was selected after the outcomes.', '',
        f'![All method losses]({out / "position_mse_all_budgets.png"})', '',
        f'![Cost and forecast loss]({out / "work_and_forecast_loss.png"})', '',
        f'![Descriptive bank and history contrasts]({out / "bank_history_contrasts.png"})', '',
        f'Analysis lock: `{result["analysis_lock_sha256"]}`. Reproduction commands are in the separately locked analysis plan.', '']
    with Path(path).open('x') as stream:
        stream.write('\n'.join(lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze', 'run', 'verify'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--report-prefix', type=Path)
    args = p.parse_args()
    if args.stage == 'freeze':
        freeze(args.out)
    elif args.stage == 'run':
        if args.report_prefix is None:
            p.error('--report-prefix is required for run')
        run(args.out, args.report_prefix)
    else:
        verify_analysis(args.out)
        print('Independent analysis lock verified')
