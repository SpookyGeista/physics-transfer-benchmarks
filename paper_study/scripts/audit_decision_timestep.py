"""Independent read-only recomputation of the scalar decision timestep diagnostic."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/decision-timestep-independent-mpl')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'study_v2/src'))
sys.path.insert(0, str(REPO))
import argparse
import ast
import copy
import hashlib
import json
import time
import numpy as np
import pandas as pd
from decision_study.reference import Reference
from paper_study.costed_benchmark.geometry import cam

ROOT = REPO / 'paper_study'
DATA = {'h': ROOT / 'decision_study/results/native_development_v1',
        'h2': ROOT / 'results/decision_refinement_h2',
        'h4': ROOT / 'results/decision_refinement_h4'}
RESULT = ROOT / 'results/decision_comparative_timestep_v1'
BRANCHES = ('release', 'hold100', 'continue100')
KEY = ['grid', 'preparation_seed', 'normal_world', 'friction_world', 'episode']
CHANNELS = ['position_mse_m2', 'velocity_mse', 'normal_mse_N2', 'friction_mse_N2']
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(mapping):
    bad = [p for p, digest in mapping.items() if sha(p) != digest]
    if bad:
        raise ValueError('Changed source/input/output: ' + repr(bad))


def native_loop_ast(path):
    function = next(x for x in ast.parse(path.read_text()).body
                    if isinstance(x, ast.FunctionDef) and x.name == 'run_triplet')
    function.body = [x for x in function.body if not
        (isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'noise' for t in x.targets))]
    return ast.dump(function, include_attributes=False)


def check_columns(actual, reported, keys, columns, *, rtol=2e-11):
    left = actual.set_index(keys).sort_index()
    right = reported.set_index(keys).sort_index()
    if not left.index.is_unique or not right.index.is_unique or not left.index.equals(right.index):
        raise ValueError('Missing, duplicated or unpaired result rows')
    errors = {}
    for column in columns:
        a, b = left[column].to_numpy(float), right[column].to_numpy(float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite reported/recomputed metric')
        errors[column] = float(np.max(abs(a - b)))
        if not np.allclose(a, b, rtol=rtol, atol=2e-18):
            raise ValueError('Raw/CSV mismatch: ' + column + str(errors[column]))
    return errors


def run(prefix):
    started = time.perf_counter()
    prefix = Path(prefix).resolve()
    for suffix in ('.json', '.md', '.png'):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError('Exclusive fresh audit reports required')
    lock = read(RESULT / 'lock.json')
    manifest = read(RESULT / 'manifest.json')
    process = read(RESULT / 'process.json')
    verify(lock['sources_inputs_sha256'])
    verify(manifest['outputs_sha256'])
    if process['status'] != 'complete' or (process['rows'], process['branch_rows']) != (27648, 2592):
        raise ValueError('Full original numeric evaluation is required')
    own_sources = {str(Path(__file__).resolve()): sha(__file__)}
    source = ROOT / 'scripts/decision_refinement.py'
    base = ROOT / 'decision_study/generate.py'
    if native_loop_ast(source) != native_loop_ast(base):
        raise ValueError('Copied native loop differs beyond the declared noise-array construction')
    reported_policy = pd.read_csv(RESULT / 'policy_episodes.csv')
    reported_branch = pd.read_csv(RESULT / 'branch_episodes.csv')
    branch_records, policy_records, drift_records = [], [], []
    maxima = dict(normal_replay_N=0., sensor_noise_abs=0., common_sensor_noise_between_grids=0.,
                  coupled_delta_m=0., compression_rate_m_s=0., force_coupling_N=0.,
                  actuator_work_J=0., normal_drive_work_J=0., friction_work_J=0., absolute_work_J=0.,
                  probability_sum=0.)
    comparisons, fixture_hashes, candidate_guard_counts = [], [], []
    native_guards = 0
    per_cell = {}
    for item in lock['triplets']:
        seed, nw, fw = item['seed'], item['normal'], item['friction']
        rel = Path(f'p{seed}/{nw}_{fw}')
        fixture = {label: sha(data / rel / 'preparation_audit.json') for label, data in DATA.items()}
        if len(set(fixture.values())) != 1:
            raise ValueError('Fixture commands/initial histories differ between time grids')
        fixture_hashes.append(dict(seed=seed, normal=nw, friction=fw, hashes=fixture))
        preparation = read(DATA['h'] / rel / 'preparation_audit.json')
        if preparation['preparation']['design']['dt'] != .0025:
            raise ValueError('External fixture clock changed')
        for label, data in DATA.items():
            cfg = lock['configs'][label]
            divisor = {'h': 1, 'h2': 2, 'h4': 4}[label]
            dt = cfg['dt']
            if dt != .0025 / divisor or cfg['observation_steps'] != 10 * divisor:
                raise ValueError('Coupled/observation clock mismatch')
            planned_path = RESULT / label / rel / 'planned.npz'
            planned_meta = read(planned_path.with_suffix('.json'))
            if planned_meta['npz_sha256'] != sha(planned_path):
                raise ValueError('Persisted plan hash changed')
            with np.load(planned_path, allow_pickle=False) as z:
                plan = {k: z[k].copy() for k in z.files}
            with np.load(RESULT / label / rel / 'forecasts.npz', allow_pickle=False) as z:
                forecasts = {k: z[k].copy() for k in z.files}
            raw_by_branch = {}
            for branch in BRANCHES:
                path = data / rel / (branch + '.npz')
                meta = read(path.with_suffix('.json'))
                if meta['npz_sha256'] != sha(path) or meta['status'] != 'complete':
                    raise ValueError('Native hash/status failure')
                with np.load(path, allow_pickle=False) as z:
                    raw = {k: z[k].copy() for k in z.files}
                raw_by_branch[branch] = raw
                release = int(raw['release_step'])
                common = int(raw['common_step'])
                expected_release = (20 if branch == 'release' else 60) * divisor
                if common != 20 * divisor or release != expected_release:
                    raise ValueError('Decision/release event mismatch')
                if raw['u'].shape != (36, release + 100 * divisor):
                    raise ValueError('Incomplete native horizon')
                if not np.array_equal(raw['observation_steps'], np.arange(0, release + 1, 10 * divisor)):
                    raise ValueError('Missing physical25ms sensor clock')
                target = np.concatenate((raw['next_state'][:, release:],
                    raw['normal'][:, release:, None], raw['friction'][:, release:, None]), axis=-1)
                prediction = forecasts[branch + '_forecast']
                if prediction.shape != target.shape:
                    raise ValueError('Forecast/target clock or channel mismatch')
                mse = np.mean((prediction - target) ** 2, axis=1)
                g, slope = cam(raw['state'][..., 0], raw['amplitude'][:, None], raw['geometry'][:, None])
                fields = dict(coupled_delta_m=(raw['delta'], raw['d'] + g),
                    compression_rate_m_s=(raw['compression_rate'], raw['d_dot'] + slope * raw['state'][..., 1]),
                    force_coupling_N=(raw['net_force'], raw['u'] - slope * raw['normal'] + raw['friction']),
                    actuator_work_J=(raw['actuator_work_J'], raw['u'] * raw['state'][..., 1] * dt),
                    normal_drive_work_J=(raw['normal_drive_work_J'], raw['normal'] * raw['d_dot'] * dt),
                    friction_work_J=(raw['friction_work_J'], raw['friction'] * raw['state'][..., 1] * dt),
                    absolute_work_J=(raw['absolute_work_J'],
                        (abs(raw['u'] * raw['state'][..., 1]) + abs(raw['normal'] * raw['d_dot'])) * dt))
                for name, (a, b) in fields.items():
                    maxima[name] = max(maxima[name], float(np.max(abs(a - b))))
                    if not np.allclose(a, b, rtol=1e-13, atol=1e-15):
                        raise ValueError('Native unit/coupling/work mismatch: ' + name)
                # Reconstruct the independent fresh normal law, including old-rate boundary samples.
                normal = Reference(nw, 36, cfg['initial_compression_m'])
                noise = np.stack([np.random.default_rng(np.random.SeedSequence(k)).normal(size=(61, 3))
                                  for k in meta['noise_stream_keys']])
                expected_noise = noise[:, np.arange(0, expected_release // divisor + 1, 10)] * np.asarray(cfg['sensor_sigma'])
                residual_noise = np.empty_like(raw['observations'])
                obs_index = 0
                for t in range(raw['u'].shape[1]):
                    xv = raw['state'][:, t]
                    if t <= release and t % (10 * divisor) == 0:
                        old_n = copy.deepcopy(normal).force(cfg['initial_compression_m'] + g[:, t],
                            slope[:, t] * xv[:, 1], xv[:, 1], dt=dt, friction_enabled=False)[0]
                        residual_noise[:, obs_index] = raw['observations'][:, obs_index] - np.column_stack((xv, old_n))
                        obs_index += 1
                    n = normal.force(raw['delta'][:, t], raw['compression_rate'][:, t],
                                     xv[:, 1], dt=dt, friction_enabled=False)[0]
                    maxima['normal_replay_N'] = max(maxima['normal_replay_N'], float(np.max(abs(n - raw['normal'][:, t]))))
                maxima['sensor_noise_abs'] = max(maxima['sensor_noise_abs'], float(np.max(abs(residual_noise - expected_noise))))
                native_guards += int(raw['guard'].sum())
                cellkey = (seed, nw, fw, branch)
                per_cell.setdefault(cellkey, {})[label] = dict(
                    target=target[:, divisor - 1::divisor],
                    force_left=target[:, ::divisor, 2:],
                    u=raw['u'][:, ::divisor], noise=residual_noise,
                    d=raw['d'][:, ::divisor], rate=raw['d_dot'][:, ::divisor])
                for e in range(36):
                    identity = dict(grid=label, preparation_seed=seed, normal_world=nw, friction_world=fw, episode=e)
                    record = dict(**identity, branch=branch, **dict(zip(CHANNELS, map(float, mse[e]))),
                        probe_seconds=release * dt, absolute_work_J=float(raw['absolute_work_J'][e].sum()),
                        probe_absolute_work_J=float(raw['absolute_work_J'][e, :release].sum()),
                        extra_probe_work_J=float(raw['absolute_work_J'][e, common:release].sum()),
                        native_guard_count=int(raw['guard'][e].sum()))
                    branch_records.append(record)
            # Independent bitwise prefix check of raw states, forces, work and observations.
            for branch in BRANCHES:
                for field in ('state', 'next_state', 'normal', 'friction', 'u', 'd', 'd_dot',
                              'absolute_work_J', 'guard', 'net_force'):
                    if not np.array_equal(raw_by_branch[branch][field][:, :20 * divisor],
                                          raw_by_branch['release'][field][:, :20 * divisor]):
                        raise ValueError('Native common physical prefix differs across branches')
                if not np.array_equal(raw_by_branch[branch]['observations'][:, :3],
                                      raw_by_branch['release']['observations']):
                    raise ValueError('Common noisy sensors differ across branches')
            current = branch_records[-108:]
            costs = {b: [r for r in current if r['branch'] == b] for b in BRANCHES}
            for budget in cfg['budgets']:
                eligibility = (plan['extra_seconds'] / budget['time_scale_s'] <= budget['max_scaled_time'] + 1e-14) & (
                    plan['planned_extra_probe_work_J'] / budget['work_scale_J'] <= budget['max_scaled_work'] + 1e-14)
                if not np.array_equal(eligibility, plan[budget['name'] + '_eligible']):
                    raise ValueError('Saved budget eligibility differs from declared expected costs')
                for method in ('fixed_release', 'fixed_hold100', 'fixed_continue100', 'uniform',
                               'uniform_expected', 'bayes_risk', 'information_gain', 'physical_only'):
                    probability = plan[budget['name'] + '__' + method]
                    maxima['probability_sum'] = max(maxima['probability_sum'], float(np.max(abs(probability.sum(axis=1) - 1.))))
                    if np.any(probability < 0.) or np.any(probability[~eligibility] != 0.):
                        raise ValueError('Infeasible or negative action probability')
                    for e in range(36):
                        record = {k: costs['release'][e][k] for k in KEY}
                        record.update(budget=budget['name'], method=method)
                        metrics = [*CHANNELS, 'probe_seconds', 'absolute_work_J', 'probe_absolute_work_J',
                                   'extra_probe_work_J', 'native_guard_count']
                        for metric in metrics:
                            record[metric] = float(sum(probability[e, a] * costs[b][e][metric]
                                                      for a, b in enumerate(BRANCHES)))
                        over = np.array([costs[b][e]['extra_probe_work_J'] / budget['work_scale_J'] >
                                         budget['max_scaled_work'] + 1e-14 for b in BRANCHES])
                        record['real_extra_work_budget_violation_probability'] = float(probability[e] @ over)
                        record['chosen_branch'] = BRANCHES[int(np.argmax(probability[e]))] if np.count_nonzero(probability[e]) == 1 else 'probability_mixture'
                        policy_records.append(record)
    recomputed_branch = pd.DataFrame(branch_records)
    recomputed_policy = pd.DataFrame(policy_records)
    metrics = [*CHANNELS, 'probe_seconds', 'absolute_work_J', 'probe_absolute_work_J', 'extra_probe_work_J', 'native_guard_count']
    branch_errors = check_columns(recomputed_branch, reported_branch, KEY + ['branch'], metrics)
    policy_errors = check_columns(recomputed_policy, reported_policy, KEY + ['budget', 'method'],
                                  [*metrics, 'real_extra_work_budget_violation_probability'])
    keys = KEY + ['budget', 'method']
    if not recomputed_policy.set_index(keys).chosen_branch.sort_index().equals(reported_policy.set_index(keys).chosen_branch.sort_index()):
        raise ValueError('Saved action probability/CSV selected branch mismatch')
    for (seed, nw, fw, branch), grids in per_cell.items():
        for label in ('h2', 'h4'):
            a, b = grids['h'], grids[label]
            maxima['common_sensor_noise_between_grids'] = max(maxima['common_sensor_noise_between_grids'], float(np.max(abs(a['noise'] - b['noise']))))
            if not np.array_equal(a['d'], b['d']) or not np.array_equal(a['rate'], b['rate']):
                # Same analytic sampling can differ only by floating arithmetic roundoff.
                if not np.allclose(a['d'], b['d'], atol=1e-17, rtol=1e-14) or not np.allclose(a['rate'], b['rate'], atol=1e-15, rtol=1e-14):
                    raise ValueError('Compression commands differ beyond sampling arithmetic')
            drift_records.append(dict(seed=seed, normal=nw, friction=fw, branch=branch, grid=label,
                native_pooled_drift_um=float(np.sqrt(np.mean((b['target'][..., 0] - a['target'][..., 0]) ** 2)) * 1e6),
                velocity_drift_m_s=float(np.sqrt(np.mean((b['target'][..., 1] - a['target'][..., 1]) ** 2))),
                left_normal_drift_N=float(np.sqrt(np.mean((b['force_left'][..., 0] - a['force_left'][..., 0]) ** 2))),
                left_friction_drift_N=float(np.sqrt(np.mean((b['force_left'][..., 1] - a['force_left'][..., 1]) ** 2))),
                max_force_change_N=float(np.max(abs(b['u'] - a['u'])))))
    drift = pd.DataFrame(drift_records)
    drift_errors = check_columns(drift, pd.read_csv(RESULT / 'native_changes.csv'),
        ['seed', 'normal', 'friction', 'branch', 'grid'], ['native_pooled_drift_um', 'max_force_change_N'],
        rtol=4 * np.finfo(np.float32).eps)
    pooled = recomputed_policy.groupby(['grid', 'budget', 'method'])[CHANNELS + ['absolute_work_J', 'probe_seconds']].mean().reset_index()
    b3 = pooled[pooled.budget == 'B3'].copy()
    b3['position_rmse_um'] = np.sqrt(b3.position_mse_m2) * 1e6
    choices = []
    for (budget, method), group in recomputed_policy.groupby(['budget', 'method']):
        paired = group.pivot(index=['preparation_seed', 'normal_world', 'friction_world', 'episode'],
                             columns='grid', values='chosen_branch')
        choices.append(dict(budget=budget, method=method, contexts=len(paired),
                            h_h2_changed=int((paired.h != paired.h2).sum()),
                            h_h4_changed=int((paired.h != paired.h4).sum())))
    for name in ('sensor_noise_abs', 'normal_replay_N', 'common_sensor_noise_between_grids'):
        if maxima[name] > 5e-12:
            raise ValueError('Native reference/noise reconstruction failed: ' + name + str(maxima[name]))
    verify(lock['sources_inputs_sha256'])
    verify(manifest['outputs_sha256'])
    verify(own_sources)
    summary = dict(status='complete', scope='Paired-noise development; two history blocks, four compositions, one noisy fitted bank; no held-out decision data',
        source_input_hashes_verified=len(lock['sources_inputs_sha256']), output_hashes_verified=len(manifest['outputs_sha256']),
        source_sha256=own_sources, source_evaluation_lock_sha256=sha(RESULT / 'lock.json'),
        source_manifest_sha256=sha(RESULT / 'manifest.json'), native_branches_per_grid=864,
        policy_rows=len(recomputed_policy), branch_rows=len(recomputed_branch), plan_count=24,
        native_loop_ast_equal_except_noise_assignments=True, canonical_fixture_audits=fixture_hashes,
        drift_precision_note='Original native_changes subtracts/means float32 x arrays; independent audit promotes to float64. Drift-table roundoff allowance is4float32eps relative; all forecast/cost comparisons retain2e-11 relative.',
        external_fixture_dt=.0025, model_native_dt=[.0025, .00125, .000625], observation_dt=.025,
        maxima=maxima, native_guards=native_guards, branch_csv_max_absolute_errors=branch_errors,
        policy_csv_max_absolute_errors=policy_errors, drift_csv_max_absolute_errors=drift_errors,
        native_drift=drift_records, choices=choices, b3=b3.to_dict('records'),
        all_pooled=pooled.to_dict('records'), seconds=time.perf_counter() - started,
        command=sys.argv, no_population_inference=True,
        limitations=['Two development blocks and one model bank only; no Main3 effect-size guarantee.',
            'Same noisy feedback policy, but realized commands differ with timestep.',
            'Native x/v drift uses aligned end states; four-channel forecast MSE uses each native grid, with forces at left/pre-step time.',
            'Finite classical mixture and known constitutive equation prior, no model-uncertainty or global-optimality guarantee.',
            'All plans execute before later sensor assimilation or target scoring; integrity hashes and structural validation can read complete files.'])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix('.json').open('x') as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for method in ('fixed_continue100', 'information_gain', 'bayes_risk', 'physical_only'):
        values = b3[b3.method == method].set_index('grid').loc[['h', 'h2', 'h4']]
        axes[0].plot([2.5, 1.25, .625], values.position_rmse_um, 'o-', label=method)
    axes[0].set(xlabel='Coupled step (ms)', ylabel='B3 pooled future x RMSE (µm)')
    axes[0].legend(fontsize=7)
    d4 = drift[drift.grid == 'h4'].native_pooled_drift_um
    axes[1].hist(d4, bins=8, color='#476b8c')
    axes[1].set(xlabel='Native h → h/4 position drift (µm)', ylabel='Composition/block/branch cells')
    fig.savefig(prefix.with_suffix('.png'), dpi=180)
    plt.close(fig)
    lines = ['# Independent scalar-decision timestep audit', '',
        'All **27,648 policy rows and2,592 branch rows** were independently recomputed from saved forecasts, native targets, saved action probabilities and raw work signals. '
        'All four forecast channels, weighted randomized-action losses, probe duration, total/probe/additional work, native guards and realized work-budget violations agree with the reported CSVs.', '',
        f"All {summary['source_input_hashes_verified']} source/input hashes, {summary['output_hashes_verified']} output hashes and24 explicit plan hashes verify.",
        'The copied native loop has the same AST after removing only its declared noise-array assignments. '
        'All eight fixture audits are byte-identical across grids; the fixture remains at2.5ms. Coupled/model steps are2.5/1.25/.625ms and sensors remain25ms. '
        'Replaying the fresh normal law verifies old-rate boundary measurements and reconstructs the same Gaussian sensor errors at each physical sample.', '',
        f"Maximum reconstructed noise error: {maxima['sensor_noise_abs']:.3g}; cross-grid noise difference: {maxima['common_sensor_noise_between_grids']:.3g}; "
        f"maximum normal-force replay discrepancy: {maxima['normal_replay_N']:.3g}N. Native guards: {native_guards}.", '',
        f"Native h→h/4 future position drift ranges **{d4.min():.3f}–{d4.max():.3f}µm** across24 paired cells. "
        'The noisy feedback policy is unchanged, but actual actuator commands differ with the changed native states. This is policy-level timestep sensitivity, not one identical force-sequence experiment.', '',
        f"The independent float64 drift calculation differs from the original float32 drift table by at most{drift_errors['native_pooled_drift_um']:.3g}µm; "
        'this is arithmetic roundoff. Forecast/cost rows agree at the separately tighter tolerance and are not adjusted.', '',
        '| B3 method | h RMSE (µm) | h/2 | h/4 |', '|---|---:|---:|---:|']
    for method in b3.method.unique():
        values = b3[b3.method == method].set_index('grid').loc[['h', 'h2', 'h4'], 'position_rmse_um']
        lines.append(f'| {method} | {values.iloc[0]:.3f} | {values.iloc[1]:.3f} | {values.iloc[2]:.3f} |')
    lines += ['', 'Bayes-risk remains worse than fixed continuation at all three grids under B3. '
        'The gap changes with timestep, so small apparent decision improvements must not be presented as numerically robust or generalized to the held-out eight-bank study. '
        'The full JSON retains every budget, method and changed-choice count; no favorable subset, confidence interval or positive-success declaration is introduced.', '',
        'This is a descriptive check on two development history blocks and one fitted bank. '
        'Four-channel forecast losses are recomputed on each grid at the correct native timing: post-step x/v, pre-step N/Ft. '
        'Native drift aligns end states at shared2.5ms times; it is not a claim of pointwise force convergence or calibrated uncertainty.', '',
        f'Reproduction: `{sys.executable} {Path(__file__).resolve()} --report-prefix /absolute/fresh/report_prefix`.',
        f'![Numerical sensitivity]({prefix.with_suffix(".png")})', '']
    prefix.with_suffix('.md').write_text('\n'.join(lines))
    print(json.dumps({k: summary[k] for k in ('status', 'policy_rows', 'branch_rows', 'maxima', 'seconds')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-prefix', type=Path, required=True)
    run(p.parse_args().report_prefix)
