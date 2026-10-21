"""Independent saved-array audit; no production analysis or model imports.

Both replay and analysis must be complete before reading any result array.
The audit reconstructs arithmetic, factor coverage and descriptive summaries.
It neither reruns dynamics nor tests a new scientific hypothesis.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import itertools
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
CHANNELS = ('position_m', 'velocity_m_s', 'normal_N', 'friction_N')
IDENTITY = ('bank', 'history', 'normal', 'friction', 'probe', 'episode')
Q = (0, .25, .5, .75, .9, .95, .99, 1)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify(mapping):
    for path, expected in mapping.items():
        if digest(path) != expected:
            raise ValueError('Changed or incorrectly bound input: ' + str(path))


def pair_metrics(past, observed, sigma, future, target, objectives, selected):
    """Reconstruct distances and scores from arrays, without model execution."""
    n = len(past)
    if past.ndim != 4 or past.shape[1] != 2 or past.shape[-1] != 3:
        raise ValueError('Expected n x 2 x observations x 3 past array')
    if sigma.shape != (3,) or np.any(sigma <= 0):
        raise ValueError('Positive three-channel likelihood scales required')
    if observed.shape != past[:, 0].shape or future.shape[:2] != (n, 2):
        raise ValueError('Candidate/observation shape mismatch')
    if target.shape != future[:, 0].shape or future.shape[-1] != 4:
        raise ValueError('Future/target shape mismatch')
    for a in (past, observed, sigma, future, target, objectives):
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite saved array')
    actual_choice = np.argmin(objectives, axis=1)
    if not np.array_equal(actual_choice, selected):
        raise ValueError('Original selection violates minimum objective / first tie')
    delta = past[:, 1].astype(np.longdouble) - past[:, 0].astype(np.longdouble)
    whitened = (delta / sigma.astype(np.longdouble)).reshape(n, -1)
    squared = np.sum(whitened * whitened, axis=1)
    residual = ((past - observed[:, None]) / sigma).reshape(n, 2, -1)
    replay_objective = np.array([[np.dot(r, r) for r in row] for row in residual])
    return {
        'past_mahalanobis': np.asarray(np.sqrt(squared), float),
        'past_squared_distance': np.asarray(squared, float),
        'original_gain': objectives[:, 0] - objectives[np.arange(n), actual_choice],
        'selected_start': actual_choice,
        'past_channel_rms': np.sqrt(np.mean((past[:, 1] - past[:, 0]) ** 2, axis=1)),
        'future_channel_rms': np.sqrt(np.mean((future[:, 1] - future[:, 0]) ** 2, axis=1)),
        'candidate_mse': np.mean((future - target[:, None]) ** 2, axis=2),
        'replay_objectives': replay_objective,
    }


def quantiles(values):
    return {str(q): float(np.quantile(values, q)) for q in Q} if len(values) else None


def aggregate(rows):
    """Independent row selection and aggregation, matching the declared schema."""
    def block(records):
        def a(key):
            return np.asarray([r[key] for r in records])
        distance = a('past_mahalanobis')
        future_m = a('future_distance_position_m')
        future = future_m * 1000
        close = distance <= 1
        return dict(
            pairs=len(records), past_curve_distance=quantiles(distance),
            future_distance_mm=quantiles(future),
            normalized_state_distance=quantiles(a('normalized_initial_distance')),
            past_close_count=int(np.count_nonzero(close)),
            close_past_divergent_future_count=int(np.count_nonzero(close & (future_m >= 1e-4))),
            future_distance_mm_when_past_close=quantiles(future[close]),
            future_distance_mm_when_past_far=quantiles(future[~close]),
            boundary_position_distance_um=quantiles(a('boundary_x_distance_m') * 1e6),
            original_vs_replay_selection_changes=int(np.count_nonzero(a('selected_start') != a('replay_float64_choice'))),
            replay_float64_vs_extended_selection_changes=int(np.count_nonzero(a('replay_float64_choice') != a('same_residual_extended_choice'))),
            same_residual_vs_formula_selection_changes=int(np.count_nonzero(a('same_residual_extended_choice') != a('stable_choice'))),
            fixed_pair_noisy_vs_clean_selection_changes=int(np.count_nonzero(a('stable_choice') != a('clean_fixed_pair_choice'))),
        )
    selected = [r for r in rows if r['selected_start'] == 1]
    groups = {
        'all_pairs': rows,
        'original_second_selected': selected,
        'second_selected_gain_at_most_1e_8': [r for r in selected if r['original_gain'] <= 1e-8],
        'second_selected_gain_above_1e_8': [r for r in selected if r['original_gain'] > 1e-8],
    }
    return {name: block(records) for name, records in groups.items()}


def keyed(row):
    return tuple(int(row[k]) if k in ('bank', 'history', 'episode') else row[k] for k in IDENTITY)


def assert_close(actual, expected, label, errors, *, atol=2e-11, rtol=2e-9):
    x, y = np.asarray(actual), np.asarray(expected)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid shapes/nonfinite comparison: ' + label)
    difference = float(np.max(abs(x-y))) if x.size else 0.
    errors[label] = max(errors.get(label, 0.), difference)
    if not np.allclose(x, y, atol=atol, rtol=rtol):
        raise ValueError(f'{label} differs (max {difference:g})')


def compare_summary(actual, expected, errors, path='summary'):
    if isinstance(actual, dict):
        if not isinstance(expected, dict) or actual.keys() != expected.keys():
            raise ValueError('Summary key mismatch: ' + path)
        for key in actual:
            compare_summary(actual[key], expected[key], errors, path + '/' + key)
    elif actual is None:
        if expected is not None:
            raise ValueError('Empty-group mismatch: ' + path)
    elif isinstance(actual, int):
        if actual != expected:
            raise ValueError(f'Exact count mismatch {path}: {actual} != {expected}')
    else:
        assert_close(actual, expected, path, errors)


def run(data, out_prefix):
    data, out_prefix = Path(data).resolve(), Path(out_prefix).resolve()
    analysis = data / 'analysis'
    # Gate before arrays, CSV, summary, lock inspection or report creation.
    if not (data/'process.json').exists() or not (analysis/'process.json').exists():
        raise RuntimeError('WAIT: both complete process records are required')
    processes = [read(data/'process.json'), read(analysis/'process.json')]
    if any(p.get('status') != 'complete' for p in processes):
        raise RuntimeError('WAIT: replay and analysis must both be complete')
    outputs = [out_prefix.with_suffix(ext) for ext in ('.json', '.md')]
    if any(p.exists() for p in outputs):
        raise FileExistsError('Audit output already exists; use a fresh prefix')
    started = time.perf_counter()
    controls = [data/'process.json', analysis/'process.json', data/'analysis_lock.json',
                data/'artifact_manifest.json', data/'pairs.csv', analysis/'summary.json',
                analysis/'artifact_manifest.json', Path(__file__)]
    bindings = {str(p): digest(p) for p in controls}
    lock = read(data/'analysis_lock.json')
    pilot_manifest = read(data/'artifact_manifest.json')
    analysis_manifest = read(analysis/'artifact_manifest.json')
    lock_sha = digest(data/'analysis_lock.json')
    if any(p['lock_sha256'] != lock_sha for p in processes + [pilot_manifest, analysis_manifest]):
        raise ValueError('Replay/analysis manifests bind different locks')
    for mapping in (lock['source_and_input_sha256'], pilot_manifest['outputs_sha256'],
                    analysis_manifest['outputs_sha256']):
        for path, h in mapping.items():
            if path in bindings and bindings[path] != h:
                raise ValueError('Conflicting input hash: ' + path)
            bindings[path] = h
    verify(bindings)
    with (data/'pairs.csv').open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    original = {keyed(r): r for r in csv_rows}
    if len(original) != len(csv_rows):
        raise ValueError('Duplicate CSV factor identity')
    expected = set(itertools.product((1,2,3), range(53201,53205), ('maxwell','boucwen'),
        ('balanced','wide_scales','compliant','stiff_high_capacity'),
        ('hold_150','gentle_150','strong_300'), range(36)))
    if set(original) != expected or len(expected) != 10368 or len(lock['batches']) != 288:
        raise ValueError('Incomplete or additional declared factor cells')
    expected_paired = {data/'paired_past'/f"bank{b['bank']['bank']}"/b['relative'] for b in lock['batches']}
    if set((data/'paired_past').rglob('*.npz')) != expected_paired or len(expected_paired) != 288:
        raise ValueError('Paired-array inventory mismatch')
    errors, rebuilt, seen = {}, [], set()
    for entry in lock['batches']:
        bank = entry['bank']['bank']
        native, prediction = Path(entry['native']), Path(entry['prediction'])
        paired = data/'paired_past'/f'bank{bank}'/entry['relative']
        meta = read(native.with_suffix('.json'))
        if meta.get('status') != 'complete' or read(prediction.with_suffix('.json')).get('status') != 'complete':
            raise ValueError('Incomplete original source batch')
        with np.load(paired, allow_pickle=False) as p, np.load(prediction, allow_pickle=False) as f, np.load(native, allow_pickle=False) as n:
            past, obs, clean, sigma = [p[k] for k in ('past_predictions','observed','clean_native_samples','effective_sigma')]
            future, objectives, choice = [f[k] for k in ('mle_candidate_forecasts','mle_candidate_objectives','mle_selected_start')]
            end = int(n['probe_steps'])
            oi = np.asarray(n['observation_steps'])
            causal = (oi > 0) & (oi <= end)
            sample = oi[causal]
            if not np.array_equal(p['observation_steps'], sample) or sample[-1] != end:
                raise ValueError('Past observation clock mismatch')
            for label, a, b in (
                ('original_objectives', p['original_objectives'], objectives),
                ('original_selection', p['original_selection'], choice),
                ('initial_pi', p['initial_pi'], f['mle_candidate_initial_pi']),
                ('effective_sigma', sigma, f['effective_sigma']),
                ('observed', obs, n['observations'][:,causal]),
                ('clean_state', clean[:,:,:2], n['state'][:,sample]),
                ('clean_normal', clean[:,:,2], n['normal'][:,sample]),
            ):
                if not np.array_equal(a, b):
                    raise ValueError('Saved input mismatch: ' + label)
            sensor = np.array([meta['sensor_parameters'][k] for k in ('sigma_x_m','sigma_v_m_s','sigma_normal_N')])
            assert_close(sigma, np.hypot(sensor, [1e-6,1e-5,1e-4]), 'likelihood_sigma', errors, atol=1e-15)
            target = np.concatenate((n['next_state'][:,end:], n['normal'][:,end:,None], n['friction'][:,end:,None]), axis=2)
            metrics = pair_metrics(past, obs, sigma, future, target, objectives, choice)
            assert_close(metrics['replay_objectives'], objectives, 'past_objective_replay', errors, atol=1e-7, rtol=1e-9)
            pi_path = REPO/'paper_study/multistate_learning/results/models'/meta['friction_world']/f"r{entry['bank']['pi_seed']}"/'noisy/model.json'
            normal_path = REPO/'research_next/results/round1/models/state_id'/meta['normal_world']/f"r{entry['bank']['normal_seed']}"/'noisy/blind_selection/model.json'
            for weight_path in (pi_path, normal_path):
                if str(weight_path) not in bindings:
                    raise ValueError('Unbound model coefficient file')
            fitted_pi = read(pi_path)
            normal = read(normal_path)['normal_model']
            pars = normal['normal']; delta0 = meta['initial_compression_m']
            N0 = pars['k_eq']*delta0 if normal['world']=='maxwell' else pars['k']*delta0+pars['F']*np.tanh(delta0/pars['dy'])
            scale = np.asarray(fitted_pi['yield_per_normal_m_N'])*max(N0,0.)
            normalized_distance = np.linalg.norm((p['initial_pi'][:,1]-p['initial_pi'][:,0])/scale, axis=1)
            if len(past) != 36 or len(meta['factor_rows']) != 36:
                raise ValueError('Wrong episode count')
            factors = {(r['profile_index'],r['geometry'],r['excitation_sign']) for r in meta['factor_rows']}
            if factors != set(itertools.product(range(6), range(3), (-1,1))):
                raise ValueError('Incomplete within-batch factors')
            for k, suffix in (('xv','xv'),('normal','normal'),('pi','pi')):
                assert_close(p['boundary_'+k], f['mle_candidate_boundary_'+suffix], 'boundary_'+k, errors, atol=1e-10, rtol=1e-10)
            for i in range(36):
                identity = (bank,meta['preparation_seed'],meta['normal_world'],meta['friction_world'],meta['probe']['name'],i)
                if identity in seen or identity not in original:
                    raise ValueError('Duplicate or unexpected raw pair')
                seen.add(identity)
                csv_row = original[identity]
                row = {k:(int(csv_row[k]) if k in ('bank','history','episode') else csv_row[k]) for k in IDENTITY}
                values = {key:float(metrics[key][i]) for key in ('past_mahalanobis','past_squared_distance','original_gain')}
                values['selected_start'] = int(choice[i])
                values['normalized_initial_distance'] = float(normalized_distance[i])
                values['boundary_x_distance_m'] = float(abs(p['boundary_xv'][i,1,0]-p['boundary_xv'][i,0,0]))
                values['boundary_v_distance_m_s'] = float(abs(p['boundary_xv'][i,1,1]-p['boundary_xv'][i,0,1]))
                values['boundary_normal_state_l2'] = float(np.linalg.norm(p['boundary_normal'][i,1]-p['boundary_normal'][i,0]))
                values['boundary_pi_state_l2_m'] = float(np.linalg.norm(p['boundary_pi'][i,1]-p['boundary_pi'][i,0]))
                for c, channel in enumerate(CHANNELS):
                    values['future_distance_'+channel] = float(metrics['future_channel_rms'][i,c])
                    values['start1_mse_'+channel] = float(metrics['candidate_mse'][i,0,c])
                    values['start2_mse_'+channel] = float(metrics['candidate_mse'][i,1,c])
                    if c<3: values['past_distance_'+channel] = float(metrics['past_channel_rms'][i,c])
                for key, value in values.items():
                    assert_close(value, float(csv_row[key]), 'csv/'+key, errors)
                if int(csv_row['geometry']) != int(n['geometry'][i]) or int(csv_row['sign']) != meta['factor_rows'][i]['excitation_sign']:
                    raise ValueError('CSV geometry/sign mismatch')
                row.update(values)
                # Arithmetic-choice labels and solver/replay summaries are
                # aggregation-only checks, not independently rerun dynamics.
                for key in ('replay_float64_choice','same_residual_extended_choice','stable_choice','clean_fixed_pair_choice'):
                    row[key] = int(csv_row[key])
                rebuilt.append(row)
    if seen != expected:
        raise ValueError('Raw coverage mismatch')
    actual = aggregate(rebuilt)
    for category, fields in (('probe',('probe',)),('bank',('bank',)),('history',('history',)),('composition',('normal','friction'))):
        groupkeys = sorted({tuple(r[k] for k in fields) for r in rebuilt})
        actual['by_'+category] = {'/'.join(map(str,key)):aggregate([r for r in rebuilt if tuple(r[k] for k in fields)==key]) for key in groupkeys}
    actual['numerics'] = {k:max(float(r[k]) for r in csv_rows) for k in ('replay_objective_max_abs','replay_future_max_abs','replay_boundary_max_abs','noise_reconstruction_max_abs','solver_max_abs_N','objective_identity_error','noise_identity_error')}
    mse1 = np.array([r['start1_mse_position_m'] for r in rebuilt])
    mse2 = np.array([r['start2_mse_position_m'] for r in rebuilt])
    selected = np.array([r['selected_start'] for r in rebuilt])
    actual['original_position_rmse_mm'] = dict(start1=float(np.sqrt(mse1.mean())*1000),two_start=float(np.sqrt(np.where(selected==1,mse2,mse1).mean())*1000))
    compare_summary(actual, read(analysis/'summary.json'), errors)
    verify(bindings)
    report = dict(status='complete',scope='Independent saved-array arithmetic and inventory audit; exploratory descriptive pilot only',
        started_from_complete_processes=True,source_sha256=digest(__file__),data_directory=str(data),
        seconds=time.perf_counter()-started,pairs=len(rebuilt),batches=288,
        unique_input_and_source_hashes=len(bindings),hash_checks_before_and_after=2*len(bindings),
        independently_recalculated=['whole whitened past-curve distance','three past channel RMS differences','original objective selection and gain',
            'four future candidate RMS differences','both candidate MSEs against native future in four channels','initial normalized state distance',
            'boundary state differences','complete factor coverage','all gain/D/future-distance stratum counts and quantiles'],
        aggregation_only=['extended-precision/clean-choice labels from protected CSV','recorded solver, replay and noise maxima'],
        no_model_or_simulator_execution=True,no_new_hypothesis_or_significance_test=True,
        no_original_files_changed=True,maximum_differences=errors,independent_summary=actual,input_sha256=bindings)
    lines=['# Independent mechanism-pilot raw-data audit','',
        '**All required saved-array and completeness checks passed.** The replay and analysis were both complete before any result array was opened.','',
        f'All {len(rebuilt):,} pairs in 288 batches were checked: three banks, four history blocks, eight compositions, three probes and 36 episode factors. Original selection uses the exact minimum past objective with first-start ties. No pair was removed or reselected.','',
        'Whitened full past-curve distances, all three past channel distances, all four future candidate distances and both candidates’ four-channel MSEs were reconstructed independently from saved arrays. Forecast separation is distinguished from error against native targets. Original commands, models and scientific results were not changed.','',
        '| Group | Pairs | D≤1 | D≤1 and future position RMS separation≥0.1 mm |','|---|---:|---:|---:|']
    for key in ('all_pairs','original_second_selected','second_selected_gain_at_most_1e_8','second_selected_gain_above_1e_8'):
        r=actual[key];lines.append(f'| {key} | {r["pairs"]} | {r["past_close_count"]} | {r["close_past_divergent_future_count"]} |')
    lines+=['', 'All subgroup counts and quantiles by probe, bank, history and composition agree with `analysis/summary.json`. The D≤1, 0.1 mm and gain strata are descriptive scales from an exploratory post-analysis protocol, not hypothesis tests or new selection rules.','',
        f'The original pooled position RMSE remains {actual["original_position_rmse_mm"]["start1"]:.5f} mm for start 1 and {actual["original_position_rmse_mm"]["two_start"]:.5f} mm for the original two-start choice.','',
        'This audit performs no simulator or model rollout. It checks recorded boundary states and the re-evaluated past objective, but does not independently reproduce full dynamical trajectories, solver residuals or the physical meaning of the MLE box states. Extended-arithmetic selection labels and recorded solver/replay/noise maxima are checked for aggregation consistency only; their algorithms are not independently rerun here. It therefore establishes arithmetic and coverage, not physical identifiability, parameter-bias isolation or causal attribution solely to friction memory.','',
        f'All {len(bindings):,} unique input/source bindings verified before and after calculation. Elapsed audit wall time: {report["seconds"]:.3f} s. The JSON includes exact hashes, maximum discrepancies and the complete independently reconstructed descriptive summary.','']
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with outputs[1].open('x') as stream: stream.write('\n'.join(lines))
    report['markdown_sha256']=digest(outputs[1])
    with outputs[0].open('x') as stream: json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('status','pairs','batches','seconds','unique_input_and_source_hashes')},indent=2))


def self_test():
    p=np.array([[[[-1.,0.,0.]],[[1.,0.,0.]]]])
    y=np.zeros((1,1,3));s=np.ones(3);f=np.zeros((1,2,2,4));t=np.ones((1,2,4))
    r=pair_metrics(p,y,s,f,t,np.ones((1,2)),np.zeros(1,dtype=int))
    assert r['past_mahalanobis'][0]==2 and r['original_gain'][0]==0
    assert np.all(r['candidate_mse']==1) and np.all(r['future_channel_rms']==0)
    try: pair_metrics(p,y,s,f,t,np.ones((1,2)),np.ones(1,dtype=int))
    except ValueError: pass
    else: raise AssertionError('Incorrect tie accepted')
    print('Synthetic arithmetic, shape and exact-selection self-check passed; no result data opened.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'results/pilot_v1')
    parser.add_argument('--out-prefix',type=Path,default=ROOT/'reports/pilot_audit')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test: self_test()
    else: run(args.data,args.out_prefix)
