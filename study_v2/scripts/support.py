"""Calibration-only support fit and frozen exact-local-replay audit.

Nearest-neighbour distance is a diagnostic; measured profile matching is the
primary local-coverage check. No replay file is read before lock verification.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.spatial import cKDTree
from decision_study.common import CFG, FI, NI, ROOT, digest, dump


TOLERANCES = dict(feature_max_abs=1e-4, delta_m=2e-6, rate_m_s=1e-5,
                  velocity_m_s=5e-5, normal_n=1e-4, friction_n=1e-4)


def load_data(folder, split, fitting=False):
    if fitting and split not in ('train', 'val'):
        raise ValueError('support fitting may access only isolated train and val')
    path = Path(folder)/f'{split}.npz'
    with np.load(path) as stored:
        arrays = {key: stored[key] for key in stored.files}
    metadata = json.loads(path.with_suffix('.json').read_text())
    assert len(metadata['rows']) == len(arrays['features']), f'metadata length mismatch: {path}'
    return arrays, metadata, {str(path): digest(path), str(path.with_suffix('.json')): digest(path.with_suffix('.json'))}


def mode_indices(rows, mode, budget=None):
    return [i for i, row in enumerate(rows) if row['mode'] == mode and
            (budget is None or row['profile'] < budget)]


def fit_arrays(train, train_rows, val, val_rows, budgets, quantile):
    """Use every selected training timestep and independent validation trajectories."""
    assert all(row['mode'] in ('A', 'B') for row in train_rows+val_rows)
    assert set(row['id'] for row in train_rows).isdisjoint(row['id'] for row in val_rows)
    saved, info = {}, {}
    for budget in budgets:
        for head, mode, columns in (('normal', 'B', NI), ('friction', 'A', FI)):
            ti = mode_indices(train_rows, mode, budget)
            vi = mode_indices(val_rows, mode)
            assert ti and vi, f'missing isolated {mode} calibration or validation'
            points = train['features'][ti][:, :, columns].reshape(-1, len(columns)).astype(np.float64)
            queries = val['features'][vi][:, :, columns].reshape(-1, len(columns)).astype(np.float64)
            assert np.isfinite(points).all() and np.isfinite(queries).all()
            distance = cKDTree(points).query(queries, k=1, workers=1)[0]
            raw_threshold = float(np.quantile(distance, quantile, method='linear'))
            key = f'{head}_b{budget}'
            saved[key] = points
            saved[key+'_val_distances'] = distance
            info[key] = dict(threshold=max(1e-6, raw_threshold), raw_quantile=raw_threshold,
                             quantile=quantile, quantile_method='linear', threshold_floor=1e-6,
                             columns=columns, train_points=len(points), val_points=len(queries),
                             train_ids=[train_rows[i]['id'] for i in ti], val_ids=[val_rows[i]['id'] for i in vi],
                             val_distance_mean=float(distance.mean()), val_distance_max=float(distance.max()))
    return saved, info


def verify_lock(lock_path, root=ROOT):
    lock_path = Path(lock_path)
    assert lock_path.exists(), 'freeze the protocol before reading replay data'
    lock = json.loads(lock_path.read_text())
    assert lock.get('files'), 'empty protocol lock'
    for name, expected in lock['files'].items():
        path = Path(name)
        if not path.is_absolute():
            path = Path(root)/path
        assert path.exists() and digest(path) == expected, f'changed or missing frozen file: {name}'
    return lock


def _profile_maps(rows):
    maps = {'A': {}, 'B': {}}
    for i, row in enumerate(rows):
        assert row['mode'] in maps
        assert row['profile'] not in maps[row['mode']], 'duplicate isolated profile ID'
        maps[row['mode']][row['profile']] = i
    assert set(maps['A']) == set(maps['B']), 'unpaired isolated profile libraries'
    for profile in maps['A']:
        ar, br = rows[maps['A'][profile]], rows[maps['B'][profile]]
        assert ar['id'].endswith('-A') and br['id'].endswith('-B')
        assert ar['id'][:-2] == br['id'][:-2], 'different A/B source profile identities'
    return maps


def audit_exact(train, train_rows, replay, replay_rows, amplitudes):
    """Match recorded local signals, never the desired/stimulus profile library."""
    maps = _profile_maps(train_rows)
    profiles = [row['profile'] for row in train_rows if row['mode'] == 'B']
    expected = [(profile, float(amplitude)) for amplitude in amplitudes for profile in profiles]
    observed = [(row['profile'], float(row['amplitude'])) for row in replay_rows]
    assert observed == expected, 'replay metadata order or complete profile-by-cam grid differs'
    assert replay['features'].shape[1] == train['features'].shape[1], 'different recorded trajectory lengths'
    results = []
    for i, row in enumerate(replay_rows):
        assert row['mode'] == 'AB'
        ai, bi = maps['A'][row['profile']], maps['B'][row['profile']]
        base = train_rows[bi]['id'][:-2]
        assert row['id'] == f'{base}-cam{row["amplitude"]:+.3f}', 'replay profile identity does not match isolated calibration'
        def error(a, b):
            return float(np.max(np.abs(np.asarray(a, dtype=np.float64)-np.asarray(b, dtype=np.float64))))
        values = dict(
            exact_normal_feature_max_abs=error(replay['features'][i][:, NI], train['features'][bi][:, NI]),
            exact_friction_feature_max_abs=error(replay['features'][i][:, FI], train['features'][ai][:, FI]),
            exact_delta_max_abs_m=error(replay['delta'][i], train['delta'][bi]),
            exact_rate_max_abs_m_s=error(replay['compression_rate'][i], train['rate'][bi]),
            exact_velocity_max_abs_m_s=error(replay['state'][i, :, 1], train['state'][ai, :, 1]),
            exact_normal_max_abs_n=error(replay['normal'][i], train['normal'][bi]),
            exact_friction_max_abs_n=error(replay['friction'][i], train['friction'][ai]),
            cascade_normal_max_abs_n=error(train['normal'][ai], train['normal'][bi]))
        checks = (
            values['exact_normal_feature_max_abs'] <= TOLERANCES['feature_max_abs'],
            values['exact_friction_feature_max_abs'] <= TOLERANCES['feature_max_abs'],
            values['exact_delta_max_abs_m'] <= TOLERANCES['delta_m'],
            values['exact_rate_max_abs_m_s'] <= TOLERANCES['rate_m_s'],
            values['exact_velocity_max_abs_m_s'] <= TOLERANCES['velocity_m_s'],
            values['exact_normal_max_abs_n'] <= TOLERANCES['normal_n'],
            values['exact_friction_max_abs_n'] <= TOLERANCES['friction_n'],
            values['cascade_normal_max_abs_n'] <= TOLERANCES['normal_n'])
        results.append(dict(**values, exact_match=int(all(checks)), matched_normal_id=train_rows[bi]['id'],
                            matched_friction_id=train_rows[ai]['id']))
    return results


def distance_matrix(replay, points, columns):
    shape = replay['features'].shape[:2]
    query = replay['features'][:, :, columns].reshape(-1, len(columns)).astype(np.float64)
    assert np.isfinite(query).all()
    return cKDTree(points).query(query, k=1, workers=1)[0].reshape(shape)


def distance_summary(distance, threshold):
    return [dict(coverage=float(np.mean(row <= threshold)), tail_fraction=float(np.mean(row > threshold)),
                 nearest_distance_min=float(row.min()), nearest_distance_mean=float(row.mean()),
                 nearest_distance_p995=float(np.quantile(row, .995, method='linear')),
                 nearest_distance_max=float(row.max()), threshold=threshold) for row in distance]


def episode_support(replay, points, threshold, columns):
    return distance_summary(distance_matrix(replay, points, columns), threshold)


def fit_stage(args):
    assert not (ROOT/'results/protocol_lock.json').exists(), 'support thresholds are frozen once the main protocol is locked'
    code_paths = [Path(__file__).resolve(), ROOT/'src/decision_study/common.py', ROOT/'configs/experiment.json',
                  ROOT/'reports/support_protocol.md']
    code = {str(path): digest(path) for path in code_paths}
    for world in args.worlds:
        for seed in args.seeds:
            folder = Path(args.data)/world/f'r{seed}'
            train, tm, tsha = load_data(folder, 'train', fitting=True)
            val, vm, vsha = load_data(folder, 'val', fitting=True)
            spec = dict(world=world, replicate=seed, sources={**tsha, **vsha}, code=code,
                        budgets=args.budgets, quantile=CFG['support_quantile'], fixed_unit_features=True)
            fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
            dest = Path(args.out)/world/f'r{seed}.npz'; meta_path = dest.with_suffix('.json')
            if meta_path.exists():
                old = json.loads(meta_path.read_text())
                assert old['fingerprint'] == fingerprint and digest(dest) == old['npz_sha256'], 'support fit changed: use fresh --out'
                print('SKIP verified support', world, seed, flush=True)
                continue
            arrays, heads = fit_arrays(train, tm['rows'], val, vm['rows'], args.budgets, CFG['support_quantile'])
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, **arrays)
            dump(meta_path, dict(**spec, fingerprint=fingerprint, heads=heads, npz_sha256=digest(dest),
                                 tolerances=TOLERANCES, selection='all selected profile timesteps; no random subsampling',
                                 interpretation='empirical local proximity; exact measured replay audit is decisive'))
            print('FIT SUPPORT', world, seed, {key: value['threshold'] for key, value in heads.items()}, flush=True)


def evaluate_stage(args):
    verify_lock(args.lock)
    all_rows, sources = [], {}
    for world in args.worlds:
        for seed in args.seeds:
            folder = Path(args.data)/world/f'r{seed}'
            support_path = Path(args.out)/world/f'r{seed}.npz'
            meta = json.loads(support_path.with_suffix('.json').read_text())
            assert digest(support_path) == meta['npz_sha256'], 'support points changed after fitting'
            for name, expected in meta['sources'].items():
                assert digest(name) == expected, f'calibration/validation changed: {name}'
            train, tm, tsha = load_data(folder, 'train')
            replay, rm, rsha = load_data(folder, 'replay')
            sources.update(tsha); sources.update(rsha)
            assert rm['calibration_sha256'] == tsha[str(folder/'train.npz')], 'replay used a different calibration file'
            exact = audit_exact(train, tm['rows'], replay, rm['rows'], CFG['cam_amplitudes'])
            with np.load(support_path) as support:
                for budget in args.budgets:
                    diagnostics = {}
                    for head, columns in (('normal', NI), ('friction', FI)):
                        key = f'{head}_b{budget}'
                        diagnostics[head] = episode_support(replay, support[key], meta['heads'][key]['threshold'], columns)
                    for i, episode in enumerate(rm['rows']):
                        row = dict(world=world, seed=seed, budget=budget, episode=episode['id'], profile=episode['profile'],
                                   amplitude=episode['amplitude'], primary=int(budget == CFG['profiles_train']),
                                   local_profile_in_budget=int(episode['profile'] < budget), **exact[i])
                        for head, records in diagnostics.items():
                            row.update({head+'_'+key: value for key, value in records[i].items()})
                        row['nn_coverage_pass'] = int(min(row['normal_coverage'], row['friction_coverage']) >= CFG['coverage_required'])
                        row['primary_local_replay_valid'] = int(bool(row['primary'] and row['exact_match'] and row['local_profile_in_budget']))
                        all_rows.append(row)
            print('AUDIT SUPPORT', world, seed, 'exact', sum(row['exact_match'] for row in exact), '/', len(exact), flush=True)
    assert all_rows
    csv_path = Path(args.csv); csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    groups = []
    for world in args.worlds:
        for seed in args.seeds:
            for budget in args.budgets:
                rows = [row for row in all_rows if row['world'] == world and row['seed'] == seed and row['budget'] == budget]
                groups.append(dict(world=world, seed=seed, budget=budget, episodes=len(rows),
                                   primary=budget == CFG['profiles_train'], exact_match_episodes=sum(row['exact_match'] for row in rows),
                                   local_profile_in_budget_episodes=sum(row['local_profile_in_budget'] for row in rows),
                                   normal_coverage_mean=float(np.mean([row['normal_coverage'] for row in rows])),
                                   friction_coverage_mean=float(np.mean([row['friction_coverage'] for row in rows])),
                                   normal_feature_error_max=max(row['exact_normal_feature_max_abs'] for row in rows),
                                   friction_feature_error_max=max(row['exact_friction_feature_max_abs'] for row in rows)))
    primary = [row for row in all_rows if row['primary']]
    dump(Path(args.out)/'summary.json', dict(tolerances=TOLERANCES, groups=groups, csv=str(csv_path), csv_sha256=digest(csv_path),
         sources=sources, primary_episodes=len(primary), primary_all_exact_match=bool(primary and all(row['primary_local_replay_valid'] for row in primary)),
         no_episodes_excluded=True, decisive_check='measured isolated B/A profile correspondence',
         distance_diagnostic='calibration-only .995 val quantile; empirical proximity, not a support guarantee',
         partial_budget='budget8 results are secondary; all main replay episodes retained even if their profile was not fitted'))


def control_stage(args):
    """Diagnostic coverage of all true controller trajectories; never a selection rule."""
    verify_lock(args.lock)
    lock_hash = digest(args.lock)
    budget = CFG['profiles_train']
    all_rows, sources, cache = [], {}, {}
    for meta_path in sorted(Path(args.control).glob('*.json')):
        metadata = json.loads(meta_path.read_text())
        episodes = metadata['rows']
        assert episodes, f'empty controller metadata: {meta_path}'
        world, seed = episodes[0]['world'], episodes[0]['seed']
        if world not in args.worlds or seed not in args.seeds:
            continue
        assert metadata['lock_sha256'] == lock_hash, f'controller used different lock: {meta_path}'
        path = meta_path.with_suffix('.npz')
        with np.load(path) as stored:
            data = {key: stored[key] for key in ('features', 'state')}
        assert len(episodes) == len(data['features'])
        assert [row['episode'] for row in episodes] == list(range(len(episodes))), 'controller episode indices do not match array rows'
        assert all(row['world'] == world and row['seed'] == seed for row in episodes)
        if (world, seed) not in cache:
            support_path = Path(args.out)/world/f'r{seed}.npz'
            meta = json.loads(support_path.with_suffix('.json').read_text())
            assert digest(support_path) == meta['npz_sha256']
            with np.load(support_path) as stored:
                points = {head: stored[f'{head}_b{budget}'] for head in ('normal', 'friction')}
            thresholds = {head: meta['heads'][f'{head}_b{budget}']['threshold'] for head in points}
            cache[(world, seed)] = points, thresholds
        points, thresholds = cache[(world, seed)]
        distances = {head: distance_matrix(data, points[head], columns) for head, columns in (('normal', NI), ('friction', FI))}
        summaries = {head: distance_summary(distances[head], thresholds[head]) for head in distances}
        for i, episode in enumerate(episodes):
            f = data['features'][i]
            row = dict(world=world, seed=seed, budget=budget, noise=episode['noise'], method=episode['method'],
                       episode=episode['episode'], amplitude=episode['amplitude'], waveform=episode['waveform'],
                       delta_min_m=float(np.min(f[:, 0]*.02)), delta_max_m=float(np.max(f[:, 0]*.02)),
                       velocity_max_abs_m_s=float(np.max(np.abs(data['state'][i, :, 1]))),
                       compression_rate_max_abs_m_s=float(np.max(np.abs(f[:, 1]*.1))),
                       normal_min_n=float(np.min(f[:, 9]*20)), normal_max_n=float(np.max(f[:, 9]*20)),
                       joint_coverage=float(np.mean((distances['normal'][i] <= thresholds['normal']) &
                                                    (distances['friction'][i] <= thresholds['friction']))),
                       excluded=0)
            for head in summaries:
                row.update({head+'_'+key: value for key, value in summaries[head][i].items()})
            all_rows.append(row)
        sources[str(path)] = digest(path); sources[str(meta_path)] = digest(meta_path)
    assert all_rows, 'no completed controller files matched the requested worlds/seeds'
    csv_path = Path(args.control_csv); csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    dump(Path(args.out)/'control_summary.json', dict(episodes=len(all_rows), budget=budget, csv=str(csv_path),
         csv_sha256=digest(csv_path), source_files=sources, no_episodes_excluded=True,
         observed_state='true reference features with true causal history, including in noisy-feedback conditions',
         joint_coverage_mean=float(np.mean([row['joint_coverage'] for row in all_rows])),
         interpretation='diagnostic only; controller results are never filtered by support'))
    print('CONTROL SUPPORT', len(all_rows), 'episodes; no exclusions', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['fit', 'evaluate', 'control'], required=True)
    parser.add_argument('--data', default=str(ROOT/'results/data'))
    parser.add_argument('--out', default=str(ROOT/'results/support'))
    parser.add_argument('--csv', default=str(ROOT/'results/support_metrics.csv'))
    parser.add_argument('--lock', default=str(ROOT/'results/protocol_lock.json'))
    parser.add_argument('--control', default=str(ROOT/'results/control'))
    parser.add_argument('--control-csv', default=str(ROOT/'results/control_support_metrics.csv'))
    parser.add_argument('--worlds', nargs='+', default=CFG['worlds'])
    parser.add_argument('--seeds', type=int, nargs='+', default=CFG['seeds'])
    parser.add_argument('--budgets', type=int, nargs='+', default=CFG['budgets'])
    args = parser.parse_args()
    {'fit': fit_stage, 'evaluate': evaluate_stage, 'control': control_stage}[args.stage](args)


if __name__ == '__main__':
    main()
