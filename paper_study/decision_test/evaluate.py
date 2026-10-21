"""Eight disjoint test-bank workers, with decisions persisted before target access."""
import argparse
import os
import signal
import sys
import time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from paper_study.decision_test.common import (HERE, STUDY_ID, BANKS, METHODS, BRANCHES,
    read, write_json, replace_json, sha, verify_hashes, load_config, bank_config,
    fresh_output, triplets)
from paper_study.decision_test.lock import verify_test_lock
from paper_study.decision_test.planning import branch_values
from paper_study.decision_study import evaluate as original
from paper_study.decision_study.inference import (load_model, load_prior, observable_prefix,
    filter_prefix, forecast_branch, select_policies, actual_forecast)

VALUE_KEYS = ('risk_m2', 'information_nats', 'physical_risk_m2',
              'planned_extra_probe_work_J', 'extra_seconds', 'fantasy_risk_mcse_m2',
              'fantasy_particle_indices', 'fantasy_noise')
IDENTITY = ['preparation_seed', 'normal_world', 'friction_world', 'episode', 'model_seed']
SCORE_KEYS = ('release_step', 'next_state', 'normal', 'friction', 'absolute_work_J', 'guard')


def native_inventory(data, cfg, digest):
    """Complete factor/clock/hash validation. No model scoring or history inspection."""
    data = Path(data).resolve()
    generation = read(data / 'generation_lock.json')
    process = read(data / 'process.json')
    if (generation.get('study_id') != STUDY_ID or generation.get('split') != 'test'
            or generation.get('config') != cfg or generation.get('test_lock_sha256') != digest
            or generation.get('smoke') is not False or process.get('status') != 'complete'
            or process.get('test_lock_sha256') != digest
            or (process.get('triplets'), process.get('contexts'), process.get('native_branches')) != (32, 1152, 3456)):
        raise ValueError('A complete authorized 32-triplet native test is required')
    manifest = read(data / 'artifact_manifest.json')
    if manifest.get('test_lock_sha256') != digest:
        raise ValueError('Native manifest authorization mismatch')
    verify_hashes(manifest['outputs_sha256'])
    expected = {data / f'p{i["seed"]}/{i["normal"]}_{i["friction"]}/{b}.npz'
                for i in triplets(cfg) for b in BRANCHES}
    if set(data.rglob('*.npz')) != expected:
        raise ValueError('Missing or unexpected native NPZ in the complete test grid')
    files = {p for p in data.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'}
    expected_files = expected | {p.with_suffix('.json') for p in expected}
    expected_files |= {p.parent / name for p in expected
                       for name in ('preparation_audit.json', 'triplet_audit.json')}
    expected_files |= {data / 'generation_lock.json', data / 'process.json'}
    if files != expected_files:
        raise ValueError('The native inventory requires every preparation/triplet audit and no extra files')
    if {str(p) for p in files} != set(manifest['outputs_sha256']):
        raise ValueError('Native manifest must bind every file, including audits and process')
    for path in sorted(expected):
        meta = read(path.with_suffix('.json'))
        if (meta.get('study_id') != STUDY_ID or meta.get('split') != 'test'
                or meta.get('test_lock_sha256') != digest or meta.get('status') != 'complete'
                or meta.get('dt') != cfg['dt'] or meta.get('mass_kg') != cfg['mass_kg']
                or not meta.get('prefix_audit', {}).get('bitwise_identical')
                or meta.get('npz_sha256') != sha(path)
                or meta.get('preparation_audit_sha256') != sha(path.parent / 'preparation_audit.json')
                or meta.get('recorded_steps') != meta.get('expected_steps')):
            raise ValueError('Incomplete/incorrect test branch metadata: ' + str(path))
        original.validate_native_branch(path, meta, cfg)
        # Structural checks of public arrays; future targets are not read here.
        with np.load(path, allow_pickle=False) as raw:
            e, t = raw['u'].shape
            for key in ('u', 'd', 'd_dot'):
                if raw[key].shape != (e, t) or not np.isfinite(raw[key]).all():
                    raise ValueError('Invalid public command array: ' + key)
            if not np.isfinite(raw['observations']).all():
                raise ValueError('Invalid public sensor array')
    return {str(p): sha(p) for p in sorted(files | {data / 'artifact_manifest.json'})}


def freeze(data, out, config_path, test_lock):
    locked = verify_test_lock(config_path, test_lock)  # before any native read
    cfg = locked['config']
    inputs = native_inventory(data, cfg, sha(test_lock))
    destination = fresh_output(out)
    write_json(destination / 'evaluation_lock.json', dict(study_id=STUDY_ID, config=cfg,
        config_path=str(Path(config_path).resolve()), test_lock=str(Path(test_lock).resolve()),
        test_lock_sha256=sha(test_lock), data_directory=str(Path(data).resolve()),
        inputs_sha256=inputs, triplets=triplets(cfg), model_seeds=BANKS,
        expected_policy_rows=294912, expected_branch_rows=27648, created_unix=time.time()))
    write_json(destination / 'process.json', dict(status='frozen', study_id=STUDY_ID,
                                                test_lock_sha256=sha(test_lock)))
    return destination


def verify_evaluation(out):
    out = Path(out).resolve()
    lock = read(out / 'evaluation_lock.json')
    authorized = verify_test_lock(lock['config_path'], lock['test_lock'])
    if (lock.get('study_id') != STUDY_ID or lock.get('config') != authorized['config']
            or lock.get('test_lock_sha256') != sha(lock['test_lock'])
            or lock.get('triplets') != triplets(lock['config']) or lock.get('model_seeds') != BANKS):
        raise ValueError('Evaluation is not bound to the full authorized test design')
    verify_hashes(lock['inputs_sha256'])
    actual = native_inventory(lock['data_directory'], lock['config'], lock['test_lock_sha256'])
    if actual != lock['inputs_sha256']:
        raise ValueError('Evaluation must retain the entire complete native inventory')
    return lock


def prepare(source, nw, fw, cfg):
    # The only native semantic input to planning is the common 0..50ms prefix.
    with np.load(Path(source) / 'release.npz', allow_pickle=False) as raw:
        public = observable_prefix({k: raw[k] for k in original.PUBLIC_RAW}, cfg)
    model = load_model(nw, fw, cfg)
    grid, prior = load_prior(nw, fw, cfg)
    snapshot = filter_prefix(model, grid, prior, public, cfg)
    branches = {b: forecast_branch(model, snapshot, public, b, cfg) for b in BRANCHES}
    return snapshot, branches, grid, prior


def write_plan(destination, snapshot, branches, grid, prior, cfg):
    started = time.perf_counter()
    values = branch_values(snapshot, branches, cfg)
    decisions = select_policies(values, cfg)
    payload = dict(**values, decision_weights=snapshot['weights'],
                   weight_trace=snapshot['weight_trace'], grid=grid, prior=prior)
    for budget, decision in decisions.items():
        payload[budget + '_eligible'] = decision['eligible']
        for method, probabilities in decision['policies'].items():
            payload[budget + '__' + method] = probabilities
    destination.mkdir(parents=True)
    with (destination / 'planned.npz').open('xb') as stream:
        np.savez_compressed(stream, **payload)
    write_json(destination / 'planned.json', dict(status='planned_before_later_sensor_assimilation_or_target_scoring',
        model_seed=cfg['model_seed'], particles=cfg['particles'], fantasies=cfg['fantasies'],
        seconds=time.perf_counter() - started, npz_sha256=sha(destination / 'planned.npz')))


def read_plan(destination, cfg):
    if read(destination / 'planned.json')['npz_sha256'] != sha(destination / 'planned.npz'):
        raise ValueError('Persisted plan was changed')
    with np.load(destination / 'planned.npz', allow_pickle=False) as raw:
        values = {k: raw[k].copy() for k in VALUE_KEYS}
        decisions = {b['name']: dict(eligible=raw[b['name'] + '_eligible'].copy(),
            policies={m: raw[b['name'] + '__' + m].copy() for m in METHODS}) for b in cfg['budgets']}
    return values, decisions


def begin_attempt(bankdir, digest, retry=False):
    """New exclusive attempts preserve every failed plan, error, timing and partial score."""
    bankdir = Path(bankdir)
    bankdir.mkdir(parents=True, exist_ok=True)
    claim = bankdir / '.active_worker'
    # Atomic cross-process exclusion. SIGKILL claims require explicit operator recovery.
    with claim.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        pointer = bankdir / 'process.json'
        previous = []
        if pointer.exists():
            old = read(pointer)
            if not retry or old.get('status') != 'failed':
                raise ValueError('Only an explicitly requested failed-bank retry is permitted')
            previous = old['attempts']
            for item in previous:
                verify_hashes(item['outputs_sha256'])
                if item['evaluation_lock_sha256'] != digest:
                    raise ValueError('Attempt lock changed')
        elif retry:
            raise ValueError('Cannot retry a bank without a retained failed attempt')
        number = len(previous) + 1
        attempt = bankdir / 'attempts' / f'attempt_{number:03d}'
        attempt.mkdir(parents=True)
        process = dict(status='running', pid=os.getpid(), evaluation_lock_sha256=digest,
                       attempt_index=number, prior_attempts=previous, started_unix=time.time(),
                       seconds_scope='Current attempt only; all earlier attempts retained in full')
        replace_json(pointer, dict(**process, attempts=previous))
        return attempt, process, claim
    except BaseException:
        claim.unlink()
        raise


def validate_rows(frame, branchframe, banks):
    contexts = 32 * 36 * len(banks)
    if len(frame) != contexts * 32 or len(branchframe) != contexts * 3:
        raise ValueError('Incomplete policy/branch Cartesian product')
    for table, extra in ((frame, ['budget', 'method']), (branchframe, ['branch'])):
        if table.duplicated(IDENTITY + extra).any():
            raise ValueError('Duplicate scored row keys')
        if set(table.model_seed) != set(banks) or not np.isfinite(table.select_dtypes(include=np.number)).all().all():
            raise ValueError('Wrong bank set or nonfinite scored row')
    expected = {(s, n, f, e, b) for s in range(55201, 55209)
                for n in ('maxwell', 'boucwen') for f in ('maxwell', 'boucwen')
                for e in range(36) for b in banks}
    for table, extra, levels in ((frame, ['budget', 'method'],
                                  {(b, m) for b in ('B0', 'B1', 'B2', 'B3') for m in METHODS}),
                                 (branchframe, ['branch'], {(b,) for b in BRANCHES})):
        if set(map(tuple, table[IDENTITY].drop_duplicates().to_numpy())) != expected:
            raise ValueError('The full bank/history/composition/episode inventory is required')
        for _, group in table.groupby(IDENTITY, sort=False):
            if set(map(tuple, group[extra].to_numpy())) != levels:
                raise ValueError('Missing or unexpected decision factor')


def run_bank(out, bank, *, retry=False):
    out = Path(out).resolve()
    lock = verify_evaluation(out)  # before native semantic reads
    cfg = bank_config(lock['config'], bank)
    digest = sha(out / 'evaluation_lock.json')
    bankdir = out / f'banks/r{bank}'
    attempt, process, claim = begin_attempt(bankdir, digest, retry)
    started = time.perf_counter()
    rows, branchrows = [], []

    def budget():
        if time.perf_counter() - started > cfg['evaluation_wall_seconds']:
            raise TimeoutError('Per-bank attempt CPU wall budget exhausted')

    def stopped(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    signal.signal(signal.SIGINT, stopped)
    signal.signal(signal.SIGTERM, stopped)
    try:
        # Phase 1: all32 decisions before later sensor assimilation or target scoring.
        for item in lock['triplets']:
            budget()
            seed, nw, fw = item['seed'], item['normal'], item['friction']
            source = Path(lock['data_directory']) / f'p{seed}/{nw}_{fw}'
            snapshot, branches, grid, prior = prepare(source, nw, fw, cfg)
            write_plan(attempt / f'p{seed}/{nw}_{fw}', snapshot, branches, grid, prior, cfg)
        plans = sorted(attempt.glob('p*/*/planned.npz'))
        if len(plans) != 32:
            raise ValueError('All32 plans must precede target access')
        planhashes = {str(p): sha(p) for p in plans}
        write_json(attempt / 'planning_complete.json', dict(plans=32,
            future_targets_entered_scoring=False, later_sensors_entered_assimilation=False,
            plans_sha256=planhashes, seconds=time.perf_counter() - started))
        # Phase 2: identical public reconstruction; saved choices are used verbatim.
        for item in lock['triplets']:
            budget()
            seed, nw, fw = item['seed'], item['normal'], item['friction']
            source = Path(lock['data_directory']) / f'p{seed}/{nw}_{fw}'
            destination = attempt / f'p{seed}/{nw}_{fw}'
            snapshot, branches, _, _ = prepare(source, nw, fw, cfg)
            values, decisions = read_plan(destination, cfg)
            predictions, scores = {}, {}
            for branch in BRANCHES:
                with np.load(source / (branch + '.npz'), allow_pickle=False) as raw:
                    sensors = {k: raw[k] for k in ('observations', 'observation_steps')}
                    prediction, weights = actual_forecast(snapshot, branches[branch], sensors, cfg)
                    targets = {k: raw[k] for k in SCORE_KEYS}
                scores[branch] = original.branch_scores(prediction, targets, cfg)
                predictions[branch + '_forecast'] = prediction
                predictions[branch + '_weights'] = weights
            with (destination / 'forecasts.npz').open('xb') as stream:
                np.savez_compressed(stream, **predictions)
            row, brow = original.episode_rows(read(source / 'release.json'), scores,
                                              decisions, values, branches, cfg)
            # Per-cell persistence keeps partial outputs and failures reviewable.
            pd.DataFrame(row).to_csv(destination / 'policy_episodes.csv', index=False)
            pd.DataFrame(brow).to_csv(destination / 'branch_episodes.csv', index=False)
            rows.extend(row)
            branchrows.extend(brow)
            print(f'DECISION TEST SCORE bank{bank} p{seed} {nw}/{fw}', flush=True)
        frame, branchframe = pd.DataFrame(rows), pd.DataFrame(branchrows)
        validate_rows(frame, branchframe, [bank])
        frame.to_csv(attempt / 'policy_episodes.csv', index=False)
        branchframe.to_csv(attempt / 'branch_episodes.csv', index=False)
        verify_hashes(planhashes)
        verify_evaluation(out)
        process.update(status='complete', policy_rows=len(rows), branch_rows=len(branchrows))
    except BaseException as exc:
        process.update(status='failed', error=repr(exc), policy_rows=len(rows), branch_rows=len(branchrows))
        raise
    finally:
        process['seconds'] = time.perf_counter() - started
        write_json(attempt / 'process.json', process)
        files = [p for p in attempt.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json']
        manifest = dict(evaluation_lock_sha256=digest, outputs_sha256={str(p): sha(p) for p in files})
        write_json(attempt / 'artifact_manifest.json', manifest)
        complete_binding = dict(manifest, outputs_sha256=dict(manifest['outputs_sha256'],
            **{str(attempt / 'artifact_manifest.json'): sha(attempt / 'artifact_manifest.json')}))
        replace_json(bankdir / 'process.json', dict(status=process['status'],
            evaluation_lock_sha256=digest, selected_attempt=str(attempt),
            attempts=[*process['prior_attempts'], complete_binding],
            attempt_index=process['attempt_index'], seconds=process['seconds']))
        claim.unlink()


def aggregate(out):
    out = Path(out).resolve()
    lock = verify_evaluation(out)
    digest = sha(out / 'evaluation_lock.json')
    destination = out / 'analysis'
    if destination.exists():
        raise FileExistsError('Analysis output must be new; no result replacement')
    frames, branchframes, inputs, bank_processes = [], [], {}, []
    for bank in BANKS:
        bankdir = out / f'banks/r{bank}'
        if (bankdir / '.active_worker').exists():
            raise ValueError('A bank worker is still active')
        status = read(bankdir / 'process.json')
        if status['status'] != 'complete' or status['evaluation_lock_sha256'] != digest:
            raise ValueError('Every bank must be complete under this evaluation lock')
        for attempt in status['attempts']:
            verify_hashes(attempt['outputs_sha256'])
            inputs.update(attempt['outputs_sha256'])
        selected = Path(status['selected_attempt'])
        if not selected.is_relative_to(bankdir / 'attempts'):
            raise ValueError('Invalid selected attempt path')
        frames.append(pd.read_csv(selected / 'policy_episodes.csv'))
        branchframes.append(pd.read_csv(selected / 'branch_episodes.csv'))
        bank_processes.append(status)
        inputs[str(bankdir / 'process.json')] = sha(bankdir / 'process.json')
    frame, branchframe = pd.concat(frames, ignore_index=True), pd.concat(branchframes, ignore_index=True)
    validate_rows(frame, branchframe, BANKS)
    destination.mkdir()
    frame.to_csv(destination / 'policy_episodes.csv', index=False)
    branchframe.to_csv(destination / 'branch_episodes.csv', index=False)
    metrics = ['position_mse_m2', 'velocity_mse', 'normal_mse_N2', 'friction_mse_N2',
        'probe_seconds', 'absolute_work_J', 'probe_absolute_work_J',
        'real_extra_work_budget_violation_probability', 'native_guard_count', 'planning_candidate_guard_count']
    for name, keys in [('pooled', ['budget', 'method']),
                       ('by_bank', ['model_seed', 'budget', 'method']),
                       ('by_history', ['preparation_seed', 'budget', 'method']),
                       ('by_bank_history', ['model_seed', 'preparation_seed', 'budget', 'method']),
                       ('by_composition', ['normal_world', 'friction_world', 'budget', 'method'])]:
        summary = frame.groupby(keys, sort=True)[metrics].mean().reset_index()
        summary['pooled_position_rmse_um'] = 1e6 * np.sqrt(summary.position_mse_m2)
        rms = frame.assign(episode_position_rmse_um=1e6 * np.sqrt(frame.position_mse_m2))
        means = rms.groupby(keys, sort=True).episode_position_rmse_um.mean().reset_index()
        summary = summary.merge(means.rename(columns={'episode_position_rmse_um':
            'mean_episode_position_rmse_um'}), on=keys, validate='one_to_one')
        summary.to_csv(destination / (name + '.csv'), index=False)
    write_json(destination / 'summary.json', dict(study_id=STUDY_ID,
        evaluation_lock_sha256=digest, policy_rows=len(frame), branch_rows=len(branchframe),
        contexts=1152, model_banks=8, bank_processes=bank_processes,
        no_population_confidence_intervals=True, no_positive_success_assumption=True,
        primary='Descriptive pooled MSE/costs; equal weight per crossed bank/history/composition/context',
        limits='Known scalar equation prior; fixture-reachable histories; finite mixture/fantasies; '
               'model-dependent expected budgets; shared contexts and banks are not independent rows; '
               'forecasting rather than tracking; native timestep sensitivity remains material'))
    verify_hashes(inputs)
    verify_evaluation(out)
    files = [p for p in destination.iterdir() if p.is_file()]
    write_json(destination / 'artifact_manifest.json', dict(evaluation_lock_sha256=digest,
        inputs_sha256=inputs, outputs_sha256={str(p): sha(p) for p in files}))
    replace_json(out / 'process.json', dict(status='complete', study_id=STUDY_ID,
        evaluation_lock_sha256=digest, policy_rows=len(frame), branch_rows=len(branchframe),
        analysis_manifest_sha256=sha(destination / 'artifact_manifest.json')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze', 'run-bank', 'aggregate', 'verify'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data', type=Path)
    p.add_argument('--config', type=Path, default=HERE / 'config.json')
    p.add_argument('--test-lock', type=Path)
    p.add_argument('--bank', type=int, choices=BANKS)
    p.add_argument('--retry', action='store_true')
    args = p.parse_args()
    if args.stage == 'freeze':
        if args.data is None or args.test_lock is None:
            p.error('--data and --test-lock required')
        print(freeze(args.data, args.out, args.config, args.test_lock))
    elif args.stage == 'run-bank':
        if args.bank is None:
            p.error('--bank required')
        run_bank(args.out, args.bank, retry=args.retry)
    elif args.stage == 'aggregate':
        aggregate(args.out)
    else:
        verify_evaluation(args.out)
        print('Verified complete test evaluation input binding')


if __name__ == '__main__':
    main()
