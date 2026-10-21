"""Synthetic contracts only. Never draws a reserved preparation or starts Isaac."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from paper_study.decision_test import common, lock, generate, planning, evaluate
from paper_study.decision_study import generate as basegen, inference as baseinf


def test_final_config_and_exact_runtime_source_union():
    cfg = common.load_config(final=True)
    assert cfg['particles'] == 201 and cfg['fantasies'] == 1024
    assert len(common.triplets(cfg)) == 32
    paths = lock.source_paths(common.HERE / 'config.json')
    assert len(paths) == len(set(paths))
    for suffix in ('decision_test/generate.py', 'decision_test/evaluate.py',
                   'decision_test/planning.py', 'decision_test/lock.py',
                   'src/preparation_v2.py', 'src/preparation.py',
                   'decision_study/reference.py', 'decision_study/isaac_rig.py',
                   'study_v2/configs/experiment.json'):
        assert any(str(p).endswith(suffix) for p in paths)
    assert sum(p.name == 'model.json' for p in paths) == 16
    assert sum(p.suffix == '.npz' for p in paths) == 32
    lock.validate_priors(cfg)
    changed = copy.deepcopy(cfg)
    changed['dt'] /= 2
    with pytest.raises(ValueError, match='unchanged'):
        common.validate_config(changed)
    changed = copy.deepcopy(cfg)
    changed['resolution_status'] = 'pending'
    with pytest.raises(ValueError, match='Resolution'):
        common.validate_config(changed, final=True)


def test_lock_requires_full_union_and_exact_review(tmp_path, monkeypatch):
    cfg = common.load_config()
    file = tmp_path / 'runtime.py'
    file.write_text('complete generator and evaluator union')
    review = tmp_path / 'review.json'
    review.write_text('{}')
    sources = {str(file): common.sha(file)}
    monkeypatch.setattr(lock, 'snapshot', lambda _: {str(file): common.sha(file)})
    authorization = dict(role=lock.ROLE, study_id=common.STUDY_ID, status='locked', split='test',
        preparation_seeds=common.SEEDS, model_seeds=common.BANKS, config=cfg,
        sources_sha256=sources, review_sha256={str(review): common.sha(review)})
    path = tmp_path / 'authorization.json'
    path.write_text(json.dumps(authorization))
    lock.verify_test_lock(common.HERE / 'config.json', path)
    authorization['study_id'] = 'another_study'
    path.write_text(json.dumps(authorization))
    with pytest.raises(ValueError, match='Wrong study'):
        lock.verify_test_lock(common.HERE / 'config.json', path)
    authorization['study_id'] = common.STUDY_ID
    path.write_text(json.dumps(authorization))
    file.write_text('changed evaluator')
    with pytest.raises(ValueError, match='union changed'):
        lock.verify_test_lock(common.HERE / 'config.json', path)
    with pytest.raises(ValueError, match='Explicit'):
        lock.verify_test_lock(None, None)


def test_direct_generator_denies_before_any_draw(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Reserved preparation was accessed without authorization')
    monkeypatch.setattr(generate, 'authorized_preparation', forbidden)
    with pytest.raises(ValueError, match='Explicit'):
        generate.run_triplet(None, 'maxwell', 'boucwen', 55201, common.load_config())
    cfg = common.load_config()
    monkeypatch.setattr(generate, 'verify_test_lock', lambda *a: {'config': cfg})
    with pytest.raises(FileExistsError):
        generate.run_triplet(None, 'maxwell', 'boucwen', 55201, cfg, tmp_path,
                             config_path='stub', test_lock='stub')


def test_factory_checks_every_call_and_does_not_mutate_original(monkeypatch):
    cfg = common.load_config()
    original_globals = dict(basegen.run_triplet.__globals__)
    original_code = basegen.run_triplet.__code__
    events = []

    def guard(*args):
        events.append('guard')
        return {'config': cfg}

    def stub_reference(world, indices, seed, split, **kwargs):
        # A sentinel only. It never calls any preparation generator.
        events.append(('draw_sentinel', world, seed, split, kwargs))
        return object()

    monkeypatch.setattr(generate, 'verify_test_lock', guard)
    monkeypatch.setattr(generate, 'original', SimpleNamespace(
        factors=basegen.factors, referencepair_indices=stub_reference))
    factory = generate.authorized_preparation('cfg', 'lock', cfg, 'maxwell', 'boucwen', 55201)
    indices = np.array([r['preparation_index'] for r in basegen.factors(55201, cfg)])
    factory('boucwen', indices, 55201, 'dev', normal_world='maxwell')
    assert events[0] == 'guard'
    assert events[1][3:] == ('test', {'normal_world': 'maxwell', 'test_lock': 'lock'})
    with pytest.raises(ValueError, match='authorized test cell'):
        factory('boucwen', indices[:-1], 55201, 'dev', normal_world='maxwell')
    assert events[-1] == 'guard'
    assert basegen.run_triplet.__code__ is original_code
    assert original_globals.keys() == basegen.run_triplet.__globals__.keys()
    assert all(basegen.run_triplet.__globals__[k] is value for k, value in original_globals.items())


def test_adapter_uses_same_function_code_without_global_mutation():
    original = basegen.run_triplet
    old = original.__globals__['referencepair_indices']
    replacement = lambda *a, **k: None
    adapted = common.copied_function(original, {'referencepair_indices': replacement})
    assert adapted.__code__ is original.__code__
    assert adapted.__kwdefaults__ == original.__kwdefaults__
    assert adapted.__globals__ is not original.__globals__
    assert adapted.__globals__['referencepair_indices'] is replacement
    assert original.__globals__['referencepair_indices'] is old


def synthetic_candidates(e=3, p=7, horizon=9):
    rng = np.random.default_rng(98765)
    weights = rng.uniform(.1, 1., (e, p))
    weights /= weights.sum(axis=1, keepdims=True)
    snapshot = dict(weights=weights, logweight=np.log(weights), weight_trace=weights[:, None, :])
    branches = {}
    for branch in common.BRANCHES:
        j = 0 if branch == 'release' else 4
        future = rng.normal(0, .001, (e, p, horizon, 4))
        branches[branch] = dict(future=future, sensors=rng.normal(0, .001, (e, p, j, 3)),
            probe_work=np.full((e, p), 0. if j == 0 else .001),
            extra_seconds=np.full(e, 0. if j == 0 else .1),
            sensor_steps=np.arange(30, 61, 10) if j else np.array([], dtype=int),
            candidate_guard_steps=np.zeros((e, p), dtype=int))
    return snapshot, branches


@pytest.mark.parametrize('chunk', [1, 17, 32])
def test_chunked_fantasies_bitwise_same_equations_draws_and_decisions(chunk):
    cfg = common.load_config()
    cfg['posterior_chunk_size'] = chunk
    snapshot, branches = synthetic_candidates()
    global_posterior = baseinf.branch_values.__globals__['posterior']
    expected = baseinf.branch_values(snapshot, branches, cfg)
    actual = planning.branch_values(snapshot, branches, cfg)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    first, second = baseinf.select_policies(expected, cfg), baseinf.select_policies(actual, cfg)
    for budget in first:
        for method in first[budget]['policies']:
            np.testing.assert_array_equal(first[budget]['policies'][method], second[budget]['policies'][method])
    assert baseinf.branch_values.__globals__['posterior'] is global_posterior


def test_attempt_exclusion_retention_and_tampering(tmp_path):
    bank = tmp_path / 'bank'
    attempt, process, claim = evaluate.begin_attempt(bank, 'digest')
    with pytest.raises(FileExistsError):
        evaluate.begin_attempt(bank, 'digest')
    record = attempt / 'error.json'
    common.write_json(record, {'status': 'failed', 'seconds': 3., 'error': 'synthetic'})
    binding = dict(evaluation_lock_sha256='digest', outputs_sha256={str(record): common.sha(record)})
    common.replace_json(bank / 'process.json', {'status': 'failed', 'attempts': [binding]})
    claim.unlink()
    second, status, secondclaim = evaluate.begin_attempt(bank, 'digest', retry=True)
    assert second.name == 'attempt_002' and status['prior_attempts'] == [binding]
    assert common.read(record)['error'] == 'synthetic'
    secondclaim.unlink()
    common.replace_json(bank / 'process.json', {'status': 'failed', 'attempts': [binding]})
    record.write_text('tampered old failure')
    with pytest.raises(ValueError, match='hash mismatch'):
        evaluate.begin_attempt(bank, 'digest', retry=True)
    assert not (bank / '.active_worker').exists()


def test_full_bank_two_phase_worker_synthetic_no_histories(tmp_path, monkeypatch):
    """32 cells and36864 policy rows; lightweight candidates, no material draws."""
    cfg = common.load_config()
    cfg['fantasies'] = 4  # synthetic workload only; production config validator rejects it
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'evaluation_lock.json').write_text('{}')
    data = tmp_path / 'data'
    evidence = dict(config=cfg, data_directory=str(data), triplets=common.triplets(cfg))
    monkeypatch.setattr(evaluate, 'verify_evaluation', lambda _: evidence)
    events = []

    def fake_prepare(source, nw, fw, local):
        snapshot, branches = synthetic_candidates(36, 3, 100)
        # Keep actual sensors/forecast finite; all inputs are synthetic.
        return snapshot, branches, np.arange(3), np.ones(3) / 3

    monkeypatch.setattr(evaluate, 'prepare', fake_prepare)
    original_read_plan = evaluate.read_plan

    def observed_plan(*args):
        assert len(list((out / 'banks/r1/attempts/attempt_001').glob('p*/*/planned.npz'))) == 32
        events.append('score')
        return original_read_plan(*args)

    monkeypatch.setattr(evaluate, 'read_plan', observed_plan)
    for item in evidence['triplets']:
        seed, nw, fw = item['seed'], item['normal'], item['friction']
        source = data / f'p{seed}/{nw}_{fw}'
        source.mkdir(parents=True)
        metadata = dict(preparation_seed=seed, normal_world=nw, friction_world=fw,
                        rows=basegen.factors(seed, cfg))
        common.write_json(source / 'release.json', metadata)
        for branch in common.BRANCHES:
            release = 20 if branch == 'release' else 60
            total = release + 100
            with (source / (branch + '.npz')).open('xb') as stream:
                np.savez_compressed(stream, observations=np.zeros((36, release // 10 + 1, 3)),
                    observation_steps=np.arange(0, release + 1, 10), release_step=np.array(release),
                    next_state=np.zeros((36, total, 2)), normal=np.zeros((36, total)),
                    friction=np.zeros((36, total)), absolute_work_J=np.zeros((36, total)),
                    guard=np.zeros((36, total), dtype=int))
    evaluate.run_bank(out, 1)
    status = common.read(out / 'banks/r1/process.json')
    assert status['status'] == 'complete' and len(events) == 32
    attempt = Path(status['selected_attempt'])
    policies = pd.read_csv(attempt / 'policy_episodes.csv')
    branches = pd.read_csv(attempt / 'branch_episodes.csv')
    evaluate.validate_rows(policies, branches, [1])
    assert len(policies) == 36864 and len(branches) == 3456
    common.verify_hashes(status['attempts'][0]['outputs_sha256'])
    assert not (out / 'banks/r1/.active_worker').exists()
    with pytest.raises(ValueError, match='failed-bank retry'):
        evaluate.run_bank(out, 1)
    policies.loc[0, 'method'] = 'missing'
    with pytest.raises(ValueError, match='decision factor'):
        evaluate.validate_rows(policies, branches, [1])


def test_evaluation_freeze_denies_before_native_read(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Native data was opened before the full union guard')
    monkeypatch.setattr(evaluate, 'native_inventory', forbidden)
    with pytest.raises(ValueError, match='Explicit'):
        evaluate.freeze('unread', 'unused', None, None)


def test_adapter_and_complete_native_inventory_without_reserved_draw(tmp_path, monkeypatch):
    """Synthetic rig uses ONLY dev34001, then tests schema labels on copies."""
    from paper_study.decision_study.smoke import SyntheticRig
    cfg = common.load_config()
    test_lock = tmp_path / 'stub-lock.json'
    test_lock.write_text('{}')
    calls = []
    untouched = basegen.referencepair_indices

    def dev_fixture(world, indices, seed, split, *, normal_world, test_lock):
        # Assert wrapper authorization arguments, but do not draw that reserved seed.
        assert split == 'test' and seed == 55201
        calls.append(34001)
        return untouched(world, indices, 34001, 'dev', normal_world=normal_world)

    proxy = SimpleNamespace(**basegen.__dict__)
    proxy.referencepair_indices = dev_fixture
    monkeypatch.setattr(generate, 'original', proxy)
    monkeypatch.setattr(generate, 'verify_test_lock', lambda *a: {'config': cfg})
    template = tmp_path / 'template'
    arrays, metadata = generate.run_triplet(SyntheticRig(108, cfg['dt']), 'maxwell', 'boucwen',
        55201, cfg, template, config_path='stub', test_lock=test_lock)
    assert calls == [34001] and basegen.referencepair_indices is untouched
    assert common.read(template / 'release.json')['study_id'] == common.STUDY_ID
    assert metadata['prefix_audit']['bitwise_identical']
    digest = common.sha(test_lock)
    data = tmp_path / 'synthetic_native'
    for item in common.triplets(cfg):
        seed, nw, fw = item['seed'], item['normal'], item['friction']
        source = data / f'p{seed}/{nw}_{fw}'
        source.mkdir(parents=True)
        # Opaque audit preservation is checked without exposing it to inference.
        (source / 'preparation_audit.json').write_bytes((template / 'preparation_audit.json').read_bytes())
        (source / 'triplet_audit.json').write_bytes((template / 'triplet_audit.json').read_bytes())
        for branch in common.BRANCHES:
            raw = {k: value.copy() for k, value in arrays[branch].items()}
            raw['preparation_seed'][:] = seed
            path = source / (branch + '.npz')
            with path.open('xb') as stream:
                np.savez_compressed(stream, **raw)
            meta = common.read(template / (branch + '.json'))
            meta.update(preparation_seed=seed, normal_world=nw, friction_world=fw,
                rows=basegen.factors(seed, cfg), npz_sha256=common.sha(path),
                backend='SYNTHETIC SCHEMA COPIES: NOT NATIVE OR RESERVED MATERIAL EVIDENCE')
            common.write_json(path.with_suffix('.json'), meta)
    common.write_json(data / 'generation_lock.json', dict(study_id=common.STUDY_ID,
        split='test', config=cfg, test_lock_sha256=digest, smoke=False))
    common.write_json(data / 'process.json', dict(status='complete', test_lock_sha256=digest,
        triplets=32, contexts=1152, native_branches=3456))

    def manifest():
        path = data / 'artifact_manifest.json'
        if path.exists():
            path.unlink()
        common.write_json(path, dict(test_lock_sha256=digest,
            outputs_sha256={str(p): common.sha(p) for p in data.rglob('*') if p.is_file()}))

    manifest()
    inventory = evaluate.native_inventory(data, cfg, digest)
    assert sum(p.endswith('.npz') for p in inventory) == 96
    file = data / 'p55201/maxwell_maxwell/release.npz'
    with np.load(file, allow_pickle=False) as raw:
        wrong = {key: raw[key].copy() for key in raw.files}
    wrong['geometry'][0] = 2
    with file.open('wb') as stream:
        np.savez_compressed(stream, **wrong)
    meta = common.read(file.with_suffix('.json'))
    meta['npz_sha256'] = common.sha(file)
    common.replace_json(file.with_suffix('.json'), meta)
    manifest()
    with pytest.raises(ValueError, match='factor array mismatch'):
        evaluate.native_inventory(data, cfg, digest)
