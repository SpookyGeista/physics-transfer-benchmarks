"""Guarded native test wrapper around the unchanged three-body-group loop."""
import argparse
import os
import signal
import sys
import time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.dont_write_bytecode = True
from paper_study.decision_test.common import (HERE, STUDY_ID, BRANCHES, load_config,
    copied_function, fresh_output, read, write_json, replace_json, sha, triplets)
from paper_study.decision_test.lock import verify_test_lock
from paper_study.decision_study import generate as original


def authorized_preparation(config_path, test_lock, cfg, nw, fw, seed):
    """Only the split/authorization adapter differs from the immutable native loop."""
    expected = [row['preparation_index'] for row in original.factors(seed, cfg)]

    def factory(world, indices, called_seed, split, *, normal_world=None):
        locked = verify_test_lock(config_path, test_lock)  # before EVERY reserved draw
        if (locked['config'] != cfg or called_seed != seed or world != fw or normal_world != nw
                or split != 'dev' or list(indices) != expected
                or seed not in cfg['preparation_seeds'] or [nw, fw] not in cfg['compositions']):
            raise ValueError('Preparation call differs from the complete authorized test cell')
        return original.referencepair_indices(world, indices, seed, 'test',
                                             normal_world=nw, test_lock=test_lock)
    return factory


def run_triplet(rig, nw, fw, seed, cfg, destination=None, *, config_path=None,
                test_lock=None, deadline=None):
    locked = verify_test_lock(config_path, test_lock)
    if locked['config'] != cfg or seed not in cfg['preparation_seeds'] or [nw, fw] not in cfg['compositions']:
        raise ValueError('Direct calls require the exact authorized configuration and test factors')
    if destination is not None and Path(destination).exists():
        raise FileExistsError('Never draw or replace a previously existing native triplet')
    rows = original.factors(seed, cfg)
    factory = authorized_preparation(config_path, test_lock, cfg, nw, fw, seed)
    adapted = copied_function(original.run_triplet, {'referencepair_indices': factory})
    digest = sha(test_lock)
    try:
        arrays, metadata = adapted(rig, nw, fw, seed, cfg, destination,
                                   factor_rows=rows, deadline=deadline)
    finally:
        # Enrich only this call's NEW metadata. Never edit the imported source or old results.
        if destination is not None and Path(destination).exists():
            for name in ('triplet_audit.json', *(b + '.json' for b in BRANCHES)):
                path = Path(destination) / name
                if path.is_file():
                    meta = read(path)
                    meta.update(study_id=STUDY_ID, split='test', test_lock_sha256=digest,
                                preparation_access='Explicit full-union authorization; physical fixture replay')
                    replace_json(path, meta)
        verify_test_lock(config_path, test_lock)
    metadata.update(study_id=STUDY_ID, split='test', test_lock_sha256=digest)
    return arrays, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--test-lock', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.device != 'cpu':
        parser.error('Only CPU PhysX is permitted')
    locked = verify_test_lock(args.config, args.test_lock)
    cfg = load_config(args.config, final=True)
    out = fresh_output(args.out)
    digest = sha(args.test_lock)
    write_json(out / 'generation_lock.json', dict(study_id=STUDY_ID, split='test',
        test_lock=str(args.test_lock.resolve()), test_lock_sha256=digest,
        config_path=str(args.config.resolve()), config=cfg,
        sources_sha256=locked['sources_sha256'], smoke=False, expected_triplets=32))
    started = time.perf_counter()
    process = dict(status='running', pid=os.getpid(), command=sys.argv, started_unix=time.time(),
                   test_lock_sha256=digest, study_id=STUDY_ID)
    write_json(out / 'process.json', process)
    app = None
    complete = 0

    def stopped(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    signal.signal(signal.SIGINT, stopped)
    signal.signal(signal.SIGTERM, stopped)
    try:
        app = AppLauncher(args).app
        signal.signal(signal.SIGINT, stopped)
        signal.signal(signal.SIGTERM, stopped)
        original.validate_legacy_contract(cfg)
        from decision_study.isaac_rig import IsaacRig
        rig = IsaacRig(108, device='cpu', dt=cfg['dt'])
        for item in triplets(cfg):
            seed, nw, fw = item['seed'], item['normal'], item['friction']
            run_triplet(rig, nw, fw, seed, cfg, out / f'p{seed}/{nw}_{fw}',
                        config_path=args.config, test_lock=args.test_lock,
                        deadline=started + cfg['native_wall_seconds'])
            complete += 1
            print(f'DECISION TEST NATIVE {complete}/32 p{seed} {nw}/{fw}', flush=True)
        verify_test_lock(args.config, args.test_lock)
        process.update(status='complete')
    except BaseException as exc:
        process.update(status='failed', error=repr(exc))
        raise
    finally:
        process.update(seconds=time.perf_counter() - started, triplets=complete,
                       contexts=complete * 36, native_branches=complete * 108)
        replace_json(out / 'process.json', process)
        files = [p for p in out.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json']
        write_json(out / 'artifact_manifest.json', dict(test_lock_sha256=digest,
            outputs_sha256={str(p): sha(p) for p in files}))
        if app is not None:
            # Persisted process/manifest first. Avoid known hanging renderer teardown.
            app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == '__main__':
    main()
