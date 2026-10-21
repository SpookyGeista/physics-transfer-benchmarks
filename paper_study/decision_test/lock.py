"""Single prospective generator/evaluator/model/prior union authorization."""
import argparse
import time
from pathlib import Path
from paper_study.decision_test.common import (HERE, REPO, STUDY_ID, SEEDS, BANKS,
    read, sha, write_json, verify_hashes, load_config, bank_config)
from paper_study.decision_study.inference import model_paths, prior_path, load_prior

ROLE = 'paper_study_native_v2_test_authorization'


def source_paths(config_path):
    """Full runtime union, including immutable mechanics and every selected bank."""
    cfg = load_config(config_path)
    base = REPO / 'paper_study/decision_study'
    paths = [*HERE.glob('*.py'), *(HERE / 'tests').glob('*.py'), HERE / 'protocol.md',
             Path(config_path).resolve(), *base.glob('*.py'), *(base / 'tests').glob('*.py'),
             base / 'protocol.md', base / 'config.json',
             REPO / 'paper_study/costed_benchmark/geometry.py',
             REPO / 'paper_study/src/preparation.py', REPO / 'paper_study/src/preparation_v2.py',
             REPO / 'study_v2/src/decision_study/reference.py',
             REPO / 'study_v2/src/decision_study/common.py',
             REPO / 'study_v2/src/decision_study/isaac_rig.py',
             REPO / 'study_v2/configs/experiment.json', REPO / 'research_next/src/state_id.py']
    prior_root = REPO / cfg['prior_bank_directory']
    prior_lock = read(prior_root / 'prior_lock.json')
    paths += [prior_root / 'prior_lock.json', prior_root / 'process.json']
    paths += [Path(p) for p in prior_lock['sources_sha256']]
    paths += [Path(p) for p in cfg['resolution_evidence_sha256']]
    for bank in BANKS:
        local = bank_config(cfg, bank)
        for nw, fw in cfg['compositions']:
            paths += [*model_paths(nw, fw, local), prior_path(nw, fw, local)]
    return sorted(set(p.resolve() for p in paths))


def snapshot(config_path):
    return {str(p): sha(p) for p in source_paths(config_path)}


def validate_priors(cfg):
    root = REPO / cfg['prior_bank_directory']
    if read(root / 'process.json').get('status') != 'complete':
        raise ValueError('A completed public prior bank is required')
    verify_hashes(read(root / 'prior_lock.json')['sources_sha256'])
    expected = {prior_path(nw, fw, bank_config(cfg, bank)).resolve()
                for bank in BANKS for nw, fw in cfg['compositions']}
    # Other conditions (e.g. clean calibration) may coexist, but the noisy grid is exact.
    if set(root.glob('*_*/r*/noisy.npz')) != expected:
        raise ValueError('Public prior must contain the exact 32 noisy composition/bank files')
    for bank in BANKS:
        for nw, fw in cfg['compositions']:
            load_prior(nw, fw, bank_config(cfg, bank))


def verify_test_lock(config_path, test_lock):
    if config_path is None or test_lock is None:
        raise ValueError('Explicit configuration and complete test authorization required')
    cfg = load_config(config_path, final=True)
    locked = read(test_lock)
    if (locked.get('role') != ROLE or locked.get('study_id') != STUDY_ID
            or locked.get('status') != 'locked' or locked.get('split') != 'test'
            or locked.get('preparation_seeds') != SEEDS or locked.get('model_seeds') != BANKS
            or locked.get('config') != cfg):
        raise ValueError('Wrong study, incomplete design, or configuration authorization')
    current = snapshot(config_path)
    if locked.get('sources_sha256') != current:
        raise ValueError('Full generator/evaluator/model/prior authorization union changed')
    review = locked.get('review_sha256')
    if not review:
        raise ValueError('Independent source review must be bound')
    verify_hashes(review)
    return locked


def freeze(config_path, destination, review_path):
    cfg = load_config(config_path, final=True)
    validate_priors(cfg)
    sources = snapshot(config_path)
    review_path = Path(review_path).resolve()
    review = read(review_path)
    if (review.get('study_id') != STUDY_ID or review.get('approved_for_test') is not True
            or review.get('sources_sha256') != sources
            or review.get('reserved_histories_drawn') is not False):
        raise ValueError('Independent review of this exact complete union is required')
    destination = Path(destination).resolve()
    root = REPO / 'paper_study/results'
    if not destination.is_relative_to(root) or destination.exists():
        raise ValueError('An exclusive new authorization file below results is required')
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, dict(role=ROLE, study_id=STUDY_ID, status='locked', split='test',
        preparation_seeds=SEEDS, model_seeds=BANKS, config=cfg, sources_sha256=sources,
        review_sha256={str(review_path): sha(review_path)}, created_unix=time.time(),
        reserved_histories_drawn=False, expected_triplets=32, expected_contexts=1152,
        expected_native_branches=3456, expected_policy_rows=294912, expected_branch_rows=27648))
    verify_test_lock(config_path, destination)
    return sha(destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze', 'verify'])
    p.add_argument('--config', type=Path, default=HERE / 'config.json')
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--review', type=Path)
    args = p.parse_args()
    if args.stage == 'freeze':
        if args.review is None:
            p.error('--review is required')
        print(freeze(args.config, args.lock, args.review))
    else:
        verify_test_lock(args.config, args.lock)
        print('Verified complete decision-test union', sha(args.lock))


if __name__ == '__main__':
    main()
