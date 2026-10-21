"""Test inventory and validation; this module never draws material histories."""
from pathlib import Path
import copy
import json
import os
import types
import numpy as np
from paper_study.decision_study import common as original

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STUDY_ID = 'decision_test_v1'
SEEDS = list(range(55201, 55209))
BANKS = list(range(1, 9))
METHODS = ('fixed_release', 'fixed_hold100', 'fixed_continue100', 'uniform',
           'uniform_expected', 'bayes_risk', 'information_gain', 'physical_only')
BRANCHES = original.BRANCHES
sha, write_json, verify_hashes = original.sha, original.write_json, original.verify_hashes
read = lambda path: json.loads(Path(path).read_text())


def copied_function(function, replacements):
    """Local namespace adapter; the original function/code/globals remain untouched."""
    unknown = set(replacements) - set(function.__globals__)
    if unknown:
        raise ValueError(f'Cannot replace unknown globals: {unknown}')
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    result = types.FunctionType(function.__code__, namespace, function.__name__,
                                function.__defaults__, function.__closure__)
    result.__kwdefaults__ = copy.copy(function.__kwdefaults__)
    return result


def load_config(path=None, *, final=False):
    cfg = read(path or HERE / 'config.json')
    validate_config(cfg, final=final)
    return cfg


def validate_config(cfg, *, final=False):
    base = read(original.HERE / 'config.json')
    changed = {'role', 'version', 'preparation_seeds', 'prior_bank_directory',
               'particles', 'fantasies', 'native_wall_seconds', 'evaluation_wall_seconds'}
    extras = {'study_id', 'model_seeds', 'resolution_status', 'resolution_evidence_sha256',
              'resolution_reason', 'posterior_chunk_size'}
    if set(cfg) != set(base) | extras:
        raise ValueError('Configuration key inventory differs from the declared extension')
    for key in set(base) - changed:
        if cfg[key] != base[key]:
            raise ValueError('The physical/decision design must stay unchanged: ' + key)
    if cfg['study_id'] != STUDY_ID or cfg['preparation_seeds'] != SEEDS or cfg['model_seeds'] != BANKS:
        raise ValueError('All eight reserved blocks and all eight fixed banks required')
    if cfg['particles'] not in (201, 401) or cfg['fantasies'] not in (64, 256, 1024):
        raise ValueError('Unsupported prospectively allowed numerical resolution')
    chunk = cfg['posterior_chunk_size']
    if isinstance(chunk, bool) or not isinstance(chunk, int) or not 1 <= chunk <= 64:
        raise ValueError('Posterior memory chunk must be an integer in [1,64]')
    for key in ('native_wall_seconds', 'evaluation_wall_seconds'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(key)
    prior = (REPO / cfg['prior_bank_directory']).resolve()
    if not prior.is_relative_to(REPO / 'paper_study/results') or prior == REPO / 'paper_study/results':
        raise ValueError('Public prior bank must be a named results subdirectory')
    if cfg['resolution_status'] not in ('pending', 'final'):
        raise ValueError('Unknown resolution status')
    if final:
        if cfg['resolution_status'] != 'final' or not cfg['resolution_reason'].strip():
            raise ValueError('Resolution must be declared after development and before test access')
        if not cfg['resolution_evidence_sha256']:
            raise ValueError('Bind the completed development resolution evidence')
        verify_hashes(cfg['resolution_evidence_sha256'])


def bank_config(cfg, bank):
    if isinstance(bank, bool) or bank not in BANKS:
        raise ValueError('One of the eight frozen banks required')
    return dict(copy.deepcopy(cfg), model_seed=int(bank))


def triplets(cfg):
    return [dict(seed=s, normal=n, friction=f) for s in cfg['preparation_seeds']
            for n, f in cfg['compositions']]


def replace_json(path, value):
    """Atomic update of an owned status/metadata file, never a source or old archive."""
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    write_json(temporary, value)
    temporary.replace(path)


def fresh_output(path):
    path = Path(path).resolve()
    root = REPO / 'paper_study/results'
    if not path.is_relative_to(root) or path == root or path.exists():
        raise ValueError('A fresh named output below paper_study/results is required')
    path.mkdir(parents=True)
    return path
