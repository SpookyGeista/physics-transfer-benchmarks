"""Local protocol/source lock; no original study source is changed."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
LOCK = ROOT/'design_lock.json'
SOURCES = ('config.json', 'protocol.md', 'model.py', 'profiles.py', 'generate.py',
           'fit.py', 'summarize.py', 'lock.py', 'test_learning.py')
REFERENCE = REPO/'paper_study/alternatives/multistate_friction.py'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    data = json.loads(LOCK.read_text())
    for relative, expected in data['files'].items():
        path = REPO/relative
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f'locked source changed or missing: {relative}')
    return digest(LOCK)


def freeze():
    if LOCK.exists():
        raise RuntimeError('refusing to overwrite existing design lock')
    paths = [ROOT/name for name in SOURCES] + [REFERENCE]
    files = {str(path.relative_to(REPO)): digest(path) for path in paths}
    LOCK.write_text(json.dumps(dict(scope='local exploratory identification, frozen before fits',
                                    files=files), indent=2)+'\n')
    return verify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'verify'])
    args = parser.parse_args()
    print(freeze() if args.action == 'freeze' else verify(), flush=True)
