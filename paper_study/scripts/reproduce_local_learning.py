"""Re-run every local PI fit in an isolated copy; preserve all original locks."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'results'):
        raise ValueError('Use a fresh destination within paper_study/results')
    original = ROOT/'multistate_learning'
    locked = json.loads((original/'design_lock.json').read_text())['files']
    for relative, digest in locked.items():
        if sha(REPO/relative) != digest:
            raise ValueError('Original source changed: '+relative)
    out.mkdir(parents=True)
    work = out/'repository'
    for relative in locked:
        dest = work/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/relative, dest)
    local = work/'paper_study/multistate_learning'
    commands = [
        [sys.executable, '-m', 'pytest', str(local/'test_learning.py'), '-q'],
        [sys.executable, str(local/'lock.py'), 'freeze'],
        [sys.executable, str(local/'generate.py')],
        [sys.executable, str(local/'fit.py'), '--workers', '2'],
        [sys.executable, str(local/'summarize.py')],
        [sys.executable, str(local/'lock.py'), 'verify'],
    ]
    env = os.environ | {'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1',
        'PYTHONDONTWRITEBYTECODE':'1', 'NUMBA_CACHE_DIR':str(out/'numba_cache')}
    record = dict(status='running', started_unix=time.time(), source_sha256=sha(__file__),
        original_lock_sha256=sha(original/'design_lock.json'), original_sources_sha256=locked,
        commands=[], scope='Reproduction of fixed fits; no new material data or test inference')
    process = out/'process.json'
    start = time.perf_counter()
    try:
        for i, command in enumerate(commands):
            process.write_text(json.dumps(record, indent=2)+'\n')
            with (out/f'step{i+1}.log').open('x') as log:
                result = subprocess.run(command, cwd=work, env=env, stdout=log,
                    stderr=subprocess.STDOUT, timeout=600)
            record['commands'].append(dict(argv=command, returncode=result.returncode))
            if result.returncode:
                raise RuntimeError(f'Reproduction step{i+1} failed; retained log')
        # Metadata contain new paths/runtimes. Compare only learned coefficients,
        # and separately retain all newly generated calibration and fit files.
        comparisons = []
        for model in sorted((local/'results/models').rglob('model.json')):
            rel = model.relative_to(local/'results/models')
            old = json.loads((original/'results/models'/rel).read_text())
            new = json.loads(model.read_text())
            difference = max(abs(a-b) for key in ('stiffness_N_m','friction_capacity')
                for a,b in zip(old[key],new[key],strict=True))
            comparisons.append(dict(model=str(rel), max_coefficient_absolute_difference=difference))
        if len(comparisons) != 24:
            raise ValueError('Missing reproduction fits')
        record.update(status='complete', fit_count=len(comparisons), comparisons=comparisons,
            max_coefficient_absolute_difference=max(x['max_coefficient_absolute_difference'] for x in comparisons))
    except BaseException as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        record['seconds'] = time.perf_counter()-start
        for relative,digest in locked.items():
            if sha(REPO/relative) != digest:
                record.update(status='failed', error='Original source changed during reproduction')
        process.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({key:record[key] for key in ('status','fit_count','seconds','max_coefficient_absolute_difference')},indent=2))


if __name__ == '__main__':
    main()
