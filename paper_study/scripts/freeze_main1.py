"""Create the local prospective Main1 lock before drawing any test history.

This is a local timestamped source snapshot, not an external preregistration.
It generates no profiles and does not start a simulator or fit a model.
"""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
sys.path.insert(0,str(ROOT/'scripts'))
from generate_native_v2 import source_paths,verify_test_lock


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    target=ROOT/'locks/main1'
    if target.exists(): raise FileExistsError('Existing Main1 lock retained')
    for folder in ('native_main1','observer_main1','statistics_main1'):
        if (ROOT/'results'/folder).exists():
            raise ValueError(f'Main1 output already exists before freeze: {folder}')
    cfg_path=ROOT/'configs/main1_native.json'
    cfg=json.loads(cfg_path.read_text())
    if cfg['preparation_seeds']!=list(range(51201,51209)) or cfg['split']!='test':
        raise ValueError('Unexpected reserved histories')
    prior=ROOT/'results/prior_main1'
    if json.loads((prior/'process.json').read_text())['status']!='complete':
        raise ValueError('Prior incomplete')
    banks=list(prior.rglob('*.npz'))
    if len(banks)!=32: raise ValueError('Expected 32 public-protocol prior banks')
    prior_lock=json.loads((prior/'prior_lock.json').read_text())
    for filename,digest in prior_lock['sources_sha256'].items():
        if sha(filename)!=digest:raise ValueError(f'Prior source changed: {filename}')
    grid=json.loads((ROOT/'reports/grid_resolution.json').read_text())
    if not grid['all_passed']:raise ValueError('Grid development criterion failed')
    for filename,digest in grid['sources_sha256'].items():
        if sha(filename)!=digest:raise ValueError('Grid check input changed')
    review=ROOT/'reports/main1_pretest_review.md'
    # Exact reviewed source digests must be in the final independent review.
    reviewed=['scripts/main1_statistics.py','tests/test_main1_statistics.py',
              'reports/main1_protocol.md','configs/main1_native.json','configs/main1_observer.json']
    if not all(sha(ROOT/relative) in review.read_text() for relative in reviewed):
        raise ValueError('Independent review does not cover current final sources')
    files=set(source_paths(cfg_path))
    files.update(Path(p) for p in prior_lock['sources_sha256'])
    files.update(banks)
    files.update([prior/'prior_lock.json',prior/'process.json',Path(__file__),
        REPO/'study_v2/configs/experiment.json',ROOT/'scripts/evaluate_native_v2.py',
        ROOT/'src/models.py',ROOT/'src/observer.py',ROOT/'src/joint_observer.py',
        review,ROOT/'reports/numerical_refinement.md',ROOT/'reports/numerical_refinement.json',
        ROOT/'scripts/analyze_refinement.py',ROOT/'reports/grid_resolution.md',
        ROOT/'reports/grid_resolution.json',ROOT/'configs/observer_grid401.json',
        ROOT/'configs/prior_grid401.json'])
    files.update(ROOT/p for p in reviewed)
    hashes={str(p.resolve()):sha(p) for p in sorted(files)}
    lock=dict(schema_version=1,role='paper_study_native_v2_test_authorization',status='locked',
        split='test',preparation_seeds=cfg['preparation_seeds'],config_sha256=sha(cfg_path),
        sources_sha256=hashes,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Local prospective Main1 held-out history replication; no external registration',
        expected_native_episodes=12288,expected_forecast_rows=294912,
        conclusions_limited_to_discrete_simulation=True,
        native_output=str(ROOT/'results/native_main1'),
        evaluation_output=str(ROOT/'results/observer_main1'),
        analysis_output=str(ROOT/'results/statistics_main1'))
    target.mkdir(parents=True)
    for filename,digest in hashes.items():
        source=Path(filename);dest=target/'snapshot'/source.relative_to(REPO)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        if sha(dest)!=digest:raise ValueError('Snapshot mismatch')
    lock_path=target/'test_authorization.json'
    with lock_path.open('x') as stream:stream.write(json.dumps(lock,indent=2)+'\n')
    verify_test_lock(cfg,cfg_path,{str(p.resolve()):sha(p) for p in source_paths(cfg_path)},lock_path)
    print(json.dumps(dict(lock=str(lock_path),sha256=sha(lock_path),files=len(hashes)),indent=2))


if __name__=='__main__':main()
