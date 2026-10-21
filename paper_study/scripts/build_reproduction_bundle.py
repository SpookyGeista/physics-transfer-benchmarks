"""Create deterministic local source/model and result archives after completion.

No download, installation, external upload or source modification. Original
absolute-path locks remain untouched; archive members preserve repo hierarchy.
"""
from pathlib import Path
import argparse,gzip,hashlib,io,json,tarfile,time

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
PRIMARY=['native_main1','observer_main1','statistics_main1','cost_tradeoffs_main1_fast',
    'main1_final_audit','native_main2','observer_main2','statistics_main2',
    'decision_test_native','decision_test_evaluation','statistics_decision_test','decision_comparative_timestep_v1',
    'pi_comparative_timestep_v1','decision_resolution_v1','decision_resolution_report_v1',
    'costed_pi_convergence_v1','costed_pi_mle_v1','costed_pi_mle_resolution_v1',
    'costed_pi_pooled_error_addendum_v1','pi_spread_diagnostic']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def verify_maps(mapping):
    for name,digest in mapping.items():
        if sha(name)!=digest:raise ValueError('Locked file changed: '+name)


def check_complete():
    paths=[ROOT/'results/observer_main1/process.json',ROOT/'results/native_main2/process.json',
        ROOT/'results/decision_test_native/process.json',ROOT/'results/decision_test_evaluation/process.json',
        ROOT/'results/decision_comparative_timestep_v1/process.json',ROOT/'results/pi_comparative_timestep_v1/process.json']
    paths.extend(ROOT/f'results/observer_main2/bank{bank}/process.json' for bank in (1,2,3))
    paths.extend(ROOT/f'results/{name}/process.json' for name in
        ('statistics_main2','statistics_decision_test'))
    for p in paths:
        if read(p).get('status')!='complete':raise ValueError('Incomplete required run: '+str(p))
    text=(ROOT/'reports/manuscript.md').read_text()
    if any(s in text for s in ('will be inserted here','pending numerical-resolution assessment','reported separately when complete')):
        raise ValueError('Manuscript still contains unfinished results')
    if not (ROOT/'reports/final_scientific_review.md').exists():raise ValueError('Final scientific review required')
    return paths


def archive(destination,paths,compressed):
    records=[]
    with destination.open('xb') as raw:
        compressor=gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=1,mtime=0) if compressed else raw
        with tarfile.open(fileobj=compressor,mode='w|',format=tarfile.PAX_FORMAT) as tar:
            for path in sorted(set(paths)):
                path=path.resolve()
                if not path.is_relative_to(REPO):raise ValueError('Refuse packaging outside repository: '+str(path))
                name=str(path.relative_to(REPO));digest=sha(path)
                info=tarfile.TarInfo(name);info.size=path.stat().st_size;info.mtime=0;info.mode=0o644
                with path.open('rb') as source:tar.addfile(info,source)
                records.append(dict(path=name,size=info.size,sha256=digest))
        if compressed:compressor.close()
    return dict(file=destination.name,size=destination.stat().st_size,sha256=sha(destination),members=records)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'deliverables')
    p.add_argument('--include-raw',action='store_true',help='Also archive native and forecast NPZ files; several GB')
    a=p.parse_args();out=a.out.resolve()
    if out.exists() or out.parent!=ROOT or not out.name.startswith('deliverables'):
        raise ValueError('Fresh deliverables-prefixed directory directly inside paper_study required')
    processes=check_complete()
    lockfiles=[ROOT/'locks/main1/test_authorization.json',ROOT/'locks/main2/test_authorization.json',
        ROOT/'results/decision_test_authorization.json']
    bound={}
    for lock in lockfiles:
        sources=read(lock)['sources_sha256'];verify_maps(sources);bound.update(sources)
    source=set(Path(p) for p in bound)|set(lockfiles)|set(processes)
    documents=set();rawdata=set()
    # Preserve the actual local-learning entry points and calibration inputs,
    # not merely the coefficient JSONs required by an inference run.
    for directory in (REPO/'research_next/src', REPO/'research_next/scripts',
        REPO/'research_next/configs', REPO/'research_next/tests',
        REPO/'study_v2/src/decision_study', REPO/'study_v2/configs'):
        source.update(f for f in directory.rglob('*') if f.is_file()
            and f.suffix in ('.py','.json','.md','.sh') and '__pycache__' not in f.parts)
    source.add(REPO/'research_next/README.md')
    source.update((REPO/'STATUS.md', REPO/'HANDOFF.md'))
    for directory in (REPO/'research_next/results/round1/calibration',
        REPO/'research_next/results/round1/models/state_id'):
        for f in directory.rglob('*'):
            if f.is_file() and not f.is_symlink() and '__pycache__' not in f.parts:
                (rawdata if f.suffix=='.npz' else documents).add(f)
    for f in ROOT.rglob('*'):
        if not f.is_file() or f.is_symlink():continue
        relative=f.relative_to(ROOT)
        if relative.parts[0].startswith('deliverables'):continue
        if any(part in ('__pycache__','.pytest_cache','numba_cache','deliverables') for part in relative.parts):continue
        if 'results' not in relative.parts and relative.parts[0] not in ('output','locks'):
            if f.suffix in ('.py','.sh','.json','.md','.txt','.toml'):source.add(f)
            elif relative.parts[0] in ('reports','videos'):documents.add(f)
    documents.add(ROOT/'output/pdf/manuscript.pdf');documents.add(ROOT/'output/pdf/manuscript.provenance.json')
    for name in PRIMARY:
        directory=ROOT/'results'/name
        if not directory.exists():raise ValueError('Missing declared result: '+str(directory))
        for f in directory.rglob('*'):
            if not f.is_file() or f.is_symlink() or '__pycache__' in f.parts:continue
            if f.suffix=='.npz':rawdata.add(f)
            else:documents.add(f)
    for directory in (ROOT/'decision_study/results/native_development_v1',ROOT/'decision_study/results/evaluation_development_v1',
        ROOT/'results/decision_refinement_h2',ROOT/'results/decision_refinement_h4',
        ROOT/'results/native_development_v2',ROOT/'results/costed_pi_development',
        ROOT/'results/pi_native_refinement_h2',ROOT/'results/pi_native_refinement_h4',
        ROOT/'multistate_learning/results'):
        for f in directory.rglob('*'):
            if f.is_file() and not f.is_symlink():
                (rawdata if f.suffix=='.npz' else documents).add(f)
    # Preserve every development, failed and superseded artifact as well as
    # successful results; their status records keep the evidence distinguishable.
    for directory in (ROOT/'results',ROOT/'decision_study/results'):
        for f in directory.rglob('*'):
            if f.is_file() and not f.is_symlink() and not any(part in ('__pycache__','.pytest_cache','numba_cache') for part in f.parts):
                (rawdata if f.suffix=='.npz' else documents).add(f)
    inventory=source|documents|(rawdata if a.include_raw else set())
    frozen_inventory={str(f):sha(f) for f in sorted(inventory)}
    out.mkdir(parents=True)
    products=[archive(out/'source_and_models.tar.gz',source,True),archive(out/'reports_and_metrics.tar.gz',documents-source,True)]
    if a.include_raw:products.append(archive(out/'native_and_forecast_arrays.tar',rawdata-source,False))
    verify_maps(bound)
    verify_maps(frozen_inventory)
    manifest=dict(status='complete',created_unix=time.time(),repository_root=str(REPO),complete_process_sha256={str(p):sha(p) for p in processes},
        prospective_lock_sha256={str(p):sha(p) for p in lockfiles},archives=products,
        inventory_sha256=frozen_inventory,
        raw_arrays_included=a.include_raw,raw_arrays_available_in_workspace=len(rawdata),
        relocation='Members preserve repo-relative paths. Original prospective locks are retained unchanged; separately document any path-map relocation.',
        runtime='Existing Isaac/Unitree installation required; third-party runtime binaries are not bundled.')
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    summary='\n'.join(f"{x['sha256']}  {x['file']}" for x in products)+'\n'
    (out/'SHA256SUMS').write_text(summary)
    print(json.dumps({x['file']:dict(bytes=x['size'],members=len(x['members']),sha256=x['sha256']) for x in products},indent=2))


if __name__=='__main__':main()
