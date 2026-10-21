"""Reproducible staged calibration/training and prospective lock creation."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from round1_lock import OUT,save_lock,verify_code_lock,verify_model_lock,verify_test_lock,digest

def artifact_manifest(folder,create=False):
    path=folder/'artifact_manifest.json'
    if create:
        files=[p for p in folder.iterdir() if p.is_file() and p!=path]
        with path.open('x') as f:json.dump({p.name:digest(p) for p in files},f,indent=2)
    if not path.exists():raise RuntimeError(f'Completed job has no artifact manifest; retain and review: {folder}')
    for name,h in json.loads(path.read_text()).items():
        if digest(folder/name)!=h:raise RuntimeError(f'Training artifact changed: {folder/name}')

def verify_calibration(cfg,seeds):
    for world in cfg['worlds']:
        for seed in seeds:
            for split in ['train','val']:
                p=OUT/f'calibration/{world}/r{seed}/{split}.npz';m=json.loads(p.with_suffix('.json').read_text())
                if m['data_sha256']!=digest(p):raise RuntimeError(f'Calibration changed: {p}')
                for source,h in m['source_sha256'].items():
                    if digest(source)!=h:raise RuntimeError(f'Calibration source changed: {source}')
                expected_base=cfg['calibration_source_bases'][split]
                if any(row['source_base']!=expected_base or row['replicate']!=seed for row in m['rows']):raise RuntimeError('Wrong calibration identity')

def command(argv,log,timeout=None):
    log.parent.mkdir(parents=True,exist_ok=True)
    print('RUN',' '.join(map(str,argv)),flush=True)
    with log.open('a') as f:
        result=subprocess.run(list(map(str,argv)),cwd=ROOT.parent,env=os.environ|{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2'},stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode:raise RuntimeError(f'Command exit {result.returncode}; see {log}')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['lock-code','calibration','neural','classical','lock-models','lock-test','verify']);p.add_argument('--seeds',type=int,nargs='+');p.add_argument('--device',default='cuda');a=p.parse_args()
    cfg=json.loads((ROOT/'configs/round1.json').read_text());seeds=cfg['replicates'] if a.seeds is None else a.seeds
    if not set(seeds)<=set(cfg['replicates']):p.error('Only predefined replicates permitted')
    OUT.mkdir(parents=True,exist_ok=True)
    if a.stage=='lock-code':
        if cfg['status']!='READY_TO_LOCK':raise RuntimeError('Finish protocol review first')
        if (OUT/'test').exists():raise RuntimeError('Test output must not predate source lock')
        files=[*ROOT.glob('src/*.py'),*ROOT.glob('scripts/*.py'),*ROOT.glob('tests/*.py'),ROOT/'configs/round1.json',ROOT/'reports/round1_protocol.md',ROOT/'reports/round1_analysis_plan.md',*ROOT.parent.glob('study_v2/src/decision_study/*.py'),ROOT.parent/'study_v2/configs/experiment.json']
        print(save_lock(OUT/'code_lock.json',files,role='prospective algorithm and protocol before new calibration and test',created_unix=time.time()),flush=True);return
    code=verify_code_lock()
    if a.stage=='calibration':
        if all((OUT/f'calibration/{world}/r{seed}/{split}.npz').exists() for world in cfg['worlds'] for seed in seeds for split in ['train','val']):
            verify_calibration(cfg,seeds);print('Calibration exists; verified and retained');return
        command([sys.executable,ROOT/'scripts/generate_development.py','--headless','--device','cpu','--seeds',*seeds,'--out',OUT/'calibration','--train-source-base',cfg['calibration_source_bases']['train'],'--val-source-base',cfg['calibration_source_bases']['val']],OUT/'logs/calibration.log',cfg['resources']['native_evaluation_wall_seconds_limit'])
        verify_calibration(cfg,seeds)
    elif a.stage in ['neural','classical']:
        verify_calibration(cfg,seeds);started=time.perf_counter();deadline=cfg['resources']['fit_wall_seconds_limit']
        for seed in seeds:
            if time.perf_counter()-started>=deadline:raise TimeoutError('Predeclared fitting wall budget reached')
            if a.stage=='neural':
                for noise in cfg['noise']:
                    for world in cfg['worlds']:
                        dest=OUT/f'models/recurrent/r{seed}/{noise}/{world}'
                        if (dest/'model.pt').exists() and (dest/'metadata.json').exists():
                            m=json.loads((dest/'metadata.json').read_text())
                            for path,h in m['source_sha256'].items():assert digest(path)==h
                            artifact_manifest(dest)
                            print('SKIP verified',dest,flush=True);continue
                        command([sys.executable,ROOT/'scripts/train_recurrent_diagnostic.py','--data-root',OUT/'calibration','--out',dest.parent,'--device',a.device,'--updates',cfg['recurrent']['updates'],'--hidden',cfg['recurrent']['hidden'],'--seed',seed,'--noise',noise,'--worlds',world],OUT/f'logs/neural_r{seed}_{noise}_{world}.log',max(1,deadline-(time.perf_counter()-started)))
                        artifact_manifest(dest,create=True)
            else:
                command([sys.executable,ROOT/'scripts/fit_state_id_diagnostic.py','--data',OUT/'calibration','--out',OUT/'models/state_id','--seeds',seed,'--noise',*cfg['noise'],'--blind-selection','--max-seconds',cfg['identification']['max_seconds_total_both_families'],'--restarts',cfg['identification']['restarts']],OUT/f'logs/state_id_r{seed}.log',max(1,deadline-(time.perf_counter()-started)))
                for world in cfg['worlds']:
                    for noise in cfg['noise']:
                        dest=OUT/f'models/state_id/{world}/r{seed}/{noise}/blind_selection'
                        artifact_manifest(dest,create=not (dest/'artifact_manifest.json').exists())
                for world in cfg['worlds']:
                    for noise in cfg['noise']:
                        for method in ['old_flex','path_ridge']:
                            dest=OUT/f'models/features/{world}/r{seed}/{noise}/{method}'
                            if all((dest/name).exists() for name in ['model.json','metrics.json','predictions.npz','declaration.json']):
                                m=json.loads((dest/'declaration.json').read_text())
                                for path,h in (m['source_sha256']|m['input_sha256']).items():assert digest(path)==h
                                artifact_manifest(dest)
                                print('SKIP verified',dest,flush=True);continue
                            command([sys.executable,ROOT/'scripts/fit_feature_baselines.py','--data',OUT/'calibration','--out',OUT/'models/features','--seed',seed,'--worlds',world,'--noise',noise,'--methods',method],OUT/f'logs/features_r{seed}_{world}_{noise}_{method}.log',max(1,deadline-(time.perf_counter()-started)))
                            artifact_manifest(dest,create=True)
        (OUT/f'{a.stage}_completion.json').write_text(json.dumps(dict(code_lock_sha256=code,seeds=seeds,seconds=time.perf_counter()-started),indent=2))
    elif a.stage=='lock-models':
        verify_calibration(cfg,cfg['replicates'])
        for seed in cfg['replicates']:
            for world in cfg['worlds']:
                for noise in cfg['noise']:
                    required=[OUT/f'models/recurrent/r{seed}/{noise}/{world}/model.pt',OUT/f'models/state_id/{world}/r{seed}/{noise}/blind_selection/model.json',*[OUT/f'models/features/{world}/r{seed}/{noise}/{method}/model.json' for method in ['old_flex','path_ridge']]]
                    for path in required:
                        if not path.exists():raise RuntimeError(f'Missing required fitted model: {path}')
                        artifact_manifest(path.parent)
        if (OUT/'test').exists():raise RuntimeError('Model lock must precede test generation')
        files=[p for folder in ['calibration','models'] for p in (OUT/folder).rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.pt']]
        print(save_lock(OUT/'model_lock.json',files,code_lock_sha256=code,created_unix=time.time(),role='all local calibration/validation and selected model artifacts before held-out composition generation'),flush=True)
    elif a.stage=='lock-test':
        model=verify_model_lock();files=[]
        for nw,fw in cfg['compositions']:
            for seed in cfg['replicates']:
                path=OUT/f'test/{nw}_{fw}/r{seed}.npz';meta=path.with_suffix('.json');m=json.loads(meta.read_text())
                if m['model_lock_sha256']!=model or m['npz_sha256']!=digest(path):raise RuntimeError('Test provenance mismatch')
                files.extend([path,meta])
        print(save_lock(OUT/'test_lock.json',files,model_lock_sha256=model,created_unix=time.time(),role='reference commands and metadata locked before prediction/controller scores'),flush=True)
    else:print('VERIFIED',verify_test_lock() if (OUT/'test_lock.json').exists() else verify_model_lock(),flush=True)
    assert verify_code_lock()==code

if __name__=='__main__':main()
