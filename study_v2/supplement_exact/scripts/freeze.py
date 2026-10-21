import sys,json,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from decision_study.common import ROOT,digest,dump
from decision_study.locking import verify_lock
parent_sha=verify_lock();SUP=ROOT/'supplement_exact';assert not list((SUP/'results/data').rglob('replay.npz'));assert not (SUP/'lock.json').exists()
files=[SUP/'config.json',SUP/'protocol.md',ROOT/'reports/numerical_amendment_review.md']+list((SUP/'scripts').glob('*.py'))
dump(SUP/'lock.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_lock_sha256=parent_sha,files={str(p.resolve()):digest(p) for p in sorted(files)},original_model_controller_metrics_inspected=False,trigger='two isolated/replay normal-force discrepancies .000113N > .000100N; no performance-based model or arrangement selection',one_supplement_only=True))
print('SUPPLEMENT_FROZEN',len(files),'files')
