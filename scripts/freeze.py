import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from null_physics_transfer.common import CFG,digest,dump
assert not Path('results/data/sls_test.npz').exists(),'Test already exists; use a new protocol/test seed for a new study.'
expected=2*len(CFG['training_seeds'])*len(CFG['budgets'])*8
metadata=list(Path('results/models').glob('*/b*/s*/*/metadata.json'))
assert len(metadata)==expected,(len(metadata),expected)
files=[]
for pattern in ['src/**/*.py','scripts/train.py','scripts/generate.py','scripts/evaluate.py','scripts/control.py','configs/experiment.json','reports/experiment_protocol.md','results/models/**/*.pt','results/models/**/classical.json','results/models/**/metadata.json']:
 files.extend(Path('.').glob(pattern))
dump('results/protocol_lock.json',dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),files={str(p):digest(p) for p in sorted(set(files))},test_design='configs/experiment.json',test_seen=False))
print('FROZEN',len(files),'files;',len(metadata),'model runs')
