"""Fail closed before the first new-arrangement or controller trial."""
import sys,json,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from decision_study.common import ROOT,CFG,dump,digest
lockpath=ROOT/'results/protocol_lock.json'
if lockpath.exists():
 from decision_study.locking import verify_lock
 print('Existing lock verified',verify_lock());raise SystemExit(0)
assert not list((ROOT/'results/data').rglob('replay.npz')), 'test already exists before freeze'
assert not list((ROOT/'results/control').glob('*.npz')), 'control already exists before freeze'
expected=len(CFG['worlds'])*len(CFG['seeds'])*len(CFG['budgets'])*len(CFG['noise_conditions'])*(len(CFG['methods'])-1)
metadata=list((ROOT/'results/models').rglob('metadata.json'));assert len(metadata)==expected,(len(metadata),expected)
files=[ROOT/'configs/experiment.json']+list((ROOT/'src/decision_study').glob('*.py'))+[ROOT/'scripts'/name for name in ['generate.py','train.py','support.py','evaluate.py','control.py','mujoco_replay.py','report.py','freeze.py']]+list((ROOT/'tests').glob('*.py'))
files += [ROOT/'reports'/name for name in ['experiment_protocol.md','reference_notes.md','design_audit.md','support_protocol.md','implementation_review.md']]
files+=list((ROOT/'results/data').glob('*/*/train.*'))+list((ROOT/'results/data').glob('*/*/val.*'))+list((ROOT/'results/data').glob('*/*/privileged.*'))
for m in metadata:
 data=json.loads(m.read_text());assert data['test_data_used'] is False and data['frozen'] and data['updates']==CFG['updates']
 for file,sha in data['code'].items():assert digest(file)==sha,f'Training code changed: {file}'
 for file,sha in data['model_hashes'].items():assert digest(file)==sha;files.append(Path(file))
 files.append(m);files+=list(m.parent.glob('learning.json'))+list(m.parent.glob('training_state.pt'))
support=list((ROOT/'results/support').glob('*/r*.npz'))+list((ROOT/'results/support').glob('*/r*.json'))
assert len(support)==20,len(support);files+=support
v1root=ROOT.parent;v1path=v1root/'results/protocol_lock.json';v1lock=json.loads(v1path.read_text()) if v1path.exists() else {'files':{}}
for name,sha in v1lock['files'].items():assert digest(v1root/name)==sha,f'Original v1 frozen file changed: {name}'
dump(lockpath,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={str(p.resolve()):digest(p) for p in sorted(set(files))},models=expected,v1_files_verified=len(v1lock['files']),test_data_available_at_lock=False,decision='Paired-seed95t confidence intervals, bothworlds andbothanalyticchampions, see frozen experiment_protocol.md'))
print('FROZEN',len(set(files)),'files;',expected,'models; original v1 verified')
