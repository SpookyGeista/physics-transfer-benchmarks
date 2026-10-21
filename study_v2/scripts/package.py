"""Create separate reproducible-source and complete measured-result bundles."""
import hashlib,json,sys,tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from decision_study.common import ROOT,digest,dump
from decision_study.locking import verify_lock
verify_lock();assert json.loads((ROOT/'results/verification.json').read_text())['passed']
base=['README.md','HANDOFF.md','STATUS.md']
source=[ROOT/name for name in base]+[p for folder in ['src','scripts','tests','configs','reports'] for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
result_dirs=['results/data','results/models','results/evaluation','results/traces','results/control','results/support','results/mujoco','figures','videos']
source += [p for p in (ROOT/'supplement_exact').rglob('*') if p.is_file() and 'results' not in p.relative_to(ROOT/'supplement_exact').parts and p.name!='lock.json' and '__pycache__' not in p.parts and p.suffix!='.pyc']
complete=source+[p for folder in result_dirs for p in (ROOT/folder).rglob('*') if p.is_file()]
complete += [p for p in (ROOT/'results').glob('*') if p.is_file() and ('pre_review' not in p.name and 'smoke' not in p.name and 'pilot' not in p.name and 'packaging' not in p.name)]
complete += [p for p in (ROOT/'supplement_exact/results').rglob('*') if p.is_file()]+[ROOT/'supplement_exact/lock.json']
complete=sorted(set(complete));dest=ROOT/'deliverables';dest.mkdir(exist_ok=True)
manifest=dest/'SHA256SUMS';manifest.write_text(''.join(f'{digest(p)}  {p.relative_to(ROOT)}\n' for p in complete))
artifacts={}
for label,files in [('source',source),('complete',complete+[manifest])]:
 path=dest/f'physics-transfer-v2-{label}.tar.gz';files=sorted(set(files))
 with tarfile.open(path,'w:gz') as archive:
  for p in files:archive.add(p,arcname=Path('study_v2')/p.relative_to(ROOT),recursive=False)
 with tarfile.open(path) as archive:
  for member in archive.getmembers():
   original=ROOT/Path(member.name).relative_to('study_v2')
   assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==digest(original),member.name
 artifacts[path.name]=dict(bytes=path.stat().st_size,sha256=digest(path),files=len(files),verified_against_current_files=True)
dump(dest/'archives.json',artifacts);print(json.dumps(artifacts,indent=2))
