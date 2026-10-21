"""Create local source and complete-result archives; never publish or push."""
import json,tarfile,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'deliverables';out.mkdir(exist_ok=True)
files=[]
for top in ['README.md','HANDOFF.md','STATUS.md','pyproject.toml','configs','src','scripts','tests','reports','results','videos']:
 p=root/top
 files.extend([p] if p.is_file() else [x for x in p.rglob('*') if x.is_file() and '__pycache__' not in x.parts and '.pytest_cache' not in x.parts])
for name,selection in [('null-physics-transfer-code.tar.gz',[p for p in files if p.relative_to(root).parts[0] not in ['results','videos'] and (p.relative_to(root).parts[0]!='reports' or p.name in ['experiment_protocol.md','decisions.md','novelty_matrix.md','environment.md'])]),('null-physics-transfer-complete.tar.gz',files)]:
 with tarfile.open(out/name,'w:gz') as tar:
  for p in sorted(selection):tar.add(p,arcname=str(Path('null-physics-transfer')/p.relative_to(root)),recursive=False)
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
manifest.update({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.tar.gz')})
(out/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
print('PACKAGED',len(files),'files')
