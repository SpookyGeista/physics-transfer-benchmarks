"""Read-only verification of delivered numerical artifacts and immutable model files."""
import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import imageio.v2 as imageio
from null_physics_transfer.common import CFG,digest,dump

lock=json.loads(Path('results/protocol_lock.json').read_text())
for path,sha in lock['files'].items():assert digest(path)==sha,f'Frozen file changed: {path}'
p=pd.read_csv('results/evaluation/prediction.csv');c=pd.read_csv('results/control/control.csv');native=pd.read_csv('results/evaluation_native/prediction.csv')
expected=2*5*2*9*2*24
assert len(p)==expected and len(native)==expected
assert len(c)==2*5*9*12
assert p[['world','budget','seed','method','noise','episode']].duplicated().sum()==0
assert c[['world','seed','method','episode']].duplicated().sum()==0
assert np.isfinite(p.select_dtypes('number')).all().all();assert np.isfinite(c.select_dtypes('number')).all().all()
assert set(p.method)==set(CFG['methods'])
assert (p.positive_friction_work<1e-8).all()
vids={}
for path in Path('videos').glob('*.mp4'):
 reader=imageio.get_reader(path);count=reader.count_frames();meta=reader.get_meta_data();assert count>=50;first=reader.get_data(0);last=reader.get_data(count-1);reader.close();assert np.mean(np.abs(first.astype(float)-last.astype(float)))>0.1
 vids[str(path)]={'frames':count,'fps':meta['fps'],'size':list(meta['size']),'sha256':digest(path)}
assert len(vids)>=2
proc=subprocess.run([sys.executable,'-m','pytest','-q','tests'],capture_output=True,text=True);Path('results/tests_final.log').write_text(proc.stdout+proc.stderr);assert proc.returncode==0
result=dict(frozen_files=len(lock['files']),prediction_rows=len(p),native_prediction_rows=len(native),controller_rows=len(c),videos=vids,tests=proc.stdout.strip(),passed=True)
dump('results/verification.json',result);print(json.dumps(result,indent=2))
