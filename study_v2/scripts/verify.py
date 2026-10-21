"""Final data, lock, finite-result, trajectory and media checks."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import imageio.v2 as imageio
from decision_study.common import ROOT,CFG,DT,MASS,dump
from decision_study.locking import verify_lock
lock=verify_lock();checks={}
models=list((ROOT/'results/models').rglob('metadata.json'));assert len(models)==360;checks['models']=len(models)
files=list((ROOT/'results/data').glob('*/*/*.npz'));assert len(files)==40
maxnewton=0.
for p in files:
 with np.load(p) as d:
  for key in d.files:assert np.isfinite(d[key]).all(),(p,key)
  residual=d['next_state'][:,:,1]-d['state'][:,:,1]-DT*d['force']/MASS;maxnewton=max(maxnewton,float(abs(residual).max()))
assert maxnewton<2e-5;checks.update(raw_data_files=len(files),max_native_Newton_residual=maxnewton)
for name,expected in [('prediction',38400),('control',1600),('support',1920),('control_support',1600)]:
 table=pd.read_csv(ROOT/f'results/{name}_metrics.csv');assert len(table)==expected,(name,len(table));assert np.isfinite(table.select_dtypes('number')).all().all(),name;checks[name+'_rows']=len(table)
control=list((ROOT/'results/control').glob('*.npz'));assert len(control)==200
for p in control:
 with np.load(p) as d:
  for key in d.files:assert np.isfinite(d[key]).all(),(p,key)
  assert np.max(abs(d['u']))<=CFG['controller']['force_limit']+1e-10
checks['control_files']=len(control)
media={}
for p in sorted((ROOT/'videos').glob('*.mp4')):
 reader=imageio.get_reader(p);frames=[frame for frame in reader];reader.close();assert len(frames)==120 and np.mean(abs(frames[20].astype(float)-frames[90]))>.1
 media[p.name]=dict(frames=len(frames),height=frames[0].shape[0],width=frames[0].shape[1])
assert len(media)==4,len(media);checks['videos']=media
supp=ROOT/'supplement_exact';sl=json.loads((supp/'lock.json').read_text())
from decision_study.common import digest
for name,sha in sl['files'].items():assert digest(name)==sha
assert sl['parent_lock_sha256']==lock
sp=pd.read_csv(supp/'results/prediction_metrics.csv');ss=pd.read_csv(supp/'results/support_metrics.csv');assert len(sp)==19200 and len(ss)==960
assert ss.exact_match.eq(1).all() and ss[['normal_coverage','friction_coverage']].eq(1).all().all()
assert np.isfinite(sp.select_dtypes('number')).all().all()
sfiles=list((supp/'results/data').glob('*/*/replay.npz'));assert len(sfiles)==10
for p in sfiles:
 with np.load(p) as d:
  for key in d.files:assert np.isfinite(d[key]).all(),(p,key)
  assert np.max(abs(d['next_state'][:,:,1]-d['state'][:,:,1]-DT*d['force']/MASS))<2e-5
checks['supplement']=dict(lock_sha256=digest(supp/'lock.json'),predictions=len(sp),reference_episodes=len(ss),exact_matches=int(ss.exact_match.sum()),model_training_repeated=False)
checks['protocol_lock_sha256']=lock;checks['passed']=True;dump(ROOT/'results/verification.json',checks);print(json.dumps(checks,indent=2))
