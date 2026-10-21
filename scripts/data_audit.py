import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from null_physics_transfer.common import DT,MASS,dump
from null_physics_transfer.models import normal_basis,friction_basis
report={}
for path in Path('results/data').glob('*.npz'):
 if '_test' in path.name:continue
 d=np.load(path);meta=json.loads(path.with_suffix('.json').read_text());err=d['next_state'][:,:,1]-d['state'][:,:,1]-DT*d['force']/MASS
 f=d['features'].reshape(-1,10);a=np.repeat([r['mode']=='A' for r in meta['rows']],d['features'].shape[1]);b=np.repeat([r['mode']=='B' for r in meta['rows']],d['features'].shape[1])
 item={'finite':bool(all(np.isfinite(d[k]).all() for k in d.files)), 'max_newton_error_m_s':float(abs(err).max()),'positive_friction_work_J':float(np.maximum(0,d['friction']*d['state'][:,:,1]).sum()*DT),'compression_range_m':[float(d['compression'].min()),float(d['compression'].max())]}
 if a.any() and b.any():
  classical=np.column_stack([f[b,0]*.02,f[b,1]*.1]);item['classical_normal_rank']=int(np.linalg.matrix_rank(classical));item['classical_normal_condition']=float(np.linalg.cond(classical));item['normal_dictionary_rank']=int(np.linalg.matrix_rank(normal_basis(f[b])));item['friction_dictionary_rank']=int(np.linalg.matrix_rank(friction_basis(f[a])));item['state_note']='Ideal known mass, prescribed excitation and direct force observations; dictionary columns may be correlated.'
 report[path.name]=item
 assert item['finite'] and item['max_newton_error_m_s']<1e-5 and item['positive_friction_work_J']<1e-9
 dump('results/data_quality.json',report)
print(json.dumps(report,indent=2))
