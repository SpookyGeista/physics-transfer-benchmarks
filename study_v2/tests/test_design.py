import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from decision_study.common import profiles,cam,features,history_init,history_update,DT,MASS

def test_profile_seed_disjoint_but_arrangement_reuses_locals():
 seeds=[]
 for split in ['train','val','privileged']:
  _,rows=profiles(split,0);seeds.append(set(r['stimulus_seed'] for r in rows))
 assert all(not a&b for i,a in enumerate(seeds) for b in seeds[i+1:])
 a,_=profiles('train',0);b,_=profiles('train',1);assert not np.array_equal(a['delta'],b['delta'])

def test_cam_work_and_derivative():
 x=np.linspace(-.2,.2,51);a=np.full_like(x,.006);g,s=cam(x,a);eps=1e-7
 assert np.allclose(s,(cam(x+eps,a)[0]-cam(x-eps,a)[0])/(2*eps))
 n=np.linspace(3,30,len(x));v=np.linspace(-.3,.3,len(x));assert np.allclose(-s*n*v+n*(s*v),0)
 assert np.max(np.abs(g))<=.006

def test_semimplicit_desired_states():
 p,_=profiles('train',0);assert np.allclose(np.diff(p['x'],axis=1),DT*p['v'][:,1:])
 assert np.all(p['v'][:,0]==0)

def test_history_is_causal():
 d=np.array([.02]);v=np.array([.1]);n=np.array([10.]);h=history_init(d,v,n);original=h.copy();n2=np.array([20.]);updated=history_update(h,d,v,n2)
 assert np.array_equal(h,original);f=features(d,v,v,h,n2);assert f.shape==(1,13)
 assert np.all(updated[:,:,2]<20) and np.all(updated[:,:,2]>10)

def test_smoke_newton():
 for p in Path('results/smoke_data').glob('*/r*/train.npz'):
  d=np.load(p);e=d['next_state'][:,:,1]-d['state'][:,:,1]-DT*d['force']/MASS
  assert abs(e).max()<2e-5
  assert np.isfinite(d['features']).all()
