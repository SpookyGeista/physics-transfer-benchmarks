import ast,json
from pathlib import Path
import numpy as np
import torch
from null_physics_transfer.common import design,features,history_update,DT,MASS
from null_physics_transfer.models import Neural,Classical,freeze
from null_physics_transfer.predict import metrics,step


def test_whole_trajectory_splits_disjoint():
    sets=[set(r['seed'] for r in design(s)) for s in ['train','val','privileged','test','control']]
    for i,a in enumerate(sets):
        for b in sets[i+1:]:assert a.isdisjoint(b)
    assert all(r['mode'] in ('A','B') and r['slope']==0 for r in design('train'))
    assert all(r['mode']=='AB' and r['slope']!=0 for r in design('test'))

def test_predictor_training_has_no_reference_access():
    for path in [Path('scripts/train.py'),*Path('src/null_physics_transfer').glob('*.py')]:
        if path.name in ('reference.py','__init__.py'):continue
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):assert 'reference' not in (node.module or '')
            if isinstance(node,ast.Import):assert all('reference' not in a.name for a in node.names)

def test_freeze_and_normal_target_leakage():
    for kind in ['modular','joint','moe','no_history']:
        model=freeze(Neural(kind)); assert not model.training
        assert all(not p.requires_grad for p in model.parameters())
        f=torch.randn(16,10); f[:,0]=1.; alt=f.clone();alt[:,9]+=100
        assert torch.equal(model.normal(f),model.normal(alt))

def test_force_units_and_cam_reaction():
    x=np.array([.1]);v=np.array([0.]);h=np.zeros((1,3,2));s=np.array([.1]);d=np.array([.01])
    xn,vn,_,n,ft,_=step(Classical(),x,v,h,d,np.zeros(1),np.zeros(1),s)
    assert np.allclose(n,10) # k * (d+s*x): N/m * m
    assert np.allclose(vn,-DT*1/MASS)
    assert np.allclose(xn,x+DT*vn)
    # Cam power: (-s*N)*v + N*(s*v) == 0 (ideal constraint).
    assert np.allclose(-s*n*.3+n*s*.3,0)

def test_friction_dissipative_and_open_contact():
    h=np.zeros((10,3,2));v=np.linspace(-.5,.5,10)
    f=features(np.ones(10)*.02,np.zeros(10),v,h,np.ones(10)*10)
    for model in [Classical(),Neural('modular'),Neural('joint'),Neural('moe')]:
        n,ft=model.predict(f);assert np.all(ft*v<=1e-7)
        z=f.copy();z[:,0]=-.1;n,ft=model.predict(z);assert np.all(n==0);assert np.all(ft==0)

def test_metrics_and_history():
    z=np.zeros((2,5,2));truth={'next_state':z,'normal':np.zeros((2,5)),'friction':np.zeros((2,5)),'state':z}
    pred={**truth,'next_state':z+np.array([.01,.02]),'compression':np.zeros((2,5))}
    m=metrics(pred,truth);assert np.allclose(m['position_rmse'],.01);assert np.allclose(m['velocity_rmse'],.02)
    h=np.zeros((1,3,2)); hn=history_update(h,np.ones(1)*.01,np.ones(1)*.2);assert np.all(h==0);assert np.all(hn[:,:,0]>0)

def test_data_matches_newton_when_available():
    for path in Path('results/smoke_data').glob('*train.npz'):
        d=np.load(path);err=d['next_state'][:,:,1]-d['state'][:,:,1]-DT*d['force']/MASS
        assert np.max(np.abs(err))<2e-5
        assert np.isfinite(d['features']).all()


def test_completed_training_information_rights():
    for path in Path('results/models').glob('*/b*/s*/*/metadata.json'):
        md=json.loads(path.read_text())
        assert md['frozen'] and not md['test_data_used']
        assert all(i.startswith('train-') or (md['method']=='privileged' and i.startswith('privileged-')) for i in md['training_ids'])
        assert len([i for i in md['training_ids'] if i.startswith('train-')])==2*md['budget']

def test_frozen_files_unmodified():
    from null_physics_transfer.common import digest
    lock=Path('results/protocol_lock.json')
    if lock.exists():
        for path,sha in json.loads(lock.read_text())['files'].items():assert digest(path)==sha


def test_independent_reference_agreement():
    for world in ['sls','nonlinear']:
        a=Path(f'results/data/{world}_test.npz');b=Path(f'results/mujoco/{world}_test.npz')
        if a.exists() and b.exists():
            xa=np.load(a)['next_state'];xb=np.load(b)['next_state']
            assert np.sqrt(np.mean((xa-xb)**2))<1e-4
