import json,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from online import Composed,Nominal,SensorPersistent,motion_step
from controller_preflight import DT,FORCES,HORIZON,REPLAN
import round1_lock

class Traced:
    def __init__(self,normal,mu):self.normal,self.mu=normal,mu
    def initial(self,delta,v,normal_sensor):return {'elapsed':np.zeros(len(delta))}
    def step(self,delta,rate,v,state,normal_override=None,dt=.0025):
        n=np.full(len(delta),self.normal) if normal_override is None else np.asarray(normal_override)
        return n,-self.mu*n*np.tanh(np.asarray(v)/.04),{'elapsed':state['elapsed']+dt}

def test_cross_composition_uses_normal_then_current_load_for_friction():
    model=Composed(Traced(13,.9),Traced(21,.4));d=np.array([.02,.02]);v=np.array([.1,-.1]);h=model.initial(d,v,np.array([13,13]))
    n,f,new=model.step(d,v*0,v,h)
    np.testing.assert_array_equal(n,[13,13]);np.testing.assert_allclose(f,-.4*n*np.tanh(v/.04))
    assert all(np.all(x==0) for x in h.values())
    assert all(np.all(x==DT) for x in new.values())

def test_sensor_control_retains_friction_state_but_freezes_last_observation():
    model=SensorPersistent(Traced(13,.4));d=np.array([.02]);v=np.array([.1]);h=model.initial(d,v,np.array([17.]))
    n,_,h=model.step(d,v*0,v,h);np.testing.assert_array_equal(n,[17.])
    n,_,h=model.step(d,v*0,v,h,normal_override=np.array([19.]));np.testing.assert_array_equal(n,[19.])
    n,_,h=model.step(d,v*0,v,h);np.testing.assert_array_equal(n,[19.]);np.testing.assert_allclose(h['elapsed'],3*DT)

def test_locked_config_matches_actual_fixed_planner():
    cfg=json.loads((ROOT/'configs/round1.json').read_text());c=cfg['controller']
    assert cfg['dt']==DT and cfg['mass']==2 and c['horizon_steps']==HORIZON and c['replan_steps']==REPLAN
    assert c['candidates']==len(FORCES) and FORCES[0]==-c['force_limit'] and FORCES[-1]==c['force_limit']

def test_mechanical_coupling_uses_virtual_work_sign_and_mass():
    model=Traced(10,0);s=np.array([[0.,0.]]);h=model.initial(np.array([.02]),np.zeros(1),np.array([10.]))
    result,*_=motion_step(model,s,h,np.array([0.]),np.array([.02]),np.array([0.]),np.array([.004]))
    expected_v=DT*(-.004/.04*10)/2
    np.testing.assert_allclose(result,[[DT*expected_v,expected_v]])

def test_source_and_model_locks_detect_mutation(tmp_path,monkeypatch):
    monkeypatch.setattr(round1_lock,'ROOT',tmp_path);monkeypatch.setattr(round1_lock,'OUT',tmp_path/'results')
    source=tmp_path/'algorithm.py';source.write_text('fixed')
    code=round1_lock.save_lock(tmp_path/'results/code_lock.json',[source])
    model=tmp_path/'weights.bin';model.write_bytes(b'fixed weights')
    round1_lock.save_lock(tmp_path/'results/model_lock.json',[model],code_lock_sha256=code)
    round1_lock.verify_model_lock();model.write_bytes(b'changed')
    with pytest.raises(RuntimeError):round1_lock.verify_model_lock()
    model.write_bytes(b'fixed weights');source.write_text('changed source')
    with pytest.raises(RuntimeError):round1_lock.verify_model_lock()
