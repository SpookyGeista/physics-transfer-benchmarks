"""Scientific integration checks; no fitted-model outcome is a test fixture."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from online import Nominal,clone_state,motion_step
from run_model_development import plan
from controller_preflight import FORCES,stage_cost,DT

def test_candidate_batched_plan_matches_separate_rollouts():
    model=Nominal();s=np.array([[.001,.02],[-.002,-.01]])
    d=np.tile([.02,.021,.022],(2,1));rate=d*0;target=d*.3;amp=np.array([.005,-.005]);normal=np.array([13.,17.])
    u,cost,bad=plan(model,s,{},normal,d,rate,target,amp)
    expected=np.zeros_like(cost)
    for k,force in enumerate(FORCES):
        ss=s.copy();h={}
        for t in range(3):
            ss,h,*_=motion_step(model,ss,h,np.full(2,force),d[:,t],rate[:,t],amp,normal_override=normal if t==0 else None)
            expected[:,k]+=DT*stage_cost(ss,np.full(2,force),target[:,t])
    np.testing.assert_allclose(cost,expected,rtol=1e-12,atol=1e-14)
    np.testing.assert_array_equal(u,FORCES[np.argmin(expected,axis=1)])
    assert not bad.any()

def test_parent_state_isolated_from_candidate_mutation():
    state={'normal':np.arange(6).reshape(2,3),'friction':np.arange(4).reshape(2,2)}
    before={k:v.copy() for k,v in state.items()};clone=clone_state(state,41)
    clone['normal'][0,0]=100
    for k in state:np.testing.assert_array_equal(state[k],before[k])
