import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from decision_study.common import DT,MASS,history_init
from decision_study.models import Classical
from decision_study.prediction import step,rollout,plan

def test_native_semiimplicit_equation():
 s=np.zeros((3,2));h=history_init(np.full(3,.02),s[:,1],np.full(3,10));u=np.arange(3.)
 ns,_,n,f,bad=step(Classical(),s,h,np.full(3,10),u,np.full(3,.02),np.zeros(3),np.zeros(3))
 assert np.allclose(ns[:,1],DT*u/MASS) and np.allclose(ns[:,0],DT*ns[:,1]);assert not bad.any()
def test_rollout_uses_only_commands_and_common_initialization():
 commands={'u':np.ones((2,10)),'d':np.full((2,10),.02),'rate':np.zeros((2,10))};s=np.zeros((2,2));n=np.full(2,10.)
 a=rollout(Classical(),commands,np.zeros(2),s,n,'noisy',3);b=rollout(Classical(),commands,np.zeros(2),s,n,'noisy',3)
 for key in a:assert np.array_equal(a[key],b[key])
 assert not np.array_equal(a['state'][0],a['state'][1])
def test_force_limit_in_common_planner():
 s=np.zeros((2,2));h=history_init(np.full(2,.02),s[:,1],np.full(2,10.));u=plan(Classical(),s,h,np.full(2,10.),np.full((2,100),.02),np.zeros((2,100)),np.zeros(2))
 assert np.all(np.abs(u)<=18) and np.all(u>0)
