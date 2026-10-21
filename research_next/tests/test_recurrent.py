import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from recurrent import Recurrent

def test_sequence_matches_causal_step():
 torch.manual_seed(8);m=Recurrent();x=torch.randn(3,30,2);x[...,0]=1
 with torch.no_grad():
  m.normal_head.readout[-1].weight.normal_()
  m.friction_head.readout[-1].weight.normal_()
 y=m.normal_sequence(x);h=None;steps=[]
 for t in range(x.shape[1]):
  v,h=m.normal_step(x[:,t],h);steps.append(v)
 assert torch.allclose(y,torch.stack(steps,1),atol=1e-5)
 f=torch.randn(3,30,3);f[...,2]=1;h=None;steps=[]
 for t in range(f.shape[1]):
  value,h=m.friction_step(f[:,t],h);steps.append(value)
 assert torch.allclose(m.friction_sequence(f),torch.stack(steps,1),atol=1e-5)
def test_future_does_not_change_past_force():
 torch.manual_seed(3);m=Recurrent()
 with torch.no_grad():m.normal_head.readout[-1].weight.normal_()
 x=torch.randn(4,50,2);x[...,0]=1;y=m.normal_sequence(x);x[:,25:]+=1
 assert torch.allclose(y[:,:25],m.normal_sequence(x)[:,:25])
def test_no_target_normal_input_and_friction_sign():
 m=Recurrent();x=torch.randn(5,20,3);x[...,2]=1
 f=m.friction_sequence(x);assert torch.all(f*x[...,0]<=1e-7)
 assert m.normal_head.cell.input_size==2
