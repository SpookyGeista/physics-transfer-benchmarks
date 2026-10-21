"""Independent causal-input firewall and complete-likelihood checks."""
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'tests')]
from joint_observer import JointGridObserver
from test_observer import ToyMaterial
from evaluate_native import observable_inputs, infer_and_forecast
from models import load_model
import json


def joint():
    return JointGridObserver(ToyMaterial(), [.1,.5,.9], [0.,.04], .02, 10.,
                             sigma_x=.001,sigma_v=.01,sigma_N=.2)


def test_joint_normal_measurement_changes_likelihood_not_dynamics_input():
    a, b = joint(), joint()
    for ob in (a,b):
        ob.advance(2., .02, .001, .005, steps=10)
    state_before = {k:v.copy() for k,v in a.state.items()}
    x, v = a.grid_xv.reshape(-1,2).T
    z = np.tanh(x/.04)
    predicted_n = 500*(.02+.005*z)
    observed = np.array([.0005,.03])
    expected = -.5*(np.sum(((observed-a.grid_xv[0])/a.effective_sigma)**2,axis=1)
                   + ((10.2-predicted_n)/a.effective_sigma_N)**2)
    a.observe(*observed,10.2,d=.02,d_dot=.001,amplitude=.005)
    b.observe(*observed,10.8,d=.02,d_dot=.001,amplitude=.005)
    np.testing.assert_allclose(a.log_likelihood[0],expected)
    assert not np.array_equal(a.weights,b.weights)
    for k in state_before:
        np.testing.assert_array_equal(a.state[k],state_before[k])
    for ob in (a,b):
        ob.advance(2.,.02,.001,.005,steps=10)
    np.testing.assert_array_equal(a.grid_xv,b.grid_xv)
    for k in a.state:
        np.testing.assert_array_equal(a.state[k],b.state[k])
    with pytest.raises(ValueError):
        c=joint(); c.observe(0.,.04,10.,d=.02,d_dot=0.,amplitude=0.)


def test_future_measurements_and_true_labels_do_not_enter_forecast():
    file=ROOT/'results/native_smoke/boucwen_boucwen/gentle_150/noisy.npz'
    with np.load(file) as f:
        a={k:f[k] for k in f.files}
    altered={k:v.copy() for k,v in a.items()}
    for key in ('normal','friction','state','next_state','net_force'):
        altered[key][...] = 12345
    altered['observations'][:,a['observation_steps']>int(a['probe_steps'])] = 98765
    inputs=observable_inputs(a)
    other=observable_inputs(altered)
    for key in inputs:
        np.testing.assert_array_equal(inputs[key],other[key])
    assert set(inputs)=={'u','d','d_dot','amplitude','observation_steps','observations','probe_steps'}
    meta=json.loads(file.with_suffix('.json').read_text())
    cfg=json.loads((ROOT/'configs/observer_development.json').read_text())
    cfg['grid_size']=9
    model=load_model('boucwen','boucwen')
    for kind in cfg['filters']:
        first,detail=infer_and_forecast(model,inputs,meta,cfg,kind)
        second,other_detail=infer_and_forecast(model,other,meta,cfg,kind)
        for key in first:
            np.testing.assert_array_equal(first[key],second[key])
        assert detail['weight_trace'].shape[1] == len(inputs['observation_steps'])
