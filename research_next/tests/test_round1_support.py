"""Synthetic audits of marginal coverage, force timing and fixed video cases."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_round1_support as support
import render_round1_video as video


def test_zero_velocity_hold_joins_warmup_and_motion():
    assert support.longest_hold(np.array([.001,0.,0.,0.,.002]),.1)==pytest.approx(.3)
    assert support.longest_hold(np.array([0.,0.,.001]),.1)==pytest.approx(.2)
    assert support.longest_hold(np.array([.001,-.001]),.1)==0
    warm=np.zeros(40);motion=np.r_[np.zeros(3),np.ones(9)*.02]
    assert support.longest_hold(np.r_[warm,motion],.1)==pytest.approx(4.3)


def test_clean_bounds_use_correct_isolated_heads():
    rows=[dict(mode=mode,profile=i,id=f'{mode}{i}') for mode in ['B','A'] for i in range(24)]
    data=dict(delta=np.full((48,8),.02),rate=np.zeros((48,8)),state=np.zeros((48,8,2)),normal=np.full((48,8),12.))
    data['delta'][24:]=99 # A compression is not a normal-head support source.
    data['state'][:24,:,1]=99 # B velocity is not a friction-head support source.
    data['normal'][:24]=99 # Friction normal bounds come from measured A loads.
    data['state'][24:,:,1]=.02;data['state'][24,2:6,1]=0.
    result=support.calibration_summary(data,rows,.1)
    assert result['bounds']['delta']['max']==.02
    assert result['bounds']['velocity']['max']==.02
    assert result['bounds']['normal']['max']==12.
    assert result['friction_A_longest_hold_s']==pytest.approx(.4)


def test_prediction_features_use_pre_step_state_and_own_normal():
    pred=dict(state=np.array([[[.01,.2],[.02,.3],[.03,.4]]]),normal=np.array([[11.,12.,13.]]))
    test=dict(state=np.array([[[.005,.1],[90.,90.],[90.,90.]]]),d=np.full((1,3),.02),rate=np.full((1,3),.01),amplitude=np.array([.005]),normal=np.full((1,3),900.))
    result=support.predicted_signals(pred,test,.0025)
    x=np.array([[.005,.01,.02]]);v=np.array([[.1,.2,.3]]);z=np.tanh(x/.04)
    np.testing.assert_allclose(result['delta'],.02+.005*z)
    np.testing.assert_allclose(result['rate'],.01+.005*(1-z*z)/.04*v)
    np.testing.assert_array_equal(result['velocity'],v)
    np.testing.assert_array_equal(result['normal'],pred['normal'])


def test_all_outside_rows_retained_and_boundaries_included():
    cal=dict(bounds={k:dict(min=0.,max=1.) for k in support.FEATURES},friction_A_longest_hold_s=.2)
    motion={k:np.array([[0.,1.,2.,-1.]]) for k in support.FEATURES}
    warm={k:np.zeros((1,3)) for k in support.FEATURES}
    rows=support.audit_case(motion,warm,[dict(episode=0)],cal,cal,dict(stage='synthetic'),.1)
    assert len(rows)==3
    selected=next(r for r in rows if r['segment']=='motion')
    assert selected['delta_outside_fraction']==.5
    assert selected['delta_below_fraction']==.25 and selected['delta_above_fraction']==.25
    full=next(r for r in rows if r['segment']=='full')
    assert full['samples']==7 and full['longest_zero_velocity_hold_s']==pytest.approx(.4)
    assert full['hold_exceeds_training_max']
    assert full['delta_min']==-1 and full['delta_max']==2


def test_nonfinite_support_is_not_silently_discarded():
    cal=dict(bounds={k:dict(min=0.,max=1.) for k in support.FEATURES},friction_A_longest_hold_s=.2)
    motion={k:np.zeros((1,3)) for k in support.FEATURES};motion['normal'][0,0]=np.nan
    warm={k:np.zeros((1,3)) for k in support.FEATURES}
    with pytest.raises(ValueError):support.audit_case(motion,warm,[{}],cal,cal,{},.1)


def synthetic_video_data():
    dt=.1;warmsteps=40;steps=12
    test=dict(warm_state=np.zeros((1,warmsteps,2)),warm_normal=np.full((1,warmsteps),12.),warm_delta=np.full((1,warmsteps),.02))
    result={}
    for i,method in enumerate(video.METHODS):
        t=np.arange(1,steps+1)*dt
        state=np.column_stack((.02*np.sin(2*np.pi*t/1.2)*(1-i*.05),.04*np.cos(2*np.pi*t/1.2)))
        controller=dict(state=np.concatenate((np.zeros((1,1,2)),state[None,:-1]),axis=1),
                        next_state=state[None],normal=np.full((1,steps),12.+i),delta=np.full((1,steps),.02),
                        target=np.array([[.03]*6+[-.03]*6]))
        result[method]=video.timeline(test,controller,0,dt)
    return result


def test_video_fixed_cases_and_native_time_alignment():
    assert video.COMPOSITIONS==(('maxwell','boucwen'),('boucwen','maxwell'))
    assert video.METHODS==('recurrent','blind_state_id','path_ridge','nominal')
    d=synthetic_video_data()['recurrent']
    assert len(d['time'])==53 and d['time'][-1]==pytest.approx(5.2)
    np.testing.assert_array_equal(d['state'][:41],np.zeros((41,2)))
    assert d['state'][41,0]!=0 and d['normal'][40]==12
    assert d['target'][39]==0 and d['target'][40]==.03 and d['target'][46]==-.03


def test_schematic_render_is_explicit_and_uses_all_fixed_methods(tmp_path):
    import matplotlib.pyplot as plt
    data=synthetic_video_data();fig,update=video.create_figure(data,.005,'maxwell','boucwen')
    update(4.8);path=tmp_path/'synthetic_fixed_video_frame.png';fig.savefig(path,dpi=110)
    assert path.stat().st_size>20000
    labels=' '.join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert all(video.LABELS[m] in labels for m in video.METHODS)
    assert 'gespeicherter nativer PhysX' in fig._supxlabel.get_text()
    assert 'kein RGB' in fig._supxlabel.get_text()
    plt.close(fig)
