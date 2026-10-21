"""Synthetic analysis contracts; no material histories or reserved inputs drawn."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC=importlib.util.spec_from_file_location('matched_analysis',Path(__file__).resolve().parents[1]/'analyze.py')
A=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(A)


def cfg():
    return dict(kind='development',banks=[1],normals=['synthetic'],cases=['synthetic'],
        history_seeds=[80001,80002],profiles_per_history=2,sign_assignment=[-1,1],
        particles=2,dt=.0025,durations=[.15,.30],strengths_N=[0.,3.,12.],
        future_amplitudes_N=[2.,10.],future_duration=.005,mle_enabled=True,
        sensor_sigma={'clean':[0.,0.,0.],'noisy':[.1,.2,.3]},numerical_floor=[.01,.01,.01],
        bootstrap={'seed':76001,'draws':1000,'interval':[.005,.995]},maximum_solver_residual_N=1e-7)


def synthetic(c=None,sign=-1,duration=.15,strength=0.,history=80001):
    c=c or cfg();indices=np.flatnonzero(np.array(c['sign_assignment'])==sign)
    n,f,e,m,p=2,2,len(indices),5,c['particles'];t=round(c['future_duration']/.0025);s=round(duration/.025)
    truth=np.zeros((f,e,t,4));pred=np.zeros((n,f,e,m,t,4))
    # Deliberately different errors, to expose MSE/RMSE pooling and orientation.
    for mi,error in enumerate([4.,2.,3.,1.,0.]):pred[:,:,:,mi]=error*(1+strength/12)*(duration/.15)
    prior=np.full((f,t,4),4.*(1+strength/12)*(duration/.15))
    post=np.broadcast_to(prior[None,:,None]*.5,(n,f,e,t,4)).copy()
    diagnostics=[]
    for future in c['future_amplitudes_N']:
        d={k:np.full(e,strength*.01 if k.startswith('prep_') else future*.01).tolist() for k in A.WORK}
        d['max_solver_residual_N']=1e-12;diagnostics.append(d)
    meta=dict(bank=1,normal='synthetic',case='synthetic',history=history,sign=sign,duration=duration,
        strength=strength,dt=c['dt'],particles=p,sensors=list(c['sensor_sigma']),futures=c['future_amplitudes_N'],
        methods=list(A.METHODS),channel_order=list(A.CHANNELS),ensemble_path='/synthetic/ensemble',ensemble_sha256='a'*64,diagnostics=diagnostics)
    arr=dict(truth_future=truth,predictions=pred,truth_past=np.zeros((e,s,3)),observed=np.zeros((n,e,s,3)),
        noise_standard=np.zeros((e,s,3)),effective_sigma=np.array([np.hypot(v,c['numerical_floor']) for v in c['sensor_sigma'].values()]),
        prior_variance=prior,posterior_variance=post,weights=np.full((n,e,p),1/p),profile_indices=indices,
        mle_cost=np.broadcast_to([2.,1.],(n,e,2)).copy(),mle_chosen=np.ones((n,e)),mle_initial=np.zeros((n,e,2,3)))
    return arr,meta


def test_mse_pooling_and_shape():
    truth=np.zeros((1,2,2,4));pred=np.zeros((1,1,2,1,2,4))
    pred[:,:,0]=1.;pred[:,:,1]=3.
    score=A.score_arrays(pred,truth)
    assert score['mse'].mean()==5.
    assert np.sqrt(score['mse'].mean()) != score['rmse'].mean()
    with pytest.raises(ValueError):A.score_arrays(pred,truth[:,:1])
    pred.flat[0]=np.nan
    with pytest.raises(ValueError):A.score_arrays(pred,truth)


def test_weights_zero_entropy_and_rejection():
    result=A.weight_metrics(np.array([[[1.,0.],[.5,.5]]]))
    np.testing.assert_array_equal(result['ess'],[[1.,2.]])
    np.testing.assert_allclose(result['entropy'],[[0.,np.log(2.)]])
    with pytest.raises(ValueError):A.weight_metrics(np.array([[[1.,1.]]]))


def test_batch_mle_choice_and_variance_can_increase():
    c=cfg();arr,meta=synthetic(c)
    arr['posterior_variance']*=4.
    rows,info,physical=A.batch_rows(arr,meta,c)
    assert len(rows)==2*2*1*5*4
    assert all(r['posterior_variance']>r['prior_variance'] for r in info)
    assert len(physical)==8
    arr['mle_chosen'][:]=0
    with pytest.raises(ValueError,match='minimum past'):A.batch_rows(arr,meta,c)
    arr['mle_cost'][:]=1.
    A.batch_rows(arr,meta,c) # exact tie retains first


def test_physical_dedup_and_disagreement():
    arr,meta=synthetic();_,_,raw=A.batch_rows(arr,meta,cfg())
    frame=pd.DataFrame(raw+copy.deepcopy(raw))
    assert len(A.deduplicate_physical(frame))==8
    frame.loc[8,'prior_variance']+=1e-10
    with pytest.raises(ValueError,match='changed across histories'):A.deduplicate_physical(frame)


def test_crossed_ratio_of_means_not_mean_ratios():
    f=pd.DataFrame([dict(bank=b,history=h,numerator=b+h,denominator=b*h) for b in [1,2,3] for h in [1,2,3,4]])
    r=A.crossed_summary(f,'numerator','denominator',cfg())
    assert r['ratio']==pytest.approx(f.numerator.mean()/f.denominator.mean())
    assert r['ratio']!=pytest.approx((f.numerator/f.denominator).mean())
    assert r==A.crossed_summary(f,'numerator','denominator',cfg())
    f.numerator=2*f.denominator
    r=A.crossed_summary(f,'numerator','denominator',cfg())
    assert r['ratio_low']==r['ratio_high']==2.
    with pytest.raises(ValueError,match='Incomplete crossed'):A.crossed_summary(f.iloc[:-1],'numerator','denominator',cfg())
    with pytest.raises(ValueError,match='Unbalanced'):A.crossed_summary(pd.concat([f,f.iloc[:1]]),'numerator','denominator',cfg())


def test_zero_denominator_no_epsilon_or_dropped_draws():
    f=pd.DataFrame([dict(bank=b,history=h,x=1.,y=0.) for b in [1,2] for h in [1,2]])
    r=A.crossed_summary(f,'x','y',cfg())
    assert r['ratio'] is None and r['ratio_low'] is None
    assert r['undefined_ratio_draws']==1000
    assert r['difference_low']==r['difference_high']==1.
    r=A.crossed_summary(f[f.bank==1],'x','y',cfg())
    assert r['bootstrap_draws']==0 and r['ratio_high'] is None


def test_all_prespecified_comparisons_and_physical_counts():
    c=cfg();rows=[];infos=[];physical=[]
    for duration in c['durations']:
        for strength in c['strengths_N']:
            for sign in [-1,1]:
                for history in c['history_seeds']:
                    arr,meta=synthetic(c,sign,duration,strength,history)
                    meta['ensemble_path']=f'/{duration}/{strength}/{sign}'
                    r,i,p=A.batch_rows(arr,meta,c);rows+=r;infos+=i;physical+=p
    df=pd.DataFrame(rows);physical=A.deduplicate_physical(pd.DataFrame(physical))
    mse,info,phys=A.comparisons(df,pd.DataFrame(infos),physical,c)
    expected={'posterior/prior MSE','mle_start1/posterior MSE','mle_two_start/posterior MSE','mle_two_start/mle_start1 MSE',
              '3/0 N MSE','12/0 N MSE','12/3 N MSE','0.3/0.15 s MSE'}
    assert set(mse.contrast)==expected
    assert set(phys.contrast)=={'3/0 N prior variance','12/0 N prior variance','12/3 N prior variance','0.3/0.15 s prior variance'}
    assert len(A.truth_work(df))==2*3*2*2*2
    assert len(physical)==2*3*2*2*4
    v=mse[(mse.contrast=='mle_two_start/mle_start1 MSE')]
    np.testing.assert_allclose(v.ratio,1/9)
    with pytest.raises(ValueError,match='Incomplete paired'):A.paired(df.iloc[1:], 'posterior','prior','method','mse')


def fake_run(tmp_path,kind='development',bind=False):
    data=tmp_path/'data';data.mkdir();c=cfg();c.update(kind=kind,durations=[.15],strengths_N=[0.,12.])
    paths,identities=A.expected_inventory(data,c)
    for path in paths:
        path.parent.mkdir(parents=True,exist_ok=True)
        if path not in identities:np.savez_compressed(path,synthetic=np.array([1.]))
    for path,identity in identities.items():
        arr,meta=synthetic(c,identity['sign'],identity['duration'],identity['strength'],identity['history'])
        meta.update(identity);meta['ensemble_sha256']=A.sha(meta['ensemble_path'])
        np.savez_compressed(path,**arr,metadata_json=np.array(json.dumps(meta)))
    sources={str(Path(A.__file__).resolve()):A.sha(A.__file__)} if bind else {}
    A.write(data/'lock.json',dict(config=c,source_sha256=sources))
    locksha=A.sha(data/'lock.json')
    A.write(data/'process.json',dict(status='complete',lock_sha256=locksha,batches=len(identities)))
    A.write(data/'manifest.json',dict(lock_sha256=locksha,outputs_sha256={str(p):A.sha(p) for p in paths}))
    return data


def test_full_synthetic_formatter_inventory_and_manifest(tmp_path):
    data=fake_run(tmp_path);out=tmp_path/'analysis';A.run(data,out)
    process=A.read(out/'process.json');summary=A.read(out/'summary.json')
    assert process['status']=='complete'
    assert summary['metric_rows']==8*2*2*5*4
    assert summary['unique_ensemble_future_contexts']==4*2
    assert not summary['analyzer_frozen_before_generation']
    A.check(A.read(out/'manifest.json')['outputs_sha256'])
    assert (out/'truth_preparation_work.pdf').is_file()
    with pytest.raises(FileExistsError):A.run(data,out)


def test_main_requires_prospective_analyzer_binding(tmp_path):
    data=fake_run(tmp_path,kind='main');out=tmp_path/'analysis'
    with pytest.raises(ValueError,match='frozen in the source lock'):A.run(data,out)
    assert A.read(out/'process.json')['status']=='failed'
    assert not (out/'manifest.json').exists()


def test_late_input_tamper_is_failed_not_complete(tmp_path,monkeypatch):
    data=fake_run(tmp_path,kind='main',bind=True);out=tmp_path/'analysis'
    def mutate(*args):
        path=next(data.rglob('*.npz'));path.write_bytes(path.read_bytes()+b'tamper')
        return []
    monkeypatch.setattr(A,'figures',mutate)
    with pytest.raises(ValueError,match='Changed bound source/input'):A.run(data,out)
    assert A.read(out/'process.json')['status']=='failed'
    assert not (out/'manifest.json').exists()

