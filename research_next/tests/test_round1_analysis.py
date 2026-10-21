"""Synthetic-only checks for the predeclared Round-1 analysis; no main reads."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import analyze_round1 as analysis


@pytest.fixture(scope='module')
def config():
    return json.loads((ROOT/'configs/round1.json').read_text())


def make_fixture(config):
    keys=analysis.expected_keys(config).to_frame(index=False)
    comp=keys.normal_world.eq('boucwen').astype(float)*2+keys.friction_world.eq('boucwen').astype(float)
    noise=keys.noise.eq('noisy').astype(float)
    offset=keys.method.map({'nominal':.003,'old_flex':0.,'path_ridge':-.001,
                           'recurrent':-.002,'blind_state_id':-.004,
                           'sensor_state_id':-.0015,'state_reset':.001})
    interaction=keys.method.eq('blind_state_id')*keys.seed*.00005
    prediction=keys.copy()
    prediction['position_rmse']=.01+keys.seed*.003+comp*.0002+noise*.0005+keys.episode*.00001+offset+interaction
    prediction['normal_rmse']=.1+prediction.position_rmse*2
    prediction['friction_rmse']=.03+prediction.position_rmse
    prediction['guard_steps']=0
    controller=keys.copy()
    controller['position_squared_integral']=prediction.position_rmse*.01
    controller['integral_cost']=controller.position_squared_integral+.0002
    controller['force_squared_integral']=.5+keys.episode*.01
    controller['guard_steps']=0
    for i,col in enumerate(analysis.SUCCESS):
        controller[col]=(keys.episode < 2*i+3).astype(int)
    controller['plan_seconds']=.01+keys.seed*.0001
    return prediction,controller


def complete_numerics(config):
    return {'complete':True,'observed':{k:float(v)*.5 for k,v in config['numerical_validity'].items()
                                       if isinstance(v,(float,int))},'source_sha256':{'synthetic':'fixture'}}


def test_full_design_and_pairing_are_exact(config):
    pred,control=make_fixture(config)
    result=analysis.analyze_frames(pred,control,config,complete_numerics(config))
    assert result['decision']['prediction_rows']==7168
    assert result['decision']['controller_rows']==7168
    assert len(result['paired_seed_differences'])==40
    assert result['decision']['all_five_advantages']
    assert result['decision']['complete_technical_success']
    assert result['decision']['numerical_validity']['status']=='pass'
    observed=result['paired_seed_differences'].query('contrast == 1').sort_values('seed').difference.to_numpy()
    np.testing.assert_allclose(observed,-.004+np.arange(1,9)*.00005,atol=1e-17)
    expected=analysis.paired_interval(observed)
    got=result['primary_contrasts'].iloc[0]
    assert got.ci_lower==pytest.approx(expected['ci_lower'])
    assert got.standard_error==pytest.approx(np.std(observed,ddof=1)/np.sqrt(8))


def test_cell_then_seed_weighting_and_input_order(config):
    pred,control=make_fixture(config)
    result=analysis.analyze_frames(pred.sample(frac=1,random_state=1),control.sample(frac=1,random_state=2),config)
    row=result['seed_aggregates'].query('dataset == "prediction" and metric == "position_rmse" and seed == 3 and method == "blind_state_id"').iloc[0]
    expected=pred.query('seed == 3 and method == "blind_state_id"').position_rmse.mean()
    assert row.value==pytest.approx(expected)
    assert len(result['cell_aggregates'].query('dataset == "prediction" and metric == "position_rmse" and seed == 3 and method == "blind_state_id"'))==8


@pytest.mark.parametrize('kind',['missing','duplicate','unexpected','nonfinite','negative','fractional_guard','invalid_success'])
def test_invalid_input_is_rejected_without_dropping(config,kind):
    pred,control=make_fixture(config)
    dataset='prediction';frame=pred
    if kind=='missing':frame=pred.iloc[:-1]
    elif kind=='duplicate':frame=pd.concat([pred.iloc[:-1],pred.iloc[[0]]])
    elif kind=='unexpected':frame.loc[0,'episode']=16
    elif kind=='nonfinite':frame.loc[0,'position_rmse']=np.nan
    elif kind=='negative':frame.loc[0,'normal_rmse']=-1
    elif kind=='fractional_guard':frame['guard_steps']=frame.guard_steps.astype(float);frame.loc[0,'guard_steps']=.5
    elif kind=='invalid_success':dataset='controller';frame=control;frame.loc[0,'success_020']=2
    with pytest.raises(ValueError):analysis.validate_metrics(frame,dataset,config)


def test_fixed_bonferroni_t_and_all_decision_directions():
    ci=analysis.paired_interval(np.arange(1,9,dtype=float))
    assert ci['critical_t']==pytest.approx(4.029337177642779,rel=1e-13)
    assert ci['individual_ci_coverage']==.995
    assert ci['alpha_per_contrast']==.005
    assert ci['ci_lower']==pytest.approx(4.5-4.029337177642779*np.std(np.arange(1,9),ddof=1)/np.sqrt(8))
    assert analysis.paired_interval(np.full(8,-1.))['decision']=='advantage'
    assert analysis.paired_interval(np.full(8,1.))['decision']=='disadvantage'
    zero=analysis.paired_interval(np.zeros(8))
    assert zero['decision']=='unresolved' and zero['p_two_sided']==1 and zero['ci_upper']==0
    assert analysis.paired_interval(np.array([-4,-3,-2,-1,1,2,3,4]))['decision']=='unresolved'
    with pytest.raises(ValueError):analysis.paired_interval(np.zeros(128))


def test_guards_are_retained_and_veto_complete_technical_claim(config):
    pred,control=make_fixture(config)
    index=pred.index[pred.method=='blind_state_id'][0]
    pred.loc[index,'guard_steps']=19;pred.loc[index,'position_rmse']=1.5
    control.loc[1,'guard_steps']=3
    result=analysis.analyze_frames(pred,control,config,complete_numerics(config))
    assert len(result['frames']['prediction'])==7168
    assert result['decision']['guarded_episodes']==2
    assert not result['decision']['complete_technical_success']
    assert result['decision']['technical_claim_status']=='inconclusive_numerical_validity'
    assert result['guards'].reported_guard_count.sum()==22
    key=pred.loc[index]
    selected=result['seed_aggregates'].query('dataset == "prediction" and metric == "position_rmse"')
    aggregate=selected[(selected.seed==key.seed)&(selected.method==key.method)].value.iloc[0]
    assert aggregate==pytest.approx(pred[(pred.seed==key.seed)&(pred.method==key.method)].position_rmse.mean())


def test_comparator_guards_do_not_veto_valid_main_method(config):
    pred,control=make_fixture(config)
    index=pred.index[pred.method=='old_flex'][0]
    pred.loc[index,'guard_steps']=19
    control.loc[index,'guard_steps']=3
    result=analysis.analyze_frames(pred,control,config,complete_numerics(config))
    assert result['decision']['guarded_episodes']==2
    assert result['decision']['guarded_episodes_main_method']==0
    assert result['decision']['main_method_no_guard_activations']
    assert not result['decision']['no_guard_activations']
    assert result['decision']['complete_technical_success']


def test_numerical_pending_failure_and_threshold_equality(config):
    pred,control=make_fixture(config)
    pending=analysis.analyze_frames(pred,control,config)
    assert pending['decision']['all_five_advantages'] and not pending['decision']['complete_technical_success']
    assert pending['decision']['technical_claim_status']=='pending_numerical_validity'
    data=complete_numerics(config)
    k='native_velocity_tracking_max';data['observed'][k]=config['numerical_validity'][k]
    assert analysis.numerical_status(data,config)['status']=='pass'
    data['observed'][k]*=1.01
    assert analysis.numerical_status(data,config)['status']=='failed'
    del data['observed'][k]
    assert analysis.numerical_status(data,config)['status']=='failed'


def test_standard_boolean_csv_spellings(config):
    pred,control=make_fixture(config)
    for col in analysis.SUCCESS:control[col]=control[col].map({0:'False',1:'True'})
    got=analysis.validate_metrics(control,'controller',config)
    assert set(got.success_020)=={0,1}


def test_config_cannot_reduce_seeds_or_change_primary_targets(config):
    changed=copy.deepcopy(config);changed['replicates']=list(range(1,8))
    with pytest.raises(ValueError):analysis.validate_config(changed)
    changed=copy.deepcopy(config);changed['primary_pairs'][0][1]='recurrent'
    with pytest.raises(ValueError):analysis.validate_config(changed)


def test_synthetic_tables_report_and_fixed_figures(tmp_path,config):
    pred,control=make_fixture(config)
    result=analysis.analyze_frames(pred,control,config,complete_numerics(config))
    paths=analysis.make_figures(result,config,tmp_path/'figures')
    assert len(paths)==4 and all(Path(path).stat().st_size>10000 for path in paths)
    report=tmp_path/'synthetic_report.md'
    analysis.write_report(result,config,tmp_path,paths,report)
    assert '7168' in report.read_text() and '99,5%' in report.read_text()
    analysis.write_json(tmp_path/'decision.json',result['decision'])
    assert json.loads((tmp_path/'decision.json').read_text())['complete_technical_success']
