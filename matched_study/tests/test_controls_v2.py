"""Regression for actual locked plan configuration, without result access."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from matched_study import compare_controls as original
from matched_study import compare_controls_v2 as corrected
from matched_study import run


def planned(kind, **kwargs):
    # plan reads the real config only; it draws no histories and opens no data.
    return dict(config=run.plan(kind,**kwargs),batches={('synthetic_pair',):'unused'})


@pytest.mark.parametrize('particles',[256,1024])
def test_real_resolution_plan_accepted_with_explicit_mle_disabled(particles):
    main=planned('main');control=planned('resolution',particles=particles)
    assert isinstance(main['config']['mle'],dict)
    assert control['config']['mle'] is False and control['config']['mle_enabled'] is False
    with pytest.raises(ValueError,match='Changed non-control factor: mle'):
        original.validate_pair(main,control,'resolution')
    corrected.validate_pair(main,control,'resolution')


@pytest.mark.parametrize('dt',[.00125,.000625])
def test_real_refinement_plan_still_requires_same_mle(dt):
    main=planned('main');control=planned('refinement',dt=dt)
    corrected.validate_pair(main,control,'refinement')
    changed=copy.deepcopy(control);changed['config']['mle']['max_nfev']+=1
    with pytest.raises(ValueError,match='Changed non-control factor: mle'):
        corrected.validate_pair(main,changed,'refinement')
    changed=copy.deepcopy(control);changed['config']['mle']=False
    with pytest.raises(ValueError,match='Changed non-control factor: mle'):
        corrected.validate_pair(main,changed,'refinement')


@pytest.mark.parametrize('change',[
    {'mle_enabled':True},
    {'mle':{'max_nfev':121,'max_residual_calls':512}},
    {'mle':{'max_nfev':120,'max_residual_calls':512}},
    {'mle':0},
    {'mle_enabled':0},
])
def test_resolution_rejects_enabled_or_budgeted_mle(change):
    main=planned('main');control=planned('resolution',particles=256)
    control['config'].update(change)
    with pytest.raises(ValueError,match='explicitly disable MLE'):
        corrected.validate_pair(main,control,'resolution')


def test_physical_factor_guard_unchanged():
    main=planned('main');control=planned('resolution',particles=256)
    control['config']['cam_amplitude_m']*=2
    with pytest.raises(ValueError,match='Changed non-control factor: cam_amplitude_m'):
        corrected.validate_pair(main,control,'resolution')
