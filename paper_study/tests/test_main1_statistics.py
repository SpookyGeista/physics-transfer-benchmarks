"""Synthetic crossed-cluster contracts; never imports held-out data."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from main1_statistics import contrasts,crossed_bootstrap,loss_cube,COMPONENTS


def test_contrasts_are_ratios_of_pooled_mses():
    means=np.array([2.,4.,3.,4.,5.])
    np.testing.assert_array_equal(contrasts(means),[.5,.25,.4])
    with pytest.raises(ValueError):contrasts([1,0,1,1,1])


def test_constant_clusters_have_exact_constant_bootstrap():
    cube=np.broadcast_to([2.,4.,3.,4.,5.],(8,8,5)).copy()
    r=crossed_bootstrap(cube,repetitions=300)
    np.testing.assert_allclose(r['bootstrap_values'],np.tile([.5,.25,.4],(300,1)),atol=1e-14)
    np.testing.assert_allclose(r['interval'],[[.5,.25,.4]]*2,atol=1e-14)


def test_crossed_bootstrap_is_symmetric_and_not_timestep_bootstrap():
    rng=np.random.default_rng(315)
    cube=rng.uniform(.5,2,(8,8,5))
    a=crossed_bootstrap(cube,repetitions=1000)
    b=crossed_bootstrap(cube,repetitions=1000)
    np.testing.assert_array_equal(a['bootstrap_values'],b['bootstrap_values'])
    assert np.all(a['interval'][1]>a['interval'][0])
    with pytest.raises(ValueError):crossed_bootstrap(cube[:1])


def test_episode_aggregation_preserves_model_history_blocks():
    rows=[]
    for bank in [1,2]:
        for history in [31001,31002]:
            for friction,probe,estimator in COMPONENTS:
                for nw in ['maxwell','boucwen']:
                    for ep in range(4):
                        rows.append(dict(sensor='noisy',calibration_noise='noisy',forecast_start='filtered_particles',
                            filter_kind='joint_sensor_likelihood',model_seed=bank,preparation_seed=history,
                            normal_world=nw,friction_world=friction,probe=probe,episode=ep,estimator=estimator,
                            future_position_rmse_m=bank*.01+(history-31000)*.001+ep*.0001))
    d=pd.DataFrame(rows)
    cube,banks,histories=loss_cube(d)
    assert cube.shape==(2,2,5)
    expected=np.mean((.01+.001+np.arange(4)*.0001)**2)
    np.testing.assert_allclose(cube[0,0],expected)
    with pytest.raises(ValueError,match='Duplicate'):loss_cube(pd.concat([d,d.iloc[:1]]))
    with pytest.raises(ValueError):loss_cube(d,main=True)
    d.loc[0,'future_position_rmse_m']=-.001
    with pytest.raises(ValueError,match='nonnegative'):loss_cube(d)


def test_main_rejects_unrecognized_categories_before_counts():
    rows=[]
    for friction,probe,estimator in COMPONENTS:
        for normal in ['maxwell','boucwen']:
            rows.append(dict(sensor='noisy',calibration_noise='noisy',forecast_start='filtered_particles',
                filter_kind='joint_sensor_likelihood',normal_world=normal,friction_world=friction,
                estimator=estimator,model_seed=1,preparation_seed=31001,probe=probe,episode=0,
                future_position_rmse_m=.001))
    # Complete permitted estimator set before corrupting one label.
    extra=rows[0].copy();extra['estimator']='privileged_full_history';rows.append(extra)
    frame=pd.DataFrame(rows)
    frame.loc[frame.estimator=='privileged_full_history','estimator']='renamed_unknown_method'
    with pytest.raises(ValueError,match='categories: estimator'):loss_cube(frame,main=True)
    frame.loc[frame.estimator=='renamed_unknown_method','estimator']='privileged_full_history'
    frame.loc[frame.normal_world=='maxwell','normal_world']='unknown_material'
    with pytest.raises(ValueError,match='categories: normal_world'):loss_cube(frame,main=True)


def test_completed_csv_digest_is_verified(tmp_path):
    import json
    from main1_statistics import sha,verify_evaluation_artifact
    data=tmp_path/'evaluation';data.mkdir()
    (data/'evaluation_lock.json').write_text('{}\n')
    (data/'episodes.csv').write_text('a,b\n1,2\n')
    manifest=dict(evaluation_lock_sha256=sha(data/'evaluation_lock.json'),
                  outputs_sha256={str(data/'episodes.csv'):sha(data/'episodes.csv')})
    (data/'artifact_manifest.json').write_text(json.dumps(manifest))
    verify_evaluation_artifact(data)
    (data/'episodes.csv').write_text('a,b\n1,3\n')
    with pytest.raises(ValueError,match='Episode CSV'):verify_evaluation_artifact(data)
