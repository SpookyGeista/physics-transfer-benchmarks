"""Main2 authorization/causality/schema tests; no reserved history is drawn."""
from pathlib import Path
import copy,sys
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import evaluate_main2 as main2


def test_fixed_budgets_and_calibration_bank_pairing():
    cfg=main2.read(HERE/'main2_evaluation.json');main2.validate_config(cfg)
    for key,value in [('particles',4096),('banks',main2.BANKS[:2]),('methods',list(main2.METHODS[:4]))]:
        bad=copy.deepcopy(cfg);bad[key]=value
        with pytest.raises(ValueError,match='mismatch'):main2.validate_config(bad)
    for bank in cfg['banks']:
        for world in main2.WORLDS:
            for case in main2.CASES:
                pi,normal=main2.model_paths(case,world,'noisy',bank['pi_seed'],bank['normal_seed'])
                assert pi.is_file() and normal.is_file()
                assert f"r{bank['pi_seed']}" in str(pi) and f"r{bank['normal_seed']}" in str(normal)


def test_authorization_requires_every_model_source_before_native_read(tmp_path,monkeypatch):
    sources={str(p):main2.sha(p) for p in main2.source_paths()}
    authorization=dict(role='paper_study_main2_test_authorization',status='locked',schema_version=1,split='test',
        config_sha256=main2.sha(HERE/'main2_native.json'),preparation_seeds=[53201,53202,53203,53204],sources_sha256=sources)
    lock=tmp_path/'synthetic_authorization.json';main2.write_json(lock,authorization)
    main2.verify_authorization(lock)
    # Intentionally break a weight entry and make all native reads observable.
    model=next(path for path in sources if path.endswith('model.json'))
    authorization['sources_sha256'].pop(model);main2.write_json(lock,authorization)
    def forbidden(*args):raise AssertionError('Native input opened before authorization')
    monkeypatch.setattr(main2,'native_inventory',forbidden)
    with pytest.raises(ValueError,match='omits/mismatches'):
        main2.freeze(tmp_path/'does_not_exist',tmp_path/'output',lock)


def test_two_start_selection_uses_past_only_and_retains_start1(monkeypatch):
    events=[];counter=[0]
    def fitted(*args,**kwargs):
        i=counter[0];counter[0]+=1;events.append('fit')
        return {'candidate':i,'diagnostics':[{'best_objective':v} for v in ([1.,3.] if i==0 else [2.,0.])],
            'initial_pi_state':np.full((2,3),i),'boundary_xv':np.full((2,2),i),
            'boundary_state':{'pi':np.full((2,3),i),'normal':np.full((2,3),i)}}
    def future(model,inputs,candidate,**kwargs):
        events.append('future')
        values=np.full((2,4,4),10. if candidate['candidate']==0 else 999.)
        return {'filtered_particles/bounded_initial_state_mle':values},{'filtered_particles_max_solver_residual_N':0.}
    monkeypatch.setattr(main2,'fit_initial_state',fitted);monkeypatch.setattr(main2,'mle_forecast',future)
    means,detail,_,_=main2.two_start_mle(None,None,None,np.zeros((3,3)),np.ones((2,3)),{'mle':{}})
    assert events==['fit','fit','future','future']
    np.testing.assert_array_equal(detail['mle_selected_start'],[0,1])
    np.testing.assert_array_equal(means['bounded_mle_start1'],10.)
    np.testing.assert_array_equal(means['bounded_mle_two_start'][:,0,0],[10.,999.])


def test_episode_mse_is_available_for_primary_pooled_rmse():
    count,steps=2,4
    prediction=np.broadcast_to(np.array([1.,3.])[:,None,None],(count,steps,4)).copy()
    truth=dict(probe_steps=0,next_state=np.zeros((count,steps,2)),normal=np.zeros((count,steps)),friction=np.zeros((count,steps)),
        geometry=np.array([0,2]),guard=np.zeros((count,steps),dtype=bool),actuator_work_J=np.zeros((count,steps)),
        normal_drive_work_discrete_J=np.zeros((count,steps)))
    metadata=dict(normal_world='maxwell',friction_world='balanced',preparation_seed=0,probe=dict(name='synthetic',duration_s=0.),
        factor_rows=[dict(episode_id=f'synthetic_{i}',profile_index=i,excitation_sign=1) for i in range(count)])
    detail={'weights':np.full((count,4),.25),'mle_selected_start':np.array([0,1])}
    mle=[[dict(best_objective=0.,budget_exhausted=False,solver_success=True) for _ in range(count)] for _ in range(2)]
    rows=main2.score({method:prediction for method in main2.METHODS},truth,metadata,main2.BANKS[0],detail,mle)
    assert len(rows)==10
    selected=[row for row in rows if row['method']=='trajectory_assimilation']
    assert np.mean([row['rmse_position_m'] for row in selected])==2.
    assert np.sqrt(np.mean([row['mse_position_m'] for row in selected]))==pytest.approx(np.sqrt(5.))


def test_resume_rejects_partial_wrong_lock_or_hash(tmp_path):
    path=tmp_path/'synthetic.npz'
    with path.open('wb') as stream:np.savez(stream,value=np.ones(1))
    meta=dict(status='complete',evaluation_lock_sha256='lock',input_npz_sha256='input',npz_sha256=main2.sha(path),metrics=[{}]*180)
    main2.write_json(path.with_suffix('.json'),meta)
    assert len(main2.reuse_batch(path,'lock','input')[0])==180
    with pytest.raises(ValueError,match='resume'):main2.reuse_batch(path,'wrong','input')
    path.write_bytes(b'tamper')
    with pytest.raises(ValueError,match='resume'):main2.reuse_batch(path,'lock','input')


def test_raw_factor_and_sensor_clock_checks_without_reserved_data():
    count,steps=2,220
    meta=dict(count=count,expected_steps=steps,probe=dict(duration_s=.15),dt=.0025,
        factor_rows=[dict(amplitude=.003,geometry=i,excitation_sign=(-1 if i==0 else 1)) for i in range(count)])
    obs=np.arange(0,steps+1,10)
    arrays=dict(probe_steps=np.array(60),observation_steps=obs,
        amplitude=np.full(count,.003),geometry=np.arange(count),excitation_sign=np.array([-1.,1.]),
        state=np.zeros((count,steps,2)),next_state=np.zeros((count,steps,2)),observations=np.zeros((count,len(obs),3)),
        **{name:np.zeros((count,steps)) for name in ('u','d','d_dot','normal','friction','guard','actuator_work_J','normal_drive_work_discrete_J')})
    main2.validate_native_arrays(arrays,meta,{'observation_steps':10})
    for key,value,pattern in [('geometry',np.array([1,0]),'factor'),('excitation_sign',np.ones(2),'factor'),
        ('probe_steps',np.array(59),'probe clock'),('observation_steps',obs[:-1],'observation clock'),
        ('observations',np.zeros((count,len(obs)-1,3)),'shape'),('normal',np.full((count,steps),np.nan),'nonfinite')]:
        bad={**arrays,key:value}
        with pytest.raises(ValueError,match=pattern):main2.validate_native_arrays(bad,meta,{'observation_steps':10})


def test_failed_resume_preserves_complete_immutable_attempt_record(tmp_path):
    failed=dict(status='failed',seconds=12.3,failed_batch='synthetic',error='original error',partial_outputs_sha256={})
    history=main2.preserve_failed_attempt(tmp_path,failed)
    assert len(history)==1
    path=Path(next(iter(history)))
    assert main2.read(path)==failed
    second={**failed,'seconds':4.5,'prior_attempts_sha256':history}
    second_history=main2.preserve_failed_attempt(tmp_path,second)
    assert len(second_history)==2 and main2.read(path)==failed
    main2.verify_hashes(second_history)
    with pytest.raises(FileExistsError):main2.preserve_failed_attempt(tmp_path,failed)
    path.write_text('tamper')
    with pytest.raises(ValueError,match='locked file'):main2.preserve_failed_attempt(tmp_path,second)
