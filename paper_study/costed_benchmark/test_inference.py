"""Synthetic CPU contracts; no native batch or reserved test input is read."""
from pathlib import Path
import sys

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from inference import load_model,build_prior,ParticleObserver
sys.path.insert(0,str(HERE.parent/'src'))
from implicit_stop_coupling import solve_velocity
from geometry import cam


@pytest.mark.parametrize('normal_world',['maxwell','boucwen'])
def test_numba_step_matches_root_solver_and_frozen_normal_model(normal_world):
    model=load_model('balanced',normal_world)
    count=3;amplitude=np.full(count,.003);geometry=np.arange(3)
    xv=np.zeros((count,2));state=model.initial(count)
    state['pi']=model.pi.step(model.pi.initial(count),.04,model.prefix_load(),.4)[1]
    normal_state=model.normal_model.initial(np.full(count,.02),np.zeros(count),np.zeros(count))
    for step in range(20):
        u=np.array([.2,-.4,.6])*np.cos(step*.4)
        d=np.full(count,.02+.0001*np.sin(step*.2));rate=np.full(count,.001)
        g,slope=cam(xv[:,0],amplitude,geometry)
        expected_N,_,new_normal=model.normal_model.step(d+g,rate+slope*xv[:,1],xv[:,1],normal_state,dt=model.dt)
        original_pi=state['pi'].copy()
        def average(dx):
            return model.pi.step(original_pi,dx/model.dt,expected_N,model.dt)[0]
        vn,expected_ft,residual=solve_velocity(xv[:,1],u,slope,expected_N,
            sum(model.pi.mu)*expected_N,average,dt=model.dt,mass=model.mass,iterations=40)
        new_xv,new_state,N,Ft,compiled_residual=model.step(xv,state,u,d,rate,amplitude,geometry)
        np.testing.assert_allclose(new_xv[:,1],vn,atol=2e-14,rtol=0)
        np.testing.assert_allclose(new_xv[:,0],xv[:,0]+model.dt*vn,atol=2e-15,rtol=0)
        np.testing.assert_allclose(N,expected_N,atol=2e-12,rtol=0)
        np.testing.assert_allclose(Ft,expected_ft,atol=2e-11,rtol=0)
        np.testing.assert_array_equal(state['pi'],original_pi)
        if normal_world=='maxwell':
            np.testing.assert_allclose(new_state['normal'],new_normal['relaxed'],atol=1e-15,rtol=0)
        else:
            np.testing.assert_allclose(new_state['normal'][:,0],new_normal['z'],atol=1e-14,rtol=0)
        assert max(abs(compiled_residual))<1e-9
        xv,state,normal_state=new_xv,new_state,new_normal


def test_prior_prefix_stability_and_independent_model_preparation():
    model=load_model('balanced','maxwell')
    small,meta=build_prior(model,particles=8,seed=43991)
    large,_=build_prior(model,particles=16,seed=43991)
    np.testing.assert_array_equal(small,large[:8])
    assert small.shape==(8,3)
    assert meta['prefix_load_N']==pytest.approx(model.prefix_load())
    assert meta['prefix_load_N']!=15.
    assert np.all(abs(small)<=model.pi.alpha[None,:]*model.prefix_load()+1e-12)


def test_causal_weights_material_states_unchanged_by_observation():
    model=load_model('balanced','maxwell')
    hypotheses,_=build_prior(model,particles=16,seed=43992)
    observer=ParticleObserver(model,hypotheses,np.zeros((1,2)),sensor_sigma=(0.,0.,0.))
    plant_state=model.initial(1,pi_state=hypotheses[[5]])
    plant_xv=np.zeros((1,2))
    amplitude=np.array([.003]);geometry=np.array([0])
    with pytest.raises(ValueError,match='initial'):
        observer.observe(0.,0.,model.prefix_load(),d=.02,d_dot=0.,amplitude=amplitude,geometry=geometry)
    for t in range(10):
        u=2*np.sin(t*.3)
        plant_xv,plant_state,_,_,_=model.step(plant_xv,plant_state,u,.02,0.,amplitude,geometry)
        observer.advance(u,.02,0.,amplitude,geometry)
    before=observer.snapshot()
    N=model.normal_force(plant_xv,plant_state,.02,0.,amplitude,geometry)
    weight=observer.observe(plant_xv[:,0],plant_xv[:,1],N,d=.02,d_dot=0.,amplitude=amplitude,geometry=geometry)
    after=observer.snapshot()
    assert weight[0,5]>=observer.prior[0,5]
    np.testing.assert_allclose(weight.sum(axis=1),1.,rtol=0,atol=1e-14)
    np.testing.assert_array_equal(before['xv'],after['xv'])
    for key in ('normal','pi'):
        np.testing.assert_array_equal(before['state'][key],after['state'][key])
    with pytest.raises(ValueError,match='duplicate'):
        observer.observe(0.,0.,N,d=.02,d_dot=0.,amplitude=amplitude,geometry=geometry)


@pytest.mark.parametrize('normal_world',['maxwell','boucwen'])
@pytest.mark.parametrize('case',['balanced','wide_scales','compliant','stiff_high_capacity'])
@pytest.mark.parametrize('noise',['clean','noisy'])
def test_all_existing_fitted_pairs_construct(normal_world,case,noise):
    model=load_model(case,normal_world,noise)
    x,state,N,Ft,residual=model.step(np.zeros((1,2)),model.initial(1),0.,.02,0.,.003,np.array([0]))
    assert np.isfinite(x).all() and np.isfinite(Ft).all()
    assert N[0]>0 and abs(residual[0])<1e-7


def synthetic_inputs(probe_steps=10,future_steps=10):
    total=probe_steps+future_steps
    times=np.array(sorted(set([0,probe_steps,total])),dtype=int)
    return dict(u=np.zeros((2,total)),d=np.full((2,total),.02),d_dot=np.zeros((2,total)),
        amplitude=np.full(2,.003),geometry=np.array([0,2]),probe_steps=np.array(probe_steps),
        observations=np.zeros((2,len(times),3)),observation_steps=times,
        # Sentinels are deliberately nonfinite; none belongs to regular input.
        state=np.full((2,total,2),np.nan),normal=np.full((2,total),np.nan),
        friction=np.full((2,total),np.nan),stop_state_before=np.full((2,total,3),np.nan),
        preparation_seed=np.full(2,33001),preparation_index=np.array([0,1]))


def test_public_whitelist_future_sensor_and_true_state_invariance():
    from evaluate import observable_inputs,PUBLIC_KEYS
    arrays=synthetic_inputs()
    arrays['observations'][:,-1]=np.nan
    public=observable_inputs(arrays)
    assert set(public)==set(PUBLIC_KEYS)|{'probe_steps','observations','observation_steps'}
    assert public['observations'].shape==(2,2,3)
    assert np.isfinite(public['observations']).all()
    arrays['u'][0,0]=999.
    assert public['u'][0,0]==0.
    arrays['observation_steps']=np.array([0,0,20])
    with pytest.raises(ValueError,match='unique'):observable_inputs(arrays)


def test_no_probe_prior_equals_assimilation_and_initial_N_does_not_prepare_prior():
    from evaluate import observable_inputs,infer_and_forecast,DEFAULT_CONFIG
    model=load_model('balanced','maxwell')
    hypotheses,_=build_prior(model,particles=8,seed=43993)
    arrays=synthetic_inputs(probe_steps=0)
    arrays['observations'][:,0,2]=999.  # initial N not a historical clamp
    public=observable_inputs(arrays)
    meta=dict(dt=.0025,mass_kg=2.,initial_compression_m=.02,
              sensor_parameters=dict(sigma_x_m=0.,sigma_v_m_s=0.,sigma_normal_N=0.))
    predictions,details=infer_and_forecast(model,public,meta,DEFAULT_CONFIG,hypotheses)
    for start in DEFAULT_CONFIG['forecast_starts']:
        np.testing.assert_array_equal(predictions[start+'/prior_propagation'],predictions[start+'/trajectory_assimilation'])
    np.testing.assert_array_equal(details['weights'],details['prior'])
    np.testing.assert_array_equal(details['boundary_pi_state'],np.tile(hypotheses,(2,1,1)))
    with pytest.raises(ValueError,match='whitelist'):
        infer_and_forecast(model,public,{**meta,'prepared_reference_audit':{}},DEFAULT_CONFIG,hypotheses)


def test_particle_forecast_aggregation_matches_single_particle_paths():
    from evaluate import observable_inputs,forecast_moments
    model=load_model('balanced','boucwen')
    inputs=observable_inputs(synthetic_inputs(probe_steps=0,future_steps=7))
    hypotheses,_=build_prior(model,particles=3,seed=43994)
    xv=np.zeros((6,2));state=model.initial(6,pi_state=np.tile(hypotheses,(2,1)))
    weights=np.tile([.2,.3,.5],(2,1))
    means,variances,_=forecast_moments(model,xv,state,inputs,{'posterior':weights})
    individual=[]
    for p in range(3):
        s=model.initial(2,pi_state=np.tile(hypotheses[p],(2,1)))
        a,_,_=forecast_moments(model,np.zeros((2,2)),s,inputs,{'one':np.ones((2,1))})
        individual.append(a['one'])
    tensor=np.stack(individual,axis=1)
    expected=np.einsum('ep,eptk->etk',weights,tensor)
    np.testing.assert_allclose(means['posterior'],expected,atol=1e-14)
    np.testing.assert_allclose(variances['posterior'],np.einsum('ep,eptk->etk',weights,(tensor-expected[:,None])**2),atol=1e-14)


def test_weight_collapse_is_finite_diagnostic_not_dropped():
    from evaluate import weight_diagnostics
    d=weight_diagnostics(np.array([[0.,0.,1.],[1/3,1/3,1/3]]))
    np.testing.assert_allclose(d['ess'],[1.,3.])
    np.testing.assert_allclose(d['entropy'],[0.,np.log(3.)])
    np.testing.assert_array_equal(d['positive_particles'],[1,3])


def test_native_inventory_requires_complete_process_and_hashes(tmp_path):
    from evaluate import native_inventory,write_json,sha
    cfg=dict(preparation_seeds=[33001],normal_worlds=['boucwen'],friction_parameter_sets=['balanced'],
        probes=[dict(name='gentle_050')],sensor_conditions={'noisy':{}},episodes_per_batch=2)
    write_json(tmp_path/'generation_lock.json',dict(config=cfg,smoke=True,sources_sha256={}))
    write_json(tmp_path/'process.json',dict(status='running'))
    with pytest.raises(ValueError,match='complete'):native_inventory(tmp_path)
    path=tmp_path/'p33001/boucwen_balanced/gentle_050/noisy.npz'
    path.parent.mkdir(parents=True)
    with path.open('wb') as stream:np.savez(stream,synthetic_only=np.zeros(2))
    meta=dict(status='complete',npz_sha256=sha(path),count=2,recorded_steps=20,expected_steps=20,
        factor_rows=[{},{}],preparation_seed=33001,normal_world='boucwen',friction_world='balanced',
        probe=dict(name='gentle_050'),sensor='noisy',friction_family='three_positive_stops',split='dev')
    write_json(path.with_suffix('.json'),meta)
    write_json(tmp_path/'process.json',dict(status='complete',batches=1,episodes=2))
    _,metadata=native_inventory(tmp_path)
    assert len(metadata)==1
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):native_inventory(tmp_path)


def test_synthetic_freeze_run_artifacts_and_tamper_gate(tmp_path,monkeypatch):
    """Temporary synthetic layout only; does not read native development data."""
    import json
    import evaluate
    model=load_model('balanced','boucwen')
    arrays=synthetic_inputs(probe_steps=20,future_steps=4)
    total=arrays['u'].shape[1]
    current=np.zeros((2,2));material=model.initial(2)
    states=[];next_states=[];normal=[];friction=[];samples=[]
    for t in range(total+1):
        if t in arrays['observation_steps']:
            n=model.normal_force(current,material,.02,0.,arrays['amplitude'],arrays['geometry'])
            samples.append(np.column_stack((current,n)))
        if t==total:break
        states.append(current.copy())
        current,material,N,Ft,_=model.step(current,material,0.,.02,0.,arrays['amplitude'],arrays['geometry'])
        next_states.append(current.copy());normal.append(N);friction.append(Ft)
    arrays.update(state=np.stack(states,axis=1),next_state=np.stack(next_states,axis=1),
        normal=np.stack(normal,axis=1),friction=np.stack(friction,axis=1),
        observations=np.stack(samples,axis=1),guard=np.zeros((2,total),dtype=bool))
    data=tmp_path/'synthetic_native'
    path=data/'p33001/boucwen_balanced/gentle_050/noisy.npz'
    path.parent.mkdir(parents=True)
    with path.open('wb') as stream:np.savez_compressed(stream,**arrays)
    sensor=dict(sigma_x_m=.00005,sigma_v_m_s=.002,sigma_normal_N=.15)
    factors=[dict(episode_id=f'synthetic_{i}',profile_index=i,excitation_sign=1) for i in range(2)]
    metadata=dict(status='complete',npz_sha256=evaluate.sha(path),count=2,recorded_steps=total,
        expected_steps=total,factor_rows=factors,preparation_seed=33001,normal_world='boucwen',
        friction_world='balanced',probe=dict(name='gentle_050',duration_s=.05),sensor='noisy',
        sensor_parameters=sensor,friction_family='three_positive_stops',split='dev',dt=.0025,
        mass_kg=2.,initial_compression_m=.02,prepared_reference_audit=dict(commands={
            'v':np.zeros((2,3)),'normal':np.full((2,3),model.prefix_load())}))
    evaluate.write_json(path.with_suffix('.json'),metadata)
    generation_cfg=dict(preparation_seeds=[33001],normal_worlds=['boucwen'],friction_parameter_sets=['balanced'],
        probes=[dict(name='gentle_050')],sensor_conditions={'noisy':sensor},episodes_per_batch=2,dt=.0025,mass_kg=2.)
    evaluate.write_json(data/'generation_lock.json',dict(config=generation_cfg,smoke=True,sources_sha256={}))
    evaluate.write_json(data/'process.json',dict(status='complete',batches=1,episodes=2))
    # Only output-directory policy changes in the temporary fixture; fitted
    # models remain ordinary pre-existing development artifacts, read-only.
    sources=evaluate.source_paths()
    monkeypatch.setattr(evaluate,'REPO',tmp_path)
    monkeypatch.setattr(evaluate,'source_paths',lambda:sources)
    out=tmp_path/'paper_study/results/synthetic_evaluation'
    cfg={**evaluate.DEFAULT_CONFIG,'particles':4,'calibration_noise':['clean']}
    evaluate.freeze(data,out,cfg)
    assert json.loads((out/'process.json').read_text())['status']=='locked'
    evaluate.run(out)
    process=json.loads((out/'process.json').read_text())
    assert process['status']=='complete' and process['rows']==12
    assert (out/'episodes.csv').is_file()
    with pytest.raises(ValueError,match='unused locked'):evaluate.run(out)
    prior=next((out/'priors').rglob('*.npz'))
    prior.write_bytes(b'changed')
    with pytest.raises(ValueError,match='locked file'):evaluate.verify_lock(out)
