"""Independent fixed-scope CPU audit; reads native data but never starts Isaac.

Four cases chosen structurally: balanced PI x both normal worlds x none/strong300,
ideal sensors, first profile crossed over three geometries/two signs, clean fit.
No model fitting, hyperparameter selection, native rerun or existing output edit.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/pi-inference-independent-numba')
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import copy
import hashlib
import json
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'costed_benchmark'))
import evaluate as ev
from inference import load_model,build_prior,model_paths
sys.path.insert(0,str(ROOT/'src'))
from implicit_stop_coupling import solve_velocity
from paper_study.costed_benchmark.geometry import cam


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def arrays_equal(first,second):
    return first.keys()==second.keys() and all(np.array_equal(first[k],second[k]) for k in first)


def main():
    started=time.perf_counter()
    data=ROOT/'results/costed_pi_development'
    output=ROOT/'reports/pi_inference_review.json'
    if output.exists():raise FileExistsError('retain existing independent audit')
    generation=json.loads((data/'generation_lock.json').read_text())
    process=json.loads((data/'process.json').read_text())
    assert process['status']=='complete' and process['batches']==128 and process['episodes']==4608
    hashes={str(Path(__file__).resolve()):sha(__file__)}
    for path,expected in generation['sources_sha256'].items():
        assert sha(path)==expected
        hashes[path]=expected
    for path in [*ev.source_paths(), ROOT/'costed_benchmark/test_inference.py',data/'generation_lock.json',data/'process.json']:
        hashes[str(path)]=sha(path)
    cfg=generation['config']
    expected={f'p{s}/{n}_{c}/{p["name"]}/{o}.npz' for s in cfg['preparation_seeds']
              for n in cfg['normal_worlds'] for c in cfg['friction_parameter_sets']
              for p in cfg['probes'] for o in cfg['sensor_conditions']}
    assert {str(p.relative_to(data)) for p in data.rglob('*.npz')}==expected
    guards=0
    for relative in sorted(expected):
        path=data/relative;metadata=json.loads(path.with_suffix('.json').read_text())
        assert metadata['status']=='complete' and metadata['count']==36
        assert metadata['recorded_steps']==metadata['expected_steps']==round((metadata['probe']['duration_s']+.4)/.0025)
        assert metadata['dt']==.0025 and metadata['mass_kg']==2.
        assert sha(path)==metadata['npz_sha256']
        hashes[str(path)]=metadata['npz_sha256'];hashes[str(path.with_suffix('.json'))]=sha(path.with_suffix('.json'))
        with np.load(path,allow_pickle=False) as stored:
            assert stored['u'].shape==(36,metadata['expected_steps'])
            guards+=int(stored['guard'].sum())
    assert guards==0
    replays=[]
    for normal_world in ('maxwell','boucwen'):
        model=load_model('balanced',normal_world,'clean')
        for path in model_paths('balanced',normal_world,'clean'):hashes[str(path)]=sha(path)
        prior,prior_meta=build_prior(model,particles=256,seed=43001)
        for probe in ('none','strong_300'):
            relative=f'p33001/{normal_world}_balanced/{probe}/ideal.npz'
            path=data/relative;metadata=json.loads(path.with_suffix('.json').read_text())
            with np.load(path,allow_pickle=False) as stored:raw={k:stored[k] for k in stored.files}
            # First prepared profile, all three geometries and both signs.
            raw={k:(v[:6].copy() if v.ndim and v.shape[0]==36 else v.copy()) for k,v in raw.items()}
            public=ev.observable_inputs(raw)
            poisoned=copy.deepcopy(raw)
            poisoned['observations'][:,poisoned['observation_steps']>int(poisoned['probe_steps'])]=np.nan
            for key in ('state','next_state','normal','friction','stop_state_before','stop_state_after'):
                poisoned[key]=np.full_like(poisoned[key],np.nan)
            poisoned['preparation_seed'][:]=-1;poisoned['preparation_index'][:]=-1
            assert arrays_equal(public,ev.observable_inputs(poisoned))
            public_meta={key:metadata[key] for key in ev.PUBLIC_META}
            predictions,details=ev.infer_and_forecast(model,public,public_meta,ev.DEFAULT_CONFIG,prior)
            prior_before=prior.copy()
            private_meta=copy.deepcopy(metadata)
            for key in ('v','normal'):
                private_meta['prepared_reference_audit']['commands'][key]=np.asarray(private_meta['prepared_reference_audit']['commands'][key])[:6]
            privileged,extra=ev.privileged_forecasts(model,public,private_meta,ev.DEFAULT_CONFIG)
            assert np.array_equal(prior,prior_before)
            assert np.max(abs(details['weights'].sum(axis=1)-1))<1e-13
            assert np.isfinite(details['weights']).all()
            for key in ('pi','normal'):
                assert np.isfinite(details['boundary_'+key+'_state']).all()
            if probe=='none':
                assert np.array_equal(details['weights'],details['prior'])
                for start in ev.STARTS:
                    assert np.array_equal(predictions[start+'/prior_propagation'],predictions[start+'/trajectory_assimilation'])
            end=int(raw['probe_steps'])
            target=np.concatenate((raw['next_state'][:,end:],raw['normal'][:,end:,None],raw['friction'][:,end:,None]),axis=2)
            combined={**predictions,**privileged}
            errors={name:np.sqrt(np.mean((value-target)**2,axis=1)).mean(axis=0).tolist() for name,value in combined.items()}
            # An independent explicit API assembly versus the accelerated kernel.
            xv=np.zeros((6,2));state=model.initial(6,pi_state=prior[:6])
            normal_state=model.normal_model.initial(np.full(6,.02),np.zeros(6),np.zeros(6))
            max_kernel=dict(velocity=0.,position=0.,normal=0.,friction=0.)
            for step in range(min(40,raw['u'].shape[1])):
                g,slope=cam(xv[:,0],raw['amplitude'],raw['geometry'])
                N,_,normal_next=model.normal_model.step(raw['d'][:,step]+g,
                    raw['d_dot'][:,step]+slope*xv[:,1],xv[:,1],normal_state,dt=model.dt)
                initial_pi=state['pi'].copy()
                force=lambda dx:model.pi.step(initial_pi,dx/model.dt,N,model.dt)[0]
                vn,Ft,_=solve_velocity(xv[:,1],raw['u'][:,step],slope,N,np.sum(model.pi.mu)*N,
                                      force,dt=model.dt,mass=model.mass,iterations=40)
                new,new_state,actualN,actualFt,_=model.step(xv,state,raw['u'][:,step],raw['d'][:,step],
                    raw['d_dot'][:,step],raw['amplitude'],raw['geometry'])
                for key,value in dict(velocity=abs(new[:,1]-vn).max(),position=abs(new[:,0]-xv[:,0]-model.dt*vn).max(),
                                      normal=abs(actualN-N).max(),friction=abs(actualFt-Ft).max()).items():
                    max_kernel[key]=max(max_kernel[key],float(value))
                xv,state,normal_state=new,new_state,normal_next
            assert max(max_kernel.values())<1e-9
            # Diagnose stored native float32 bookkeeping, without changing it.
            current=raw['state'].astype(float);next_state=raw['next_state'].astype(float)
            dx=raw['actual_displacement'].astype(float)
            delta_v=next_state[:,:,1]-current[:,:,1]
            energy=.5*2.*(next_state[:,:,1]**2-current[:,:,1]**2)
            numerical=.5*2.*delta_v**2
            external=raw['u']-raw['slope']*raw['normal']
            momentum=2.*delta_v/.0025-raw['net_force']
            kinematic=dx-.0025*next_state[:,:,1]
            residual=external*dx-energy-(raw['stop_energy_after_J']-raw['stop_energy_before_J'])-raw['stop_dissipation_J']-numerical
            decomposed=raw['friction_commit_minus_applied']*dx+raw['net_force']*kinematic-.0025*momentum*next_state[:,:,1]
            # Original code divided a float32 dx by dt before the material
            # converted velocity to float64. Track the actual committed path.
            committed_dx=(raw['actual_displacement']/.0025).astype(float)*.0025
            corrected_decomposition=decomposed+raw['friction_commit']*(committed_dx-dx)
            precision=dict(
                stored_energy_decomposition_error_J=float(abs(raw['energy_decomposition_error_J']).max()),
                recomputed_float64_decomposition_error_J=float(abs(residual-decomposed).max()),
                actual_commit_displacement_difference_m=float(abs(committed_dx-dx).max()),
                decomposition_with_actual_commit_displacement_error_J=float(abs(residual-corrected_decomposition).max()),
                recomputed_float64_energy_residual_J=float(abs(residual).max()),
                stored_energy_residual_J=float(abs(raw['tangential_balance_residual_J']).max()))
            assert precision['decomposition_with_actual_commit_displacement_error_J']<1e-13
            replays.append(dict(cell=relative,episode_ids=raw['episode_id'].tolist(),particles=256,
                prior_seed=43001,model_case='balanced',calibration_noise='clean',prior_metadata=prior_meta,
                input_whitelist_poison_invariant=True,prior_unchanged_after_privileged=True,
                error_metrics_order=['position_m','velocity_m_s','normal_N','friction_N'],
                descriptive_replay_rmse=errors,max_kernel_vs_public_solver_difference=max_kernel,
                max_inference_solver_residual_N=float(details['max_solver_residual_N']),
                max_privileged_solver_residual_N=float(extra['privileged_max_solver_residual_N']),
                final_ess=ev.weight_diagnostics(details['weights'])['ess'].tolist(),
                native_bookkeeping_precision=precision))
            print('AUDITED PI',relative,flush=True)
    for path,expected_hash in hashes.items():assert sha(path)==expected_hash,path
    result=dict(status='complete',scope='read-only independent fixed-scope audit; no tuning/fit/native execution',
        command=sys.argv,seconds=time.perf_counter()-started,native_batches=128,native_episodes=4608,
        native_guard_steps=guards,source_and_input_sha256=hashes,read_only_replays=replays)
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print('COMPLETE',output,flush=True)


if __name__=='__main__':main()
