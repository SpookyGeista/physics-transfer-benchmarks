"""Independent PI h/h2/h4 audit of complete native records; CPU only.

Both preparation and coupled phase are refined by the recorded native wrapper.
All 24 declared ideal-sensor cells and all 36 episodes are retained. Fitted
full-history replays are diagnostics, not particle inference or a new decision.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/pi-refinement-independent-numba')
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'costed_benchmark'))
import evaluate as ev
from inference import load_model,model_paths


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    with Path(path).open('x') as stream:
        json.dump(ev.jsonable(value),stream,indent=2,allow_nan=False)
        stream.write('\n')


def stats(error):
    error=np.asarray(error,dtype=float)
    rms=np.sqrt(np.mean(error**2,axis=1))
    return dict(episode_rmse=rms,mean_episode_rmse=np.mean(rms,axis=0),
                max_absolute=np.max(np.abs(error),axis=(0,1)))


def precision(raw,dt,mass):
    current=raw['state'].astype(float);following=raw['next_state'].astype(float)
    # Native coordinate differences are recomputed in float64. Stored float32
    # dx and its float32 division determine the actual material commit path.
    dx=following[:,:,0]-current[:,:,0]
    committed_dx=(raw['actual_displacement']/dt).astype(float)*dt
    dv=following[:,:,1]-current[:,:,1];vn=following[:,:,1]
    kinetic=.5*mass*(vn**2-current[:,:,1]**2)
    numerical=.5*mass*dv**2
    external=raw['u']-raw['slope']*raw['normal']
    momentum=mass*dv/dt-raw['net_force']
    position=dx-dt*vn
    residual=external*dx-kinetic-(raw['stop_energy_after_J']-raw['stop_energy_before_J'])-raw['stop_dissipation_J']-numerical
    base=raw['friction_commit_minus_applied']*dx+raw['net_force']*position-dt*momentum*vn
    corrected=base+raw['friction_commit']*(committed_dx-dx)
    result=dict(max_native_momentum_residual_N=float(abs(momentum).max()),
        max_native_position_residual_m=float(abs(position).max()),
        max_recorded_dx_rounding_m=float(abs(dx-raw['actual_displacement']).max()),
        max_pi_commit_displacement_difference_m=float(abs(committed_dx-dx).max()),
        max_float64_energy_residual_J=float(abs(residual).max()),
        max_stored_energy_residual_J=float(abs(raw['tangential_balance_residual_J']).max()),
        max_energy_decomposition_without_commit_correction_J=float(abs(residual-base).max()),
        max_corrected_energy_decomposition_error_J=float(abs(residual-corrected).max()),
        max_stored_energy_decomposition_error_J=float(abs(raw['energy_decomposition_error_J']).max()),
        max_native_solver_residual_N=float(abs(raw['solver_residual_N']).max()),
        max_friction_commit_minus_applied_N=float(abs(raw['friction_commit_minus_applied']).max()),
        min_stop_dissipation_J=float(raw['stop_dissipation_J'].min()),
        min_stop_load_release_J=float(raw['stop_load_release_J'].min()),
        min_stop_slip_dissipation_J=float(raw['stop_slip_dissipation_J'].min()),
        max_stop_material_residual_J=float(abs(raw['stop_material_residual_J']).max()),
        max_normal_geometry_discretization_J=float(abs(raw['normal_geometry_discretization_J']).max()),
        episode_total_float64_energy_residual_J=np.sum(residual,axis=1))
    assert result['max_corrected_energy_decomposition_error_J']<1e-12
    assert result['max_native_solver_residual_N']<1e-7
    assert min(result[k] for k in ('min_stop_dissipation_J','min_stop_load_release_J','min_stop_slip_dissipation_J'))>=-1e-14
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/pi_refinement_independent_audit')
    parser.add_argument('--report-prefix',type=Path,default=ROOT/'reports/pi_numerical_refinement')
    args=parser.parse_args();out=args.out.resolve();report=args.report_prefix.resolve()
    if out.exists() or any(report.with_suffix(s).exists() for s in ('.md','.json','.png')):
        raise FileExistsError('retain prior independent audit outputs; choose fresh output paths')
    roots={'h':ROOT/'results/costed_pi_development','h2':ROOT/'results/pi_native_refinement_h2','h4':ROOT/'results/pi_native_refinement_h4'}
    divisors={'h':1,'h2':2,'h4':4};hashes={};locks={}
    def capture(path,expected=None):
        digest=sha(path)
        if expected is not None:assert digest==expected,str(path)
        hashes[str(Path(path).resolve())]=digest
    capture(__file__)
    for path in ev.source_paths():capture(path)
    for level,path in roots.items():
        process=json.loads((path/'process.json').read_text())
        assert process['status']=='complete',(level,process)
        assert process['batches']==(128 if level=='h' else 24)
        assert process['episodes']==(4608 if level=='h' else 864)
        lock=json.loads((path/'generation_lock.json').read_text());locks[level]=lock
        for source,expected in lock['sources_sha256'].items():capture(source,expected)
        capture(path/'generation_lock.json');capture(path/'process.json')
        cfg=lock['config']
        expected={f'p{s}/{n}_{c}/{p["name"]}/{sensor}.npz' for s in cfg['preparation_seeds'] for n in cfg['normal_worlds'] for c in cfg['friction_parameter_sets'] for p in cfg['probes'] for sensor in cfg['sensor_conditions']}
        assert {str(p.relative_to(path)) for p in path.rglob('*.npz')}==expected
    selected=sorted(str(p.relative_to(roots['h2'])) for p in roots['h2'].rglob('*.npz'))
    assert len(selected)==24
    out.mkdir(parents=True)
    write_json(out/'status.json',dict(status='running',command=sys.argv))
    start=time.perf_counter();cells=[];output_hashes={};guards=0
    try:
        for relative in selected:
            metadata={};arrays={};predictions={};level_details={}
            for level,path in roots.items():
                file=path/relative;meta=json.loads(file.with_suffix('.json').read_text())
                capture(file,meta['npz_sha256']);capture(file.with_suffix('.json'))
                with np.load(file,allow_pickle=False) as archive:raw={k:archive[k].copy() for k in archive.files}
                q=divisors[level];dt=.0025/q
                assert meta['status']=='complete' and meta['count']==36
                assert meta['dt']==dt and meta['sensor']=='ideal'
                assert meta['recorded_steps']==meta['expected_steps']==round((meta['probe']['duration_s']+.4)/dt)
                assert raw['next_state'].shape==(36,meta['recorded_steps'],2)
                assert all(np.isfinite(a).all() for a in raw.values() if a.dtype.kind in 'fiu')
                guards+=int(raw['guard'].sum())
                assert np.array_equal(raw['normal_updates'],np.broadcast_to(np.arange(1,meta['recorded_steps']+1),(36,meta['recorded_steps'])))
                commands=meta['prepared_reference_audit']['commands']
                times=np.asarray(commands['time']);v=np.asarray(commands['v']);x=np.asarray(commands['x'])
                assert len(times)==round(4/dt)+1 and abs(times[-1]-4)<1e-14
                assert np.max(abs(np.diff(times)-dt))<1e-14
                assert v.shape==x.shape==(36,len(times))
                assert np.max(abs(v[:,:-1]*dt-np.diff(x)))<1e-15
                assert all(r['dt']==dt and r['duration_s']==4 for r in meta['prepared_reference_audit']['rows'])
                assert np.array_equal(np.asarray(meta['initial_truth_audit']['deflection']),raw['stop_state_before'][:,0])
                for k in ('amplitude','geometry','excitation_sign','preparation_seed','preparation_index','episode_id'):
                    if level!='h':assert np.array_equal(raw[k],arrays['h'][k]),(level,k)
                model=load_model(meta['friction_world'],meta['normal_world'],'clean',dt=dt)
                for p in model_paths(meta['friction_world'],meta['normal_world'],'clean'):capture(p)
                public=ev.observable_inputs(raw)
                cfg={**ev.DEFAULT_CONFIG,'forecast_starts':['filtered_particles']}
                forecast,details=ev.privileged_forecasts(model,public,meta,cfg)
                pred=forecast['filtered_particles/privileged_full_history']
                assert float(details['privileged_max_solver_residual_N'])<1e-7
                begin=int(raw['probe_steps'])
                target=np.concatenate((raw['next_state'][:,begin:],raw['normal'][:,begin:,None],raw['friction'][:,begin:,None]),axis=2).astype(float)
                # State fields are poststep; normal is pre-update and Ft is the
                # interval mean. Common-grid comparison respects both clocks.
                common_pred=np.concatenate((pred[:,q-1::q,:2],pred[:,::q,2:]),axis=2)
                common_target=np.concatenate((target[:,q-1::q,:2],target[:,::q,2:]),axis=2)
                destination=out/'privileged'/level/relative;destination.parent.mkdir(parents=True,exist_ok=True)
                with destination.open('xb') as stream:np.savez_compressed(stream,prediction=pred,episode_id=raw['episode_id'],dt=dt,probe_steps=begin)
                output_hashes[str(destination)]=sha(destination)
                level_details[level]=dict(dt=dt,guards=int(raw['guard'].sum()),precision=precision(raw,dt,meta['mass_kg']),
                    privileged_own_grid=stats(pred-target),privileged_common_grid=stats(common_pred-common_target),
                    privileged_solver_residual_N=float(details['privileged_max_solver_residual_N']))
                metadata[level]=meta;arrays[level]=raw;predictions[level]=pred
            comparisons={};prefix={}
            for a,b in (('h','h2'),('h2','h4'),('h','h4')):
                ratio=divisors[b]//divisors[a];aa,bb=arrays[a],arrays[b]
                ca,cb=[metadata[k]['prepared_reference_audit']['commands'] for k in (a,b)]
                rowsa,rowsb=[metadata[k]['prepared_reference_audit']['rows'] for k in (a,b)]
                assert all(ra['waypoints_m']==rb['waypoints_m'] and ra['knot_times_s']==rb['knot_times_s'] for ra,rb in zip(rowsa,rowsb))
                xdiff=np.max(abs(np.asarray(ca['x'])-np.asarray(cb['x'])[:,::ratio]))
                ndiff=np.max(abs(np.asarray(ca['normal'])-np.asarray(cb['normal'])[:,::ratio]))
                state_diff=np.max(abs(aa['stop_state_before'][:,0]-bb['stop_state_before'][:,0]))
                prefix[a+'_'+b]=dict(max_common_position_difference_m=float(xdiff),max_normal_load_difference_N=float(ndiff),
                    max_initial_pi_state_difference_m=float(state_diff),all_36_waypoints_and_turning_times_identical=True)
                assert xdiff<1e-15 and ndiff==0 and state_diff<1e-13
                postdiff=bb['next_state'][:,ratio-1::ratio].astype(float)-aa['next_state']
                normaldiff=bb['normal'][:,::ratio]-aa['normal']
                # Ft is an interval average, so report both left-aligned small
                # interval force and time-averaged fine force on coarse bins.
                friction_left=bb['friction'][:,::ratio]-aa['friction']
                friction_mean=bb['friction'].reshape(36,-1,ratio).mean(axis=2)-aa['friction']
                end=int(aa['probe_steps'])
                matrix=np.concatenate((postdiff,normaldiff[:,:,None],friction_left[:,:,None]),axis=2)
                ddiff=np.max(abs(bb['d'][:,::ratio]-aa['d']));ratediff=np.max(abs(bb['d_dot'][:,::ratio]-aa['d_dot']))
                assert ddiff<1e-14 and ratediff<1e-12
                comparisons[a+'_'+b]=dict(full=stats(matrix),future=stats(matrix[:,end:]),
                    future_time_averaged_friction_difference=stats(friction_mean[:,end:,None]),
                    max_common_d_difference_m=float(ddiff),max_common_rate_difference_m_s=float(ratediff),
                    max_common_u_difference_N=float(abs(bb['u'][:,::ratio]-aa['u']).max()),
                    max_future_bias_difference_N=float(abs(bb['future_bias']-aa['future_bias']).max()),
                    max_observation_time_difference_s=float(abs(bb['observation_steps']*.0025/divisors[b]-aa['observation_steps']*.0025/divisors[a]).max()))
            cell=dict(relative_path=relative,episode_ids=arrays['h']['episode_id'].tolist(),normal_world=metadata['h']['normal_world'],
                friction_world=metadata['h']['friction_world'],probe=metadata['h']['probe']['name'],prefix_comparisons=prefix,
                native_comparisons=comparisons,levels=level_details)
            denom=np.asarray(comparisons['h_h2']['future']['mean_episode_rmse'])
            numerator=np.asarray(comparisons['h2_h4']['future']['mean_episode_rmse'])
            cell['future_D24_over_D12']=numerator/np.maximum(denom,1e-300)
            cells.append(cell)
            print('AUDITED PI REFINEMENT',relative,flush=True)
        assert guards==0
        for p,h in hashes.items():assert sha(p)==h,p
        result=dict(status='complete',command=sys.argv,seconds=time.perf_counter()-start,
            source_input_sha256=hashes,output_sha256=output_hashes,cells=cells,selected_batches_per_grid=24,
            episodes_per_grid=864,guard_steps=guards,components_order=['position_m','velocity_m_s','normal_N','friction_N'],
            scope='Descriptive native timestep study; one prefix seed, all 24 declared ideal-sensor cells. Both phases refined. Pure fitted privileged history only, no particle filter/decision test.')
        write_json(report.with_suffix('.json'),result)
        output_hashes[str(report.with_suffix('.json'))]=sha(report.with_suffix('.json'))
        make_report(result,report)
        for suffix in ('.md','.png'):output_hashes[str(report.with_suffix(suffix))]=sha(report.with_suffix(suffix))
        write_json(out/'artifacts.json',dict(status='complete',files_sha256=output_hashes))
        # This newly created status file is ours; original sources/results stay
        # immutable. Retain process failure information if the audit aborts.
        (out/'status.json').write_text(json.dumps(dict(status='complete',seconds=time.perf_counter()-start),indent=2)+'\n')
        print('COMPLETE PI REFINEMENT',report,flush=True)
    except BaseException as exc:
        (out/'status.json').write_text(json.dumps(dict(status='failed',error=repr(exc),completed_cells=len(cells)),indent=2)+'\n')
        raise


def make_report(result,prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cells=result['cells'];levels=('h','h2','h4')
    state_drift=np.array([c['native_comparisons']['h_h4']['future']['mean_episode_rmse'] for c in cells])
    ratios=np.array([c['future_D24_over_D12'] for c in cells])
    priv=np.array([[c['levels'][level]['privileged_common_grid']['mean_episode_rmse'] for level in levels] for c in cells])
    maxprefix=max(v['max_initial_pi_state_difference_m'] for c in cells for v in c['prefix_comparisons'].values())
    maxu=max(c['native_comparisons']['h_h4']['max_common_u_difference_N'] for c in cells)
    fig,axs=plt.subplots(1,2,figsize=(12,4.7))
    for c,p in zip(cells,priv):
        axs[0].plot([2.5,1.25,.625],p[:,0]*1e6,alpha=.6,lw=1)
    axs[0].set(xlabel='Coupled and fixture timestep (ms)',ylabel='Privileged position RMSE (µm)',yscale='log',title='24 cells; 36 episodes per cell')
    labels=[f"{c['normal_world'][:2]}/{c['friction_world']}/{c['probe']}" for c in cells]
    axs[1].scatter(np.arange(len(cells)),state_drift[:,0]*1e6,label='Native h versus h/4')
    axs[1].scatter(np.arange(len(cells)),priv[:,0,0]*1e6,label='Privileged error at h')
    axs[1].set(yscale='log',ylabel='Position RMSE (µm)',title='Future trajectory: discrete fit versus drift')
    axs[1].set_xticks(np.arange(len(cells)));axs[1].set_xticklabels(labels,rotation=90,fontsize=6)
    axs[1].legend(fontsize=8);fig.tight_layout();fig.savefig(prefix.with_suffix('.png'),dpi=180);plt.close(fig)
    lines=['# Independent PI numerical refinement — 2026-09-15','',
        'All 24 declared ideal-sensor cells (eight material compositions × three probes), each with 36 episodes, are retained at h=2.5 ms, h/2 and h/4. Both the four-second C2 material fixture and coupled phase were refined. This is a descriptive one-seed numerical study, not a population test, a particle-inference comparison or a new stopping-policy claim. No experimental source, model or original record was edited.','',
        f"Raw process/completeness metadata and {len(result['source_input_sha256'])} unique source/input hashes were checked before/after the audit. All 2,592 selected episode-runs have zero guard steps. Per-cell/episode results and artifact hashes are in `{prefix.name}.json` and `results/pi_refinement_independent_audit/artifacts.json`.",'',
        '## Preparation and time alignment','',
        f'All 36 rows in every cell retain exactly the same C2 waypoints and turning times across grids; common-grid positions and imposed normal loads agree. The largest initial PI-deflection difference is {maxprefix:.3g} m. Preparation duration remains exactly four seconds; its sample count and dt change together. Thus the existing privileged replay uses the correct preparation clock here. This does not remove its earlier limitation for a different design with fixed coarse preparation and refined coupling.','',
        'Post-step mechanical states are aligned with fine indices `(q−1)::q`; normal force and interval-start records use `::q`. Tangential force is an interval mean, so the JSON separately reports both the left-aligned fine-interval comparison and a fine-force average over each coarse interval. Observation times and exogenous compression/rate agree. Actuator commands respond to native feedback, and future bias responds to the boundary state: these are the same policy at finer integration, not an identical applied-force replay.','',
        f'Maximum common-grid actuator-command change between h and h/4: {maxu:.6g} N.','',
        '## Native trajectory drift and matched discrete prediction','',
        'The following ranges are across the 24 cell means of episode future RMSE, without pooling episodes across cells. D24/D12 compares h/2→h/4 with h→h/2.','',
        '| Quantity | Native h→h/4 range | D24/D12 range |','|---|---:|---:|']
    for j,(name,factor,unit) in enumerate((('Position',1e6,'µm'),('Velocity',1,'m/s'),('Normal force',1,'N'),('Tangential interval-start mean',1,'N'))):
        lines.append(f'| {name} | {state_drift[:,j].min()*factor:.6g}–{state_drift[:,j].max()*factor:.6g} {unit} | {ratios[:,j].min():.6g}–{ratios[:,j].max():.6g} |')
    lines+=['','Pure fitted privileged-history forecasts were rerun for all 24 cells at each matching dt, using fixed PI seed61001 clean and normal seed1 clean. These replays have no future observations and never copy active native material state. State/force comparisons use their respective clocks.','',
        '| Timestep | Mean cell privileged position RMSE | Largest cell mean |','|---|---:|---:|']
    for j,level in enumerate(levels):lines.append(f'| {level} | {priv[:,j,0].mean()*1e6:.6g} µm | {priv[:,j,0].max()*1e6:.6g} µm |')
    lines+=['','A small matched discrete forecast error does not establish equally small continuum physical error: the model and native integrator can share discretization effects. Cellwise refinement ratios and raw drift must be considered, including exceptions to monotone convergence. No posterior-information-gain or MLE effect was tested here.','',
        '## Energy accounting','',
        'Native positions and velocities are converted to float64 before differences. The frozen native PI commit uses `ds=float64(float32(dx_recorded/dt))*dt`; the position displacement is recomputed as `dx=x_next−x`. The audit uses `RE=(Fc−Fa)dx+Fnet(dx−dt*vn)−dt*Rm*vn+Fc(ds−dx)`. This last term and float64 arithmetic correct bookkeeping; they do not remove actual native energy residuals.','',
        '| Grid | Max native energy residual | Max corrected identity error | Max commit displacement mismatch |','|---|---:|---:|---:|']
    for level in levels:
        p=[c['levels'][level]['precision'] for c in cells]
        lines.append(f"| {level} | {max(x['max_float64_energy_residual_J'] for x in p):.6g} J | {max(x['max_corrected_energy_decomposition_error_J'] for x in p):.6g} J | {max(x['max_pi_commit_displacement_difference_m'] for x in p):.6g} m |")
    lines+=['','Implicit scalar residuals satisfy the original 1e−7 N tolerance; stored dissipation, load-release and slip terms are nonnegative within rounding. The JSON also retains native momentum/position residuals, applied-versus-committed force, stored-versus-recomputed energy errors and normal-geometry discretization work. This tangential energy identity is not a proof of whole-system passivity for a moving cam, varying normal load and driven compression. Ideal sensors avoid confounding the comparison with differently indexed random noise. Only one prefix seed and these three probes were studied; no universal convergence or transfer guarantee follows.','',
        '## Reproduction','',
        'From `/home/loya2/physics-transfer`:','',
        '```bash','PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/pi-refinement-independent-numba OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/loya2/miniconda3/envs/unitree_sim_env/bin/python paper_study/scripts/analyze_pi_refinement.py','```','',
        'Existing audit outputs are exclusive. For another run, specify fresh `--out` and `--report-prefix` paths. All native jobs must be complete; this script never starts a simulator.','']
    with prefix.with_suffix('.md').open('x') as stream:stream.write('\n'.join(lines))


if __name__=='__main__':main()
