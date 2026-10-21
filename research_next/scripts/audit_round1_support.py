"""Descriptive clean-calibration marginal-range/zero-velocity-hold audit.

No model fitting or confirmation decisions. Main data are read only by the
explicit evaluate/all CLI stages after the model lock is verified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/round1'
FEATURES=('delta','rate','velocity','normal')
HOLD_SPEED=.001


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def longest_hold(velocity,dt):
    mask=np.abs(np.asarray(velocity))<HOLD_SPEED
    if mask.ndim!=1:raise ValueError('One velocity trajectory required')
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    return float(np.max(ends-starts,initial=0)*dt)


def calibration_summary(data,rows,dt):
    b=np.array([i for i,r in enumerate(rows) if r['mode']=='B'])
    a=np.array([i for i,r in enumerate(rows) if r['mode']=='A'])
    if len(b)!=24 or len(a)!=24:raise ValueError('All 24 isolated B and A training profiles required')
    for idx in (a,b):
        if sorted(rows[i]['profile'] for i in idx)!=list(range(24)):
            raise ValueError('Duplicate/missing calibration profile IDs')
    fields=dict(delta=data['delta'][b],rate=data['rate'][b],velocity=data['state'][a,:,1],normal=data['normal'][a])
    if not all(np.isfinite(x).all() for x in fields.values()):raise ValueError('Nonfinite clean calibration')
    bounds={k:dict(min=float(x.min()),max=float(x.max()),samples=x.size) for k,x in fields.items()}
    holds=[dict(profile=rows[i]['profile'],id=rows[i]['id'],longest_zero_velocity_hold_s=longest_hold(data['state'][i,:,1],dt)) for i in a]
    return dict(bounds=bounds,friction_A_longest_hold_s=max(r['longest_zero_velocity_hold_s'] for r in holds),
                training_holds=holds,scope='clean TRAIN only; normal B and friction A; no validation or target tuning')


def fit_bounds(root,cfg):
    results={};hashes={}
    for world in cfg['worlds']:
        for seed in cfg['replicates']:
            path=root/f'calibration/{world}/r{seed}/train.npz'
            meta=path.with_suffix('.json')
            with np.load(path) as z:data={k:z[k] for k in ('delta','rate','state','normal')}
            rows=json.loads(meta.read_text())['rows']
            results[f'{world}/r{seed}']=calibration_summary(data,rows,cfg['dt'])
            hashes.update({str(path):sha(path),str(meta):sha(meta)})
    result=dict(calibration=results,source_sha256=hashes,hold_speed_threshold_m_s=HOLD_SPEED,
                boundary_tolerance=0.,scope='Marginal empirical bounds only; not history support or interpolation proof')
    write_json(root/'support/calibration_bounds.json',result)
    return result


def predicted_signals(prediction,test,dt):
    """Forces at t correspond to pre-step state, not stored post-step position."""
    initial=np.asarray(test['state'])[:,0:1,:]
    before=np.concatenate((initial,np.asarray(prediction['state'])[:,:-1,:]),axis=1)
    if before.shape!=prediction['state'].shape:raise ValueError('Prediction time axis mismatch')
    x,v=before[...,0],before[...,1]
    amplitude=np.asarray(test['amplitude'])[:,None]
    z=np.tanh(x/.04);slope=amplitude*(1-z*z)/.04
    return dict(delta=np.asarray(test['d'])+amplitude*z,
                rate=np.asarray(test['rate'])+slope*v,
                velocity=v,normal=np.asarray(prediction['normal']))


def true_signals(data):
    return dict(delta=np.asarray(data['delta']),rate=np.asarray(data['compression_rate']),
                velocity=np.asarray(data['state'])[...,1],normal=np.asarray(data['normal']))


def warm_signals(test):
    velocity=test['warm_velocity'] if 'warm_velocity' in test else test['warm_state'][...,1]
    return dict(delta=test['warm_delta'],rate=test['warm_rate'],velocity=velocity,normal=test['warm_normal'])


def audit_case(motion,warm,rows,normal_cal,friction_cal,identity,dt):
    bounds={k:(normal_cal if k in ('delta','rate') else friction_cal)['bounds'][k] for k in FEATURES}
    records=[]
    for segment in ('warmup','motion','full'):
        signals=(warm if segment=='warmup' else motion if segment=='motion'
                 else {k:np.concatenate((warm[k],motion[k]),axis=1) for k in FEATURES})
        shapes={np.asarray(x).shape for x in signals.values()}
        if len(shapes)!=1 or len(next(iter(shapes)))!=2 or next(iter(shapes))[0]!=len(rows):
            raise ValueError('Signal/episode axes differ')
        if not all(np.isfinite(x).all() for x in signals.values()):raise ValueError('Nonfinite support input; retain and investigate original guard artifacts')
        for i,row in enumerate(rows):
            rec=dict(**identity,episode=i,segment=segment,samples=signals['delta'].shape[1],
                     longest_zero_velocity_hold_s=longest_hold(signals['velocity'][i],dt),
                     friction_training_max_hold_s=friction_cal['friction_A_longest_hold_s'])
            rec['hold_exceeds_training_max']=rec['longest_zero_velocity_hold_s']>rec['friction_training_max_hold_s']
            any_out=np.zeros(signals['delta'].shape[1],bool)
            for key in FEATURES:
                value=signals[key][i];low=bounds[key]['min'];high=bounds[key]['max']
                below=value<low;above=value>high;outside=below|above;any_out|=outside
                rec.update({f'{key}_min':float(value.min()),f'{key}_max':float(value.max()),
                            f'{key}_training_min':low,f'{key}_training_max':high,
                            f'{key}_below_fraction':float(below.mean()),f'{key}_above_fraction':float(above.mean()),
                            f'{key}_outside_fraction':float(outside.mean())})
            rec['any_feature_outside_fraction']=float(any_out.mean())
            records.append(rec)
    return records


def load_artifact(path,hashes):
    meta_path=path.with_suffix('.json');meta=json.loads(meta_path.read_text())
    digest=sha(path)
    if meta.get('npz_sha256',meta.get('data_sha256'))!=digest:raise ValueError(f'Artifact hash mismatch {path}')
    hashes[str(path)]=digest;hashes[str(meta_path)]=sha(meta_path)
    with np.load(path) as z:data={k:z[k] for k in z.files}
    return data,meta


def evaluate(root,cfg,bounds):
    hashes={};records=[]
    for nw,fw in cfg['compositions']:
        for seed in cfg['replicates']:
            path=root/f'test/{nw}_{fw}/r{seed}.npz'
            test,meta=load_artifact(path,hashes);rows=meta['rows']
            if len(rows)!=16:raise ValueError('All 16 fixed episodes required')
            nc=bounds['calibration'][f'{nw}/r{seed}'];fc=bounds['calibration'][f'{fw}/r{seed}']
            warm=warm_signals(test)
            common=dict(seed=seed,normal_world=nw,friction_world=fw)
            records.extend(audit_case(true_signals(test),warm,rows,nc,fc,
                                     dict(**common,stage='reference_open',noise='shared',method='reference'),cfg['dt']))
            for noise in cfg['noise']:
                for method in cfg['methods']:
                    for folder,stage in [('predictions','predicted_open'),('controllers','controller_truth')]:
                        data,md=load_artifact(root/f'{folder}/{nw}_{fw}/r{seed}/{noise}/{method}.npz',hashes)
                        if md['rows']!=rows:raise ValueError('Reference and rollout row order differ')
                        signals=predicted_signals(data,test,cfg['dt']) if folder=='predictions' else true_signals(data)
                        records.extend(audit_case(signals,warm,rows,nc,fc,
                            dict(**common,stage=stage,noise=noise,method=method),cfg['dt']))
    frame=pd.DataFrame(records)
    expected=(8*4*16+2*8*4*2*7*16)*3
    if len(frame)!=expected or frame.duplicated(['seed','normal_world','friction_world','stage','noise','method','episode','segment']).any():
        raise ValueError('Incomplete or duplicated descriptive support design')
    dest=root/'support';dest.mkdir(parents=True,exist_ok=True)
    frame.to_csv(dest/'support_metrics.csv',index=False)
    group=['stage','noise','method','segment']
    summary=frame.groupby(group).agg(episodes=('episode','size'),
        any_feature_outside_fraction=('any_feature_outside_fraction','mean'),
        longest_zero_velocity_hold_s=('longest_zero_velocity_hold_s','max'),
        hold_exceeds_training_fraction=('hold_exceeds_training_max','mean'),
        **{key+'_outside_fraction':(key+'_outside_fraction','mean') for key in FEATURES}).reset_index()
    summary.to_csv(dest/'support_summary.csv',index=False)
    write_json(dest/'provenance.json',dict(source_sha256=hashes,calibration_source_sha256=bounds['source_sha256'],
        audit_source_sha256=sha(Path(__file__)),scope='descriptive clean-train marginal bounds and measured zero-speed hold durations; not a decision gate'))
    report(summary,frame,bounds)
    return dict(episode_segment_rows=len(frame),unique_rollout_episodes=len(frame)//3,summary_rows=len(summary))


def report(summary,frame,bounds):
    text="""# Runde 1: marginale Datenabdeckung und Haltezeiten

Dieser Audit ist **deskriptiv**. Einzelne Merkmalsminima/-maxima beweisen weder
Unterstützung gemeinsamer Merkmalskombinationen noch Unterstützung ganzer
Historien, Interpolation oder Validität einer kausalen Zustandsdarstellung.
Keine Episode wird anhand dieses Audits entfernt und keine primäre
Entscheidungsregel wird geändert.

Die Grenzen stammen ausschließlich aus den sauberen lokalen **Trainingsdaten**
des jeweiligen neuen Replikats: δ/δ̇ aus B der Normalwelt, v/N aus A der
Reibungswelt. Validierungsdaten erweitern diese Grenzen nicht. Verrauschte
Verfahren werden gegen dieselben sauberen physikalischen Trainingsbereiche
diagnostiziert. Grenzgleichheit gilt als innerhalb; es gibt keine angepasste
Toleranz. Die Rohdateien enthalten für jedes Merkmal und jede Episode Minimum,
Maximum, Trainingsgrenzen sowie Anteile unter-/oberhalb und außerhalb.

Berichtet werden alle wahren offenen Referenzfahrten, alle echten Controller-
Trajektorien und alle vorhergesagten offenen Trajektorien. Für letztere wird
der Zustand **vor** Kraftschritt t aus dem initialen Testzustand und den
gespeicherten Nachfolgezuständen bis t−1 rekonstruiert: δ=d+g(x̂),
δ̇=ḋ+g′(x̂)v̂. N ist der vom Prädiktor gespeicherte eigene Kraftwert, keine
künftige wahre Kraft. Der beobachtete viersekündige Präfix ist dagegen für alle
Verfahren identisch und benutzt die tatsächlich gemessenen Präfixsignale.

Haltezeit ist die längste zusammenhängende Folge |v|<0,001 m/s, mit dt gewichtet.
Verglichen wird mit dem längsten solchen Intervall aller 24 lokalen A-
Trainingsprofile der zugehörigen Reibungswelt. Die kombinierte Folge enthält
den gesamten viersekündigen Warmup und die anschließenden 1,2 s Bewegung;
ein Intervall über diese Grenze wird nicht künstlich geteilt. Geschwindigkeit
null bedeutet nicht, dass interne Alterung/Hysterese stillsteht. Längere
Haltezeit als im Training ist daher eine wichtige Verlaufsabweichung, selbst
wenn alle momentanen Werte innerhalb ihrer marginalen Grenzen liegen.

## Vollständige deskriptive Übersicht

Anteile sind gleichgewichtet über die vollständig vorhandenen Episoden.
Warmup, Bewegung und vollständige Folge stehen getrennt in den Volltabellen;
hier wird die vollständige Folge gezeigt. „shared“ ist die einmalig erzeugte
Referenzfahrt, die beiden Kalibrierungsrauschbedingungen gemeinsam ist.

| Trajektorie | Rauschen | Methode | δ außerhalb | δ̇ außerhalb | v außerhalb | N außerhalb | Irgendein Merkmal außerhalb | Max. Ruhe s | Episoden mit längerer Ruhe als Training |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for _,r in summary[summary.segment=='full'].iterrows():
        text+=f"| {r.stage} | {r.noise} | {r.method} | "+' | '.join(f'{r[key+"_outside_fraction"]*100:.3f}%' for key in FEATURES)+f" | {r.any_feature_outside_fraction*100:.3f}% | {r.longest_zero_velocity_hold_s:.4f} | {r.hold_exceeds_training_fraction*100:.2f}% |\n"
    holds=[v['friction_A_longest_hold_s'] for v in bounds['calibration'].values()]
    text+=f"\nTrainingsmaxima der |v|<0,001-Haltezeit über die 16 Welt-/Seed-Kalibrierungen: {min(holds):.4f}–{max(holds):.4f} s. Ausgewertet wurden {len(frame)//3} Trajektorien-Episoden mit jeweils drei Segmenten.\n\n"
    text+=f"Rohdaten: [support_metrics.csv]({OUT/'support/support_metrics.csv'}), [support_summary.csv]({OUT/'support/support_summary.csv'}), [calibration_bounds.json]({OUT/'support/calibration_bounds.json'}), [Provenienz]({OUT/'support/provenance.json'}). Reproduktion: `python research_next/scripts/audit_round1_support.py all` nach vollständiger Datenerzeugung.\n"
    (ROOT/'reports/round1_support.md').write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['fit','evaluate','all']);args=parser.parse_args()
    sys.path.insert(0,str(ROOT/'src'))
    from round1_lock import verify_code_lock,verify_model_lock,verify_test_lock
    verify_code_lock()
    test_lock=verify_test_lock() if args.stage in ('evaluate','all') else None
    cfg=json.loads((ROOT/'configs/round1.json').read_text())
    if args.stage in ('fit','all'):bounds=fit_bounds(OUT,cfg)
    else:bounds=json.loads((OUT/'support/calibration_bounds.json').read_text())
    for path,expected in bounds['source_sha256'].items():
        if sha(path)!=expected:raise ValueError('Clean training source changed')
    if args.stage in ('evaluate','all'):
        lock=verify_model_lock();print(json.dumps(evaluate(OUT,cfg,bounds),indent=2))
        assert verify_model_lock()==lock and verify_test_lock()==test_lock
    print('ROUND1_SUPPORT_COMPLETE',args.stage,flush=True)


if __name__=='__main__':main()
