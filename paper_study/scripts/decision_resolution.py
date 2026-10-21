"""Prospectively bound numerical diagnostic; imports frozen decision equations unchanged."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse
import copy
import json
import resource
import time
import numpy as np
import pandas as pd
from paper_study.decision_study.common import BRANCHES, sha, write_json, verify_hashes
from paper_study.decision_study.inference import (load_model, load_prior, prior_path,
    observable_prefix, filter_prefix, forecast_branch, branch_values, select_policies, actual_forecast)
from paper_study.decision_study.evaluate import PUBLIC_RAW, branch_scores, episode_rows, verify_lock

VARIANTS = [('P201K64',201,64,'prior_development'),('P201K256',201,256,'prior_development'),
            ('P201K1024',201,1024,'prior_development'),('P401K1024',401,1024,'prior_grid401')]
METHODS = ['fixed_release','fixed_hold100','fixed_continue100','uniform','uniform_expected',
           'bayes_risk','information_gain','physical_only']
DETERMINISTIC = [x for x in METHODS if x not in ('uniform','uniform_expected')]
IDENTITY = ['preparation_seed','normal_world','friction_world','episode']
CRITERIA = dict(minimum_choice_agreement=.95, maximum_pooled_mse_relative_change=.05,
                risk_relative_denominator_floor_m2=1e-14, wall_seconds=600,
                baseline='P201K64', final='P401K1024', grid_control='P201K1024',
                contexts=288, deterministic_methods=DETERMINISTIC,
                mse_zero_rule='both zero:0; baseline zero and new positive:undefined/fail',
                inference='Descriptive resolution check, no population confidence intervals')
VALUE_KEYS = ('risk_m2','information_nats','physical_risk_m2','planned_extra_probe_work_J',
              'extra_seconds','fantasy_risk_mcse_m2','fantasy_particle_indices','fantasy_noise')
PROTOCOL = ROOT/'paper_study/reports/decision_resolution_protocol.md'


def prepare_candidates(source_directory,nw,fw,cfg):
    """Public causal helper; supports explicit dt/observation_steps and arbitrary source path."""
    with np.load(Path(source_directory)/'release.npz',allow_pickle=False) as raw:
        public = observable_prefix({k:raw[k] for k in PUBLIC_RAW},cfg)
    model = load_model(nw,fw,cfg)
    grid,prior = load_prior(nw,fw,cfg)
    snapshot = filter_prefix(model,grid,prior,public,cfg)
    branches = {b:forecast_branch(model,snapshot,public,b,cfg) for b in BRANCHES}
    return public,snapshot,branches


def score_candidates(source_directory,snapshot,branches,cfg):
    """Call only after choices are persisted. Observations and targets stay separate."""
    source = Path(source_directory)
    scores, predictions = {},{}
    metadata = json.loads((source/'release.json').read_text())
    for b in BRANCHES:
        with np.load(source/(b+'.npz'),allow_pickle=False) as raw:
            sensor_input = {k:raw[k] for k in ('observations','observation_steps')}
            prediction,weights = actual_forecast(snapshot,branches[b],sensor_input,cfg)
            targets = {k:raw[k] for k in ('release_step','next_state','normal','friction','absolute_work_J','guard')}
        scores[b] = branch_scores(prediction,targets,cfg)
        predictions[b+'_forecast'] = prediction
        predictions[b+'_weights'] = weights
    return scores,predictions,metadata


def frozen_config(base,particles,fantasies,bank):
    cfg = copy.deepcopy(base)
    cfg.update(particles=particles,fantasies=fantasies,prior_bank_directory='paper_study/results/'+bank)
    return cfg


def freeze(native,evaluation,out):
    native,evaluation,out = [Path(x).resolve() for x in (native,evaluation,out)]
    if out.exists() or not out.is_relative_to(ROOT/'paper_study/results'):
        raise ValueError('Require fresh output below paper_study/results')
    old = verify_lock(evaluation)
    if Path(old['data_directory']) != native: raise ValueError('Original data binding mismatch')
    if json.loads((evaluation/'process.json').read_text())['status']!='complete':
        raise ValueError('Original evaluation incomplete')
    manifest = json.loads((evaluation/'artifact_manifest.json').read_text())
    verify_hashes(manifest['outputs_sha256'])
    if manifest['evaluation_lock_sha256']!=sha(evaluation/'evaluation_lock.json'):
        raise ValueError('Original manifest lock mismatch')
    triplets = old['triplets']
    if len(triplets)!=8 or {x['seed'] for x in triplets}!={34001,34002}:
        raise ValueError('Need all eight declared development batches')
    bindings = {}
    for k,v in old.items():
        if k.endswith('_sha256') and isinstance(v,dict): bindings.update(v)
    bindings.update(manifest['outputs_sha256'])
    bindings[str(evaluation/'artifact_manifest.json')] = sha(evaluation/'artifact_manifest.json')
    variants = {}
    for label,p,k,bank in VARIANTS:
        cfg = frozen_config(old['config'],p,k,bank);variants[label]=cfg
        priorlock = ROOT/cfg['prior_bank_directory']/'prior_lock.json'
        prior_info = json.loads(priorlock.read_text());verify_hashes(prior_info['sources_sha256'])
        bindings.update(prior_info['sources_sha256'])
        for f in (priorlock,priorlock.parent/'process.json'):
            bindings[str(f)] = sha(f)
        if json.loads((priorlock.parent/'process.json').read_text())['status']!='complete':
            raise ValueError('Prior incomplete')
        for nw,fw in cfg['compositions']:
            load_prior(nw,fw,cfg)
            f=prior_path(nw,fw,cfg);bindings[str(f)]=sha(f)
    verify_hashes(bindings)
    out.mkdir(parents=True)
    memory = next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    lock = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        native_directory=str(native),original_evaluation_directory=str(evaluation),triplets=triplets,
        variants=variants,criteria=CRITERIA,input_bindings_sha256=bindings,
        source_sha256={str(Path(__file__).resolve()):sha(Path(__file__).resolve()),str(PROTOCOL):sha(PROTOCOL)},
        memory_available_before_run=memory,
        role='Development numerical sensitivity, declared before additional scores; no native run')
    write_json(out/'resolution_lock.json',lock)
    write_json(out/'process.json',dict(status='frozen'))
    return out


def verify_resolution(out):
    lock=json.loads((Path(out)/'resolution_lock.json').read_text())
    verify_hashes(lock['source_sha256']);verify_hashes(lock['input_bindings_sha256'])
    if lock['criteria']!=CRITERIA: raise ValueError('Criteria mismatch')
    return lock


def dump_plan(dest,values,decisions,snapshot,cfg,seconds):
    dest.mkdir(parents=True)
    payload = dict(**values,decision_weights=snapshot['weights'])
    for budget,result in decisions.items():
        payload[budget+'_eligible']=result['eligible']
        for method,probability in result['policies'].items():payload[budget+'__'+method]=probability
    with (dest/'planned.npz').open('xb') as f: np.savez_compressed(f,**payload)
    write_json(dest/'planned.json',dict(status='planned_before_any_target_read',seconds=seconds,
        particles=cfg['particles'],fantasies=cfg['fantasies'],npz_sha256=sha(dest/'planned.npz')))


def read_plan(dest,cfg):
    meta=json.loads((dest/'planned.json').read_text())
    if meta['npz_sha256']!=sha(dest/'planned.npz'):raise ValueError('Changed saved choices')
    with np.load(dest/'planned.npz',allow_pickle=False) as z:
        values={k:z[k].copy() for k in VALUE_KEYS}
        decisions={b['name']:dict(eligible=z[b['name']+'_eligible'].copy(),
            policies={m:z[b['name']+'__'+m].copy() for m in METHODS}) for b in cfg['budgets']}
    return values,decisions


def paired_comparison(frame,old,new):
    key=IDENTITY+['budget','method']
    a=frame[frame.variant==old].set_index(key).sort_index()
    b=frame[frame.variant==new].set_index(key).sort_index()
    if not a.index.equals(b.index) or not a.index.is_unique:raise ValueError('Nonpaired resolution table')
    rows=[]
    for (budget,method),v in a.groupby(level=['budget','method'],sort=True):
        w=b.loc[v.index];oldm=float(v.position_mse_m2.mean());newm=float(w.position_mse_m2.mean())
        change=abs(newm-oldm)/oldm if oldm>0 else (0. if newm==0 else None)
        agreement=float((v.chosen_branch==w.chosen_branch).mean())
        rows.append(dict(old=old,new=new,budget=budget,method=method,contexts=len(v),
            old_mse_m2=oldm,new_mse_m2=newm,relative_mse_change=change,
            choice_agreement=agreement,agreement_applicable=method in DETERMINISTIC,
            agreement_pass=agreement>=CRITERIA['minimum_choice_agreement'] if method in DETERMINISTIC else True,
            mse_pass=change is not None and change<=CRITERIA['maximum_pooled_mse_relative_change']))
    return rows


def baseline_check(frame,branchframe,original):
    results={}
    for name,current,file,extra in [('policy',frame,'policy_episodes.csv',['budget','method']),
                                     ('branch',branchframe,'branch_episodes.csv',['branch'])]:
        old=pd.read_csv(original/file);new=current[current.variant=='P201K64'].drop(columns='variant')
        key=IDENTITY+extra;old=old.set_index(key).sort_index();new=new.set_index(key).sort_index()
        if not old.index.equals(new.index) or set(old)!=set(new):raise ValueError('Original reproduction inventory differs')
        maximum=0.
        for column in old:
            if pd.api.types.is_numeric_dtype(old[column]):
                maximum=max(maximum,float(np.max(np.abs(old[column].to_numpy()-new[column].to_numpy()))))
                if not np.allclose(old[column],new[column],rtol=2e-12,atol=2e-15):
                    raise ValueError('Original numerical reproduction failed: '+name+column)
            elif not np.array_equal(old[column],new[column]):raise ValueError('Original categorical reproduction failed')
        results[name]=dict(rows=len(old),max_absolute_numeric_difference=maximum,passed=True)
    return results


def analyze(frame,branchframe,out,lock):
    if len(frame)!=36864 or len(branchframe)!=3456:raise ValueError('Incomplete row inventory')
    for f,extra in ((frame,['budget','method']),(branchframe,['branch'])):
        if f.duplicated(['variant']+IDENTITY+extra).any():raise ValueError('Duplicate rows')
        if not np.isfinite(f.select_dtypes(include=np.number)).all().all():raise ValueError('Nonfinite scored output')
    comparisons=[]
    pairs=[('P201K64','P401K1024'),('P201K1024','P401K1024'),
           ('P201K64','P201K256'),('P201K256','P201K1024'),('P201K64','P201K1024')]
    for a,b in pairs:comparisons.extend(paired_comparison(frame,a,b))
    pd.DataFrame(comparisons).to_csv(out/'comparisons.csv',index=False)
    risks=[]
    for a,b in pairs:
        index=IDENTITY+['branch']
        x=branchframe[branchframe.variant==a].set_index(index).sort_index()
        y=branchframe[branchframe.variant==b].set_index(index).sort_index()
        diff=(y.planned_risk_m2-x.planned_risk_m2).to_numpy()
        rel=abs(diff)/np.maximum(abs(x.planned_risk_m2.to_numpy()),1e-14)
        risks.append(dict(old=a,new=b,mean_absolute_risk_change_m2=float(abs(diff).mean()),
            max_absolute_risk_change_m2=float(abs(diff).max()),mean_relative_risk_change=float(rel.mean()),
            p95_relative_risk_change=float(np.quantile(rel,.95)),max_relative_risk_change=float(rel.max()),
            diagnostic_only=True))
    pd.DataFrame(risks).to_csv(out/'risk_diagnostics.csv',index=False)
    pooled=frame.groupby(['variant','budget','method'],sort=True)[['position_mse_m2','absolute_work_J','probe_seconds']].mean().reset_index()
    pooled['pooled_position_rmse_um']=1e6*np.sqrt(pooled.position_mse_m2)
    pooled['rank']=pooled.groupby(['variant','budget']).position_mse_m2.rank(method='min')
    pooled.to_csv(out/'pooled_summary.csv',index=False)
    comp=pd.DataFrame(comparisons);checks={}
    for a,b in pairs:
        sub=comp[(comp.old==a)&(comp.new==b)]
        checks[a+'__'+b]=dict(passed=bool(sub.agreement_pass.all() and sub.mse_pass.all()),
            minimum_deterministic_agreement=float(sub[sub.agreement_applicable].choice_agreement.min()),
            maximum_relative_mse_change=float(sub.relative_mse_change.max()),
            failed_rows=sub[~(sub.agreement_pass & sub.mse_pass)].to_dict('records'))
    baseline=baseline_check(frame,branchframe,Path(lock['original_evaluation_directory']))
    result=dict(criteria=CRITERIA,comparisons=checks,baseline_reproduction=baseline,
        contexts=288,policy_rows=len(frame),branch_rows=len(branchframe),
        native_guard_count_max=float(frame.native_guard_count.max()),
        planning_candidate_guard_count_max=float(frame.planning_candidate_guard_count.max()),
        no_population_inference=True,original_result_not_replaced=True)
    write_json(out/'decision.json',result)
    b3=pooled[pooled.budget=='B3'].pivot(index='method',columns='variant',values='pooled_position_rmse_um').reindex(METHODS)
    gate=checks['P201K64__P401K1024'];grid=checks['P201K1024__P401K1024']
    lines=['# Numerische Auflösung der endlichen Entscheidungsregel', '',
        f"Der vorab festgelegte Gesamtvergleich P201/K64 → P401/K1024 ist **{'bestanden' if gate['passed'] else 'nicht bestanden'}**. "
        f"Kleinste deterministische Wahlübereinstimmung: {100*gate['minimum_deterministic_agreement']:.2f}%; "
        f"größte Änderung eines gepoolten Methoden-/Budget-MSE: {100*gate['maximum_relative_mse_change']:.2f}%.", '',
        f"Die gesonderte Gitterkontrolle bei K1024 ist **{'bestanden' if grid['passed'] else 'nicht bestanden'}** "
        f"(Wahlminimum {100*grid['minimum_deterministic_agreement']:.2f}%, maximale MSE-Änderung {100*grid['maximum_relative_mse_change']:.2f}%).", '',
        'Alle 288 Developmentkontexte (zwei Historienblöcke, eine eingefrorene verrauscht kalibrierte Modellbank), vier Kompositionen und drei Zweige bleiben enthalten. '
        'Die 36.864 Policyzeilen enthalten vier numerische Varianten derselben Kontexte, keine unabhängigen Wiederholungen. Keine p-Werte oder populationsweiten Konfidenzintervalle.', '',
        '## Gepoolter Positions-RMSE in µm, Budget B3', '', b3.to_markdown(floatfmt='.3f'), '',
        'Dies ist Zukunfts-Prognosefehler, kein Tracking- oder Regelungserfolg. Kleineres modelliertes Bayesrisiko garantiert bei Modellfehlern keinen kleineren tatsächlichen Verlust. '
        'Alle Varianten und Rangwechsel bleiben in pooled_summary.csv; der ursprüngliche Developmentbefund wird durch diese Diagnose nicht ersetzt.', '',
        '## Paarvergleiche', '',
        '| Vergleich | Wahlminimum | maximale MSE-Änderung | Kriterien |',
        '|---|---:|---:|---|']
    for name,c in checks.items():lines.append(f"| {name} | {100*c['minimum_deterministic_agreement']:.2f}% | {100*c['maximum_relative_mse_change']:.2f}% | {'Pass' if c['passed'] else 'Fail'} |")
    lines+=['', 'Schwellen: mindestens 95% Übereinstimmung für jede deterministische Regel/Budget und höchstens 5% relative gepoolte MSE-Änderung für jede der acht Regeln/Budget. '
        'Einzelne Fehler bleiben explizit in decision.json/comparisons.csv. Die zusätzliche relative Risikodiagnostik verwendet den vorher erklärten Nennerboden 1e-14 m² und ist kein Gate.', '',
        '## Provenienz und Grenzen', '',
        'Alle Planungswerte und Wahlwahrscheinlichkeiten sämtlicher Varianten wurden vor dem ersten späteren realen Probesensor oder Zukunftsziel persistiert (planning_complete.json). '
        'P201/K64 reproduziert die originale Policy- und Branch-Tabelle innerhalb der protokollierten Rundungstoleranz. '
        'Materialgewichte, native Dateien und vorhandene Entscheidungsfunktionen bleiben unverändert. '
        'Spätere Sensorassimilation und Zielauswertung wurden je P einmal berechnet, für die drei K-Werte desselben P wiederverwendet.', '',
        'Gleicher Fantasie-Seed bedeutet bei verschiedenen K keine verschachtelten Ziehungen pro Episode: der ursprüngliche Generator zieht ganze (E,K)-Arrays. '
        'K-Vergleiche sind daher Monte-Carlo-Empfindlichkeitsvergleiche. Bei festem K1024 stimmen Uniform- und Gaußziehungen der P-Vergleiche überein; die diskrete Priorverteilung ändert sich.', '',
        f"Maximaler nativer Guardzählwert je Episode: {result['native_guard_count_max']:.0f}; maximaler gewählter Kandidaten-Guardzählwert: {result['planning_candidate_guard_count_max']:.0f}. "
        'Risiken und erwartete Arbeitsbudgets sind modellbedingt, keine kalibrierte Fehler- oder Kostenobergrenze. '
        'Diese CPU-Prüfung ersetzt keine native Zeitschrittverfeinerung.', '',
        'Reproduktion und Kriterien: [Protokoll](../../reports/decision_resolution_protocol.md). '
        'Artefakte: [Sperrliste](resolution_lock.json), [Entscheidung](decision.json), [Einzelvergleiche](comparisons.csv), '
        '[Risikodiagnostik](risk_diagnostics.csv), [alle Policyzeilen](policy_episodes.csv), [alle Zweige](branch_episodes.csv).', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    return result


def run(out):
    out=Path(out).resolve();lock=verify_resolution(out)
    if json.loads((out/'process.json').read_text())['status']!='frozen':raise ValueError('Fresh frozen output only')
    started=time.perf_counter();process=dict(status='running',pid=os.getpid(),command=sys.argv)
    (out/'process.json').write_text(json.dumps(process,indent=2)+'\n')
    data=Path(lock['native_directory']);timings=[]
    def budget():
        if time.perf_counter()-started>CRITERIA['wall_seconds']:raise TimeoutError('Declared CPU budget exceeded')
    try:
        # Phase 1: no target or actual post-decision observation access anywhere in this phase.
        for item in lock['triplets']:
            seed,nw,fw=item['seed'],item['normal'],item['friction'];source=data/f'p{seed}/{nw}_{fw}'
            for particles in (201,401):
                labels=[label for label,p,_,_ in VARIANTS if p==particles]
                cfg=lock['variants'][labels[0]];budget();t=time.perf_counter()
                public,snapshot,branches=prepare_candidates(source,nw,fw,cfg)
                construction=time.perf_counter()-t
                for label in labels:
                    cfg=lock['variants'][label];budget();t=time.perf_counter()
                    values=branch_values(snapshot,branches,cfg);decisions=select_policies(values,cfg)
                    sec=time.perf_counter()-t
                    dump_plan(out/label/f'p{seed}/{nw}_{fw}',values,decisions,snapshot,cfg,sec)
                    timings.append(dict(phase='plan',variant=label,seed=seed,normal=nw,friction=fw,seconds=sec,candidate_seconds=construction))
                    print(f'PLANNED {label} p{seed} {nw}/{fw} {sec:.3f}s',flush=True)
                    del values,decisions
                del public,snapshot,branches
        plans=sorted(out.glob('P*/p*/*/planned.npz'))
        if len(plans)!=32:raise ValueError('All variants must be planned before scoring')
        write_json(out/'planning_complete.json',dict(plans=len(plans),seconds=time.perf_counter()-started,
            targets_read=False,plans_sha256={str(p):sha(p) for p in plans}))
        # Phase 2: all saved choices are immutable before actual observations/targets are read.
        allrows,allbranches=[],[]
        for item in lock['triplets']:
            seed,nw,fw=item['seed'],item['normal'],item['friction'];source=data/f'p{seed}/{nw}_{fw}'
            for particles in (201,401):
                labels=[label for label,p,_,_ in VARIANTS if p==particles]
                cfg=lock['variants'][labels[0]];budget();t=time.perf_counter()
                public,snapshot,branches=prepare_candidates(source,nw,fw,cfg)
                scores,predictions,metadata=score_candidates(source,snapshot,branches,cfg)
                dest=out/f'actual_P{particles}/p{seed}/{nw}_{fw}';dest.mkdir(parents=True)
                with (dest/'forecasts.npz').open('xb') as f:np.savez_compressed(f,**predictions)
                for label in labels:
                    cfg=lock['variants'][label];values,decisions=read_plan(out/label/f'p{seed}/{nw}_{fw}',cfg)
                    rows,brows=episode_rows(metadata,scores,decisions,values,branches,cfg)
                    allrows.extend(dict(variant=label,**x) for x in rows)
                    allbranches.extend(dict(variant=label,**x) for x in brows)
                timings.append(dict(phase='score',variant=f'P{particles}',seed=seed,normal=nw,friction=fw,
                    seconds=time.perf_counter()-t,candidate_seconds=None))
                print(f'SCORED P{particles} p{seed} {nw}/{fw}',flush=True)
                del public,snapshot,branches,predictions
        frame,bframe=pd.DataFrame(allrows),pd.DataFrame(allbranches)
        frame.to_csv(out/'policy_episodes.csv',index=False);bframe.to_csv(out/'branch_episodes.csv',index=False)
        pd.DataFrame(timings).to_csv(out/'timings.csv',index=False)
        decision=analyze(frame,bframe,out,lock)
        verify_resolution(out)
        verify_hashes(json.loads((out/'planning_complete.json').read_text())['plans_sha256'])
        process.update(status='complete',policy_rows=len(frame),branch_rows=len(bframe),
            primary_resolution_pass=decision['comparisons']['P201K64__P401K1024']['passed'])
    except BaseException as exc:
        process.update(status='failed',error=repr(exc));raise
    finally:
        process.update(seconds=time.perf_counter()-started,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        (out/'process.json').write_text(json.dumps(process,indent=2)+'\n')
        files=[p for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json']
        write_json(out/'artifact_manifest.json',dict(resolution_lock_sha256=sha(out/'resolution_lock.json'),
            outputs_sha256={str(p):sha(p) for p in files}))


def selftest():
    # Threshold equality, no favorable zero-denominator substitution, keys strictly paired.
    rows=[]
    for variant,error in [('a',1.),('b',1.05)]:
        for e in range(100):
            rows.append(dict(variant=variant,preparation_seed=1,normal_world='n',friction_world='f',
                episode=e,budget='B',method='bayes_risk',position_mse_m2=error,
                chosen_branch='release' if variant=='a' or e<95 else 'hold100'))
    comparison=paired_comparison(pd.DataFrame(rows),'a','b')[0]
    assert comparison['choice_agreement']==.95 and comparison['agreement_pass']
    # Binary floating equality may put 1.05 fraction slightly above .05: check unambiguous sides.
    z=pd.DataFrame(rows);z.loc[z.variant=='b','position_mse_m2']=1.049
    assert paired_comparison(z,'a','b')[0]['mse_pass']
    z.loc[z.variant=='a','position_mse_m2']=0
    assert paired_comparison(z,'a','b')[0]['relative_mse_change'] is None
    assert not paired_comparison(z,'a','b')[0]['mse_pass']
    z['position_mse_m2']=0
    assert paired_comparison(z,'a','b')[0]['mse_pass']
    try:paired_comparison(z.iloc[:-1],'a','b')
    except ValueError:pass
    else:raise AssertionError('Missing pairs accepted')
    print('SELFTEST PASSED')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','run','selftest'])
    p.add_argument('--native',type=Path);p.add_argument('--evaluation',type=Path);p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.stage=='selftest':selftest();return
    if args.out is None:p.error('--out required')
    if args.stage=='freeze':
        if args.native is None or args.evaluation is None:p.error('--native and --evaluation required')
        print(freeze(args.native,args.evaluation,args.out))
    else:run(args.out)


if __name__=='__main__':main()
