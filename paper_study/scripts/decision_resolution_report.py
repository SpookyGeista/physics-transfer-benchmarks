"""Presentation-only completion after optional tabulate was absent; never refit/replan."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
from paper_study.scripts.decision_resolution import verify_resolution,IDENTITY,METHODS,DETERMINISTIC


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mdtable(frame,decimals=3):
    rows=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join('---' for _ in frame.columns)+' |']
    for values in frame.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{v:.{decimals}f}' if isinstance(v,(float,np.floating)) else str(v) for v in values)+' |')
    return '\n'.join(rows)


def main(source,out):
    source,out=Path(source).resolve(),Path(out).resolve()
    if out.exists():raise ValueError('Fresh presentation output required')
    lock=verify_resolution(source)
    manifest=json.loads((source/'artifact_manifest.json').read_text())
    checks=[];log_extension=None
    for path,expected in manifest['outputs_sha256'].items():
        actual=digest(path)
        if actual!=expected:
            if Path(path)!=source/'run.log':raise ValueError('Changed scientific output '+path)
            data=Path(path).read_bytes();lines=data.splitlines(keepends=True);h=hashlib.sha256();prefix=None
            for i,line in enumerate(lines):
                h.update(line)
                if h.hexdigest()==expected:prefix=b''.join(lines[:i+1]);break
            if prefix is None or b'ImportError' not in data[len(prefix):] or b'tabulate' not in data[len(prefix):]:
                raise ValueError('Unexpected log modification')
            log_extension=dict(path=path,locked_prefix_sha256=expected,current_sha256=actual,
                locked_prefix_bytes=len(prefix),appended_bytes=len(data)-len(prefix),
                reason='Python appended its uncaught optional-tabulate traceback after finally wrote the manifest')
        checks.append(dict(path=path,expected=expected,actual=actual,exact_match=actual==expected))
    if manifest['resolution_lock_sha256']!=digest(source/'resolution_lock.json'):raise ValueError('Lock binding mismatch')
    planning=json.loads((source/'planning_complete.json').read_text())
    if planning['plans']!=32 or planning['targets_read'] is not False:raise ValueError('Incomplete planning barrier')
    for p,h in planning['plans_sha256'].items():
        if digest(p)!=h:raise ValueError('Changed saved policy')
    decision=json.loads((source/'decision.json').read_text());process=json.loads((source/'process.json').read_text())
    if process['status']!='failed' or 'tabulate' not in process['error']:raise ValueError('Unexpected failure condition')
    policies=pd.read_csv(source/'policy_episodes.csv');branches=pd.read_csv(source/'branch_episodes.csv')
    if len(policies)!=36864 or len(branches)!=3456:raise ValueError('Incomplete numerical inventory')
    comparisons=pd.read_csv(source/'comparisons.csv')
    # Independent direct grouped recomputation of the declared gates; do not invoke analyze/paired_comparison.
    direct=[]
    for pair,c in decision['comparisons'].items():
        old,new=pair.split('__');a=policies[policies.variant==old].set_index(IDENTITY+['budget','method']).sort_index()
        b=policies[policies.variant==new].set_index(IDENTITY+['budget','method']).sort_index()
        assert a.index.equals(b.index) and a.index.is_unique
        agreements=[];changes=[]
        for budget in ['B0','B1','B2','B3']:
            for method in METHODS:
                x=a.xs((budget,method),level=['budget','method']);y=b.xs((budget,method),level=['budget','method'])
                assert len(x)==288 and x.index.equals(y.index)
                original=float(np.sum(x.position_mse_m2.to_numpy())/288)
                refined=float(np.sum(y.position_mse_m2.to_numpy())/288)
                change=abs(refined-original)/original if original>0 else (0. if refined==0 else float('inf'))
                agreement=float(np.count_nonzero(x.chosen_branch.to_numpy()==y.chosen_branch.to_numpy())/288)
                if method in DETERMINISTIC:agreements.append(agreement)
                changes.append(change)
        direct.append(dict(comparison=pair,minimum_agreement=min(agreements),maximum_relative_mse_change=max(changes),
            passed=min(agreements)>=.95 and max(changes)<=.05))
        assert direct[-1]['passed']==c['passed']
        assert abs(min(agreements)-c['minimum_deterministic_agreement'])<1e-14
        assert abs(max(changes)-c['maximum_relative_mse_change'])<2e-8
    pooled=policies.groupby(['variant','budget','method'],sort=True).position_mse_m2.mean().reset_index()
    pooled['RMSE_um']=1e6*np.sqrt(pooled.position_mse_m2)
    table=pooled[pooled.budget=='B3'].pivot(index='method',columns='variant',values='RMSE_um').reindex(METHODS).reset_index()
    risk=pd.read_csv(source/'risk_diagnostics.csv')
    summary=json.loads((source/'decision.json').read_text())['comparisons']
    root=summary['P201K64__P401K1024'];grid=summary['P201K1024__P401K1024']
    out.mkdir(parents=True)
    pd.DataFrame(checks).to_csv(out/'artifact_checks.csv',index=False)
    pd.DataFrame(direct).to_csv(out/'independent_gate_checks.csv',index=False)
    report=['# Auflösung der endlichen Entscheidungsregel', '',
      '**Beide vorab festgelegten Auflösungsprüfungen sind bestanden.** '
      f"P201/K64 → P401/K1024: kleinste deterministische Wahlübereinstimmung {100*root['minimum_deterministic_agreement']:.4f}%; "
      f"größte relative Änderung eines gepoolten Methoden-/Budget-MSE {100*root['maximum_relative_mse_change']:.4f}%. "
      f"Gitterkontrolle P201/K1024 → P401/K1024: Wahl {100*grid['minimum_deterministic_agreement']:.4f}%, MSE {100*grid['maximum_relative_mse_change']:.6f}%.", '',
      'Damit ergibt diese begrenzte Developmentprüfung keinen numerischen Grund, für den noch folgenden unabhängigen Test die vorab verwendete Auflösung P201/K64 zu ändern. '
      'Sie bestätigt keine Überlegenheit einer Auswahlregel. Die feineren Varianten bleiben vollständig erhalten.', '',
      '## Tatsächlicher Zukunfts-Prognosefehler', '',
      'Gepoolter Positions-RMSE in µm, Budget B3, über dieselben 288 Kontexte:', '',mdtable(table), '',
      'Die Bayesrisikoauswahl hat weiterhin einen höheren mittleren tatsächlichen Prognosefehler als fixed_continue100 und information_gain. '
      'Dies ist kein Regelungs-/Trackingversuch und kein Beleg für neue Methodik. Das modellierte Vorhersagerisiko ist unter Parameter- und Zustandsmodellfehlern keine kalibrierte Fehlergarantie.', '',
      '## Alle Auflösungsvergleiche', '',
      mdtable(pd.DataFrame([dict(Vergleich=x['comparison'],Wahlminimum_Prozent=100*x['minimum_agreement'],
        MSE_Aenderung_Prozent=100*x['maximum_relative_mse_change'],Kriterien='Pass' if x['passed'] else 'Fail') for x in direct]),4), '',
      'Die Kriterien gelten jeweils für jede deterministische Methode/Budget (≥95% identische Wahl) und jede der acht Methoden/Budget (≤5% Änderung des gepoolten MSE). '
      'Alle 160 Methoden-/Budget-Paarvergleiche sind gespeichert. Alle Rangfolgen stehen in pooled_summary.csv, Einzelentscheidungen in policy_episodes.csv. '
      'Für gleiche P sind tatsächliche Branchforecasts über K identisch; nur Planung und Wahl ändern sich.', '',
      'Geplante Risiken werden zusätzlich mit dem vorher deklarierten relativen Nennerboden 1e-14 m² untersucht; dies ist kein Gate:', '',
      mdtable(risk[['old','new','mean_relative_risk_change','p95_relative_risk_change','max_relative_risk_change']],5), '',
      'Unterschiedliche K verwenden denselben RNG-Seed, aber keine verschachtelten Ziehungen je Episode, weil ganze (E,K)-Arrays gezogen werden. '
      'Bei gleichem K1024 sind Uniform-/Gaußziehungen der P-Varianten gleich, die diskrete Priorverteilung ändert sich. Zwei Developmenthistorienblöcke und eine Modellbank erlauben hier keine populationsweiten p-Werte oder Konfidenzintervalle.', '',
      '## Vollständigkeit, Ressourcen und Berichtskorrektur', '',
      'Alle 288 Kontexte, 864 nativen Zweige und vier Varianten sind enthalten: 36.864 Policy- und 3.456 Branchzeilen. '
      'Sämtliche 32 Planungsdateien wurden vor dem ersten tatsächlichen späteren Probesensor oder Zukunftsziel persistiert. '
      'P201/K64 reproduziert die originale Policy-/Branch-Tabelle bis maximal 2.22e-16 numerischer Differenz. '
      'Alle nativen und gewählten Planungsguards sind null. Keine Modelle oder nativen Trajektorien wurden verändert.', '',
      f"Der Berechnungslauf dauerte {process['seconds']:.2f}s bei {process['peak_rss_MiB']:.1f}MiB gemessener Prozess-Spitze. "
      'Alle Statistiktabellen und decision.json waren bereits vollständig geschrieben, als ausschließlich die Markdown-Darstellung an der fehlenden optionalen Bibliothek tabulate scheiterte. '
      'Der ursprüngliche Lauf behält deshalb ehrlich status=failed und seinen Traceback. Dieses separate Skript rendert ohne neue Abhängigkeit und ohne Neuplanung oder Änderung der Zahlen.', '',
      'Die dadurch übersprungene abschließende Quellen-/Eingabeprüfung wurde hier vollständig nachgeholt. Alle wissenschaftlichen Artefakthashes und gespeicherten Entscheidungen stimmen; '
      'nur run.log wurde nach seiner Prüfsummenerfassung vom Python-Interpreter um den erwarteten Fehlertraceback ergänzt. Der gesperrte Logpräfix wurde bytegenau wiedergefunden; aktueller Vollhash und Erweiterung sind in completion_audit.json dokumentiert. '
      'Die Kriterien wurden zusätzlich unmittelbar aus den Einzelzeilen unabhängig nachgerechnet.', '',
      'Diese Prüfung untersucht Partikel- und Monte-Carlo-Auflösung; sie ersetzt keine native Zeitschrittverfeinerung. Erwartete Arbeitsbudgets sind keine garantierten Kostenobergrenzen.', '',
      'Quellen und vollständige Resultate: [Protokoll](../../reports/decision_resolution_protocol.md), '
      '[Originale numerische Entscheidung](../decision_resolution_v1/decision.json), '
      '[Sperrliste](../decision_resolution_v1/resolution_lock.json), '
      '[alle Policyzeilen](../decision_resolution_v1/policy_episodes.csv), '
      '[alle Zweige](../decision_resolution_v1/branch_episodes.csv), '
      '[Paarvergleiche](../decision_resolution_v1/comparisons.csv), '
      '[Abschlussaudit](completion_audit.json).', '',
      'Darstellung erneut erzeugen (frischen Ausgabeordner verwenden):', '',
      '```bash',
      'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \\',
      '/home/loya2/miniconda3/envs/unitree_sim_env/bin/python paper_study/scripts/decision_resolution_report.py \\',
      '--source paper_study/results/decision_resolution_v1 --out paper_study/results/decision_resolution_report_v2',
      '```', '']
    (out/'REPORT.md').write_text('\n'.join(report))
    audit=dict(role='Independent completion verification and presentation, no scientific source change',
        source_directory=str(source),numerical_results_complete=True,original_cli_status=process['status'],
        presentation_failure_only=True,all_scientific_hashes_match=True,log_extension=log_extension,
        source_and_input_bindings=len(lock['input_bindings_sha256'])+len(lock['source_sha256']),
        output_hash_checks=len(checks),plan_hash_checks=32,independent_gate_checks=direct,
        report_script_sha256=digest(__file__),source_lock_sha256=digest(source/'resolution_lock.json'))
    (out/'completion_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    files=sorted(p for p in out.rglob('*') if p.is_file())
    (out/'artifact_manifest.json').write_text(json.dumps(dict(outputs_sha256={str(p):digest(p) for p in files},
        script_sha256=digest(__file__)),indent=2)+'\n')
    print(json.dumps(dict(status='complete',directory=str(out),all_gates_pass=all(x['passed'] for x in direct))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.source,a.out)
