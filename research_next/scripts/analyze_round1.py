"""Frozen Round-1 analysis: strict complete cells, five paired seed contrasts.

Imports/readers have no side effects. CLI opens held-out metric CSVs only after
the experiment owner runs it. Inference never selects cells, methods or seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
KEY = ['seed', 'normal_world', 'friction_world', 'noise', 'method', 'episode']
CELL = ['seed', 'normal_world', 'friction_world', 'noise', 'method']
SUCCESS = ['success_020', 'success_040', 'success_060', 'success_080', 'success_100', 'success_120']
METRICS = {
    'prediction': ['position_rmse', 'normal_rmse', 'friction_rmse', 'guard_steps'],
    'controller': ['position_squared_integral', 'integral_cost', 'force_squared_integral',
                   'guard_steps', *SUCCESS, 'plan_seconds'],
}
PRIMARY = {'prediction': 'position_rmse', 'controller': 'position_squared_integral'}
LABELS = {'nominal':'Nominal', 'old_flex':'3-Filter-Analytik', 'path_ridge':'Verlaufs-Ridge',
          'recurrent':'Lokale GRU', 'blind_state_id':'Lokale Familienwahl',
          'sensor_state_id':'Sensor-State-ID', 'state_reset':'State-Reset'}
GERMAN_DECISION = {'advantage':'Vorteil belegt', 'disadvantage':'Nachteil belegt',
                   'unresolved':'Unentschieden'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def validate_config(config):
    if config['round'] != 1 or config['replicates'] != list(range(1, 9)):
        raise ValueError('Round 1 requires exactly the eight reserved seeds 1..8')
    expected = set(itertools.product(('maxwell', 'boucwen'), repeat=2))
    if len(config['compositions']) != 4 or set(map(tuple, config['compositions'])) != expected:
        raise ValueError('All four fixed material compositions are required')
    if len(config['methods']) != 7 or set(config['methods']) != set(LABELS):
        raise ValueError('Exactly the seven declared methods are required')
    if config['noise'] != ['clean', 'noisy'] or config['profiles_per_replicate']['evaluation'] != 16:
        raise ValueError('Both fixed noise conditions and all 16 profiles are required')
    pairs = [('prediction','blind_state_id','old_flex'), ('prediction','blind_state_id','path_ridge'),
             ('controller','blind_state_id','old_flex'), ('controller','blind_state_id','path_ridge'),
             ('controller','blind_state_id','sensor_state_id')]
    if list(map(tuple, config['primary_pairs'])) != pairs:
        raise ValueError('The five predeclared primary contrasts or their order changed')
    if config['statistics']['n'] != 8 or not np.isclose(config['statistics']['alpha_round'], .025, rtol=0, atol=1e-15):
        raise ValueError('Round-1 n or alpha allocation changed')


def expected_keys(config):
    return pd.MultiIndex.from_tuples(
        [(s, nw, fw, noise, method, ep)
         for s, (nw, fw), noise, method, ep in itertools.product(
             config['replicates'], config['compositions'], config['noise'],
             config['methods'], range(config['profiles_per_replicate']['evaluation']))], names=KEY)


def validate_metrics(frame, dataset, config):
    """Reject missing, duplicated, unexpected, nonfinite or invalid rows."""
    validate_config(config)
    if dataset not in METRICS:
        raise ValueError('Unknown dataset')
    required = KEY + METRICS[dataset]
    missing_columns = set(required)-set(frame.columns)
    if missing_columns:
        raise ValueError(f'{dataset}: missing columns {sorted(missing_columns)}')
    df = frame.copy()
    for col in ['seed', 'episode', *METRICS[dataset]]:
        if col in SUCCESS:
            # Accept literal booleans, 0/1, and their standard CSV spellings only.
            values = df[col].map(lambda x: {'true':1, 'false':0, '0':0, '1':1,
                                          '0.0':0, '1.0':1}.get(str(x).lower(), np.nan))
            if values.isna().any():
                raise ValueError(f'{dataset}: {col} must be boolean or 0/1')
            df[col] = values.astype(int)
        else:
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{dataset}: nonnumeric {col}') from exc
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f'{dataset}: nonfinite {col}; do not silently drop failed episodes')
        if (df[col] < 0).any():
            raise ValueError(f'{dataset}: negative {col}')
    for col in ('seed', 'episode', 'guard_steps'):
        if not np.equal(df[col], np.floor(df[col])).all():
            raise ValueError(f'{dataset}: {col} must be an integer')
        df[col] = df[col].astype(np.int64)
    if df[KEY].isna().any().any():
        raise ValueError(f'{dataset}: null key')
    if df.duplicated(KEY).any():
        raise ValueError(f'{dataset}: duplicate full episode keys')
    expected = expected_keys(config)
    observed = pd.MultiIndex.from_frame(df[KEY])
    absent, extra = expected.difference(observed), observed.difference(expected)
    if len(df) != 7168 or len(absent) or len(extra):
        raise ValueError(f'{dataset}: incomplete design; rows={len(df)}, missing={len(absent)}, unexpected={len(extra)}')
    return df.sort_values(KEY).reset_index(drop=True)


def aggregate(df, dataset):
    """Episode -> equally weighted composition/noise cell -> seed, never pooled SE."""
    cells = df.groupby(CELL, sort=True)[METRICS[dataset]].mean().reset_index()
    seeds = cells.groupby(['seed', 'method'], sort=True)[METRICS[dataset]].mean().reset_index()
    cell_long = cells.melt(id_vars=CELL, var_name='metric', value_name='value')
    seed_long = seeds.melt(id_vars=['seed','method'], var_name='metric', value_name='value')
    cell_long.insert(0, 'dataset', dataset)
    seed_long.insert(0, 'dataset', dataset)
    return cell_long, seed_long


def paired_interval(differences, alpha_round=.025, comparisons=5):
    d = np.asarray(differences, dtype=float)
    if d.shape != (8,) or not np.isfinite(d).all():
        raise ValueError('Exactly eight finite paired replicate differences required')
    alpha = alpha_round/comparisons
    critical = float(student_t.ppf(1-alpha/2, df=7))
    mean = float(d.mean()); sd = float(d.std(ddof=1)); se = sd/np.sqrt(8)
    low, high = mean-critical*se, mean+critical*se
    if se == 0:
        p = 1. if mean == 0 else 0.
    else:
        p = float(2*student_t.sf(abs(mean/se), df=7))
    decision = 'advantage' if high < 0 else ('disadvantage' if low > 0 else 'unresolved')
    return dict(n=8, df=7, mean_difference=mean, seed_difference_sd=sd, standard_error=float(se),
                ci_lower=float(low), ci_upper=float(high), critical_t=critical,
                alpha_round=alpha_round, alpha_per_contrast=alpha, individual_ci_coverage=1-alpha,
                p_two_sided=p, p_bonferroni=min(1., comparisons*p), decision=decision)


def primary_contrasts(seed_values, config):
    contrasts, raw = [], []
    for index, (dataset, method, comparator) in enumerate(config['primary_pairs'], 1):
        metric = PRIMARY[dataset]
        selected = seed_values[(seed_values.dataset == dataset) & (seed_values.metric == metric)]
        wide = selected.pivot(index='seed', columns='method', values='value').reindex(config['replicates'])
        left, right = wide[method].to_numpy(), wide[comparator].to_numpy()
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError('Missing paired primary seed aggregate')
        base = dict(contrast=index, dataset=dataset, metric=metric, method=method, comparator=comparator)
        contrasts.append(dict(**base, **paired_interval(left-right, config['statistics']['alpha_round'], 5),
                              method_mean=float(left.mean()), comparator_mean=float(right.mean()),
                              relative_difference_of_means=(float((left.mean()-right.mean())/right.mean())
                                                            if right.mean() > 0 else None)))
        raw.extend(dict(**base, seed=int(seed), method_value=float(a), comparator_value=float(b), difference=float(a-b))
                   for seed,a,b in zip(config['replicates'], left, right))
    return pd.DataFrame(contrasts), pd.DataFrame(raw)


def numerical_status(evidence, config):
    limits = {k:v for k,v in config['numerical_validity'].items() if isinstance(v,(int,float))}
    if not isinstance(evidence,dict) or evidence.get('complete') is not True:
        return dict(status='pending', complete=False, checks=[], reason='Independent numerical evidence missing or incomplete')
    checks=[]
    observed=evidence.get('observed',{})
    for key, limit in limits.items():
        value=observed.get(key)
        valid=isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value) and value>=0
        checks.append(dict(metric=key, observed=float(value) if valid else None, maximum=float(limit),
                           passed=bool(valid and value<=limit)))
    return dict(status='pass' if all(r['passed'] for r in checks) else 'failed',
                complete=True, checks=checks, reason='All fixed thresholds pass' if all(r['passed'] for r in checks) else 'Missing/invalid measurement or a fixed threshold exceeded')


def analyze_frames(prediction, controller, config, numerical_evidence=None):
    frames = {k:validate_metrics(v,k,config) for k,v in [('prediction',prediction),('controller',controller)]}
    parts = [aggregate(frame, kind) for kind,frame in frames.items()]
    cells = pd.concat([p[0] for p in parts], ignore_index=True)
    seeds = pd.concat([p[1] for p in parts], ignore_index=True)
    contrasts, raw = primary_contrasts(seeds, config)
    cell_summary = cells.groupby(['dataset','normal_world','friction_world','noise','method','metric']).value.agg(
        mean='mean', seed_sd='std', replicates='count').reset_index()
    method_summary = seeds.groupby(['dataset','method','metric']).value.agg(
        mean='mean', seed_sd='std', replicates='count').reset_index()
    guard = []
    for kind, df in frames.items():
        for method in config['methods']:
            x = df[df.method == method]
            guard.append(dict(dataset=kind, method=method, episodes=len(x),
                              guarded_episodes=int((x.guard_steps > 0).sum()),
                              reported_guard_count=int(x.guard_steps.sum())))
    decisions = dict(round=1, complete=True, prediction_rows=len(frames['prediction']),
                     controller_rows=len(frames['controller']), independent_replicates=8,
                     paired_observations_per_contrast=8, primary_contrasts=5,
                     alpha_round=config['statistics']['alpha_round'],
                     simultaneous_coverage_lower_bound=1-config['statistics']['alpha_round'],
                     all_five_advantages=bool((contrasts.decision == 'advantage').all()),
                     advantages=int((contrasts.decision == 'advantage').sum()),
                     disadvantages=int((contrasts.decision == 'disadvantage').sum()),
                     unresolved=int((contrasts.decision == 'unresolved').sum()),
                     neural_novelty_claim=False, architecture_superiority_claim=False,
                     guards_excluded=False, guarded_episodes=sum(r['guarded_episodes'] for r in guard),
                     decisions=contrasts.astype(object).where(pd.notna(contrasts),None).to_dict(orient='records'))
    decisions['numerical_validity']=numerical_status(numerical_evidence,config)
    decisions['no_guard_activations']=decisions['guarded_episodes']==0
    decisions['guarded_episodes_main_method']=sum(r['guarded_episodes'] for r in guard if r['method']=='blind_state_id')
    decisions['main_method_no_guard_activations']=decisions['guarded_episodes_main_method']==0
    decisions['complete_technical_success']=bool(decisions['all_five_advantages'] and decisions['main_method_no_guard_activations'] and decisions['numerical_validity']['status']=='pass')
    if not decisions['main_method_no_guard_activations'] or decisions['numerical_validity']['status']=='failed':
        decisions['technical_claim_status']='inconclusive_numerical_validity'
    elif decisions['numerical_validity']['status']=='pending':
        decisions['technical_claim_status']='pending_numerical_validity'
    elif decisions['complete_technical_success']:
        decisions['technical_claim_status']='complete_pattern_supported'
    else:
        decisions['technical_claim_status']='full_pattern_not_demonstrated'
    return dict(frames=frames, cell_aggregates=cells, seed_aggregates=seeds,
                cell_summary=cell_summary, method_summary=method_summary,
                primary_contrasts=contrasts, paired_seed_differences=raw,
                guards=pd.DataFrame(guard), decision=decisions)


def make_figures(result, config, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':10, 'savefig.dpi':180, 'axes.spines.top':False, 'axes.spines.right':False})
    outputs = []
    fig, axes = plt.subplots(1,2,figsize=(12,4.4), gridspec_kw={'width_ratios':[1,1.2]})
    for ax, dataset, factor, unit in zip(axes, ['prediction','controller'], [1000,1e6], ['mm','mm² s']):
        c = result['primary_contrasts'].query('dataset == @dataset').reset_index(drop=True)
        for i,row in c.iterrows():
            raw = result['paired_seed_differences']
            d = raw[raw.contrast == row.contrast].difference.to_numpy()*factor
            # Fixed small vertical offsets show all eight seed differences.
            ax.scatter(d, i+np.linspace(-.10,.10,8), color='#a4aab3', s=17, zorder=1)
            center=row.mean_difference*factor
            ax.errorbar(center,i,xerr=[[center-row.ci_lower*factor],[row.ci_upper*factor-center]],
                        fmt='o',color='#17645e',capsize=4,zorder=2)
        ax.axvline(0,color='black',linewidth=.8)
        ax.set_yticks(range(len(c)), [LABELS[x] for x in c.comparator]);ax.invert_yaxis()
        ax.set_xlabel(f'Lokale Familienwahl minus Vergleich [{unit}]')
        ax.set_title('Offene Positionsprognose' if dataset=='prediction' else 'Regelung: Positionsfehlerintegral')
        ax.grid(axis='x',alpha=.18)
    fig.suptitle('Fünf festgelegte Kontraste · 99,5%-Intervalle · acht gepaarte Replikate')
    fig.tight_layout()
    path=directory/'round1_primary_contrasts.png';fig.savefig(path,bbox_inches='tight');plt.close(fig);outputs.append(path)
    for dataset, metric, scale, unit in [('prediction','position_rmse',1000,'Positions-RMSE [mm]'),
                                       ('controller','position_squared_integral',1e6,'Positionsfehlerintegral [mm² s]')]:
        rows = [(nw,fw,noise) for nw,fw in config['compositions'] for noise in config['noise']]
        selected = result['cell_summary'].query('dataset == @dataset and metric == @metric')
        matrix=np.array([[selected[(selected.normal_world==nw)&(selected.friction_world==fw)&(selected.noise==noise)&(selected.method==method)]['mean'].iloc[0]*scale for method in config['methods']] for nw,fw,noise in rows])
        fig,ax=plt.subplots(figsize=(12,5.7))
        im=ax.imshow(matrix,aspect='auto',cmap='YlGnBu')
        for i in range(len(rows)):
            for j in range(len(config['methods'])):
                rgb=im.cmap(im.norm(matrix[i,j]))[:3]
                luminance=.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
                ax.text(j,i,f'{matrix[i,j]:.3g}',ha='center',va='center',fontsize=9,
                        color='white' if luminance < .5 else '#111111')
        ax.set_xticks(range(7),[LABELS[m] for m in config['methods']],rotation=24,ha='right')
        ax.set_yticks(range(len(rows)),[f'N:{nw} · Ft:{fw} · {noise}' for nw,fw,noise in rows])
        ax.set_title(f'{unit} · Mittelwerte aller acht Replikate, deskriptiv')
        fig.colorbar(im,ax=ax,label=unit,shrink=.8);fig.tight_layout()
        path=directory/f'round1_{dataset}_cells.png';fig.savefig(path,bbox_inches='tight');plt.close(fig);outputs.append(path)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,noise in zip(axes,config['noise']):
        for method in config['methods']:
            q=result['cell_aggregates'].query('dataset == "controller" and noise == @noise and method == @method')
            y=[q[q.metric==metric].value.mean()*100 for metric in SUCCESS]
            ax.plot(config['checkpoints_seconds'],y,marker='o',label=LABELS[method],linewidth=1.4,markersize=4)
        ax.set_title(noise);ax.set_xlabel('Festgelegter Checkpoint [s]');ax.set_ylim(-3,103);ax.set_xticks(config['checkpoints_seconds']);ax.grid(alpha=.18)
    axes[0].set_ylabel('Erfolg der letzten 100 ms [%]')
    axes[1].legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False)
    fig.suptitle('Alle Methoden, Kombinationen und Replikate · Kalibrierungsrauschen')
    fig.tight_layout();path=directory/'round1_success_checkpoints.png';fig.savefig(path,bbox_inches='tight');plt.close(fig);outputs.append(path)
    return [str(p) for p in outputs]


def write_report(result, config, out, figure_paths, report_path):
    decision=result['decision'];c=result['primary_contrasts'];ms=result['method_summary']
    if decision['all_five_advantages']:
        opening='Alle fünf vorab festgelegten Kontraste zeigen einen Vorteil der lokalen dynamischen Familienwahl innerhalb der untersuchten Bedingungen.'
    else:
        opening=(f"Das vollständige Vorteilsmuster über alle fünf Primärkontraste wurde nicht gezeigt: "
                 f"{decision['advantages']} Vorteile, {decision['disadvantages']} Nachteile und {decision['unresolved']} unentschiedene Kontraste.")
    technical={'complete_pattern_supported':'Das vollständige technische Erfolgsmuster besteht einschließlich der festgelegten Numerik- und Guardbedingungen.',
               'full_pattern_not_demonstrated':'Die Numerikbedingungen sind erfüllt; das vollständige technische Vorteilsmuster ist statistisch nicht gezeigt.',
               'pending_numerical_validity':'Der vollständige technische Claim bleibt bis zum Abschluss der unabhängigen Numerikkontrolle offen.',
               'inconclusive_numerical_validity':'Wegen mindestens einer Numerik-/Guardabweichung bleibt der vollständige technische Claim unentschieden; die Kontraste werden unverändert offengelegt.'}[decision['technical_claim_status']]
    text=f"""# Runde 1: Lokale Zustandsmodelle neu koppeln

**{opening}** {technical} Diese Aussage betrifft eine Bibliothek bekannter Materialfamilien
mit lokal identifizierten Koeffizienten. Sie belegt keine neuronale Neuheit,
keine allgemeine Überlegenheit rekurrenter Architekturen und keine Übertragung
auf reale Materialien oder Roboter. Frühere negative Studien bleiben bestehen.

## Festgelegter Versuch und vollständige Stichprobe

Acht reservierte Replikate (Seeds 1–8), vier Kombinationen aus Normal- und
Reibungsgesetz, zwei Bedingungen des **Kalibrierungsrauschens**, sieben
Verfahren und 16 Profile ergeben je **7168 offene Prognosen und 7168
Controller-Episoden**. Beide CSVs wurden auf genau dieses kartesische Produkt,
eindeutige vollständige Schlüssel und endliche nichtnegative Metriken geprüft.
Alle beobachteten Guards bleiben in Mittelwerten und Kontrasten enthalten.

Jede Methode beobachtet den gesamten viersekündigen Präparationsverlauf.
Während der folgenden 1,2 s werden mechanische Kopplung, Kompressionswelle und
Zielwechsel untersucht. Die offene Prognose nutzt unverändert gespeicherte,
einmalig mit privilegierter inverser Referenzregelung erzeugte Kraftbefehle;
nach dem Präfix werden keine späteren wahren Zustände oder Kräfte korrigierend
eingespeist. Die tatsächliche Regelaufgabe verwendet keine inverse Korrektur.
Alle Controller erhalten ideale aktuelle x/v/N-Signale und dieselbe konstante
Kandidatenstrategie: 41 Kräfte ±12 N, 250 ms Horizont, Neuplanung alle 50 ms.
Im ersten Planungsschritt ist aktuelles N erlaubt, danach das Lastmodell.

Die Hauptmethode wählt beide Gleichungsfamilien ausschließlich aus isolierter
Validierung; die richtige Familie steht als starker Prior in der Bibliothek.
Sensor-State-ID behält das identifizierte Reibungszustandsmodell und den
Präfix, hält aber die aktuelle Normalkraft im Horizont konstant. State-Reset
verwirft den Präfix am Bewegungsbeginn und ist eine diagnostische Ablation.
Alle Modelle wurden vor Öffnung der zurückgehaltenen Kombinationen eingefroren;
deren formale Sperrnachweise werden im Gesamtabschluss separat referenziert.

## Primärkontraste und Unsicherheit

Zuerst werden die 16 Episoden pro Kombination/Rauschbedingung gemittelt,
anschließend die acht Zellen gleich gewichtet. Jeder Kontrast hat damit exakt
**acht gepaarte Replikatdifferenzen**. Einzelne Episoden oder Zeitschritte sind
keine unabhängigen Stichproben für diese Intervalle. Das Rundenbudget α=0,025
wird Bonferroni auf fünf zweiseitige Vergleiche verteilt: jedes Intervall hat
99,5% nominelle Abdeckung, die fünf gemeinsam mindestens 97,5% unter den
Student-t-Annahmen. Verwendet werden df=7 und t={c.critical_t.iloc[0]:.9f}.
Negative Differenzen begünstigen die lokale Familienwahl. Null im Intervall
bedeutet unentschieden, nicht nachgewiesene Gleichwertigkeit.

| Aufgabe | Vergleich | Differenz | 99,5%-Intervall | Relativ zum Vergleichsmittel | Entscheidung |
|---|---|---:|---:|---:|---|
"""
    for _,r in c.iterrows():
        factor,unit=(1000,'mm') if r.dataset=='prediction' else (1e6,'mm² s')
        relative=f'{100*r.relative_difference_of_means:+.3g}%' if pd.notna(r.relative_difference_of_means) else 'nicht definiert'
        text+=f"| {'Prognose' if r.dataset=='prediction' else 'Regelung'} | {LABELS[r.comparator]} | {r.mean_difference*factor:+.5g} {unit} | [{r.ci_lower*factor:+.5g}, {r.ci_upper*factor:+.5g}] {unit} | {relative} | {GERMAN_DECISION[r.decision]} |\n"
    text+="\nDas Feld `all_five_advantages` fasst nur das statistische Muster zusammen. Vollständiger technischer Erfolg verlangt zusätzlich alle festgelegten Numerikprüfungen und null Guards der Hauptmethode blind_state_id in deren offenen Trajektorien und geplanten Kandidaten. Guards anderer Verfahren bleiben deren berichtete Fehlerfälle und verhindern allein nicht den Hauptmethoden-Claim. Ein statistisches Vorzeichen allein wird nicht als substantieller technischer Nutzen definiert; absolute und relative Größen stehen deshalb nebeneinander. Dies ist kein zusätzlicher Test und kein pauschales Go-Signal für Hardware. Zellweise Befunde oder GRU-Mittelwerte werden nicht nachträglich zu Primärhypothesen erklärt.\n\n"
    if figure_paths:
        text+=f"![Die fünf festgelegten gepaarten Kontraste]({figure_paths[0]})\n\n"
    text+="## Alle Methoden: primäre und ergänzende Mittelwerte\n\nGleichgewichtete Mittelwerte über dieselben acht Replikate; keine zusätzlichen Signifikanztests. Reibungs-/Normalkraftfehler in der Tabelle stammen aus der offenen Prognose, nicht aus konditionierter lokaler Validierung. Plannerkosten und Kraftaufwand bleiben ergänzend: Die Controller-Primärmetrik ist ausschließlich das gesamte quadratische Positionsfehlerintegral.\n\n"
    text+="| Methode | Positions-RMSE mm | Normal-RMSE N | Reibungs-RMSE N | Positionsfehlerintegral mm² s | Plannerkosten | ∫u²dt N²s | Planzeit-Anteil s |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for method in config['methods']:
        def get(dataset,metric):
            return float(ms[(ms.dataset==dataset)&(ms.method==method)&(ms.metric==metric)]['mean'].iloc[0])
        vals=[get('prediction','position_rmse')*1000,get('prediction','normal_rmse'),get('prediction','friction_rmse'),get('controller','position_squared_integral')*1e6,get('controller','integral_cost'),get('controller','force_squared_integral'),get('controller','plan_seconds')]
        text+='| '+LABELS[method]+' | '+' | '.join(f'{v:.6g}' for v in vals)+' |\n'
    text+='\nPlanzeit-Anteil ist die Summe der Wandzeiten aller 24 vektorisierten Batch-Pläne geteilt durch 16 Episoden. Das ist weder Einzelinstanzlatenz noch ein Nachweis von Echtzeitfähigkeit.\n\n## Alle Kombinationen und Rauschbedingungen\n\n'
    text+='Die vollständige Tabelle `cell_summary.csv` enthält pro Materialkombination, Rauschbedingung, Methode und Metrik den Mittelwert und die deskriptive Standardabweichung der acht Seedmittel. Kein günstiger Einzelbereich wird ausgewählt. „Noisy“ bezeichnet ausschließlich gestörte Kalibrierung/Validierung; Test- und Feedbacksensoren bleiben ideal.\n\n'
    for path in figure_paths[1:3]:
        text+=f'![Vollständige deskriptive Materialkombinationen]({path})\n\n'
    text+='## Sechs unveränderte Checkpoints\n\nErfolg verwendet das unmittelbar vor dem Checkpoint liegende 100-ms-Fenster: mittlerer absoluter Positionsfehler <4 mm und mittlere absolute Geschwindigkeit <0,04 m/s. Der Checkpoint 0,6 s bewertet das erste Ziel vor dessen Wechsel. Kein Zeitpunkt ersetzt nachträglich die primäre Integralmetrik.\n\n'
    text+='| Methode | 0,2 s | 0,4 s | 0,6 s | 0,8 s | 1,0 s | 1,2 s |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for method in config['methods']:
        vals=[float(ms[(ms.dataset=='controller')&(ms.method==method)&(ms.metric==k)]['mean'].iloc[0])*100 for k in SUCCESS]
        text+='| '+LABELS[method]+' | '+' | '.join(f'{v:.2f}%' for v in vals)+' |\n'
    if len(figure_paths)>3:
        text+=f'\n![Alle Checkpoints getrennt nach Kalibrierungsrauschen]({figure_paths[3]})\n'
    text+='\n## Guards und numerische Grenzen\n\n'
    text+=f"{decision['guarded_episodes']} der 14336 gemeldeten Episoden haben mindestens einen Guard-Eintrag; davon betreffen {decision['guarded_episodes_main_method']} die Hauptmethode blind_state_id. Diese Episoden wurden vollständig beibehalten. Nichtendliche Metriken hätten die Auswertung mit Fehler beendet; sie wären nicht stillschweigend entfernt worden. Im Prognoseteil zählt guard_steps offene Bewegungsschritte. Im Controllerteil zählt es reale Planungszeitpunkte, an denen mindestens ein Kandidat dieser Episode einen Guard hat; Rohdateien bewahren zusätzlich die einzelnen Kandidatenzähler. Dies sind keine realen PhysX-Fehlschritte. Jeder Eintrag bei blind_state_id verhindert den vollständigen technischen Claim. Guards anderer Verfahren werden als deren Robustheitsfehler berichtet, ohne sie zu entfernen oder allein die Hauptmethode für ungültig zu erklären. Vorteile gegenüber einer extrem fehlerhaften Baseline erlauben keinen universellen Leistungsclaim.\n\n"
    text+='| Aufgabe | Methode | Episoden mit Guard / alle | Gemeldete Guard-Summe |\n|---|---|---:|---:|\n'
    for _,r in result['guards'].iterrows():
        text+=f'| {r.dataset} | {LABELS[r.method]} | {r.guarded_episodes} / {r.episodes} | {r.reported_guard_count} |\n'
    text+='\n### Festgelegte unabhängige Numerikkontrolle\n\n'
    text+=f"Status: **{decision['numerical_validity']['status']}**. Die vorher reservierte saubere Replikation r1 umfasst alle vier Kombinationen und 16 Profile; MuJoCo wird bei h1/h2/h4 verglichen.\n\n"
    if decision['numerical_validity']['checks']:
        text+='| Messgröße | Beobachtet | Feste Obergrenze | Bestanden |\n|---|---:|---:|---|\n'
        for row in decision['numerical_validity']['checks']:
            value=f"{row['observed']:.6g}" if row['observed'] is not None else 'fehlt/ungültig'
            text+=f"| {row['metric']} | {value} | {row['maximum']:.6g} | {'ja' if row['passed'] else 'nein'} |\n"
    text+='\n## Geltungsbereich und Abschlussnachweise\n\n'
    text+='Der technische Befund gilt für bekannte geschlossene Kontakte, bekannten Präparationsbeginn, unveränderte synthetische Kraftgesetze, dt=2,5 ms und einen beschränkten konstanten MPC. Gleiche Kalibrierungsverteilungen garantieren keine vollständige lokale Merkmalsabdeckung unter Kopplung. Die zwei vorhandenen Gesetze mit vier technischen Zusammensetzungen ersetzen keine breite Materialpopulation. Acht Replikate begrenzen die Präzision und die Prüfung der Verteilungsannahmen. Ein praktischer Vorteil der Familienbibliothek wäre keine Entdeckung der Gleichungsfamilien aus beliebigen Daten.\n\n'
    text+='Die zwei Metrik-CSVs enthalten keine unabhängigen Backend-/Zeitschritt-, Support-, Hold- oder Videonachweise. Numerik wird nur bei vorhandener vollständiger numerical_validity.json anhand der festen Grenzen bewertet; fehlende Evidenz bleibt offen. Rohartefakte der Kontrollen sowie Support/Hold/Videos werden im Gesamtabschluss separat verlinkt. Ein echter G1 wurde nicht verwendet.\n\n'
    text+='## Reproduktion und Quellen\n\n'
    text+='```bash\nOPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/round1-mpl /home/loya2/miniconda3/envs/unitree_sim_env/bin/python research_next/scripts/analyze_round1.py\n```\n\n'
    text+=f"Die Volltabellen und Entscheidungen liegen in `{out}`: `seed_aggregates.csv`, `cell_aggregates.csv`, `cell_summary.csv`, `method_summary.csv`, `paired_seed_differences.csv`, `primary_contrasts.csv`, `guards.csv` und `decision.json`.\n\n"
    text+=f"Protokoll: [round1_protocol.md]({ROOT/'reports/round1_protocol.md'}); Auswertungsplan: [round1_analysis_plan.md]({ROOT/'reports/round1_analysis_plan.md'}); Konfiguration: [round1.json]({ROOT/'configs/round1.json'}). `provenance.json` enthält Prüfsummen der eingelesenen Daten, Quellen, Konfiguration und des Auswertungsplans. Die statistischen Entscheidungen sind vollständig aus den acht gespeicherten Differenzen pro Kontrast rekonstruierbar.\n"
    report_path.parent.mkdir(parents=True,exist_ok=True);report_path.write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/round1')
    parser.add_argument('--out',type=Path,default=ROOT/'results/round1/analysis')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/round1.json')
    parser.add_argument('--report',type=Path,default=ROOT/'reports/round1_report.md')
    parser.add_argument('--figures',type=Path,default=ROOT/'reports/figuresround1')
    parser.add_argument('--numerical',type=Path,help='Default: input/numerical_validity.json; missing file means pending')
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    sys.path.insert(0,str(ROOT/'src'))
    from round1_lock import verify_model_lock,verify_test_lock
    locks=dict(model_lock_sha256=verify_model_lock(),test_lock_sha256=verify_test_lock())
    config=json.loads(args.config.read_text());validate_config(config)
    sources=[args.input/'prediction_metrics.csv',args.input/'controller_metrics.csv',args.config,
             Path(__file__),ROOT/'reports/round1_analysis_plan.md',ROOT/'reports/round1_protocol.md']
    numerical_path=args.numerical or args.input/'numerical_validity.json'
    numerical_evidence=None
    if numerical_path.exists():
        sources.append(numerical_path);numerical_evidence=json.loads(numerical_path.read_text())
    provenance=dict(source_sha256={str(p.resolve()):sha(p) for p in sources},
                    python=sys.version,numpy=np.__version__,pandas=pd.__version__,
                    locks=locks,
                    input_scope='fresh reserved round1 metrics; no posthoc exclusions or model fitting')
    pred=pd.read_csv(sources[0]);control=pd.read_csv(sources[1])
    result=analyze_frames(pred,control,config,numerical_evidence)
    args.out.mkdir(parents=True,exist_ok=True)
    for key in ('cell_aggregates','seed_aggregates','cell_summary','method_summary','primary_contrasts','paired_seed_differences','guards'):
        result[key].to_csv(args.out/f'{key}.csv',index=False)
    write_json(args.out/'decision.json',result['decision']);write_json(args.out/'provenance.json',provenance)
    figures=[] if args.no_plots else make_figures(result,config,args.figures)
    write_report(result,config,args.out,figures,args.report)
    # Detect a concurrent producer changing an input while it is being analyzed.
    if any(sha(p)!=provenance['source_sha256'][str(p.resolve())] for p in sources):
        raise RuntimeError('Input/source changed during analysis; generated output is not certified')
    if dict(model_lock_sha256=verify_model_lock(),test_lock_sha256=verify_test_lock())!=locks:
        raise RuntimeError('Model/test lock changed during analysis')
    print(json.dumps(result['decision'],indent=2,allow_nan=False))


if __name__=='__main__':
    main()
