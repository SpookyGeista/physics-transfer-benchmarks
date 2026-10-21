"""Descriptive analysis only: no fitting or new selection policy."""
from pathlib import Path
import argparse,json,hashlib,time,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pilot import read,write,check,sha,ROOT


def quantile(x):
    a=np.asarray(x,dtype=float)
    return {str(q):float(np.quantile(a,q)) for q in (0,.25,.5,.75,.9,.95,.99,1)} if len(a) else None


def describe(d):
    selected=d[d.selected_start==1]
    small=selected[selected.original_gain<=1e-8]
    large=selected[selected.original_gain>1e-8]
    def metrics(a):
        close=a.past_mahalanobis<=1
        divergent=a.future_distance_position_m>=1e-4
        return dict(pairs=len(a),past_curve_distance=quantile(a.past_mahalanobis),future_distance_mm=quantile(a.future_distance_position_m*1000),
            normalized_state_distance=quantile(a.normalized_initial_distance),past_close_count=int(close.sum()),
            close_past_divergent_future_count=int((close&divergent).sum()),
            future_distance_mm_when_past_close=quantile(a.loc[close,'future_distance_position_m']*1000),
            future_distance_mm_when_past_far=quantile(a.loc[~close,'future_distance_position_m']*1000),
            boundary_position_distance_um=quantile(a.boundary_x_distance_m*1e6),
            original_vs_replay_selection_changes=int((a.selected_start!=a.replay_float64_choice).sum()),
            replay_float64_vs_extended_selection_changes=int((a.replay_float64_choice!=a.same_residual_extended_choice).sum()),
            same_residual_vs_formula_selection_changes=int((a.same_residual_extended_choice!=a.stable_choice).sum()),
            fixed_pair_noisy_vs_clean_selection_changes=int((a.stable_choice!=a.clean_fixed_pair_choice).sum()))
    return dict(all_pairs=metrics(d),original_second_selected=metrics(selected),
        second_selected_gain_at_most_1e_8=metrics(small),second_selected_gain_above_1e_8=metrics(large))


def analyze(a):
    """Write descriptive outputs only after the caller verifies every input."""
    d=pd.read_csv(a.data/'pairs.csv')
    if len(d)!=10368:raise ValueError('Expected all 10,368 candidate pairs')
    summary=describe(d)
    summary['by_probe']={str(k):describe(g) for k,g in d.groupby('probe')}
    summary['by_bank']={str(k):describe(g) for k,g in d.groupby('bank')}
    summary['by_history']={str(k):describe(g) for k,g in d.groupby('history')}
    summary['by_composition']={f'{n}/{f}':describe(g) for (n,f),g in d.groupby(['normal','friction'])}
    summary['numerics']={k:float(d[k].max()) for k in ('replay_objective_max_abs','replay_future_max_abs','replay_boundary_max_abs','noise_reconstruction_max_abs','solver_max_abs_N','objective_identity_error','noise_identity_error')}
    selected_mse=np.where(d.selected_start==1,d.start2_mse_position_m,d.start1_mse_position_m)
    summary['original_position_rmse_mm']=dict(start1=float(np.sqrt(d.start1_mse_position_m.mean())*1000),two_start=float(np.sqrt(selected_mse.mean())*1000))
    write(a.out/'summary.json',summary)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(10.5,4.1),sharex=True,sharey=True)
    for ax,name in zip(axes,('hold_150','gentle_150','strong_300')):
        g=d[d.probe==name]
        for bank,color in zip((1,2,3),('#0072B2','#D55E00','#009E73')):
            q=g[g.bank==bank]
            ax.scatter(q.past_mahalanobis,q.future_distance_position_m*1e3,s=9,alpha=.25,color=color,label=f'Bank {bank}')
        ax.axvline(1,color='.35',ls='--',lw=.8);ax.axhline(.1,color='.35',ls=':',lw=.8)
        ax.set_xscale('symlog',linthresh=1e-6);ax.set_yscale('symlog',linthresh=1e-5)
        ax.set_xlim(left=0);ax.set_ylim(bottom=0);ax.set_title(name.replace('_',' '));ax.grid(alpha=.15)
    axes[0].set_ylabel('Candidate future position separation (mm RMS)')
    axes[1].set_xlabel('Past x/v/N curve distance in likelihood units, D')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='upper center',frameon=False)
    fig.text(.5,.015,'All 10,368 fixed candidate pairs; dashed/dotted lines are descriptive scales, not success criteria',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.91))
    fig.savefig(a.out/'past_curves_and_future_separation.png',dpi=190);fig.savefig(a.out/'past_curves_and_future_separation.pdf');plt.close(fig)
    s=summary['original_second_selected'];q=summary['second_selected_gain_at_most_1e_8']
    large=summary['second_selected_gain_above_1e_8'];numerics=summary['numerics']
    lines=['# Mechanismus-Pilot: Messkurven statt nur Zielfunktionswerte','',
    'Explorative Nachanalyse bereits bekannter Main2-Daten. Kein frischer Bestätigungstest, keine neue Auswahlregel, kein Parameterfit und kein Robotereinsatz. Alle drei Banken, vier Historienblöcke, acht Kompositionen und drei Proben sind enthalten.','',
    f'Alle {len(d):,} Kandidatenpaare wurden unter denselben gespeicherten Befehlen mit den unveränderten Modellen rekonstruiert. Die ursprünglichen Ergebnisse bleiben erhalten: gepoolte Positions-RMSE {summary["original_position_rmse_mm"]["start1"]:.5f} mm für Start 1 und {summary["original_position_rmse_mm"]["two_start"]:.5f} mm für die ursprüngliche Zweistart-Auswahl.','',
    '## Vergangenheitskurven und Zukunft','',
    'D ist die euklidische Distanz der gesamten x/v/N-Messvorhersage nach Normierung durch die ursprünglichen Likelihoodskalen. D≤1 ist nur eine deskriptive Skala; sie ist wegen datenabhängig ausgewählter Zustände und möglichem Modellfehler kein kalibrierter Test. Die Zukunftsdistanz ist die RMS-Differenz der Kandidatenprognosen, nicht deren Fehler gegen die Wahrheit.','',
    '| Probe | ursprüngliche zweite Auswahl | davon D≤1 | D≤1 und Zukunftsabstand≥0,1 mm |','|---|---:|---:|---:|']
    for name,g in summary['by_probe'].items():
        v=g['original_second_selected'];lines.append(f'| {name} | {v["pairs"]} | {v["past_close_count"]} | {v["close_past_divergent_future_count"]} |')
    lines+=['',f'Insgesamt: {s["past_close_count"]} von {s["pairs"]} ursprünglich ausgewählten Zweitlösungen haben D≤1; davon zeigen {s["close_past_divergent_future_count"]} eine Zukunftsdifferenz von mindestens 0,1 mm. Unter den {q["pairs"]} Zweitauswahlen mit ursprünglichem Gewinn≤1e−8 sind es {q["past_close_count"]} mit D≤1 und {q["close_past_divergent_future_count"]} mit zusätzlich mindestens 0,1 mm Zukunftsdifferenz.','',
    f'Im komplementären Stratum mit ursprünglichem Gewinn>1e−8 liegen {large["pairs"]} Zweitauswahlen; davon haben {large["past_close_count"]} D≤1 und {large["close_past_divergent_future_count"]} zusätzlich mindestens 0,1 mm Zukunftsdifferenz. Beide Strata und ihre vollständigen deskriptiven Verteilungen stehen für jede Aufschlüsselung in summary.json.','',
    '![Vergangene Messkurvendistanz und zukünftige Prognosetrennung](past_curves_and_future_separation.png)','',
    '## Rauschen und numerische Reduktion','',
    'Für unveränderte Kandidaten gilt ΔJ=2r₁ᵀd+‖d‖² und ΔJ_beobachtet=ΔJ_sauber−2eᵀd. Dabei sind r₁=(p₁−y)/σ, d=(p₂−p₁)/σ und e=(y−y_sauber)/σ bereits durch dieselben Likelihoodskalen normiert. Der Rauschterm ist nur der unmittelbare Einfluss des realisierten Messrauschens auf dieses feste Paar. Die Kandidaten und Rückkopplungsbefehle wurden ursprünglich unter Rauschen erzeugt. Diese Auswertung isoliert weder den gesamten Rauscheinfluss noch Parameterbias.','',
    f'Über alle Paare ändern sich {summary["all_pairs"]["fixed_pair_noisy_vs_clean_selection_changes"]} Vorzeichenentscheidungen beim algebraischen Entfernen des direkten Rauschterms. Das sagt ohne Größenprüfung nichts über praktische Bedeutsamkeit. Originalauswahl versus neuer float64-Replay: {summary["all_pairs"]["original_vs_replay_selection_changes"]} Unterschiede. Auf exakt denselben float64-Residuen unterscheidet sich die float64-Objektivauswertung von der Auswertung mit Produkten und Summen in erweiterter Präzision in {summary["all_pairs"]["replay_float64_vs_extended_selection_changes"]} Auswahlen. Dies ist keine isolierte Untersuchung allein der Summationsrundung.','',
    f'Die Auswertung derselben float64-Residuen in erweiterter Präzision unterscheidet sich in {summary["all_pairs"]["same_residual_vs_formula_selection_changes"]} Auswahlen von der gesonderten stabilen Differenzformel mit bereits in erweiterter Präzision gebildeten Residuen. Diese zweite Differenz umfasst Residuenbildung und algebraische Auswertung. Alle drei Vergleiche werden getrennt gezählt; der dynamische Integrator bleibt float64 und die Originalmethode wird nicht ersetzt.','',
    f'Maximale Replay-Abweichungen: Zielfunktion {numerics["replay_objective_max_abs"]:.3g}, Zukunftskanal {numerics["replay_future_max_abs"]:.3g}, Boundaryzustand {numerics["replay_boundary_max_abs"]:.3g}. Direkte Rekonstruktion des ursprünglichen Rauschstroms: {numerics["noise_reconstruction_max_abs"]:.3g}. Alle Solverresiduen≤{numerics["solver_max_abs_N"]:.3g} N.','',
    '## Aussagegrenzen und nächste Kontrolle','',
    'Ein kleiner Kurvenabstand bei großer Prognosetrennung ist mit endlicher Fenstermehrdeutigkeit vereinbar. Große Kurvenabstände trotz nahezu gleicher Ziele sprechen stattdessen für kompensierende Residuen. Beides ist eine Diagnose der konkreten festen Kandidaten. Sie beweist keine globale Nichtbeobachtbarkeit, keine tatsächliche Erreichbarkeit sämtlicher MLE-Boxzustände und keine neue Theorie.','',
    'Die ganzen Boundaryzustände können sich unterscheiden. Deshalb werden mechanische, normale und Reibungszustandsabstände einzeln gespeichert. Eine reine Zuordnung zum Reibungsgedächtnis oder zu bestimmten Fließereignissen erfordert eine eigene kontrollierte Intervention.','',
    'Als nächste wissenschaftliche Stufe ist ein separat deklarierter Modellgleichheits-/Rauschkontrollversuch sinnvoll: erreichbare Anfangszustände, synthetische Wahrheit und Schätzung im exakt gleichen Modell, zunächst ohne, dann mit Rauschen, bei gleichem Befehlsprogramm. Die vorhandenen PhysX-Zeitschrittgrenzen gelten weiterhin; Ergebnisse dieses Piloten werden dem fertigen Paper nicht stillschweigend als Bestätigung hinzugefügt.','',
    'Vollständige Verteilungen und alle Bank-, Historien- und Kompositionsaufschlüsselungen: `summary.json`. Rohdiagnosen: `../pairs.csv` und `../paired_past/`.']
    (a.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return dict(pairs=len(d),original_second_selected=s,near_tie_second_selected=q,
        larger_gain_second_selected=large,numerics=numerics)


def run(a):
    a.data,a.out=Path(a.data).resolve(),Path(a.out).resolve()
    if a.out.exists():raise FileExistsError('Fresh analysis directory required')
    started=time.perf_counter()
    process=dict(status='running',started_unix=time.time(),command=sys.argv,
        source_sha256=sha(__file__),data_directory=str(a.data))
    a.out.mkdir(parents=True)
    write(a.out/'process.json',process)
    try:
        paths=[a.data/name for name in ('process.json','analysis_lock.json','artifact_manifest.json','pairs.csv')]
        control_hashes={str(path):sha(path) for path in paths}
        original_process=read(paths[0]);lock=read(paths[1]);manifest=read(paths[2])
        lock_sha=control_hashes[str(paths[1])]
        if original_process.get('status')!='complete':raise ValueError('Complete replay required')
        if original_process.get('lock_sha256')!=lock_sha or manifest.get('lock_sha256')!=lock_sha:
            raise ValueError('Pilot process/manifest must bind the same analysis lock')
        if manifest['outputs_sha256'].get(str(paths[3]))!=control_hashes[str(paths[3])]:
            raise ValueError('Candidate-pair CSV must be bound by the pilot manifest')
        if lock['source_and_input_sha256'].get(str(Path(__file__).resolve()))!=process['source_sha256']:
            raise ValueError('Pilot analysis lock must bind this analyzer source')
        check(control_hashes);check(manifest['outputs_sha256']);check(lock['source_and_input_sha256'])
        process.update(lock_sha256=lock_sha,pilot_manifest_sha256=control_hashes[str(paths[2])],
            input_sha256=control_hashes)
        write(a.out/'process.json',process,False)
        result=analyze(a)
        # No success record is written if sources, pilot controls or outputs
        # changed while the descriptive report/plots were being constructed.
        check(control_hashes);check(manifest['outputs_sha256']);check(lock['source_and_input_sha256'])
        outputs={str(f):sha(f) for f in a.out.iterdir() if f.is_file() and f.name!='process.json'}
        write(a.out/'artifact_manifest.json',dict(source_sha256=process['source_sha256'],
            lock_sha256=lock_sha,input_sha256=control_hashes,outputs_sha256=outputs))
        write(a.out/'process.json',dict(**{k:v for k,v in process.items() if k!='status'},
            status='complete',seconds=time.perf_counter()-started,pairs=result['pairs']),False)
        print(json.dumps(result,indent=2))
    except BaseException as exc:
        partial={str(f):sha(f) for f in a.out.iterdir() if f.is_file() and f.name!='process.json'}
        write(a.out/'process.json',dict(**{k:v for k,v in process.items() if k!='status'},
            status='failed',error=repr(exc),partial_outputs_sha256=partial,
            seconds=time.perf_counter()-started),False)
        raise
    finally:
        plt.close('all')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
