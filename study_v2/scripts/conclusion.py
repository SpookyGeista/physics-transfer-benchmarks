"""Presentation-only consolidated report from already frozen derived statistics.

No fitting, episode selection, confidence-interval calculation or decision
modification occurs here. Original and supplementary arrangement sets stay separate.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/physics-transfer-conclusion-matplotlib')
from pathlib import Path
from datetime import datetime, timezone
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from decision_study.common import ROOT, CFG, digest
from decision_study.locking import verify_lock

NAMES = dict(nominal='Nominell', identified='Identifikation', analytic_v1='Analytisch v1',
             analytic_flex='Analytisch flexibel', joint='Gemeinsames NN', moe='MoE-Adaption',
             modular='Modular', no_history='Ohne Historie', privileged='Privilegiert', decoupled='Entkoppelt')
WORLDS = dict(maxwell='Maxwell', boucwen='Bouc–Wen')
COLORS = dict(nominal='#8b96a1', identified='#8e7638', analytic_v1='#217b57', analytic_flex='#15639b',
              joint='#8370a7', moe='#a17447', modular='#c83c35', no_history='#6f787c',
              privileged='#879d2d', decoupled='#3b9295')


def one(table, **conditions):
    selected = table
    for key, value in conditions.items():
        selected = selected[selected[key] == value]
    assert len(selected) == 1, f'ambiguous existing statistic: {conditions}'
    return selected.iloc[0]


def ci(row, scale=1., digits=2):
    return f'{row["mean"]*scale:.{digits}f} ± {row.ci95_half*scale:.{digits}f}'


def difference(row, scale=1., digits=3):
    return f'{row["mean"]*scale:+.{digits}f} [{row.ci95_low*scale:+.{digits}f}; {row.ci95_high*scale:+.{digits}f}]'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(str(value) for value in row)+' |' for row in rows])


def save(fig, path):
    fig.savefig(path, dpi=220, bbox_inches='tight', facecolor='white'); plt.close(fig)


def plots(prediction, pairs, task, folder):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .15})
    # The overview includes every method; the adjacent panel makes the small
    # predeclared analytical/modular comparisons legible on their own scale.
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for wi, world in enumerate(CFG['worlds']):
        for col, methods in enumerate([CFG['methods'], ['analytic_v1', 'analytic_flex', 'modular']]):
            ax = axes[wi, col]
            rows = [one(prediction, world=world, method=method, noise='clean', metric='position_rmse') for method in methods]
            for i, (method, row) in enumerate(zip(methods, rows)):
                ax.errorbar(row['mean']*1000, i, xerr=row.ci95_half*1000, fmt='o', ms=6,
                            capsize=3, lw=1.6, color=COLORS[method])
                if col == 1:
                    ax.annotate(ci(row, 1000), (row['mean']*1000, i), xytext=(0, 10),
                                textcoords='offset points', ha='center', fontsize=9)
            ax.set(yticks=np.arange(len(methods)), yticklabels=[NAMES[m] for m in methods],
                   xlabel='Positions-RMSE (mm) · Mittel ± 95-%-t-Intervall',
                   title=WORLDS[world]+(' · alle Methoden' if col == 0 else ' · analytischer Vergleich'),
                   ylim=(len(methods)-.5, -.65))
    fig.suptitle('Prospektives Supplement · volles Budget · clean · fünf gepaarte Replikate', fontsize=14)
    save(fig, folder/'supplement_prediction.png')
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.5), constrained_layout=True)
    for wi, world in enumerate(CFG['worlds']):
        for col, baselines in enumerate([['analytic_v1', 'analytic_flex'], ['joint', 'moe']]):
            ax = axes[wi, col]
            for i, baseline in enumerate(baselines):
                row = one(pairs, family='prediction', world=world, noise='clean', baseline=baseline, metric='position_rmse')
                ax.errorbar(row['mean']*1000, i, xerr=row.ci95_half*1000, fmt='o', color=COLORS['modular'], capsize=4, lw=1.8)
                ax.annotate(difference(row, 1000, 2), (row['mean']*1000, i), xytext=(0, 11),
                            textcoords='offset points', ha='center', fontsize=8)
            ax.axvline(0, color='#333333', lw=1)
            ax.set(yticks=[0, 1], yticklabels=[NAMES[m] for m in baselines], ylim=(1.5, -.65),
                   xlabel='Modular minus Vergleich (mm) · negativ ist günstiger', title=WORLDS[world])
    fig.suptitle('Gepaarte Positionsdifferenzen · 95-%-t-Intervalle · getrennte Skalen', fontsize=14)
    save(fig, folder/'supplement_paired.png')
    selected = ['identified', 'analytic_v1', 'analytic_flex', 'modular']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, world in zip(axes, CFG['worlds']):
        x = np.arange(len(selected))
        for shift, noise, marker in [(-.12, 'clean', 'o'), (.12, 'noisy', 's')]:
            for i, method in enumerate(selected):
                row = one(task, world=world, method=method, noise=noise, metric='cost')
                ax.errorbar(i+shift, row['mean']*1000, yerr=row.ci95_half*1000, fmt=marker,
                            color=COLORS[method], mfc=COLORS[method] if noise == 'clean' else 'white', capsize=3, ms=6)
        ax.plot([], [], 'o', color='black', label='clean'); ax.plot([], [], 's', color='black', mfc='white', label='noisy')
        ax.set(xticks=x, xticklabels=[NAMES[m] for m in selected], ylabel='Aufgabenkosten × 10³', title=WORLDS[world])
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right'); ax.legend(fontsize=8)
    fig.suptitle('Separate ursprüngliche Controlleraufgabe · 100 % Erfolg bei allen Methoden', fontsize=13)
    save(fig, folder/'supplement_task.png')


def main():
    original_hash = verify_lock()
    sup = ROOT/'supplement_exact'
    supplemental_lock = json.loads((sup/'lock.json').read_text())
    for name, sha in supplemental_lock['files'].items():
        assert digest(name) == sha, f'changed frozen supplement file: {name}'
    paths = {name: sup/'results'/filename for name, filename in {
        'prediction': 'prediction_summary.csv', 'pairs': 'paired_differences.csv',
        'native': 'native_summary.csv', 'decision': 'decision.json'}.items()}
    assert all(path.exists() for path in paths.values()), 'run the locked supplemental analysis first'
    decision = json.loads(paths['decision'].read_text())
    assert decision['lock_sha256'] == digest(sup/'lock.json')
    pred, pairs, native = [pd.read_csv(paths[key]) for key in ('prediction', 'pairs', 'native')]
    task = pd.read_csv(ROOT/'results/report/control_summary.csv')
    costs = pd.read_csv(ROOT/'results/report/model_metadata.csv')
    original_decision = json.loads((ROOT/'results/decision.json').read_text())
    assert original_decision['outcome'] == 'inconclusive'
    assert decision['model_scores_seen_before_supplement_lock'] is False
    assert len(costs) == 360
    figures = ROOT/'reports/figures'; figures.mkdir(parents=True, exist_ok=True)
    plots(pred, pairs, task, figures)
    summary = ['# Ergebnis: zweiter Simulationsversuch und einmalige Ergänzung', '',
               'Stand: '+datetime.now(timezone.utc).isoformat(timespec='seconds')+'. '+
               'Dieser Bericht stellt bereits gespeicherte Auswertungen dar; er verändert keine statistische Regel und fasst die beiden Anordnungstests nicht zusammen.', '']
    if decision['outcome'] == 'no-go':
        summary += ['**No-Go für das Hochskalieren dieser konkreten Lernmethode im geprüften Aufbau.** '+
                    'Die streng geprüfte Ergänzung zeigt keinen durchgängigen Vorteil gegenüber analytischer Identifikation. '+
                    'Die Aussage gilt für die eingefrorenen Modelle und den Zeitschritt von **2,5 ms**; sie ist kein allgemeines Urteil über modulare Robotikmodelle.', '']
    elif decision['outcome'] == 'go':
        summary += ['**Positiver begrenzter Simulationsbefund.** Die gespeicherte Entscheidungsdatei erfüllt die vollständige vorab festgelegte Regel. '+
                    'Der Befund betrifft die eingefrorenen Modelle im synthetischen Prüfstand bei 2,5 ms und erfordert unabhängige weitere Replikation.', '']
    else:
        summary += ['**Nicht entscheidbar.** Die gespeicherte Ergänzungsentscheidung erfüllt weder die vollständige Go-Regel noch die gültige eindeutige Gegenregel. '+
                    'Fehlende Signifikanz belegt keine Gleichwertigkeit.', '']
    mflex = one(pairs, family='prediction', world='maxwell', noise='clean', baseline='analytic_flex', metric='position_rmse')
    bflex = one(pairs, family='prediction', world='boucwen', noise='clean', baseline='analytic_flex', metric='position_rmse')
    modular_m = one(pred, world='maxwell', noise='clean', method='modular', metric='position_rmse')
    analytic_m = one(pred, world='maxwell', noise='clean', method='analytic_flex', metric='position_rmse')
    modular_b = one(pred, world='boucwen', noise='clean', method='modular', metric='position_rmse')
    analytic_b = one(pred, world='boucwen', noise='clean', method='analytic_flex', metric='position_rmse')
    summary += [f'Maxwell: Modular erreicht **{1000*modular_m["mean"]:.2f} mm**, analytisch flexibel **{1000*analytic_m["mean"]:.2f} mm** Positions-RMSE. '+
                f'Der gepaarte Mehrfehler beträgt **{difference(mflex, 1000)} mm** (Mittel [95-%-CI]). '+
                f'Bouc–Wen: **{1000*modular_b["mean"]:.2f} gegenüber {1000*analytic_b["mean"]:.2f} mm**; die Differenz **{difference(bflex, 1000)} mm** ist unsicher. '+
                'In der getrennten Controlleraufgabe fehlt ein durchgängiger Kostenvorteil; alle 1.600 Ausführungen waren erfolgreich.', '',
                '## Warum es zwei v2-Auswertungen gibt', '',
                'Der ursprüngliche Versuch bleibt formal **nicht entscheidbar**. 958 von 960 Referenzepisoden bestanden den vorab definierten direkten Abgleich. '+
                'Bei zwei Bouc–Wen-Episoden betrug die Normalkraftabweichung etwa **113 µN** statt der erlaubten **100 µN**. '+
                'Alle lokalen Nähediagnosen lagen bei 100 %; dennoch wurde die strengere Grenze nicht nachträglich gelockert. '+
                'Die ursprünglichen Messungen und ihr Urteil bleiben im [Originalbericht](final_report.md) erhalten.', '',
                'Genau **eine prospektiv dokumentierte Ergänzung** wurde nach dieser Konstruktionsdiagnose und laut Änderungsprotokoll **vor Einsicht der Modell- und Controllerscores** festgelegt. '+
                'Der einmalige Referenzgenerator kompensiert die Nocke nun am tatsächlichen simulierten Zustand; seine Kommandos werden anschließend unverändert offen abgespielt. '+
                'Neue Nockenamplituden sind ±2 und ±5 mm, statt ursprünglich ±3 und ±6 mm. Alle 360 Modellstände bleiben identisch; es gibt kein Nachtraining, keine neue Checkpointauswahl und keine nachträglich geänderte Auswertungsgrenze. '+
                'Die unterschiedlichen Anordnungen können die Schwierigkeit ändern: Leistungsunterschiede zwischen beiden Durchläufen dürfen nicht allein der Konstruktionskorrektur zugeschrieben werden. '+
                '[Prospektives Ergänzungsprotokoll](../supplement_exact/protocol.md).', '',
                f'Die Ergänzung erfüllt **960/960 direkte Abgleiche**, 100 % lokale Nähediagnose und beide unabhängigen Backendprüfungen. '+
                f'Die gesperrte Entscheidung lautet `{decision["outcome"]}`. '+
                'Die ursprünglichen Controlleraufgaben bleiben eine separate Prüfung auf den ursprünglichen Geometrien; für die neuen Nocken wird kein neuer Controllernachweis behauptet.', '',
                '## Vorhersage in neuen Anordnungen', '',
                'Der Prüfstand ist eine native CPU-PhysX-Masse von 2 kg an einer Führung. Programmierte Normal- und Reibkräfte wirken über eine bekannte beschränkte Nocke. '+
                'Native Materialkontakte sind deaktiviert; es wurde kein interner Kontaktlöser gelernt. Maxwell umfasst mehrere Relaxationszweige, Bouc–Wen eine andere Hysteresedynamik. '+
                'Die Normalkraft wird in B isoliert gemessen und in A als kontrollierte Last eines getrennten Reibungsversuchs wiederholt. '+
                'Der Haupttest verwendet **bekannte vollständige lokale Verläufe unter neuen Geometrien**, keine neuen lokalen Historien. '+
                'Alle Prädiktoren erhalten dieselben gespeicherten Kommandos; nach gleicher Anfangsmessung verwenden sie ausschließlich selbst vorhergesagte Zustände und Kräfte.', '',
                'Fünf unabhängige Kalibrierdatensätze mit gepaarten Modellseeds je Welt. Die Tabelle zeigt Mittel ± halbe Breite des bestehenden 95-%-t-Intervalls bei n=5; '+
                'Episoden und Zeitpunkte vergrößern dieses n nicht. Alle Werte: Ergänzung, volles Budget, clean, Positions-RMSE in mm.', '']
    rows = []
    for method in CFG['methods']:
        rows.append([NAMES[method]]+[ci(one(pred, world=world, noise='clean', method=method, metric='position_rmse'), 1000) for world in CFG['worlds']])
    summary += [table(['Methode', 'Maxwell mm', 'Bouc–Wen mm'], rows), '',
                '![Vorhersagevergleich der Ergänzung](figures/supplement_prediction.png)', '',
                'Modular hat deutlich kleinere mittlere Fehler als die gemeinsame NN- und MoE-Adaption. '+
                'Die Intervalle des gemeinsamen Netzes sind jedoch breit und umfassen Null in den gepaarten Architekturvergleichen; das trägt keinen belastbaren Sieg über alle neuronalen Vergleichsarchitekturen. '+
                'Numerische Begrenzungen treten beim gemeinsamen Netz auf; die gespeicherte Bedingung `numerical_guard_pass` ist deshalb falsch. '+
                'Diese Ausreißer wurden nicht entfernt. Die eigene modulare Methode erreicht keine solche Begrenzung.', '',
                '![Bereits berechnete gepaarte Unterschiede](figures/supplement_paired.png)', '',
                '## Wo die Fehler entstehen', '',
                'Konditionierte Kraftdiagnosen verwenden gemessene aktuelle lokale Eingänge und Normallast. Offene Vorhersagen müssen diese Größen selbst fortschreiben. '+
                'Beide Diagnosen werden getrennt berichtet; eine konditionierte Diagnose allein ist kein Transfernachweis. Kraft-RMSE in N, clean:', '']
    force_rows = []
    for world in CFG['worlds']:
        for method in ['analytic_flex', 'modular']:
            vals = [one(pred, world=world, noise='clean', method=method, metric=metric)['mean'] for metric in
                    ['conditional_normal_rmse', 'normal_rmse', 'conditional_friction_rmse', 'friction_rmse']]
            force_rows.append([WORLDS[world], NAMES[method]]+[f'{x:.3f}' for x in vals])
    summary += [table(['Welt', 'Methode', 'N kond.', 'N offen', 'Ft kond.', 'Ft offen'], force_rows), '',
                'Es gibt lokale Approximationsfehler schon bei gemessenen Eingängen und zusätzliche Fehlerfortpflanzung im offenen Verlauf. '+
                'Der Befund ist daher nicht ausschließlich ein Fehler der mechanischen Zusammensetzung. '+
                'Drei fest gewählte Filterzeiten (0,02/0,08/0,3 s) rekonstruieren die inneren Hysteresezustände nicht zwangsläufig vollständig. '+
                'Die Aussage betrifft diese konkrete Filter-/MLP-Repräsentation und zwei Sekunden lange Kalibrierprofile; beliebige rekurrente Zustandsmodelle sind damit nicht widerlegt.', '',
                'Bei verrauschter Kalibrierung und Anfangsmessung bleiben die Modulfehler begrenzt; ein durchgängig abgesicherter Vorsprung gegenüber der Analytik entsteht auch daraus nicht. '+
                'Die offenen Rollouts erhalten nach ihrer Anfangsmessung keine weiteren Zustandskorrekturen. Werte in mm:', '']
    noise_rows = [[WORLDS[w], NAMES[m], ci(one(pred, world=w, noise='noisy', method=m, metric='position_rmse'), 1000)]
                  for w in CFG['worlds'] for m in ['analytic_v1', 'analytic_flex', 'modular']]
    summary += [table(['Welt', 'Methode', 'Noisy: Mittel ± 95-%-CI'], noise_rows), '',
                '## Die gewöhnliche Controlleraufgabe', '',
                'Derselbe Planner bewertet 41 konstante Kräfte in ±18 N über 0,25 s und plant alle 0,05 s neu. '+
                'Ziel ist 0,06 m innerhalb einer 2,5-s-Ausführung. Erfolg verlangt im letzten 100-ms-Fenster mittleren Positionsfehler <8 mm und Geschwindigkeitsbetrag <0,03 m/s. '+
                '**Alle Methoden erreichen 100 % Erfolg**; die einfache Aufgabe trennt sie anhand dieses Kriteriums nicht. Das beweist keine Gleichwertigkeit. '+
                'Das kontinuierliche Kostenmaß berücksichtigt Position, Geschwindigkeit und Kraft. Modular-minus-Vergleich, Kosten ×10³; positiv bedeutet höhere Kosten:', '']
    task_rows = []
    for world in CFG['worlds']:
        for baseline in ['identified', 'analytic_v1', 'analytic_flex']:
            task_rows.append([WORLDS[world], NAMES[baseline]]+[difference(one(pairs, family='control', world=world, noise=noise,
                baseline=baseline, metric='cost'), 1000, 3) for noise in CFG['noise_conditions']])
    summary += [table(['Welt', 'Modular minus', 'Clean: Mittel [95-%-CI]', 'Noisy: Mittel [95-%-CI]'], task_rows), '',
                '![Getrennte Controlleraufgabe](figures/supplement_task.png)', '',
                'Ein durchgängiger Kostenvorteil gegenüber starker Analytik fehlt. Die reguläre Aufgabenlinie enthält neue lokale Zustände; '+
                'ihre schlechteste mittlere gemeinsame Nähediagnose beträgt etwa 79 %. Keine Aufgabe wurde deshalb ausgeschlossen. '+
                'Kontinuierlich gestörte Beobachtungen werden hier während der Ausführung genutzt; diese Sensorbedingung unterscheidet sich vom offenen Replay.', '',
                '## Numerik, Aufwand und Grenzen', '',
                'Unabhängig implementierte Materialgleichungen wurden mit nativer MuJoCo-Dynamik und identischen gespeicherten Kommandos wiederholt. '+
                'Die folgende Kontrolle betrifft r0. Ein kleiner Unterschied bei gleichem Zeitschritt und Konvergenz bei Verfeinerung sind verschiedene Prüfungen:', '']
    nr = []
    for _, row in native.iterrows():
        nr.append([WORLDS[row.world], f'{1000*row.h1_physx_position_max_abs_m:.5f}',
                   f'{1000*row.refinement_h1_h2_rmse_m:.3f}', f'{1000*row.refinement_h2_h4_rmse_m:.3f}'])
    summary += [table(['Welt', 'Max. PhysX–MuJoCo bei h mm', 'RMS h−h/2 mm', 'RMS h/2−h/4 mm'], nr), '',
                'Beide Backendgrenzen und Verfeinerungsregeln bestehen. Die Abweichung bei gleichem h liegt weit unter einem Millimeter; '+
                'die Änderung durch kleinere Schritte ist deutlich größer und begrenzt kleine Leistungsbehauptungen. '+
                'Das No-Go gilt für die tatsächlich eingefrorene 2,5-ms-Bedingung; es ist kein Beleg einer vom Integrator unabhängigen universellen Rangfolge. '+
                'Zwei Implementierungen derselben synthetischen Gleichungen sind keine unabhängigen realen Materialmessungen.', '',
                'Die modularen Netze benötigen 1.610 Parameter; die flexible analytische Kontrolle 109 freie Koeffizienten '+
                'zuzüglich aus Kalibrierung ermittelter Wörterbuchskalen. Größen und gemessene Fitzeiten über die ursprünglichen 40 Bedingungen je Methode:', '']
    resources = []
    for method in ['identified', 'analytic_v1', 'analytic_flex', 'modular']:
        rows = costs[costs.method == method]
        resources.append([NAMES[method], int(rows.parameters.iloc[0]), len(rows), f'{rows.seconds.mean():.3f}', f'{rows.seconds.sum():.2f}'])
    summary += [table(['Methode', 'Parameter', 'Fits', 'Sekunden/Fit', 'Sekunden gesamt'], resources), '',
                'Training erfolgte auf CPU mit zwei Torch-Threads. Zeiten enthalten Scheduling-Effekte und sind keine Energie- oder Eurokosten. '+
                'Das Supplement verwendet sämtliche Modellstände unverändert; zusätzliche Geometriedaten werden nicht zum Lernen genutzt. '+
                'Die MoE ist eine begrenzte PRISM-WM-inspirierte Adaption und keine vollständige Originalreproduktion. '+
                'Der Aufbau belegt weder echte Haftreibung, Ganzkörperstabilität, reale Datenersparnis noch Sim-to-Real-Transfer. **Der echte G1 blieb ungenutzt.**', '',
                '**Praktische Folgerung:** Diesen konkreten Ansatz vorerst nicht weiter hochskalieren. '+
                'Modulare Kräfte lassen sich wiederverwenden, doch gegenüber starker analytischer Identifikation ist hier kein durchgängiger zusätzlicher Nutzen belegt. '+
                'Ein neuer Forschungsanlauf braucht ein unabhängig motiviertes Problem, eine passende beobachtbare Zustandsrepräsentation und eine Aufgabe, deren Nutzen die Kontrollen tatsächlich unterscheiden kann.', '',
                '## Artefakte und Reproduktion', '',
                'Die beiden v2-Linien enthalten zusammen **57.600 offene Modell-Episodenauswertungen** (38.400 ursprünglich, 19.200 ergänzend), '+
                '**1.600 Controller-Episoden**, **1.920 PhysX-Referenzreplays**, **1.152 MuJoCo-Episoden** über beide Linien und drei Zeitschritte sowie **360 ursprüngliche Modell-/Identifikationsläufe**. '+
                'Diese Zählungen sind Arbeitsumfang, keine unabhängigen statistischen Stichprobengrößen. Pro Welt bleibt n=5.', '',
                'Videos: [Maxwell-Replay](../videos/maxwell_replay.mp4) · [Bouc–Wen-Replay](../videos/boucwen_replay.mp4) · '+
                '[Maxwell-Controller](../videos/maxwell_control.mp4) · [Bouc–Wen-Controller](../videos/boucwen_control.mp4). '+
                'Die Replayvideos zeigen Ergänzungsdaten r0/Episode 0; die Controllervideos zeigen die getrennte ursprüngliche Aufgabenlinie. '+
                'Es sind Visualisierungen gespeicherter Verläufe, keine zusätzlichen Messungen.', '',
                'Quellen: [Originalbericht](final_report.md), [Vorabprotokoll](experiment_protocol.md), '+
                '[Ergänzungsprotokoll](../supplement_exact/protocol.md), [gesperrte Ergänzungsentscheidung](../supplement_exact/results/decision.json), '+
                '[Vorhersagetabelle](../supplement_exact/results/prediction_summary.csv), '+
                '[gepaarte Differenzen](../supplement_exact/results/paired_differences.csv), '+
                '[Controllerstatistik](../results/report/control_summary.csv), [Referenzgleichungen und Literatur](reference_notes.md), '+
                '[vorhandene Umgebung](environment.md), [README](../README.md), [STATUS](../STATUS.md).', '',
                'Vollständiger Ablauf einschließlich Ergänzung und Abschlussdarstellung:', '',
                '```bash', 'cd /home/loya2/physics-transfer/study_v2', 'bash scripts/run_all.sh', '```', '',
                'Optional nur die Ergänzung und die Abschlussdarstellung fortsetzen:', '',
                '```bash', 'cd /home/loya2/physics-transfer/study_v2', 'bash supplement_exact/scripts/run_all.sh',
                'OPENBLAS_NUM_THREADS=1 /home/loya2/miniconda3/envs/unitree_sim_env/bin/python scripts/conclusion.py', '```', '',
                'Prüfsummen der wissenschaftlichen Sperrlisten: Original `'+original_hash+'`; Ergänzung `'+digest(sup/'lock.json')+'`. '+
                'Dieser Darstellungsbericht liest die fertigen Statistikdateien; seine Ausführung trainiert keine Modelle und ändert keine Entscheidung.', '']
    report = ROOT/'reports/conclusion.md'; report.write_text('\n'.join(summary))
    print('CONCLUSION_READY', decision['outcome'], report)


if __name__ == '__main__':
    main()
