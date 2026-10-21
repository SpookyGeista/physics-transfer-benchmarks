"""Frozen aggregation/decision rules for the v2 main study; no pilot inputs.

Run only after prediction, controller, coverage and independent checks complete.
The self-test uses synthetic numbers and does not open experiment results.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from decision_study.common import CFG, DT, ROOT, digest, dump
from decision_study.locking import verify_lock

T95 = 2.776
LABELS = dict(nominal='Nominell', identified='Identifikation', analytic_v1='Analytisch v1',
              analytic_flex='Analytisch flexibel', joint='Gemeinsames NN', moe='MoE-Adaption',
              modular='Modular', no_history='Ohne Historie', privileged='Privilegiert', decoupled='Entkoppelt')
METHODS = CFG['methods']
ANALYTIC = ['analytic_v1', 'analytic_flex']
STRONG = ANALYTIC + ['joint', 'moe']
WORLD_LABELS = dict(maxwell='Maxwell', boucwen='Bouc–Wen')
COLORS = {method: plt.get_cmap('tab10')(i) for i, method in enumerate(METHODS)}
COLORS['modular'] = '#bc342e'


def interval(values):
    values = np.asarray(values, dtype=float)
    assert len(values) == 5 and np.isfinite(values).all(), 'five finite replicate means required'
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    half = T95*sd/np.sqrt(5)
    return dict(mean=mean, sd=sd, ci95_half=half, ci95_low=mean-half, ci95_high=mean+half, n=5)


def summarize(seed_means, keys, metrics):
    rows = []
    for values, group in seed_means.groupby(keys, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        assert sorted(group['seed'].tolist()) == CFG['seeds']
        for metric in metrics:
            rows.append(dict(zip(keys, values), metric=metric, **interval(group[metric])))
    return pd.DataFrame(rows)


def paired(seed_means, keys, metrics, family):
    rows = []
    for values, group in seed_means.groupby(keys, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        for metric in metrics:
            pivot = group.pivot(index='seed', columns='method', values=metric).reindex(CFG['seeds'])
            assert set(pivot.columns) == set(METHODS) and pivot.notna().all().all()
            for baseline in METHODS:
                if baseline == 'modular':
                    continue
                differences = pivot['modular']-pivot[baseline]
                row = dict(zip(keys, values), family=family, metric=metric, method='modular',
                           baseline=baseline, **interval(differences))
                row.update({f'difference_r{seed}': float(differences.loc[seed]) for seed in CFG['seeds']})
                rows.append(row)
    return pd.DataFrame(rows)


def pick(table, **conditions):
    selected = table
    for key, value in conditions.items():
        selected = selected[selected[key] == value]
    assert len(selected) == 1, f'expected one statistic for {conditions}, found {len(selected)}'
    return selected.iloc[0]


def validate_complete(prediction, control, model_records):
    expected_replay = CFG['profiles_train']*len(CFG['cam_amplitudes'])
    for world in CFG['worlds']:
        for seed in CFG['seeds']:
            for noise in CFG['noise_conditions']:
                for method in METHODS:
                    for budget in CFG['budgets']:
                        rows = prediction[(prediction.world == world) & (prediction.seed == seed) &
                                          (prediction.noise == noise) & (prediction.method == method) & (prediction.budget == budget)]
                        assert len(rows) == expected_replay and rows.id.nunique() == expected_replay, 'incomplete or duplicated main prediction cells'
                    rows = control[(control.world == world) & (control.seed == seed) &
                                   (control.noise == noise) & (control.method == method)]
                    n = len(CFG['cam_amplitudes'])*CFG['controller']['per_condition']
                    assert len(rows) == n and rows.episode.nunique() == n, 'incomplete or duplicated controller cells'
    expected_models = len(CFG['worlds'])*len(CFG['seeds'])*len(CFG['budgets'])*len(CFG['noise_conditions'])*(len(METHODS)-1)
    assert len(model_records) == expected_models, f'expected {expected_models} completed model fits'
    assert all(record['frozen'] and not record['test_data_used'] for record in model_records)


def native_statistics(results):
    rows, sources = [], {}
    for world in CFG['worlds']:
        states = {}
        summaries = {}
        for h in [1, 2, 4]:
            base = results/'mujoco'/f'{world}_r0_h{h}'
            path, meta_path = base.with_suffix('.npz'), base.with_suffix('.json')
            assert path.exists() and meta_path.exists(), f'missing planned independent check: {base}'
            with np.load(path) as arrays:
                states[h] = arrays['next_state'][:, :, 0]
            summaries[h] = json.loads(meta_path.read_text())
            sources[str(path)] = digest(path); sources[str(meta_path)] = digest(meta_path)
        d12, d24 = states[1]-states[2], states[2]-states[4]
        h1 = summaries[1]['comparison_to_recorded_physx']['next_state']
        rm12, rm24 = float(np.sqrt(np.mean(d12**2))), float(np.sqrt(np.mean(d24**2)))
        row = dict(world=world, seed=0, h1_physx_position_rmse_m=h1['position_rmse_m'],
                   h1_physx_position_max_abs_m=h1['position_max_abs_m'],
                   refinement_h1_h2_rmse_m=rm12, refinement_h2_h4_rmse_m=rm24,
                   refinement_h1_h2_max_abs_m=float(np.max(np.abs(d12))),
                   refinement_h2_h4_max_abs_m=float(np.max(np.abs(d24))),
                   h2_physx_position_rmse_m=summaries[2]['comparison_to_recorded_physx']['next_state']['position_rmse_m'],
                   h4_physx_position_rmse_m=summaries[4]['comparison_to_recorded_physx']['next_state']['position_rmse_m'],
                   h1_contact_disagreements=summaries[1]['contact_state_disagreements'],
                   maximum_friction_power_w=max(item['maximum_friction_power_W'] for item in summaries.values()),
                   native_tolerance_pass=h1['position_max_abs_m'] <= CFG['prediction_numerical_tol'],
                   refinement_pass=rm24 < rm12,
                   seconds=sum(item['seconds'] for item in summaries.values()))
        # Conservative resolution check: even the CI-supported benefit must exceed
        # measured pointwise backend/refinement discrepancies on the checked seed.
        row['numerical_resolution_m'] = max(row['h1_physx_position_max_abs_m'],
                                            row['refinement_h1_h2_max_abs_m'], row['refinement_h2_h4_max_abs_m'])
        rows.append(row)
    return pd.DataFrame(rows), sources


def decide(pairs, support, support_summary, native, prediction):
    primary_support = support[support.budget == CFG['profiles_train']]
    coverage = primary_support.groupby(['world', 'seed'])[['normal_coverage', 'friction_coverage']].mean()
    expected_groups = len(CFG['worlds'])*len(CFG['seeds'])
    coverage_pass = bool(len(coverage) == expected_groups and (coverage.values >= CFG['coverage_required']).all() and
                         support_summary['primary_all_exact_match'] and primary_support.exact_match.eq(1).all())
    native_pass = bool(native.native_tolerance_pass.all() and native.refinement_pass.all())
    valid = coverage_pass and native_pass
    tests = []
    def stat(family, world, noise, baseline, metric):
        query = dict(family=family, world=world, noise=noise, baseline=baseline, metric=metric)
        if family == 'prediction':
            query['budget'] = CFG['profiles_train']
        return pick(pairs, **query)
    for world in CFG['worlds']:
        resolution = float(native.loc[native.world == world, 'numerical_resolution_m'].iloc[0])
        for baseline in STRONG:
            clean = stat('prediction', world, 'clean', baseline, 'position_rmse')
            noisy = stat('prediction', world, 'noisy', baseline, 'position_rmse')
            tests.append(dict(world=world, baseline=baseline, criterion='clean_prediction_superiority',
                              pass_=bool(clean.ci95_high < 0), value=float(clean.ci95_high), threshold=0.))
            tests.append(dict(world=world, baseline=baseline, criterion='prediction_numerical_resolution',
                              pass_=bool(-clean.ci95_high > resolution), value=float(-clean.ci95_high), threshold=resolution))
            tests.append(dict(world=world, baseline=baseline, criterion='noisy_prediction_no_reversal',
                              pass_=bool(noisy['mean'] <= 0), value=float(noisy['mean']), threshold=0.))
        for baseline in ANALYTIC:
            clean = stat('control', world, 'clean', baseline, 'cost')
            noisy = stat('control', world, 'noisy', baseline, 'cost')
            tests.append(dict(world=world, baseline=baseline, criterion='clean_task_cost_superiority',
                              pass_=bool(clean.ci95_high < 0), value=float(clean.ci95_high), threshold=0.))
            tests.append(dict(world=world, baseline=baseline, criterion='noisy_task_cost_no_reversal',
                              pass_=bool(noisy['mean'] <= 0), value=float(noisy['mean']), threshold=0.))
            for noise in CFG['noise_conditions']:
                success = stat('control', world, noise, baseline, 'success')
                tests.append(dict(world=world, baseline=baseline, criterion=noise+'_success_nondecrease',
                                  pass_=bool(success['mean'] >= 0), value=float(success['mean']), threshold=0.))
    guard_rows = prediction[(prediction.budget == CFG['profiles_train']) & prediction.method.isin(['modular']+STRONG)]
    guard_pass = bool(guard_rows.guard_steps.eq(0).all())
    superiority = pairs[(pairs.family == 'prediction') & (pairs.budget == CFG['profiles_train']) &
                        (pairs.noise == 'clean') & (pairs.metric == 'position_rmse') & pairs.baseline.isin(ANALYTIC)]
    inferior = superiority[superiority.ci95_low > 0]
    go = valid and guard_pass and all(item['pass_'] for item in tests)
    no_go = valid and len(inferior) > 0
    outcome = 'go' if go else ('no-go' if no_go else 'inconclusive')
    return dict(outcome=outcome, verdict_de={'go': 'Positiver begrenzter Simulationsbefund',
                'no-go': 'Negativer Befund für den stärkeren Anspruch dieser konkreten Methode',
                'inconclusive': 'Nicht entscheidbar nach der vorab festgelegten Regel'}[outcome],
                valid_pipeline=valid, coverage_pass=coverage_pass, native_pass=native_pass,
                numerical_guard_pass=guard_pass, comparisons=tests,
                analytic_inferiority= inferior[['world', 'baseline', 'mean', 'ci95_low', 'ci95_high']].to_dict('records'),
                ci_rule='paired replicate mean +/- 2.776 * sample_sd / sqrt(5); df=4; two-sided 95%',
                claim_scope='unseen cam arrangements with reused observed local histories; no real transfer',
                missing_significance_is_not_equivalence=True, real_robot_used=False)


def ci_text(row, scale=1., decimals=2):
    return f'{row["mean"]*scale:.{decimals}f} ± {row["ci95_half"]*scale:.{decimals}f}'


def markdown_table(headers, rows):
    lines = ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
    lines.extend('| '+' | '.join(str(value).replace('|', '/') for value in row)+' |' for row in rows)
    return '\n'.join(lines)


def finish_plot(fig, path):
    fig.savefig(path, dpi=170, bbox_inches='tight', facecolor='white'); plt.close(fig)


def make_figures(out, ps, cs, pairs, model_records, native, results):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .16})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, world in zip(axes, CFG['worlds']):
        rows = [pick(ps, world=world, budget=24, noise='clean', method=m, metric='position_rmse') for m in METHODS]
        ax.barh(np.arange(len(METHODS)), [r['mean']*1000 for r in rows],
                xerr=[r.ci95_half*1000 for r in rows], color=[COLORS[m] for m in METHODS], capsize=2)
        ax.set(yticks=np.arange(len(METHODS)), yticklabels=[LABELS[m] for m in METHODS],
               xlabel='Offener Positions-RMSE (mm); 95-%-t-Intervall', title=WORLD_LABELS[world]); ax.invert_yaxis()
    finish_plot(fig, out/'prediction.png')
    baselines = [m for m in METHODS if m != 'modular']
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, world in zip(axes, CFG['worlds']):
        rows = [pick(pairs, family='prediction', world=world, budget=24, noise='clean', baseline=m, metric='position_rmse') for m in baselines]
        ax.errorbar([r['mean']*1000 for r in rows], np.arange(len(rows)), xerr=[r.ci95_half*1000 for r in rows], fmt='o', capsize=3, color=COLORS['modular'])
        ax.axvline(0, color='black', lw=1)
        ax.set(yticks=np.arange(len(rows)), yticklabels=[LABELS[m] for m in baselines],
               xlabel='Modular minus Vergleich (mm); negativ ist günstiger', title=WORLD_LABELS[world]); ax.invert_yaxis()
    finish_plot(fig, out/'paired_prediction.png')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for wi, world in enumerate(CFG['worlds']):
        for hi, head in enumerate(['normal', 'friction']):
            ax = axes[wi, hi]; y = np.arange(len(METHODS))
            for offset, metric, label, color in [(-.18, 'conditional_'+head+'_rmse', 'Gemessene lokale Eingänge', '#64879a'),
                                                 (.18, head+'_rmse', 'Offener Rollout', '#d28a4b')]:
                rows = [pick(ps, world=world, budget=24, noise='clean', method=m, metric=metric) for m in METHODS]
                ax.barh(y+offset, [r['mean'] for r in rows], height=.34, xerr=[r.ci95_half for r in rows], color=color, label=label, capsize=2)
            ax.set(yticks=y, yticklabels=[LABELS[m] for m in METHODS], xlabel='Kraft-RMSE (N)',
                   title=WORLD_LABELS[world]+' · '+('Normal' if head == 'normal' else 'Reibung')); ax.invert_yaxis()
            if wi == 0 and hi == 0: ax.legend(fontsize=8)
    finish_plot(fig, out/'conditional_vs_open.png')
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for wi, world in enumerate(CFG['worlds']):
        for mi, metric in enumerate(['cost', 'success']):
            ax = axes[wi, mi]; x = np.arange(len(METHODS)); scale = 1000 if metric == 'cost' else 100
            for offset, noise, color in [(-.17, 'clean', '#557e92'), (.17, 'noisy', '#cb8241')]:
                rows = [pick(cs, world=world, noise=noise, method=m, metric=metric) for m in METHODS]
                ax.bar(x+offset, [r['mean']*scale for r in rows], width=.33, yerr=[r.ci95_half*scale for r in rows], color=color, label=noise, capsize=2)
            ax.set(xticks=x, xticklabels=[LABELS[m] for m in METHODS], ylabel='Aufgabenkosten × 10³' if metric == 'cost' else 'Erfolg (%)', title=WORLD_LABELS[world])
            plt.setp(ax.get_xticklabels(), rotation=48, ha='right')
            if wi == 0 and mi == 0: ax.legend()
    finish_plot(fig, out/'control.png')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, world in zip(axes, CFG['worlds']):
        for method in METHODS:
            rows = [pick(ps, world=world, budget=b, noise='clean', method=method, metric='position_rmse') for b in CFG['budgets']]
            ax.errorbar(CFG['budgets'], [r['mean']*1000 for r in rows], yerr=[r.ci95_half*1000 for r in rows], marker='o', label=LABELS[method], color=COLORS[method], lw=1.4, capsize=2)
        ax.set(xticks=CFG['budgets'], xlabel='Kalibrierprofile je Effekt', ylabel='Positions-RMSE (mm)', title=WORLD_LABELS[world])
    axes[-1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
    finish_plot(fig, out/'data_budget.png')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for wi, world in enumerate(CFG['worlds']):
        for ni, noise in enumerate(CFG['noise_conditions']):
            ax = axes[wi, ni]
            for method in ['joint', 'moe', 'modular', 'no_history', 'privileged']:
                records = [r for r in model_records if r['world'] == world and r['noise'] == noise and r['budget'] == 24 and r['method'] == method]
                curves = [pd.DataFrame(json.loads((Path(r['_directory'])/'learning.json').read_text())) for r in records]
                curve = pd.concat(curves).groupby('update')['val_loss'].mean()
                ax.plot(curve.index, np.maximum(curve.values, 1e-12), label=LABELS[method], color=COLORS[method])
            ax.set(yscale='log', xlabel='Optimierungsupdate', ylabel='Isolierter Validierungsverlust', title=WORLD_LABELS[world]+' · '+noise)
    axes[0, 1].legend(fontsize=8)
    finish_plot(fig, out/'learning_curves.png')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, world in zip(axes, CFG['worlds']):
        path = results/'traces'/f'{world}_modular.npz'
        with np.load(path) as stored:
            observed, predicted = stored['reference'], stored['state']
        errors = np.sqrt(np.mean((predicted[:, :, 0]-observed[:, :, 0])**2, axis=1)); episode = int(np.argmax(errors))
        t = (np.arange(predicted.shape[1])+1)*DT
        ax.plot(t, observed[episode, :, 0]*1000, 'k', label='Referenz', lw=2)
        ax.plot(t, predicted[episode, :, 0]*1000, color=COLORS['modular'], label='Modular')
        with np.load(results/'traces'/f'{world}_analytic_flex.npz') as stored:
            ax.plot(t, stored['state'][episode, :, 0]*1000, color=COLORS['analytic_flex'], label='Analytisch flexibel')
        ax.set(xlabel='Zeit (s)', ylabel='Position (mm)', title=f'{WORLD_LABELS[world]} · schlechteste modulare Episode {episode}, r0')
        ax.legend(fontsize=8)
    finish_plot(fig, out/'failure_cases.png')


def build_report(decision, ps, cs, pairs, support, control_support, native, models, prediction, control, out):
    lines = ['# Abschlussbericht — Entscheidungsversuch v2', '',
             'Automatisch aus den gesperrten Hauptmessungen erstellt: '+datetime.now(timezone.utc).isoformat(timespec='seconds')+'.', '',
             '**'+decision['verdict_de']+'.** Entscheidung: **'+decision['outcome'].upper()+'**.', '']
    if decision['outcome'] == 'no-go':
        findings = []
        for row in decision['analytic_inferiority']:
            findings.append(f'{WORLD_LABELS[row["world"]]} gegenüber {LABELS[row["baseline"]]}: '+
                            f'{1000*row["mean"]:.2f} mm höhere Fehler, 95-%-CI [{1000*row["ci95_low"]:.2f}, {1000*row["ci95_high"]:.2f}] mm')
        lines += ['Die modulare Methode ist in mindestens einem gültigen Hauptvergleich klar ungenauer als eine starke analytische Kontrolle. '+
                  '; '.join(findings)+'. Dies spricht gegen das Hochskalieren dieser konkreten Filter-/Netzparametrisierung. Es beweist weder Gleichwertigkeit anderer Methoden noch die generelle Nutzlosigkeit modularer Modelle.', '']
    elif decision['outcome'] == 'go':
        lines += ['Alle vorab festgelegten Vorhersage-, Aufgaben-, Rausch- und Gültigkeitskriterien sind erfüllt. Der Befund rechtfertigt einen größeren, weiterhin begrenzten Simulationsversuch. Er belegt weder Realtransfer noch allgemeine Material- oder Trajektoriengeneralisation.', '']
    else:
        failed = [name for name in ['coverage_pass', 'native_pass', 'numerical_guard_pass'] if not decision[name]]
        lines += ['Die vollständige Go-Regel ist nicht erfüllt; der gültige eindeutige analytische Gegenbefund für ein No-Go liegt ebenfalls nicht vor. '+
                  ('Nicht erfüllte Gültigkeits-/Numerikprüfungen: '+', '.join(failed)+'. ' if failed else '')+
                  'Breite oder nullüberdeckende Intervalle sind kein Äquivalenznachweis. Vor weiterer Skalierung muss die offene Information oder Aufgabenrelevanz geklärt werden.', '']
    lines += ['## Was dieser Versuch prüft', '',
              'Ein nativer CPU-PhysX-Schlitten mit 2 kg wird durch programmierte Maxwell- beziehungsweise Bouc–Wen-Materialkräfte belastet. Eine bekannte beschränkte Nocke koppelt Kompression und Bewegung über `δ=d+g(x)` und die rückwirkende Kraft `−g′(x)N`. Native Materialkontakte sind abgeschaltet; es wird kein interner PhysX-Kontaktlöser gelernt.', '',
              'B misst die isolierte Nachgiebigkeit. A spielt anschließend die in B gemessene Normallast als kontrollierte Last in einem getrennten Reibungsversuch ein. Der Haupttest verbindet **dieselben vollständigen lokalen Profile** durch vier zuvor ungesehene Nockengeometrien. Ein Referenzorakel erzeugt die Aktionsfolgen einmalig; anschließend erhalten alle Modelle identische gespeicherte Kommandos und rollen offen. Das ist ein Test neuer Anordnungen bei wiederverwendeten lokalen Historien. Er ist kein Holdout neuer lokaler Trajektorien. Die gewöhnliche Controllerlinie verwendet eigene Kraftkommandos und neue Zustandsverläufe.', '',
              'Fünf unabhängige Kalibrierdatensätze mit gepaarten Trainingsseeds je Welt; alle Methoden teilen innerhalb eines Replikats dieselben Daten, Masken, Störungen und Aufgaben. Die beiden MLP-Module werden separat gelernt und eingefroren. Gemeinsames Netz und MoE besitzen vergleichbare Kapazität. Analytisch v1 übernimmt die frühere Wörterbuchfamilie mit neu identifizierten Koeffizienten; die flexible Kontrolle verwendet 109 systematische Polynombasiskoeffizienten und allein auf isolierter Validierung ausgewählte Ridge-Stärken. Keine Referenzparameter oder zukünftigen Testkräfte gelangen in die offenen Prädiktoren.', '',
              'Die MoE ist eine eingeschränkte PRISM-WM-inspirierte Adaption, **keine vollständige Reproduktion**. Separierbare lokale Materialgesetze und bekannte Mechanik sind vorgegebenes Strukturwissen. Der echte G1 und seine Kommunikationsschnittstellen wurden nicht verwendet.', '',
              '## Vollständigkeit und Statistik', '',
              f'{len(models)} abgeschlossene Modell-/Identifikationsläufe, {len(prediction)} ausgewertete offene Modell-Episoden und {len(control)} gewöhnliche Controller-Episoden. '+
              f'Je Welt/Replikat: {CFG["profiles_train"]} Profile × {len(CFG["cam_amplitudes"])} neue Geometrien im Replay. Am vollen Budget: 48 isolierte A/B-Trajektorien beziehungsweise 96 s Kalibrierung, zusätzlich 16 Validierungstrajektorien beziehungsweise 32 s. Privilegiert erhält acht weitere AB-Trajektorien beziehungsweise 16 s.', '',
              'Die statistische Einheit ist der Mittelwert eines vollständigen Replikats, **n=5**. Episoden und Zeitpunkte zählen nicht als unabhängige Trainingswiederholungen. Angegeben ist jeweils Mittelwert ± halbe Breite des zweiseitigen 95-%-t-Intervalls (`2.776·s/√5`, vier Freiheitsgrade). Unterschiede werden aus denselben fünf gepaarten Replikaten berechnet. Einzelintervalle sind keine simultanen familienweiten Intervalle; fünf Replikate begrenzen die Auflösung.', '',
              '## Messwerte für alle Verfahren', '',
              'Position in mm; Aufgabenkosten mit Faktor 10³, Erfolg in Prozent. Das Kostenmaß ist die protokollierte gewichtete Positions-/Geschwindigkeits-/Kraft-Zielfunktion und keine Energie in Joule. Budget 8 enthält nur acht der 24 wiederholten lokalen Profile und ist eine **sekundäre Bedingung mit unvollständiger Abdeckung**. Die Controller verwenden ausschließlich Budget 24.', '']
    for world in CFG['worlds']:
        for noise in CFG['noise_conditions']:
            rows = []
            for method in METHODS:
                p8 = pick(ps, world=world, budget=8, noise=noise, method=method, metric='position_rmse')
                p24 = pick(ps, world=world, budget=24, noise=noise, method=method, metric='position_rmse')
                cost = pick(cs, world=world, noise=noise, method=method, metric='cost')
                success = pick(cs, world=world, noise=noise, method=method, metric='success')
                rows.append([LABELS[method], ci_text(p8, 1000), ci_text(p24, 1000), ci_text(cost, 1000, 3), ci_text(success, 100, 1)])
            lines += ['### '+WORLD_LABELS[world]+' · '+noise, '', markdown_table(['Methode', 'Position b8', 'Position b24', 'Kosten ×10³', 'Erfolg %'], rows), '']
    lines += ['![Offene Hauptvorhersage](figures/prediction.png)', '', '![Gepaarte Hauptvergleiche](figures/paired_prediction.png)', '',
              'Primäre Unterschiede der modularen Methode gegenüber den vier starken Vergleichen, clean/b24. Negative Werte bedeuten geringeren Positionsfehler:', '']
    rows = []
    for world in CFG['worlds']:
        for baseline in STRONG:
            row = pick(pairs, family='prediction', world=world, budget=24, noise='clean', baseline=baseline, metric='position_rmse')
            rows.append([WORLD_LABELS[world], LABELS[baseline], f'{row["mean"]*1000:.3f}', f'[{row.ci95_low*1000:.3f}, {row.ci95_high*1000:.3f}]'])
    lines += [markdown_table(['Welt', 'Modular minus', 'Differenz mm', '95-%-CI mm'], rows), '',
              'Alle Paarungen, darunter Rauschbedingungen, Kraftmetriken, beide Budgets und Aufgabenkosten, stehen in [paired_differences.csv](../results/report/paired_differences.csv); dort bleiben auch die fünf einzelnen gepaarten Differenzen erhalten.', '',
              '## Kraftdiagnose, Rauschen und Aufgabennutzen', '',
              'Die konditionierte Kraftdiagnose verwendet wahre aktuelle lokale Zustände und gemessene Normallast. Die offenen Rollouts verwenden ab Initialisierung eigene Zustände und vorhergesagte Lasten samt Historie. Eine gute konditionierte Kraftvorhersage allein ist kein Transfernachweis. Die Differenz beider Diagnosen hilft, lokale Approximation und Fehlerfortpflanzung auseinanderzuhalten.', '',
              '![Konditionierte und offene Kräfte](figures/conditional_vs_open.png)', '',
              'In `noisy` werden Kalibrierung und Validierung samt kausaler Historie neu aus gestörten Signalen aufgebaut: 0,05 mm Weg, 0,002 m/s Geschwindigkeit, 0,001 m/s Rate, 0,15 N Normal- und 0,10 N Reibkraft. Im offenen Replay gibt es nur eine anfängliche Messstörung, anschließend keine laufende Zustandskorrektur. Der Controller erhält während der gesamten Ausführung gestörte Position, Kompression, Geschwindigkeit und Normallast. Reibkraftmessung und zusätzliche Ratenstörung werden dort nicht benutzt.', '',
              'Der unveränderte Planner prüft 41 konstante Kräfte in ±18 N über 0,25 s und plant alle 0,05 s neu. Eine Aufgabe dauert 2,5 s; Erfolg verlangt im letzten 100-ms-Fenster mittleren Positionsfehler <8 mm und mittleren Geschwindigkeitsbetrag <0,03 m/s am Ziel 0,06 m. Gleiche Erfolgsquoten belegen keine Gleichwertigkeit und ersetzen den kontinuierlichen Kostenvergleich nicht.', '',
              '![Controllerkosten und Erfolg](figures/control.png)', '',
              '## Abdeckung, numerische Grenzen und unabhängige Kontrolle', '']
    full = support[support.budget == 24]
    coverage_rows = []
    for world in CFG['worlds']:
        sub = full[full.world == world]
        seed_cov = sub.groupby('seed')[['normal_coverage', 'friction_coverage']].mean()
        coverage_rows.append([WORLD_LABELS[world], f'{int(sub.exact_match.sum())}/{len(sub)}',
                              f'{100*seed_cov.normal_coverage.min():.3f}', f'{100*seed_cov.friction_coverage.min():.3f}',
                              f'{sub.exact_normal_feature_max_abs.max():.3g}', f'{sub.exact_friction_feature_max_abs.max():.3g}'])
    lines += [markdown_table(['Welt', 'Exakte Episoden', 'Min. Seed-Abdeckung N %', 'Min. Seed-Abdeckung Ft %', 'Max. Featurefehler N', 'Max. Featurefehler Ft'], coverage_rows), '',
              'Die direkte Prüfung vergleicht aufgezeichnete B-/A-Messungen anhand von Profil-IDs einschließlich aller erlaubten Historien. Sie verwendet keine idealen Sollprofile als Messersatz. Abstandsgrenzen stammen ausschließlich aus unabhängiger isolierter Validierung. Erforderlich sind alle direkten Übereinstimmungen und ≥99 % Abdeckung je Kopf/Welt/Replikat; die Abstandsdistanz selbst bleibt eine empirische Nähediagnose.', '']
    control_cov = control_support.groupby(['world', 'noise', 'method'])[['normal_coverage', 'friction_coverage', 'joint_coverage']].mean().reset_index()
    minima = control_cov.sort_values('joint_coverage').iloc[0]
    lines += [f'Die gewöhnlichen Controllerverläufe erreichen im ungünstigsten Methoden-/Welt-/Rauschmittel eine gemeinsame lokale Abdeckung von {100*minima.joint_coverage:.1f} % '+
              f'({WORLD_LABELS[minima.world]}, {LABELS[minima.method]}, {minima.noise}). '+
              ('Damit enthält die Aufgabenlinie relevante neue lokale Zustandsbereiche; ihr Befund isoliert Komposition nicht vollständig von lokaler Extrapolation. ' if minima.joint_coverage < .99 else 'Die empirische Aufgabenabdeckung ist hoch, aber keine vollständige Zustandsraumgarantie. ')+
              'Keine Controller-Episode wurde wegen geringer Abdeckung ausgeschlossen. Einzelheiten: [control_support_metrics.csv](../results/control_support_metrics.csv).', '']
    guards = prediction[prediction.guard_steps > 0]
    lines += [f'{len(guards)} von {len(prediction)} offenen Modell-Episoden aktivierten mindestens eine numerische Begrenzung; insgesamt {int(prediction.guard_steps.sum())} betroffene Schritte. '+
              'Grenzen: |x|>2 m, |v|>20 m/s oder |N|/|Ft|>10.000 N sowie nichtendliche Werte. Fehler gehen nach der vorab festgelegten begrenzten Fortsetzung in die Metriken ein; keine Episode wird entfernt. Begrenzte Verläufe sind keine physikalisch gültigen Rollouts.', '']
    rows = [[WORLD_LABELS[row.world], f'{row.h1_physx_position_max_abs_m*1000:.6f}',
             f'{row.refinement_h1_h2_rmse_m*1000:.6f}', f'{row.refinement_h2_h4_rmse_m*1000:.6f}',
             str(bool(row.native_tolerance_pass and row.refinement_pass))] for _, row in native.iterrows()]
    lines += [markdown_table(['Welt (r0)', 'Max. PhysX–MuJoCo h1 mm', 'RMS h1–h2 mm', 'RMS h2–h4 mm', 'Native Prüfung erfüllt'], rows), '',
              'MuJoCo implementiert dieselben synthetischen Gesetze separat und spielt exakt dieselben gespeicherten Kommandos ab. Die native h1-Maximalabweichung muss ≤0,1 mm bleiben; die Verfeinerungsdifferenz h2–h4 muss kleiner als h1–h2 sein. Eine wachsende Differenz feiner Schritte zum festen groben PhysX-Log ist für sich kein Konvergenzfehler. Diese Prüfung betrifft r0 und liefert keine unabhängige reale Materialwahrheit. Ein Go verlangt zusätzlich, dass der vom Intervall gestützte Vorhersagevorteil die gemessenen numerischen Maximalabweichungen übersteigt.', '',
              '## Datenbudget, Lernverlauf und Kosten', '',
              '![Datenbudget](figures/data_budget.png)', '',
              '![Isolierte Validierungsverläufe](figures/learning_curves.png)', '',
              'Zwei Datenbudgets sind eine begrenzte Datenkontrolle und kein allgemeines Skalierungsgesetz. Die dargestellten Lernverläufe mitteln die isolierten Validierungsverluste über die fünf Replikate; Checkpointauswahl erfolgte ausschließlich auf diesen Validierungen. Der privilegierte Vergleich besitzt zusätzliche Kombinationslabels und ist keine Methode ohne Anpassungsdaten.', '']
    cost_rows = []
    for method in [m for m in METHODS if m != 'decoupled']:
        selected = [r for r in models if r['method'] == method]
        cost_rows.append([LABELS[method], selected[0]['parameters'], len(selected),
                          f'{np.mean([r["seconds"] for r in selected]):.3f}', f'{sum(r["seconds"] for r in selected):.2f}'])
    lines += [markdown_table(['Methode', 'Parameter', 'Fits', 'Sekunden/Fit', 'Sekunden gesamt'], cost_rows), '',
              f'Gesamte protokollierte Fit-/Identifikationszeit: {sum(r["seconds"] for r in models):.2f} s. Training: CPU, zwei Torch-Threads; Modellevaluation: ein Torch-Thread. Entkoppelt verwendet dieselben modularen Gewichte und erzeugt keinen zusätzlichen Fit. '+
              'Gemessene Zeiten enthalten Scheduling-Effekte und sind keine Energie- oder Eurokosten. Zusätzlich zu den 109 freien flexiblen Koeffizienten werden kalibrierungsbasierte Wörterbuchmittel/-skalen gespeichert. Ridge-Auswahl und Einzelmetadaten stehen in [model_metadata.csv](../results/report/model_metadata.csv).', '',
              '## Fehlerfälle, Grenzen und Fortsetzung', '',
              '![Vorab definierte Fehlfallauswahl](figures/failure_cases.png)', '',
              'Gezeigt wird je Welt ausschließlich die Episode mit größtem modularem Positions-RMSE bei r0, Budget 24, clean; die Vergleichskurve ist vorab festgelegt auf Analytisch flexibel. Die Auswahl dient der Fehlerdiagnose und verändert keine Hauptauswertung.', '',
              'Der Versuch betrifft zwei synthetische Materialfamilien, drei kausale Filterzeiten, einen geführten Körper und bekannte Geometrie. Ein schlechteres Ergebnis kann die konkrete Zustandsrepräsentation betreffen und widerlegt keine beliebige rekurrente Materialmethode. Es gibt keine Aussage über echte Haftreibung, Ganzkörperstabilität, reale Datenersparnis, Herstellersteuerung oder Sim-to-Real-Transfer.', '']
    if decision['outcome'] == 'no-go':
        recommendation = 'Diesen konkreten Lernansatz vorerst zurückstellen. Die starke analytische Identifikation als Arbeitsmodell behalten; weitere Forschung zuerst auf ein unabhängig motiviertes Problem richten, in dem isolierte Beobachtbarkeit und zusätzlicher Aufgabennutzen plausibel sind. Ein unverändertes Hochskalieren wird durch diesen Versuch nicht gestützt.'
    elif decision['outcome'] == 'go':
        recommendation = 'Den begrenzten Vorteil mit unabhängig gewählten Materialfällen, zusätzlicher Zustandsinformation und neuen Aufgaben replizieren. Erst danach wären ein weitergehender Robotikversuch oder ein größerer methodischer Anspruch begründet.'
    else:
        recommendation = 'Die fehlenden Gültigkeits- oder Informationsnachweise klären und eine frische zurückgehaltene Folgeprüfung festlegen. Keine Architektur- oder Referenzänderung nach Einsicht dieser Haupttestwerte als bestätigendes Ergebnis desselben Protokolls ausgeben.'
    lines += ['**Empfehlung:** '+recommendation, '',
              'Reproduktion und Einzelregeln: [README](../README.md), [STATUS](../STATUS.md), [Vorabprotokoll](experiment_protocol.md), [Supportprotokoll](support_protocol.md), [Referenzen](reference_notes.md). '+
              'Maschinenlesbare Entscheidung: [decision.json](../results/decision.json). Aggregierte Tabellen und alle Einzel-Replikatmittel: [Ergebnistabellen](../results/report/). '+
              'Videos zeigen gespeicherte Referenz-/Vorhersageverläufe und dienen der Darstellung, nicht als neue Messungen.', '']
    out.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        assert interval([1, 1, 1, 1, 1])['ci95_half'] == 0
        assert np.isclose(interval([1, 2, 3, 4, 5])['ci95_half'], T95*np.std([1, 2, 3, 4, 5], ddof=1)/np.sqrt(5))
        records = [dict(world='w', seed=s, method=m, value=float(s)+(0 if m == 'modular' else 2)) for s in CFG['seeds'] for m in METHODS]
        differences = paired(pd.DataFrame(records), ['world'], ['value'], 'synthetic')
        assert differences['mean'].eq(-2).all() and differences.ci95_half.eq(0).all()
        print('REPORT_SELF_TEST_PASS: paired replicate aggregation and fixed df4 interval; no study data read')
        return
    lock_hash = verify_lock()
    results = ROOT/'results'; report_dir = results/'report'; report_dir.mkdir(exist_ok=True)
    prediction = pd.read_csv(results/'prediction_metrics.csv'); control = pd.read_csv(results/'control_metrics.csv')
    support = pd.read_csv(results/'support_metrics.csv'); control_support = pd.read_csv(results/'control_support_metrics.csv')
    support_summary = json.loads((results/'support/summary.json').read_text())
    models = []
    for path in sorted((results/'models').glob('*/r*/b*/*/*/metadata.json')):
        record = json.loads(path.read_text()); record['_directory'] = str(path.parent); models.append(record)
    validate_complete(prediction, control, models)
    assert np.isfinite(prediction[['position_rmse', 'velocity_rmse', 'normal_rmse', 'friction_rmse', 'guard_steps']].values).all()
    assert np.isfinite(control[['cost', 'success', 'tracking_rmse']].to_numpy(dtype=float)).all()
    pm = ['position_rmse', 'velocity_rmse', 'normal_rmse', 'friction_rmse', 'conditional_normal_rmse', 'conditional_friction_rmse', 'final_error', 'guard_steps', 'seconds_per_episode']
    cm = ['cost', 'success', 'terminal_position_error', 'terminal_velocity', 'tracking_rmse']
    pred_seed = prediction.groupby(['world', 'seed', 'budget', 'noise', 'method'], as_index=False)[pm].mean()
    control_seed = control.groupby(['world', 'seed', 'noise', 'method'], as_index=False)[cm].mean()
    ps = summarize(pred_seed, ['world', 'budget', 'noise', 'method'], pm)
    cs = summarize(control_seed, ['world', 'noise', 'method'], cm)
    pairs = pd.concat([paired(pred_seed, ['world', 'budget', 'noise'], pm, 'prediction'),
                       paired(control_seed, ['world', 'noise'], cm, 'control')], ignore_index=True)
    native, native_sources = native_statistics(results)
    decision = decide(pairs, support, support_summary, native, prediction)
    decision['protocol_lock_sha256'] = lock_hash
    dump(results/'decision.json', decision)
    for name, table in [('prediction_seed_means', pred_seed), ('control_seed_means', control_seed),
                        ('prediction_summary', ps), ('control_summary', cs), ('paired_differences', pairs), ('native_summary', native)]:
        table.to_csv(report_dir/(name+'.csv'), index=False)
    pd.DataFrame([{key: r[key] for key in ['world', 'seed', 'budget', 'noise', 'method', 'parameters', 'seconds', 'optimizer_updates', 'max_rss_kib']} |
                 {'selected': json.dumps(r['selected'], sort_keys=True), 'fingerprint': r['fingerprint'], 'directory': r['_directory']} for r in models]).to_csv(report_dir/'model_metadata.csv', index=False)
    make_figures(ROOT/'reports/figures', ps, cs, pairs, models, native, results)
    build_report(decision, ps, cs, pairs, support, control_support, native, models, prediction, control, ROOT/'reports/final_report.md')
    inputs = [results/name for name in ['prediction_metrics.csv', 'control_metrics.csv', 'support_metrics.csv', 'control_support_metrics.csv', 'support/summary.json']]
    inputs += list((results/'models').glob('*/r*/b*/*/*/metadata.json'))
    dump(report_dir/'manifest.json', dict(protocol_lock_sha256=lock_hash, report_script_sha256=digest(__file__),
         inputs={str(path): digest(path) for path in inputs} | native_sources,
         output_report_sha256=digest(ROOT/'reports/final_report.md'), counts=dict(model_fits=len(models), prediction_episodes=len(prediction), controller_episodes=len(control))))
    print('REPORT_COMPLETE', decision['outcome'], ROOT/'reports/final_report.md')


if __name__ == '__main__':
    main()
