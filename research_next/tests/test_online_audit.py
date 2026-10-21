"""Independent recurrent/online audit. CPU-only; original models are read-only.

pytest runs invariant tests. Direct execution writes the audit report and
recomputes local development metrics; no combined/new confirmation is opened.
Known adapter limitations are explicit strict xfails, not silently discarded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recurrent import Recurrent, inputs
from online import RecurrentOnline, Nominal, clone_state, motion_step


def nontrivial_model():
    torch.manual_seed(113)
    model = Recurrent(hidden=7).eval()
    with torch.no_grad():
        for head in (model.normal_head, model.friction_head):
            head.readout[-1].weight.normal_(0, 0.04)
            head.readout[-1].bias.fill_(0.15)
    return model


def adapter(model):
    obj = RecurrentOnline.__new__(RecurrentOnline)
    obj.model = model
    return obj


def test_both_head_prefix_causality_and_batch_permutation():
    model = nontrivial_model()
    generator = torch.Generator().manual_seed(41)
    for dim, sequence in ((2, model.normal_sequence), (3, model.friction_sequence)):
        x = torch.randn(4, 30, dim, generator=generator)
        x[..., 0] = 1 if dim == 2 else x[..., 0]
        if dim == 3:
            x[..., 2] = 0.8
        y = sequence(x)
        changed = x.clone()
        changed[:, 15:] += 0.7
        torch.testing.assert_close(y[:, :15], sequence(changed)[:, :15])
        perm = torch.tensor([2, 0, 3, 1])
        torch.testing.assert_close(y[perm], sequence(x[perm]))
        for i in range(4):
            torch.testing.assert_close(y[i:i+1], sequence(x[i:i+1]), atol=2e-6, rtol=2e-6)


def test_online_matches_sequences_at_declared_zero_rate_start():
    model = nontrivial_model()
    online = adapter(model)
    rng = np.random.default_rng(42)
    delta = 0.02+rng.uniform(-0.003, 0.003, (3, 18))
    rate = rng.normal(0, 0.04, (3, 18)); rate[:, 0] = 0
    velocity = rng.normal(0, 0.06, (3, 18))
    sensor = rng.uniform(8, 18, (3, 18))
    xn = torch.tensor(np.stack((delta/.02, rate/.1), -1), dtype=torch.float32)
    xf = torch.tensor(np.stack((velocity/.5, np.abs(velocity)/.5, sensor/20), -1), dtype=torch.float32)
    expected_n = model.normal_sequence(xn).detach().numpy()
    expected_f = model.friction_sequence(xf).detach().numpy()
    ns = online.initial(delta[:, 0], velocity[:, 0], sensor[:, 0])
    fs = clone_state(ns)
    for t in range(18):
        n, _, ns = online.step(delta[:, t], rate[:, t], velocity[:, t], ns)
        n_sensor, f, fs = online.step(delta[:, t], rate[:, t], velocity[:, t], fs,
                                     normal_override=sensor[:, t])
        np.testing.assert_allclose(n, expected_n[:, t], atol=3e-6, rtol=2e-6)
        np.testing.assert_allclose(f, expected_f[:, t], atol=3e-6, rtol=2e-6)
        np.testing.assert_array_equal(n_sensor, sensor[:, t])
        np.testing.assert_array_equal(ns['normal'], fs['normal'])


def test_online_general_initial_rate_should_match_direct_sequence():
    model = nontrivial_model(); online = adapter(model)
    delta, rate, velocity, normal = np.array([.02]), np.array([.08]), np.array([.03]), np.array([14.])
    state = online.initial(delta, velocity, normal, rate=rate)
    actual = online.step(delta, rate, velocity, state)[0]
    x = torch.tensor([[[1., .8]]])
    expected = model.normal_sequence(x).detach().numpy()[:, 0]
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_measurement_rights_and_state_copies():
    online = adapter(nontrivial_model())
    delta, rate, v = np.full(4, .02), np.full(4, .02), np.array([-.06, -.02, .01, .08])
    state = online.initial(delta, v, np.full(4, 13.))
    before = clone_state(state)
    n1, f1, h1 = online.step(delta, rate, v, state, normal_override=np.full(4, 5.))
    n2, f2, h2 = online.step(delta, rate, v, state, normal_override=np.full(4, 18.))
    np.testing.assert_array_equal(n1, np.full(4, 5.))
    np.testing.assert_array_equal(n2, np.full(4, 18.))
    np.testing.assert_array_equal(h1['normal'], h2['normal'])
    assert not np.allclose(f1, f2) and not np.allclose(h1['friction'], h2['friction'])
    for name in state:
        np.testing.assert_array_equal(state[name], before[name])
        assert not np.shares_memory(h1[name], state[name])
    expanded = clone_state(state, 3)
    for name in state:
        np.testing.assert_array_equal(expanded[name], np.repeat(state[name], 3, axis=0))
        expanded[name][0] += 1
        np.testing.assert_array_equal(state[name], before[name])


def test_physical_nonnegative_normal_and_dissipative_tangent():
    model = nontrivial_model()
    x = torch.tensor([[[-1., 1.], [0., 1.], [.2, -1.], [1., 0.]]])
    n = model.normal_sequence(x)
    assert torch.all(n >= 0) and torch.all(n[:, :2] == 0)
    rng = torch.Generator().manual_seed(16)
    f = torch.randn(3, 80, 3, generator=rng)
    f[..., 2] = torch.abs(f[..., 2])
    ft = model.friction_sequence(f)
    assert torch.all(ft*f[..., 0] <= 1e-7)


@pytest.mark.xfail(strict=True, reason="Viscous drag remains nonzero without contact/load")
def test_unloaded_contact_should_have_zero_tangential_force():
    model = nontrivial_model()
    x = torch.tensor([[[.4, .4, 0.], [-.4, .4, 0.]]])
    torch.testing.assert_close(model.friction_sequence(x), torch.zeros(1, 2), atol=1e-8, rtol=0)


@pytest.mark.xfail(strict=True, reason="normal_override is returned without nonnegative/contact projection")
def test_negative_measured_override_should_not_be_a_tensile_normal_force():
    online = adapter(nontrivial_model())
    state = online.initial(np.array([.02]), np.array([.01]), np.array([10.]))
    normal = online.step(np.array([.02]), np.array([0.]), np.array([.01]), state,
                         normal_override=np.array([-1.]))[0]
    assert np.all(normal >= 0)


def test_motion_adapter_matches_declared_semimplicit_mechanics():
    state = np.array([[.012, .07], [-.009, -.04], [0., 0.]])
    amplitude = np.array([.005, -.005, .005])
    d = np.array([.019, .022, .02]); rate = np.array([.01, -.03, 0.])
    u = np.array([3., -2., 1.])
    x, v = state.T; z = np.tanh(x/.04); slope = amplitude*(1-z*z)/.04
    n = np.maximum(500*(d+amplitude*z), 0)
    ft = -.35*n*np.tanh(v/.04)
    vn = v+.0025*(u-slope*n+ft)/2
    expected = np.column_stack((x+.0025*vn, vn))
    actual, _, an, af, guard = motion_step(Nominal(), state, {}, u, d, rate, amplitude)
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    np.testing.assert_allclose(an, n, atol=1e-15)
    np.testing.assert_allclose(af, ft, atol=1e-15)
    assert not guard.any()


def test_real_optimizer_steps_reach_recurrent_weights():
    torch.manual_seed(22); model = Recurrent()
    optimizer = torch.optim.Adam(model.parameters(), lr=.002, weight_decay=1e-6)
    x = torch.rand(3, 32, 2); x[..., 0] += .5
    f = torch.rand(3, 32, 3); f[..., 0] -= .5
    old = model.normal_head.cell.weight_hh_l0.detach().clone()
    for _ in range(3):
        loss = ((model.normal_sequence(x)-14)/20).square().mean() + ((model.friction_sequence(f)+1)/10).square().mean()
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
    assert {int(s['step']) for s in optimizer.state.values()} == {3}
    assert torch.max(torch.abs(old-model.normal_head.cell.weight_hh_l0)).item() > 1e-7
    assert model.normal_head.cell.weight_hh_l0.grad.abs().sum().item() > 0


def test_planner_only_uses_current_normal_once_and_preserves_observer():
    sys.path.insert(0, str(ROOT/'scripts'))
    from run_model_development import plan
    from controller_preflight import FORCES

    class Probe:
        def __init__(self):
            self.overrides = []
        def step(self, delta, rate, v, state, normal_override=None, dt=.0025):
            self.overrides.append(None if normal_override is None else normal_override.copy())
            normal = np.full(len(delta), 12.) if normal_override is None else normal_override.copy()
            return normal, -.35*normal*np.tanh(v/.04), {'counter':state['counter']+1}

    probe = Probe()
    state = np.array([[.001, .02], [-.002, -.01]])
    history = {'counter':np.array([[4.], [6.]])}
    before = clone_state(history)
    d = np.full((2, 5), .02)
    n = np.array([9., 17.])
    plan(probe, state, history, n, d, d*0, d+.01, np.array([.005, -.005]))
    np.testing.assert_array_equal(probe.overrides[0], np.repeat(n, len(FORCES)))
    assert all(x is None for x in probe.overrides[1:])
    np.testing.assert_array_equal(history['counter'], before['counter'])


def evaluate_local(model_root, data_root):
    """Recompute selected local validation and train-only mirror-pair diagnostics."""
    results = []
    for world in ('maxwell', 'boucwen'):
        checkpoint = torch.load(model_root/world/'model.pt', map_location='cpu', weights_only=True)
        model = Recurrent(checkpoint['hidden']); model.load_state_dict(checkpoint['state']); model.eval()
        result = dict(world=world, model_root=str(model_root), data_root=str(data_root))
        for split in ('val', 'train'):
            path = data_root/world/'r0'/f'{split}.npz'
            with np.load(path) as z:
                data = {k:z[k] for k in z.files}
            rows = json.loads(path.with_suffix('.json').read_text())['rows']
            bi = np.array([i for i, r in enumerate(rows) if r['mode'] == 'B'])
            ai = np.array([i for i, r in enumerate(rows) if r['mode'] == 'A'])
            with torch.inference_mode():
                pn = model.normal_sequence(torch.tensor(inputs(data, 'normal')[bi])).numpy()
                pf = model.friction_sequence(torch.tensor(inputs(data, 'friction')[ai])).numpy()
            en = pn-data['normal'][bi]; ef = pf-data['friction'][ai]
            if split == 'val':
                result.update(normal_val_rmse=float(np.sqrt(np.mean(en**2))),
                              friction_val_rmse=float(np.sqrt(np.mean(ef**2))),
                              val_normal_shape=list(pn.shape), val_friction_shape=list(pf.shape))
                selected = json.loads((model_root/world/'metadata.json').read_text())['selected_update']
                log = next(x for x in json.loads((model_root/world/'learning.json').read_text()) if x['update'] == selected)
                result['logged_selected_normal_rmse'] = log['normal_rmse']
                result['logged_selected_friction_rmse'] = log['friction_rmse']
                t = np.arange(pn.shape[1])*.0025
                masks = np.zeros_like(pn, dtype=bool)
                for j, i in enumerate(bi):
                    for start, end in rows[i].get('normal_hold_intervals', []):
                        masks[j] |= (t >= start+.5) & (t < end)
                result['hold_samples_after_0p5s'] = int(masks.sum())
                if masks.any():
                    result['normal_hold_rmse'] = float(np.sqrt(np.mean(en[masks]**2)))
                    result['normal_hold_bias'] = float(np.mean(en[masks]))
                    fm = np.array([masks[next(j for j, b in enumerate(bi) if rows[b]['profile'] == rows[a]['profile'])] for a in ai])
                    result['friction_during_normal_hold_rmse'] = float(np.sqrt(np.mean(ef[fm]**2)))
            elif rows[bi[0]].get('kind') == 'mirrored_preload_pair':
                t = np.arange(pn.shape[1])*.0025
                mask = (t >= 1.5) & (t < 4.)
                gap = data['normal'][bi[0]]-data['normal'][bi[1]]
                predicted_gap = pn[0]-pn[1]
                result['train_mirror_hold_true_gap_mean'] = float(gap[mask].mean())
                result['train_mirror_hold_predicted_gap_mean'] = float(predicted_gap[mask].mean())
                result['train_mirror_hold_gap_rmse'] = float(np.sqrt(np.mean((predicted_gap[mask]-gap[mask])**2)))
                result['train_mirror_samples'] = [dict(time=t[idx], truth_gap=float(gap[idx]), predicted_gap=float(predicted_gap[idx]))
                    for idx in [round(1.5/.0025), round(2/.0025), round(3/.0025), round(4/.0025)-1]]
        results.append(result)
    return results


def training_evidence():
    evidence = []
    for root in (ROOT/'results/recurrent_diagnostic', ROOT/'results/recurrent_development_1500'):
        for world in ('maxwell', 'boucwen'):
            meta = json.loads((root/world/'metadata.json').read_text())
            saved = torch.load(root/world/'continuation.pt', map_location='cpu', weights_only=False)
            evidence.append(dict(folder=root.name, world=world, last_zero_based_update=saved['update'],
                optimizer_steps=sorted({int(v['step']) for v in saved['optimizer']['state'].values()}),
                parameters_with_optimizer_state=len(saved['optimizer']['state']),
                seconds=meta['seconds'], selected_update=meta['selected_update'], device=meta['device']))
    torch.manual_seed(7); model = Recurrent(); optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    xn = torch.rand(4, 64, 2); xn[..., 0] += .5
    xf = torch.rand(4, 64, 3); xf[..., 0] -= .5
    tic = time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        for _ in range(3):
            loss = (model.normal_sequence(xn)/20).square().mean() + (model.friction_sequence(xf)/10).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    counters = {r.key:dict(count=r.count, cpu_time_total_us=r.cpu_time_total)
                for r in profiler.key_averages() if r.key in ('aten::gru', 'Optimizer.step#Adam.step', 'aten::linear')}
    benchmark = dict(seconds=time.perf_counter()-tic, batch=4, sequence=64, updates=3,
                     threads=torch.get_num_threads(), torch=torch.__version__, profiler=counters)
    return evidence, benchmark


def write_report():
    torch.set_num_threads(2)
    old = evaluate_local(ROOT/'results/recurrent_diagnostic', ROOT.parent/'study_v2/results/data')
    new = evaluate_local(ROOT/'results/recurrent_development_1500', ROOT/'results/development')
    evidence, benchmark = training_evidence()
    native = []
    for world in ('maxwell', 'boucwen'):
        path = ROOT/'results/preflight'/f'{world}_oracle.npz'
        if path.exists():
            with np.load(path) as z:
                nv = z['state'][..., 1]+.0025*z['net_force']/2
                nx = z['state'][..., 0]+.0025*nv
                native.append(dict(world=world, max_velocity_residual=float(np.max(np.abs(z['next_state'][..., 1]-nv))),
                                   max_position_residual=float(np.max(np.abs(z['next_state'][..., 0]-nx)))))
    text = """# Unabhängiger Audit der kausalen GRU und ihres Online-Adapters

Die Achsen, die Ausgabe vor dem Historienupdate und der Optimizerpfad sind im
geprüften Code korrekt. Für die neuen Entwicklungsdaten mit δ̇(0)=v(0)=0 ist der
Online-Adapter zum sequenziellen Modell konsistent. Zwei begrenzte
Schnittstellen-/Physikabweichungen sind unten ausdrücklich ausgewiesen. Es
wurden keine Produktionsdateien oder Modelle verändert und kein frischer
Kombinations-/Bestätigungstest geöffnet.

## Befunde und konkrete Konsequenzen

1. **Anfangsrate korrigiert und geprüft:** Die zuerst gelesene Version von
   `RecurrentOnline.initial` setzte die Anfangsrate auf null. Der Hauptagent
   ergänzte während dieses Audits das explizite optionale Argument `rate`.
   Der zusätzliche Test bestätigt jetzt auch bei nichtverschwindender
   Anfangsrate die Äquivalenz. Ohne Argument gilt weiterhin Rate null; die
   neuen Pulssignale erfüllen diese Voraussetzung exakt.
2. **Reibung bei fehlendem Kontakt:** Die positive viskose Komponente bleibt
   bei N=0 bestehen. Damit ist Ft·v≤0 weiterhin garantiert, aber die
   Kontaktbedingung der Referenz (Ft=0 ohne Last/Kontakt) nicht umgesetzt.
   Sämtliche geprüften Entwicklungsprofile bleiben geschlossen und belastet;
   der Befund begrenzt Aussagen über Kontaktöffnung.
3. **Normalkraft-Override:** `normal_override` wird unverändert zurückgegeben,
   einschließlich negativer Werte. Im Reibungskopf wird N intern auf ≥0
   begrenzt; am mechanischen Ausgang kann dagegen eine negative Normalkraft
   erscheinen. Die derzeit idealen positiven Clamp-Signale sind unbetroffen.
   Vor verrauschten/open-contact Anwendungen ist eine gemeinsame explizite
   Projektionsregel erforderlich. Dasselbe gilt für den Nominal-Adapter.

Die zwei noch offenen allgemeinen Invarianten sind als **strict xfail**
sichtbar. Sie werden nicht als bestandene Tests gezählt. Die übrigen Tests
prüfen beide Köpfe unter Zukunftsänderungen, Batch-Permutation und Einzel-
Batch-Vergleich, Schritt-/Sequenzäquivalenz, Eigentum der geklonten Zustände,
Messrechte, Kraftvorzeichen, mechanische Integration und echte Lernupdates.

## Kausalität, Initialisierung und Messrechte

`Head.sequence` erzeugt zuerst h₀ aus dem ersten aktuellen Signal und verwendet
für Ausgabe t ausschließlich xₜ und hₜ. Der GRU-Ausgang nach xₜ wird erst bei
t+1 eingesetzt. `step` führt dieselbe Reihenfolge aus. `batch_first=True` und
die Tensorform [Episode, Zeit, Merkmal] bleiben durch den Datenlader erhalten;
keine Episode wird über die Batch-Achse mit einer anderen verbunden.

Der Normalkopf sieht nur Kompression und Rate, nie Kraftlabels oder deren
Historie. Der Reibungskopf sieht Geschwindigkeit, deren Betrag und die aktuelle
Normalkraft. Im isolierten A-Training ist diese gemessene Clamp-Last ein
erlaubtes lokales Eingangssignal. Die lokale Reibungsvalidierung ist deshalb
**bedingt auf die gemessene Normalkraft**, keine offene kombinierte Vorhersage.
Im bisherigen Entwicklungsrollout initialisiert nur N₀ aus dem gemeinsamen
Kraftsensor den Reibungszustand; alle späteren Lasten werden vorhergesagt.

Der neue Online-Adapter kann optional gemessene Lasten zur Zustandsassimilation
verwenden. Ob dies während Vorbereitung/Feedback erlaubt ist, muss vor einem
neuen Versuch für alle Methoden gleich feststehen. In der MPC-Vorhersage darf
kein künftiges gemessenes N eingespeist werden. Ein nach der Vorbereitung
neu initialisierter Zustand enthält die vergangene Belastung nicht: Die
Historie muss vom definierten kalten Anfang durch die gesamte Vorbereitung
geführt werden. Das ist eine erforderliche Nutzungsregel, kein erkannter
Fehler des GRU-Schrittes.

## Trainingszeit und Optimizerprüfung

"""
    text += "| Modellordner | Welt | Letzter Index | Adam-Schritte | Gemeldete Sekunden |\n|---|---|---:|---:|---:|\n"
    for r in evidence:
        text += f"| {r['folder']} | {r['world']} | {r['last_zero_based_update']} | {r['optimizer_steps']} | {r['seconds']:.3f} |\n"
    text += """
Die Checkpoints bestätigen 1500 Adam-Updates für alle Parameterzustände; die
Schleife enthält genau einen echten Vorwärts-/Rückwärts-/Adam-Schritt je Index.
Die Datenformen betragen alt 24×800 und neu 24×2400 Zeitschritte je lokalem Kopf.
Der geprüfte CPU-Test zeigt Änderungen auch an rekurrenten Gewichten und einen
nichtverschwindenden Gradienten. Die GPU-Zeit ist keine reine Warteschlangen-
Zeit: `assert torch.isfinite(loss)` erzwingt die Auswertung eines GPU-Skalars
pro Schritt; außerdem synchronisieren `.item()` und CPU-Checkpointkopien.

Ein voller erster CPU-Normalkopf-Update benötigte in separaten kurzen Checks
etwa 0,071 s (800 Schritte) und 0,190 s (2400 Schritte). Diese CPU-Zeiten sind
keine Schätzung der CUDA-Laufzeit. Die sehr kurzen CUDA-Zeiten sind auffällig,
aber aus Zähler, Tensorformen und Updatepfad folgt kein belegter Trainingsfehler.
Ein unabhängiger synchronisierter GPU-Profiler wäre für belastbare Hardware-
Kostenangaben zusätzlich nötig; dieser Audit startet bewusst keinen GPU-Job.

Die alten 1500er-Metadaten enthalten noch keine Datendatei-Hashes. Die neuere
Trainerfassung protokolliert zusätzlich Train-/Val-NPZ-Hashes. Der Audit kann
die gespeicherten Zähler und aktuellen lokalen Nachrechnungen bestätigen;
er ersetzt keine fehlende ursprüngliche Provenienzaufzeichnung.

Kurzer CPU-Profiler, beide Köpfe, drei Updates mit Batch 4 und Länge 64:
"""
    text += f"{benchmark['seconds']:.3f} s bei {benchmark['threads']} Torch-Threads; "
    text += "; ".join(f"{k}: {v['count']} Aufrufe" for k,v in benchmark['profiler'].items())+".\n\n"
    text += "## Unabhängig neu berechnete lokale Fehler\n\n"
    text += "Einheiten N; jeweils Seed 0 und der bereits ausgewählte Checkpoint. Die neuen Validierungsprofile sind von den Trainingsprofilen getrennt, wurden aber für Checkpoint-Auswahl benutzt und sind deshalb Entwicklung.\n\n"
    text += "| Daten | Welt | Normal-RMSE | Reibungs-RMSE mit gemessenem N |\n|---|---|---:|---:|\n"
    for label, items in [('alte 2-s-Profile', old), ('neue 6-s-Profile', new)]:
        for r in items:
            text += f"| {label} | {r['world']} | {r['normal_val_rmse']:.6f} | {r['friction_val_rmse']:.6f} |\n"
    max_residual = max(max(abs(r['normal_val_rmse']-r['logged_selected_normal_rmse']),abs(r['friction_val_rmse']-r['logged_selected_friction_rmse'])) for r in old+new)
    text += f"\nGrößte Abweichung der CPU-Nachrechnung von den ausgewählten CUDA-Logwerten: {max_residual:.3g} N.\n\n"
    text += "### Halteintervalle nach mindestens 0,5 s Ruhe der Kompression\n\n"
    text += "Masken werden aus den vorab gespeicherten normalen Halteintervallen gebildet: t≥Haltebeginn+0,5 s und t<Halteende. Keine Schwelle wird an den Kraftfehler angepasst. Reibung wird während desselben normalen Haltefensters ausgewertet; Geschwindigkeit darf dabei variieren.\n\n"
    text += "| Welt | Samples | Normal-RMSE | Normal-Bias | Reibungs-RMSE mit gemessenem N |\n|---|---:|---:|---:|---:|\n"
    for r in new:
        text += f"| {r['world']} | {r['hold_samples_after_0p5s']} | {r['normal_hold_rmse']:.6f} | {r['normal_hold_bias']:+.6f} | {r['friction_during_normal_hold_rmse']:.6f} |\n"
    text += "### Spiegelpaar: ausdrücklich Trainingsdiagnostik\n\n"
    text += "Die positiven/negativen Vorbereitungspulse der Trainingsprofile 0/1 führen danach zu identischen aktuellen δ und Rate. Ihre Differenz prüft, ob ein ausgewähltes Modell frühere Lasten überhaupt auseinanderhalten kann; sie ist kein Generalisierungsnachweis. Fenster 1,5≤t<4 s.\n\n"
    text += "| Welt | Mittlerer wahrer N₊−N₋-Abstand | Vorhergesagter Abstand | RMSE der Abstandsvorhersage |\n|---|---:|---:|---:|\n"
    for r in new:
        text += f"| {r['world']} | {r['train_mirror_hold_true_gap_mean']:+.6f} | {r['train_mirror_hold_predicted_gap_mean']:+.6f} | {r['train_mirror_hold_gap_rmse']:.6f} |\n"
    text += "\n| Welt | Zeit s | Wahre Normalkraftdifferenz N | Vorhergesagte Differenz N |\n|---|---:|---:|---:|\n"
    for r in new:
        for row in r['train_mirror_samples']:
            text += f"| {r['world']} | {row['time']:.4f} | {row['truth_gap']:+.6f} | {row['predicted_gap']:+.6f} |\n"
    text += "\n## Audit des neuen Entwicklungsplaners\n\n"
    text += "`run_model_development.py` klont den aktuellen Beobachterzustand vor der Planung. Ein zusätzlicher Probe-Test bestätigt: Nur der erste Horizontschritt erhält die aktuelle gemessene Normalkraft; alle späteren Kräfte stammen aus dem Prädiktor. Die Kandidaten verändern den realen Beobachterzustand nicht. Im echten Schritt wird dieser mit aktueller Kinematik und gemessenem N genau einmal weitergeführt. Die zwei Warmup-Abfragen beginnen am selben Zustand; nur der zweite zurückgegebene Zustand wird übernommen. Dies ist bei den geprüften reinen Adaptern kein Doppelupdate.\n\n"
    text += "Der neue Nominal-Controller erhält damit im ersten Planungsschritt mehr Information als der Nominal-Controller des Preflight. Für faire neue Vergleichstabellen muss er im neuen Harness erneut laufen. Die Prüfsummenliste sollte sämtliche verbrauchten Modellimplementierungen sowie Reference/common/IsaacRig und Datensplit-Metadaten umfassen. Diese Empfehlungen wurden dem Hauptagenten vor den neuen Modellläufen gemeldet.\n\n"
    text += "## Mechanische Schnittstelle\n\n`motion_step` stimmt mit dem explizit nachgerechneten Cam-Kraftterm und derselben semiimpliziten Integrationsregel überein. Die bereits vorhandenen nativen Oracle-Preflight-Trajektorien liefern folgende maximale Einschritt-Abweichungen von dieser idealen Integrationsregel; dies ist kein neu gestarteter Simulationslauf.\n\n"
    text += "| Welt | Position m | Geschwindigkeit m/s |\n|---|---:|---:|\n"
    for r in native:
        text += f"| {r['world']} | {r['max_position_residual']:.3g} | {r['max_velocity_residual']:.3g} |\n"
    text += """
Die NPZ-Kräfte und Bewegungen stammen aus PhysX, während der Adapter einen
analytischen Integrationsschritt verwendet. Daher bedeutet diese Prüfung
Übereinstimmung der diskreten Mechanik innerhalb der gezeigten Genauigkeit,
keine bitweise oder engineübergreifende Gleichheit. Kraft- und Zustandsguards
werden markiert; NaN in latenten Zuständen wird dadurch nicht repariert.

## Reproduktion

```bash
OPENBLAS_NUM_THREADS=1 /home/loya2/miniconda3/envs/unitree_sim_env/bin/python -m pytest research_next/tests/test_recurrent.py research_next/tests/test_online_audit.py -q
OPENBLAS_NUM_THREADS=1 /home/loya2/miniconda3/envs/unitree_sim_env/bin/python research_next/tests/test_online_audit.py
```

Die zweite Zeile berechnet ausschließlich lokale Train-/Val-Diagnostik und
schreibt diesen Bericht. Sie startet weder Isaac noch Training der vorhandenen
Modelle. Der dreischrittige Profiler verwendet einen separaten frischen CPU-
Testkörper, der nicht gespeichert wird. Vollständige Zahlen/Provenienz folgen
maschinenlesbar, damit spätere Berichte dieselbe Evaluation aufrufen können.

<details><summary>Maschinenlesbare Auditwerte</summary>

```json
"""
    payload = dict(old_validation=old, development_validation=new, optimizer_evidence=evidence,
                   cpu_profiler=benchmark, native_residuals=native,
                   inspected_source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (ROOT/'src/recurrent.py',ROOT/'src/online.py',ROOT/'scripts/train_recurrent_diagnostic.py',ROOT/'scripts/rollout_recurrent_diagnostic.py',ROOT/'scripts/run_model_development.py')})
    text += json.dumps(payload, indent=2, allow_nan=False)+"\n```\n\n</details>\n"
    (ROOT/'reports/recurrent_audit.md').write_text(text)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == '__main__':
    write_report()
