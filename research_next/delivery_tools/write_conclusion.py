"""Synthesize completed locked results; added after freeze, no new inference."""
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/round1"
sys.path.insert(0, str(ROOT / "src"))
from round1_lock import verify_test_lock, digest


def decimal(value, places=3):
    return format(value, "." + str(places) + "f").replace(".", ",")


def main():
    lock = verify_test_lock()
    inputs = [OUT / "analysis/decision.json", OUT / "analysis/method_summary.csv",
              OUT / "analysis/primary_contrasts.csv", OUT / "support/support_metrics.csv",
              OUT / "fit_accounting.json", OUT / "preflight_checks.json"]
    decision = json.loads(inputs[0].read_text())
    assert decision["complete"]
    means = pd.read_csv(inputs[1])
    contrasts = pd.read_csv(inputs[2])
    support = pd.read_csv(inputs[3])
    fits = json.loads(inputs[4].read_text())
    tests = json.loads(inputs[5].read_text())
    def value(method, dataset, metric):
        rows = means[(means.method == method) & (means.dataset == dataset) & (means.metric == metric)]
        assert len(rows) == 1
        return float(rows.iloc[0]["mean"])
    def position(method):
        return value(method, "prediction", "position_rmse") * 1000
    def control(method):
        return value(method, "controller", "position_squared_integral") * 1e6
    labels = {"nominal": "Nominal", "old_flex": "Kurze Filteranalytik", "path_ridge": "Breite Verlaufsregression",
              "recurrent": "Lokale rekurrente Netze (GRU)", "blind_state_id": "Lokale dynamische Identifikation",
              "sensor_state_id": "Identifizierte Reibung + konstante Sensorlast", "state_reset": "Identifikation mit zurückgesetzter Vorgeschichte"}
    verdict = ("Positiver technischer Befund innerhalb des untersuchten synthetischen Materialbaukastens."
               if decision["complete_technical_success"] else "Das vollständige technische Vorteilsmuster ist nicht nachgewiesen.")
    text = f"""# Abschluss der fortgesetzten Simulationsstudie

**{verdict}** Separat kalibrierte dynamische Materialmodelle lassen sich ohne Anpassung an die Zielkombination neu koppeln. Die festgelegte Auswertung meldet **{decision['advantages']} von fünf Primärvergleichen als Vorteil**, {decision['disadvantages']} als Nachteil und {decision['unresolved']} als unentschieden. Alle vorgesehenen Experimente sind ausgeführt: **{decision['prediction_rows']} offene Prognosen und {decision['controller_rows']} Controller-Episoden**, acht unabhängige Kalibrier-/Fitreplikate, vier Materialkombinationen und saubere beziehungsweise verrauschte Kalibrierung.

Der bessere Ansatz ist hier **klassische dynamische Identifikation mit einem vorgegebenen Katalog passender Gleichungsfamilien**. Seine Koeffizienten und Familienauswahl werden aus isolierten Messungen bestimmt. Das ist ein nutzbarer Kompositionsbefund, keine Entdeckung unbekannter Naturgesetze und kein Nachweis eines neuen neuronalen Verfahrens. Die früheren negativen v1-/v2-Ergebnisse bleiben unverändert bestehen. Der echte G1 wurde während des gesamten Auftrags nicht benutzt.

## Was gefunden wurde

Die alte Darstellung speichert nur schnell abklingende Filterzustände. Eine gezielte Vorbelastungsdiagnose zeigt ihre Schwäche bei Hysterese, die während einer Ruhephase erhalten bleibt. Breitere Pfadmerkmale und gelernte rekurrente Zustände wurden daraufhin auf Entwicklungsdaten untersucht. Erst anschließend wurden neue Modelle und ein frischer Kombinationstest festgeschrieben. Die vollständige Suchhistorie einschließlich schlechter Varianten steht im [Suchprotokoll](search_ledger.md); die mathematische Erklärung samt Grenzen in [Materialgedächtnis und zulässige Aussagen](mechanism_and_claims.md).

Im neuen Haupttest erreicht die lokale dynamische Identifikation **{decimal(position('blind_state_id'), 3)} mm mittlere offene Positions-RMSE**. Die GRU erreicht **{decimal(position('recurrent'), 3)} mm**, die kurze Filteranalytik **{decimal(position('old_flex'), 3)} mm** und die breite Verlaufsregression **{decimal(position('path_ridge'), 3)} mm**. Die Mittelwerte gewichten alle vier Kombinationen und beide Rauschbedingungen gleich. Es wird keine günstige Zelle herausgegriffen.

Die beiden Materialbibliotheken enthalten Maxwell-Relaxation und Bouc–Wen-Hysterese sowie jeweils eigene dynamische Reibung. Alle vier normalen/tangentialen Paarungen werden verwendet, darunter die beiden gekreuzten Paarungen. Die mechanische Verdrahtung ist bekannt. Der geführte 2-kg-Schlitten wird nativ in Isaac/PhysX simuliert; die synthetischen Materialkräfte werden explizit angelegt. Die Vorhersagemodelle arbeiten außerhalb des internen Kontaktlösers. Es wurde kein neuer PhysX-Kontaktlöser entwickelt und kein realer Materialversuch ersetzt.

## Vorhersage und tatsächliche Regelaufgabe

| Verfahren | Offene Positions-RMSE [mm] | Quadratisches Positionsfehlerintegral [mm² s] |
|---|---:|---:|
"""
    for method, label in labels.items():
        text += f"| {label} | {decimal(position(method), 3)} | {decimal(control(method), 3)} |\n"
    text += """
Das Controllermaß integriert den quadratischen Positionsfehler über die ganze 1,2-s-Aufgabe einschließlich Zielwechsel. Alle Modelle benutzen denselben begrenzten Planer mit 41 konstanten Kraftkandidaten, 250 ms Vorschau und Neuplanung alle 50 ms. Prozentangaben unten beziehen sich auf dieses Integral, nicht auf eine gleich große Verringerung des Positionsfehlers in Millimetern.

| Festgelegter Vergleich der Identifikation | Mittlere Differenz | 99,5%-Intervall | Relative Änderung |
|---|---:|---:|---:|
"""
    for row in contrasts.itertuples():
        scale, unit = (1000, "mm") if row.dataset == "prediction" else (1e6, "mm² s")
        title = "Prognose" if row.dataset == "prediction" else "Regelung"
        text += f"| {title} – Identifikation minus {labels[row.comparator]} | {row.mean_difference*scale:+.3f} {unit} | [{row.ci_lower*scale:+.3f}, {row.ci_upper*scale:+.3f}] {unit} | {100*row.relative_difference_of_means:+.2f}% |\n"
    text += """
Die Intervalle beruhen auf acht **gepaarten Seedmitteln**, nicht auf Tausenden vermeintlich unabhängigen Episoden. Fünf zweiseitige Vergleiche teilen das vorab festgelegte Rundenbudget α=0,025; die Einzelintervalle haben nominell 99,5% Abdeckung und gemeinsam mindestens 97,5% unter den Student-t-Annahmen. Acht Replikate erlauben nur eine begrenzte Prüfung dieser Verteilungsannahmen. Einzelne Material-/Rauschzellen und weitere Methodenvergleiche bleiben deskriptiv. Vollständige Daten, Unsicherheit und Prüfregeln stehen im [automatisch erzeugten Hauptbericht](round1_report.md) und im [unabhängigen Statistikaudit](round1_statistical_audit.md).

![Alle fünf vorher festgelegten Vergleiche](figuresround1/round1_primary_contrasts.png)

## Der Gegenbefund begrenzt den praktischen Anspruch

"""
    reset_change = 100 * (control("state_reset") / control("blind_state_id") - 1)
    text += f"""Die Variante mit verworfener Vorgeschichte hat **schlechtere offene Vorhersagen** ({decimal(position('state_reset'), 3)} statt {decimal(position('blind_state_id'), 3)} mm), erzielt im begrenzten Planer aber ein **um {decimal(abs(reset_change), 2)}% niedrigeres** mittleres Positionsfehlerintegral. Am letzten, festgelegten Checkpoint erreicht sie {decimal(100*value('state_reset','controller','success_120'), 2)}% Erfolg gegenüber {decimal(100*value('blind_state_id','controller','success_120'), 2)}% mit vollständiger Vorgeschichte. Dies ist eine deskriptive Ablation, kein nachträglich ergänzter Primärtest.

**Fünf bestätigte Primärvorteile bedeuten deshalb keinen Sieg über alle sieben Verfahren.** Der Versuch belegt nicht, dass das dynamisch genaueste Modell die besten Controllerkosten erzielt. Bereits die Entwicklung zeigte, dass Modellfehler die beschränkte Kandidatensuche gelegentlich günstig beeinflussen. Der zusätzliche Nutzen einer zukünftigen Lastprognose gegenüber der starken Sensorlast-Kontrolle ist zudem klein. Bevor ein größerer Robotiknutzen behauptet wird, muss der Zusammenhang zwischen Modellgenauigkeit und Planungsqualität gesondert untersucht werden.

## Was der Test voraussetzt

Alle Verfahren erhalten die gesamte beobachtete Präparation und ideale kausale x/v/N-Sensoren im Controller. State-Reset verwirft den fortgeschriebenen Modellzustand einmalig am Bewegungsbeginn; der physikalische Referenzzustand bleibt bestehen. Rauschen betrifft die isolierte Kalibrierung und lokale Validierung; robuste verrauschte Online-Zustandsschätzung wurde hier nicht bestätigt. Die offene Referenzfahrt wird einmalig mit einer privilegierten inversen Kraftvorgabe erzeugt, anschließend verwenden alle Prädiktoren genau diese gespeicherten Aktionen. Dieser inverse Eingriff fehlt in der echten Controllerbewertung. Im ersten Vorhersageschritt ist die gemeinsame aktuelle Normalkraftmessung erlaubt, danach bleiben die offenen Prädiktoren ohne zukünftige Messkorrekturen.

Die Identifikation kennt die richtigen Gleichungsformen als Kandidaten, einschließlich Typ/Anzahl interner Zustände. Sie wählt nicht anhand eines geheimen Weltlabels, verfügt aber über mehr spezifisches Physikwissen als die GRU. Die Bibliothek, bekannte Präparationsanfänge, geschlossene belastete Kontakte und feste Diskretisierung begrenzen den Befund. Ein gemeinsamer rekurrenter Vergleich und eine neue Zielkombinationsdaten-Kontrolle wurden in dieser Runde nicht ergänzt; die entsprechenden älteren Vergleichsreihen belegen keine neue GRU-Architekturüberlegenheit. Die [HANDOFF-Abdeckung](handoff_coverage.md) trennt historische Nachweise und neue Einschränkungen.

"""
    full = support[support.segment == "full"]
    controller_motion = support[(support.stage == "controller_truth") & (support.method == "blind_state_id") & (support.segment == "motion")]
    cross = support[(support.stage == "reference_open") & (support.segment == "motion")
                    & (support.normal_world == "boucwen") & (support.friction_world == "maxwell")]
    text += f"""Der Bereichsaudit umfasst {len(support)} Segmentzeilen. **{decimal(100*full.hold_exceeds_training_max.mean(), 0)}% der vollständigen Verläufe** enthalten eine längere Ruhephase als die lokale Reibungskalibrierung: deren längste Ruhezeiten liegen bei {decimal(support.friction_training_max_hold_s.min(), 4)}–{decimal(support.friction_training_max_hold_s.max(), 4)} s, die neuen Folgen etwa bei vier Sekunden. In der gekreuzten offenen Bouc–Wen-Normal-/Maxwell-Reibungskombination verlassen {int((cross.normal_outside_fraction>0).sum())}/{len(cross)} Episoden den lokalen Lastbereich; im Mittel betrifft das {decimal(100*cross.normal_outside_fraction.mean(), 3)}% ihrer Bewegungsproben. Die echte offene Kompression, Kompressionsrate und Geschwindigkeit bleiben dagegen innerhalb ihrer jeweiligen marginalen Trainingsbereiche. Das ist **kein ausschließlich interpolierender Test** und kein Beleg beliebiger unbekannter Vorgeschichten. Bei den echten Controllerfahrten der Hauptmethode überschreitet zusätzlich die Kompressionsrate in durchschnittlich {decimal(100*controller_motion.rate_outside_fraction.mean(), 3)}% der Bewegungsproben ihre marginalen Trainingsgrenzen. Keine dieser Episoden wurde entfernt. Details: [Bereiche und Haltezeiten](round1_support.md) und [unabhängige Bereichsnachprüfung](round1_support_audit.md).

## Numerik, Fehlerfälle und tatsächlicher Aufwand

Die festgelegte unabhängige MuJoCo-/Zeitschrittprüfung ist **{decision['numerical_validity']['status']}**. Sie umfasst die reservierte saubere Replikation r1, alle vier Kombinationen und 16 Profile bei drei Zeitschritten, insgesamt zwölf Replay-Batches mit 192 Trajektorien. Ihre Reichweite wird nicht auf alle acht Replikate vergrößert. Die zusätzliche Rohprüfung aller offenen Fahrten prüft Vor-/Nachschrittindizes, Einheiten, Nettokraftbilanz und Reibleistung. Eine negative momentane Reibleistung ist keine globale Passivitätsgarantie. Belege: [Backend](round1_backend.md), [offener Physikaudit](round1_open_physical_audit.md).

{decision['guarded_episodes']} der {decision['prediction_rows']+decision['controller_rows']} bewerteten Episoden enthalten Guard-Einträge, davon **{decision['guarded_episodes_main_method']} bei der Hauptmethode**. Sie bleiben sämtlich in der Auswertung. Offene Guardzähler zählen begrenzte Bewegungsschritte; Controller-Guardzähler zählen Planungszeitpunkte mit mindestens einem ungültigen Kandidaten und keine realen PhysX-Fehlschritte. Die vollständigen Zähler und betroffenen Methoden stehen im Hauptbericht.

Die vorhandene Isaac-Sim-5.1-/Isaac-Lab-/Unitree-Umgebung und das separate MuJoCo-3.6-Environment wurden weiterverwendet. Neue Installation oder Hardwaresteuerung war nicht erforderlich. Die neue Kalibrierung enthält {sum(c['episodes'] for c in fits['calibration_cells'])} isolierte Sechssekunden-Verläufe. Alle {len(fits['recurrent_cells'])} GRU-Fits haben tatsächlich jeweils {fits['recurrent_cells'][0]['optimizer_steps']} Adam-Schritte ausgeführt. Die protokollierten Fit-Timersummen betragen {decimal(fits['summary']['recurrent']['seconds']['sum'], 2)} s für GRU und {decimal(sum(c['seconds'] for c in fits['state_id_cells']), 2)} s für dynamische Identifikation. CPU/GPU, Timergrenzen und parallele Ausführung unterscheiden sich; dies ist keine hardwarebereinigte Geschwindigkeitsrangfolge. Nicht gemessene GPU-/Prozesspeaks werden nicht aus Konfigurationsgrenzen erfunden. [Vollständige Kostenbilanz](round1_fit_accounting.md).

Vor der Codesperre bestanden {tests['primary_environment']['passed']} Tests im Hauptenvironment; {tests['primary_environment']['skipped']} MuJoCo-Prüfungen wurden dort ausgelassen und separat im passenden Environment geprüft. Zwei erwartete GRU-Grenzfälle liegen außerhalb des deklarierten positiven geschlossenen Kontakts. Die separate MuJoCo-Suite meldet {tests['mujoco_environment']['passed']} bestandene Prüfungen; die Zahlen werden wegen Überlappungen nicht zu einer neuen Gesamtzahl addiert. Die alten und neuen Sperrlisten sowie vollständige Rohdaten werden im [abschließenden Integritätsaudit](round1_integrity_audit.md) geprüft.

## Empfehlung und gelieferte Artefakte

Die gefundene technische Grundlage ist ein kleiner wiederverwendbarer Baukasten identifizierter dynamischer Materialmodelle. Für bekannte Gesetzesfamilien ist dieser Ansatz die derzeit begründete Wahl. Der allgemeine stärkere Lern-/Neuheitsanspruch bleibt ungestützt; die [Literaturprüfung](literature_options.md) und [Vorarbeitenmatrix](../../reports/novelty_matrix.md) werden dadurch nicht widerlegt. Ein größerer Paper-Versuch braucht zunächst unbekannte Materialfamilien oder unvollständig beobachtete Vorgeschichten sowie einen stärker geprüften Planer. Dafür spricht ein neuer, begrenzter Simulationsversuch; für einen allgemeinen Durchbruch oder direkten G1-Einsatz liefert diese Runde keinen Nachweis. Die aktuelle Bestätigungsrunde ist abgeschlossen und wird nicht weiter auf ihre Testwerte abgestimmt.

- [Vollständiger Hauptbericht mit vier Diagrammen](round1_report.md)
- [Vorhersage-Messwerte](../results/round1/prediction_metrics.csv) und [Controller-Messwerte](../results/round1/controller_metrics.csv)
- [Video Maxwell/Bouc–Wen](../videosround1/round1_maxwell_boucwen_r1_clean_episode0.mp4) und [Video Bouc–Wen/Maxwell](../videosround1/round1_boucwen_maxwell_r1_clean_episode0.mp4)
- [Videoprüfung](round1_video_audit.md): schematische Visualisierung echter aufgezeichneter PhysX-Zustände, keine RGB-Aufnahme eines Roboters
- [Code, vollständige Ausführungsbefehle und Paketbeschreibung](../README.md), [Fortsetzungsstatus](../STATUS.md), [Reproduktionsaudit](round1_reproducibility_audit.md)
- [Quellenpaket](../deliverables/physics-transfer-round1-source.tar.gz), [vollständiges Paket mit v1-/v2-Begleitdaten](../deliverables/physics-transfer-round1-complete.tar.gz), [Paketprüfsummen](../deliverables/SHA256SUMS.json)

Die Paketdateien enthalten keine installierten Simulatoren. Alte Herkunftsdaten erwarten die ursprüngliche Dateisystemabbildung; erneute CUDA-Fits werden nicht als byteidentisch zugesichert. Die gelieferten eingefrorenen Gewichte und Aktionen erlauben die erneute Ergebnisberechnung. Vorhandene Endarchive und Videos werden nicht still überschrieben.

Erneute Ergebnisprüfung auf dem vorhandenen Labor-PC:

```bash
cd /home/loya2/physics-transfer
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
/home/loya2/miniconda3/envs/unitree_sim_env/bin/python research_next/scripts/prepare_round1.py verify
/home/loya2/miniconda3/envs/unitree_sim_env/bin/python research_next/scripts/evaluate_round1.py metrics
/home/loya2/miniconda3/envs/unitree_sim_env/bin/python research_next/scripts/analyze_round1.py
```

Dieser Synthesebericht wird durch `delivery_tools/write_conclusion.py` aus abgeschlossenen Ergebnisdateien erzeugt. Das nach der experimentellen Sperre ergänzte Berichts-/Lieferwerkzeug ändert keine Messergebnisse, Verfahren oder statistischen Regeln. Die eigentliche Primäranalyse war vor neuen Kalibrierungen gesperrt.
"""
    report = ROOT / "reports/conclusion.md"
    report.write_text(text)
    provenance = dict(test_lock_sha256=lock,
                      sources_sha256={str(p): digest(p) for p in inputs + [Path(__file__)]},
                      report_sha256=digest(report),
                      role="post-evaluation narrative synthesis; no new primary analysis")
    (OUT / "conclusion_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    assert verify_test_lock() == lock
    print(report)


if __name__ == "__main__":
    main()
