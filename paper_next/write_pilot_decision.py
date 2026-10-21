"""Generate the short research decision from the completed exploratory pilot."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "pilot_memory/results"


def main():
    summary = pd.read_csv(DATA / "summary.csv")
    run = json.loads((DATA / "run.json").read_text())
    audit = json.loads((ROOT / "pilot_independent_audit.json").read_text())
    assert run["status"] == "complete" and audit["passed"]

    def row(world, kind, duration, condition):
        selected = summary[(summary.world == world) & (summary.probe_kind == kind)
                           & (summary.duration_s == duration) & (summary.condition == condition)
                           & (summary.sigma_N == .2)]
        assert len(selected) == 1
        return selected.iloc[0]

    def number(value, precision=4):
        return f"{float(value):.{precision}g}".replace(".", ",")

    bw_blind = row("boucwen", "slip", .1, "blinded")
    bw_sensed = row("boucwen", "slip", .1, "sensed")
    mx_rest = row("maxwell", "rest", .1, "blinded")
    mx_slip = row("maxwell", "slip", .1, "blinded")
    bw_rest0 = row("boucwen", "rest", 0., "blinded")
    bw_rest1 = row("boucwen", "rest", 1., "blinded")
    text = f"""# Entscheidung nach dem ersten Materialpilot

14.09.2026. **Der Mechanismus ist untersuchbar; eine neue Methode ist noch nicht nachgewiesen.** Der nächste sinnvolle Schritt ist die kostenbewusste Wahl der Vorbereitung unter realistischen Messbedingungen und neuer mechanischer Kopplung. Eine größere Wiederholung der alten Modellrangliste wäre dafür nicht ausreichend.

## Tatsächlich gerechnet

{run['rows']:,} Bedingungszeilen aus zwei Materialgesetzen, Ruhe/Slip, neun Dauern, neun festen Anfangszuständen, zwei Kraft-Rauschstärken, jeweils 20 Rauschreplikaten und zwei Informationsrechten. Der einmalige vollständige CPU-Lauf dauerte {number(run['seconds'])} s. Die zwei verwendeten r1-clean-Modelle stammen unverändert aus der früheren isolierten Kalibrierung; neue Gewichtsaktualisierungen: {run['weight_updates']}. Diese Zeilen sind keine {run['rows']:,} unabhängigen Trainingsläufe.

Der Prüfstand schreibt Geschwindigkeit und Normalkraft ideal vor. Es wurden keine neue Isaac-Szene, kein Robotikcontroller und keine reale Hardware getestet. Der Beobachter ist eine klassische Gitterinferenz mit bekannten Modellfamilien und festem Prior. Anfangszustände werden für diese Diagnose direkt gesetzt; erreichbare verborgene Präparationsverläufe sind erst eine nächste Versuchsbedingung.

## Drei getrennte Befunde

1. **Messinformation kann bei gleicher Bewegung helfen.** Beim alternden Reibungsmodell ergibt eine 100-ms-Slip-Probe bei 0,20 N unabhängiger Kraft-Messstörung einen zukünftigen Kraft-RMSE von {number(bw_blind.future_force_rmse_N)} N ohne Nutzung der Messungen und {number(bw_sensed.future_force_rmse_N)} N mit ihnen. Beide Vorhersagen beginnen nach demselben physischen Versuch. Die Probe umfasst 8 mm Schlupf; vorhergesagt wird ein vorab festgelegtes 150-ms-Kraftprogramm. Das ist ein deskriptiver Vergleich über feste Zustände und Rauschwiederholungen.

2. **Bewegen kann unnötig sein.** Beim Maxwell-Reibungsmodell betragen die blinden Zukunftsfehler nach 100 ms Ruhe beziehungsweise Slip {number(mx_rest.future_force_rmse_N)} und {number(mx_slip.future_force_rmse_N)} N. Bei konstanter Last ist die Kontraktion der Zustandsdifferenz hier unabhängig von der Bewegung. Eine lange aktive Vorbereitung ist deshalb nicht schon durch einen kleineren Vorhersagefehler gerechtfertigt; die einfache Ruhekontrolle ist echte Konkurrenz.

3. **Kleinerer Kraftfehler beweist keine Zustandserkennung.** Beim alternden Modell senkt eine Sekunde Ruhe den blinden Kraftfehler von {number(bw_rest0.future_force_rmse_N)} auf {number(bw_rest1.future_force_rmse_N)} N, während der Fehler des aktuellen Alterszustands bei rund {number(bw_rest1.current_state_rmse)} s bleibt. Die Zustandsdifferenzen bleiben erhalten; die logarithmische Kraftabhängigkeit wird weniger empfindlich. Ein Bericht muss Zustand, Ausgangssensitivität und Messinformation auseinanderhalten.

## Was das für ein mögliches Paper bedeutet

Der eigenständige Untersuchungsgegenstand lautet: **Wie viel Zeit, Bewegung und Messinformation ist nötig, um ein eingefrorenes Kontaktmodell nach unbekannter Vorgeschichte wiederzuverwenden?** Die interessante Ergebnisform wäre eine validierte Entscheidungskarte für Beobachten, Sondieren und Rekonditionieren, einschließlich der Bereiche, in denen aktives Sondieren nichts bringt.

Der Pilot rechtfertigt die nächste diagnostische Stufe, weil informative kurze Proben und einfache Gegenstrategien messbar auseinanderfallen. Er zeigt weder eine neue Explorationmethode noch einen Vorteil gegenüber optimaler klassischer Vorbereitung. Die nächste Stufe muss eine auf Entwicklung gewählte feste Probe plus starken Beobachter und eine klassische kostenbewusste prädiktive Bayes-Auswahl enthalten. Eingefrorene Gesetze, Vorbereitungsstrategie, erreichbare Vorgeschichten und Rauschannahmen werden vor einer frischen Hauptauswertung festgelegt.

Eine neue Methode wäre erst begründet, wenn sie unter gleicher Informations- und Interaktionsbilanz eine verbleibende Schwäche dieser Konkurrenz löst oder ihre Rechenkosten bei gleicher Entscheidungsqualität nachweislich senkt. Andernfalls bleibt ein mechanistischer Vergleich bekannter Verfahren. Die [Forschungsentscheidung](research_direction.md) benennt Literatur, weitere Bedingungen und Abbruchkriterien. Die [Literaturmatrix](literature_matrix.md) enthält direkte Vorarbeiten; globale Neuheit ist offen.

## Reproduktion und Prüfung

[Quellcode und Laufbefehle](../pilot_memory/README.md) · [Rohmetriken](../pilot_memory/results/trials.csv) · [Zusammenfassung](../pilot_memory/results/summary.csv) · [Unabhängige Prüfung](../pilot_independent_audit.json) · [Status und Fortsetzung](../STATUS.md).

Die unabhängige Prüfung hat alle gespeicherten Quell-/Ergebnishashes bestätigt und sämtliche {audit['sensed_rows_independently_recomputed']:,} Vorhersagefehler der beobachtenden Bedingung aus gespeicherten Posteriorgewichten, Kandidatenverläufen und Referenzkräften erneut berechnet. Eine zusätzliche [Tabelle der Referenzantwortstreuung](../pilot_reference_response_spread.csv) verhindert die Gleichsetzung von Prädiktorfehler und physischer Zustandskontraktion.

Die ersten Rohplots vergrößerten im nahezu konstanten BW-Ruhe-Zustandsfehler Rundungsreste durch automatische Achsenskalierung. Sie bleiben als ursprüngliche Ausgabe erhalten; die gesondert erzeugten, visuell geprüften Lieferplots verwenden gemeinsame, sinnvolle Achsengrenzen: [Slip-Probe](../pilot_memory/figures_final/memory_mechanism_slip.png) und [Ruhekontrolle](../pilot_memory/figures_final/memory_mechanism_rest.png). Die Messwerte wurden nicht verändert. Der Renderstand ist in `pilot_memory/RESULTS.md` dokumentiert.
""".replace("25,920", "25.920").replace("12,960", "12.960")
    target = ROOT / "reports/pilot_decision.md"
    target.write_text(text)
    print(target)


if __name__ == "__main__":
    main()
