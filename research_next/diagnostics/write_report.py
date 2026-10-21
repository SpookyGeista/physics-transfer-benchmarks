"""Render the local exploratory diagnostic from its recorded results."""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE/'summary.csv')
old = pd.read_csv(HERE/'existing_five_seed_audit.csv')
names = {'base3_longopt': '3 Filter · MLP24 · länger', 'ridge_base3': '3 Filter · Ridge',
         'bank9': '9 Filter · MLP24', 'ridge_bank9': '9 Filter · Ridge',
         'bank9_path': '9 Filter + Pfad · MLP24', 'ridge_bank9_path': '9 Filter + Pfad · Ridge',
         'bank9_path_wide': '9 Filter + Pfad · MLP64'}


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


lines = ['# Lokale Diagnose: Historienrepräsentation und Optimierung', '',
         '14. September 2026. **Explorative Entwicklungsanalyse**, kein neuer zurückgehaltener Haupttest. '+
         'Eigene Dateien ausschließlich unter `research_next/diagnostics/` und dieser Bericht. '+
         'Die abgeschlossenen v1-/v2-Studien, Modelle und Urteile wurden nicht verändert.', '',
         '**Es gibt einen konkreten neuen Ansatzpunkt: Die bisherige kurze Historienrepräsentation ist wesentlich zu schwach.** '+
         'Mehr generische Filterzeiten helfen der Maxwell-Normalantwort stark; zusätzliche beobachtbare Pfadgrößen helfen besonders der Bouc–Wen-Normalhysterese und der Reibung. '+
         'Nur länger zu optimieren bringt deutlich weniger. Die stärkere analytische Kontrolle profitiert ebenfalls; das Ergebnis begründet noch keinen besonderen Vorteil neuronaler Komposition.', '',
         '## Vorgehen und erlaubte Information', '',
         'Untersucht wurden ausschließlich die bisherigen isolierten `train`-/`val`-Dateien von Replikat r0, volles Budget 24, clean, in beiden Welten. '+
         'Je Effekt 24 zweisekündige Trainingsprofile und acht getrennte Validierungsprofile. Diese Validierung ist nach wiederholter Entwicklung kein unberührter Bestätigungstest. '+
         'Die bisherige fünf-Seed-Auswertung wurde gelesen, um die alten lokalen und offenen Fehler einzuordnen; neue Modelle wurden nicht auf alten Kombinationstests ausgeführt. '+
         'Kein neuer Referenztest, kein Controllerlauf und keine Roboterschnittstelle wurden gestartet.', '',
         'Vor dem ersten Fit wurden vier Konfigurationen in [declaration.json](../diagnostics/declaration.json) festgelegt:', '',
         '1. `base3_longopt`: dieselben drei Filter und dieselbe MLP-Breite 24 wie bisher, aber längere Optimierung.',
         '2. `bank9`: generische Filterzeiten 0,005/0,01/0,02/0,04/0,08/0,16/0,32/0,64/1,28 s, Breite 24.',
         '3. `bank9_path`: dieselben neun Filter plus beobachtbare Pfadmerkmale, Breite 24.',
         '4. `bank9_path_wide`: identische Merkmale wie 3, Breite 64 als Kapazitätskontrolle.', '',
         'Die Normal-Pfadmerkmale sind Anfangskompression, bisheriges Minimum/Maximum der Kompression und das bis zum vorigen Schritt integrierte |Kompressionsrate|. '+
         'Reibung erhält zusätzlich Zeit seit Präparation, akkumulierten absoluten Schlupfweg, aktuelle Absolutgeschwindigkeit und neun Filter der Absolutgeschwindigkeit. '+
         '**Weder aktuelle noch vergangene Kraftmessungen gehen in die Normalfunktion ein.** '+
         'Reibung darf die gemessene aktuelle/kausale Normallast als lokale Schnittstellengröße verwenden. Verborgene Materialzustände und -parameter werden nicht importiert. '+
         'Eine spätere offene Verwendung müsste diese Lasthistorie mit vorhergesagten Kräften fortführen. Die neue Initial-/Pfadinformation setzt eine bekannte beobachtete Präparation voraus.', '',
         'Alle MLP-Konfigurationen starten frisch mit Modellseed 0 und erhalten 12.000 Adam-Updates: Lernrate 0,002 für 6.000 Updates, dann 0,0005; Batch 512 pro Kopf, Weight-Decay 1e-6, Gradientengrenze 5. '+
         'Normal- und Reibfehler sind wie zuvor durch 20/10 N skaliert. Der gemeinsame Checkpoint wird auf isolierter Validierung alle 200 Updates sowie bei 1.200 und 12.000 ausgewählt. '+
         'Die Ausgangskontrolle reproduziert bei 1.200 Updates den früheren Bouc–Wen-Validierungsfehler bis auf Fließkommarundung; die Checkpoint-Abstände unterscheiden sich vom früheren 50er-Raster.', '',
         'Für jeden der drei verschiedenen Merkmalsräume läuft außerdem ein systematischer starker Ridge-Vergleich: vollständige Normalmonome bis Grad 3, Reibmonome bis Grad 2 plus Absolutmerkmale, Zentrierung/Skalierung nur aus Train. '+
         'Die fünf λ-Kandidaten 1e-8/1e-6/1e-4/0,01/1 werden nur auf Val gewählt. Reibung wird direkt als Kraft angepasst; keine Division verrauschter oder niedriger Geschwindigkeitslabels. '+
         'Die Pfad-Ridge hat 1.409 freie Koeffizienten, die kompakte Pfad-MLP 2.426 Parameter, die breite MLP 11.586. Die Vergleichsrechte sind gleich; Rechenzeiten sind nicht künstlich angeglichen.', '',
         '## Gemessene lokale Fehler', '',
         'Werte sind RMSE über alle gültigen Zeitschritte der jeweiligen isolierten Profile in N. Kein Seed-Konfidenzintervall: diese Diagnose hat genau einen Datensatz- und Modellseed.', '']
for world in ['maxwell', 'boucwen']:
    rows = []
    for method in ['modular', 'analytic_flex']:
        row = old[(old.world == world)&(old.seed == 0)&(old.method == method)].iloc[0]
        rows.append(['Alt: '+method, '—', f'{row.val_normal_rmse:.6f}', '—', f'{row.val_friction_rmse:.6f}', f'{row.seconds:.3f}'])
    for _, row in df[df.world == world].iterrows():
        rows.append([names[row.variant], f'{row.train_normal_rmse:.6f}', f'{row.val_normal_rmse:.6f}',
                     f'{row.train_friction_rmse:.6f}', f'{row.val_friction_rmse:.6f}', f'{row.seconds:.3f}'])
    lines += ['### '+world, '', table(['Konfiguration', 'Train N', 'Val N', 'Train Ft', 'Val Ft', 'Fit s'], rows), '']
lines += ['![Lokale Validierungsfehler](../diagnostics/local_validation.png)', '',
          'Rot: MLP, Blau: systematische Ridge-Kontrolle. Die Achsen sind logarithmisch; Tabellen enthalten die genauen Größen.', '',
          '## Deutung und Einschränkungen', '',
          '1. **Optimierung allein erklärt den alten Fehler nicht ausreichend.** Bei Bouc–Wen sinkt Normal-Val durch längeres Training derselben drei Filter von 1,106 auf 0,808 N; Train bleibt bei 0,725 N. '+
          'Mit Pfadmerkmalen erreicht das gleich breite Netz dagegen 0,0855 N bei 0,0741 N Train. Der Effekt ist erheblich größer als die Optimierungskorrektur. '+
          'Das ist ein Hinweis auf eine unzureichende praktisch nutzbare Darstellung, keine bewiesene informationstheoretische Fehleruntergrenze der alten Features.', '',
          '2. **Maxwell braucht längere/breitere Zeitinformation, die einfache Analytik hervorragend nutzt.** Neun generische Zeiten senken MLP-Normal-Val von 0,0466 auf 0,00356 N. '+
          'Die Ridge erreicht 0,0000749 N. Die Verbesserung kann deshalb nicht als neuraler oder modularer Alleinstellungsbeleg gelten.', '',
          '3. **Reibung profitiert von Betrags- und Pfadgeschichte.** Mit Pfadmerkmalen erreicht MLP24 0,0122 N (Maxwell) und 0,0308 N (Bouc–Wen), '+
          'die Ridge desselben Eingangsraums 0,0194/0,0390 N. Das ist ein lokaler, ausgewählter Validierungsbefund dieses einen Seeds; kein statistisch bestätigter oder offener Transfer-Vorteil.', '',
          '4. **Mehr Kapazität ist hier kein überzeugender Haupthebel.** Breite 64 verändert die Fehler nur gering, während die Parameterzahl von 2.426 auf 11.586 wächst. '+
          'Bei Maxwell verschlechtert sich dabei die Normal-Val, bei Bouc–Wen verbessert sie sich minimal und die Reib-Val verschlechtert sich leicht. '+
          'Ein universeller Satz über Modellgröße folgt aus einem Seed nicht.', '',
          '5. **Die Pfadmerkmale selbst tragen Vorwissen.** Min/Max, Präparationszustand, verstrichene Zeit und kumulierter Schlupf können die bisher kurzen, einfachen Anregungen gut beschreiben. '+
          'Bei längeren, verschachtelten Umkehrfolgen oder unbekannter Präparation kann diese Beschreibung scheitern. '+
          'Ein rekurrenter Zustand sollte daher als Alternative geprüft werden; größere MLPs allein sind weniger begründet.', '',
          'Die bisherigen fünf Replikate zeigen bereits lokale Fehler, bevor eine mechanische Kombination offen gerollt wird. '+
          'Bei Bouc–Wen liegen alte modulare Normal-Val-Mittel bei 0,996 N, die konditionierte lokale Kraftdiagnose der alten Kombination bei 0,943 N; die starke Analytik bei 0,744/0,677 N. '+
          'Damit war die alte Hauptschwäche nicht ausschließlich eine fehlerhafte Nockenkopplung. '+
          'Die vollständigen zwanzig Altmodell-Einträge stehen in [existing_five_seed_audit.csv](../diagnostics/existing_five_seed_audit.csv).', '',
          '## Konkreter nächster Versuch', '',
          'Als nächste Lernmethode ist ein kleines kausales rekurrentes Modul mit expliziter Initialisierung aus der beobachteten Präparation sinnvoll. '+
          'Es sollte die Folge der erlaubten Kompressions-/Geschwindigkeits-/Lastsignale verarbeiten, ohne verborgene Materialzustände oder eigene Kraftlabels als Normalinput. '+
          'Die kompakte Pfad-MLP und starke Pfad-Ridge bleiben faire Kontrollen: Wird nur eine kurze Filterbank geschlagen, ist das kein ausreichender Methodenbeitrag.', '',
          'Vor einem neuen Kompositionstest sind längere isolierte Folgen mit unabhängigen, mehrfachen und verschachtelten Umkehrungen, Halten und neuer Präparation zu prüfen. '+
          'Neue Validierungs-/Testfolgen müssen nach dieser explorativen Auswahl frisch festgelegt werden. Neue offene Rollouts und Aufgabennutzen sind vollständig ausstehend; '+
          'der frühere No-Go-Befund wird durch diese lokale Entwicklungsverbesserung nicht umgedeutet.', '',
          '## Reproduktion, Kosten und Prozessstatus', '',
          f'14 neue Fits (acht MLP, sechs Ridge), zusammen {df.seconds.sum():.3f} s protokollierte Fit-/Validierungszeit. '+
          'CPU, zwei Torch-Threads, OPENBLAS_NUM_THREADS=1. Featureaufbau und Dateiarbeit sind in dieser Fitsumme nicht enthalten. '+
          'Alle Prozesse dieses Teilauftrags sind beendet; keine alte Datei wurde überschrieben. Die acht ursprünglichen NPZ-/JSON-Eingaben wurden nach Abschluss erneut per SHA-256 geprüft.', '',
          '```bash', 'cd /home/loya2/physics-transfer',
          'OPENBLAS_NUM_THREADS=1 /home/loya2/miniconda3/envs/unitree_sim_env/bin/python research_next/diagnostics/run_local_diagnostics.py',
          'OPENBLAS_NUM_THREADS=1 /home/loya2/miniconda3/envs/unitree_sim_env/bin/python research_next/diagnostics/write_report.py', '```', '',
          'Vorliegende Runs werden wiederverwendet; die Deklaration samt Code-/Datenhash muss unverändert sein. '+
          'Checkpointgewichte, ausgewählte Updates, Lernkurven, Ridge-λ, Vorhersagen je Profil und Laufzeiten liegen unter `diagnostics/runs/`. '+
          'Tabelle: [summary.csv](../diagnostics/summary.csv); vollständige Messwerte: [results.json](../diagnostics/results.json). '+
          'Der echte G1 blieb ungenutzt.', '']
dest = HERE.parent/'reports/local_diagnostics.md'; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_text('\n'.join(lines))
print(dest)
