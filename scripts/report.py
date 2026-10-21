"""All numeric report tables and plots are derived from machine-readable results."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student
from null_physics_transfer.common import CFG,DT,dump,digest

out=Path('reports');fig=out/'figures';fig.mkdir(exist_ok=True)
p=pd.read_csv('results/evaluation/prediction.csv');c=pd.read_csv('results/control/control.csv')
native=pd.read_csv('results/evaluation_native/prediction.csv')
methods=CFG['methods'];labels={'nominal':'Nominell','identified':'Identifikation','analytic':'Analytisch','joint':'Gemeinsames NN','moe':'MoE-Adaption','modular':'Modular','no_history':'Ohne Historie','decoupled':'Entkoppelt','privileged':'Privilegiert'}
plt.rcParams.update({'figure.dpi':140,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
colors={m:plt.cm.tab10(i%10) for i,m in enumerate(methods)}
def interval(values):
 a=np.asarray(values);mean=float(a.mean());half=float(student.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else 0.
 return mean,half

def summary(frame,metric,group):
 seed=frame.groupby(group+['seed'])[metric].mean().reset_index();rows=[]
 for key,g in seed.groupby(group):
  key=key if isinstance(key,tuple) else (key,);mean,ci=interval(g[metric]);rows.append(dict(zip(group,key),mean=mean,ci95=ci,n_seeds=len(g)))
 return pd.DataFrame(rows)

def markdown_table(df):
 return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in df.itertuples(index=False,name=None))

full=p[(p.budget==24)&(p.noise==0)];base=summary(full,'position_rmse',['world','method']);base.to_csv('results/summary_prediction.csv',index=False)
cs=summary(c,'success',['world','method']);cs.to_csv('results/summary_control.csv',index=False)
fig1,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
for ax,world in zip(axes,CFG['worlds']):
 sub=base[base.world==world].set_index('method').loc[methods];ax.bar(np.arange(len(methods)),sub['mean']*1000,color=[colors[m] for m in methods]);ax.errorbar(np.arange(len(methods)),sub['mean']*1000,yerr=sub.ci95*1000,fmt='none',ecolor='black',capsize=3);ax.set_xticks(np.arange(len(methods)),[labels[m] for m in methods],rotation=55,ha='right');ax.set_title(world);ax.set_yscale('log');ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Positions-RMSE [mm], logarithmisch');fig1.suptitle('Zurückgehaltene Kombinationen | Mittel und 95%-t-CI über 5 Trainingsseeds');fig1.tight_layout();fig1.savefig(fig/'prediction.png');fig1.savefig(fig/'prediction.pdf');plt.close(fig1)
fig2,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
for ax,world in zip(axes,CFG['worlds']):
 sub=cs[cs.world==world].set_index('method').loc[methods];means=sub['mean'].to_numpy()*100;ci=sub.ci95.to_numpy()*100
 ax.bar(np.arange(len(methods)),means,color=[colors[m] for m in methods]);ax.errorbar(np.arange(len(methods)),means,yerr=np.stack([np.minimum(ci,means),np.minimum(ci,100-means)]),fmt='none',ecolor='black',capsize=3);ax.set_xticks(np.arange(len(methods)),[labels[m] for m in methods],rotation=55,ha='right');ax.set_ylim(0,105);ax.set_title(world);ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Erfolg [%]');fig2.suptitle('Gleicher Planner und gleiche PhysX-Referenzwelt | 12 Episoden pro Seed');fig2.tight_layout();fig2.savefig(fig/'control.png');plt.close(fig2)
fig3,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
lc=summary(p[p.noise==0],'position_rmse',['world','method','budget']);lc.to_csv('results/learning_curve.csv',index=False)
for ax,world in zip(axes,CFG['worlds']):
 for method in ['identified','analytic','joint','moe','modular','no_history','privileged']:
  sub=lc[(lc.world==world)&(lc.method==method)].sort_values('budget');ax.errorbar(sub.budget*2,sub['mean']*1000,yerr=sub.ci95*1000,label=labels[method],marker='o',color=colors[method],capsize=3)
 ax.set_title(world);ax.set_xlabel('Kalibriertrajektorien insgesamt (A+B)');ax.set_xticks([12,48]);ax.set_yscale('log');ax.grid(alpha=.2)
axes[0].set_ylabel('Positions-RMSE [mm]');axes[1].legend(fontsize=8,bbox_to_anchor=(1.03,1));fig3.tight_layout();fig3.savefig(fig/'learning_curve.png');plt.close(fig3)
fig4,axes=plt.subplots(2,2,figsize=(12,7))
worst=[]
for j,world in enumerate(CFG['worlds']):
 truth=np.load(f'results/data/{world}_test.npz');pred=np.load(f'results/evaluation/{world}_modular_trajectories.npz');idx=int(np.argmax(np.sqrt(((pred['next_state'][:,:,0]-truth['next_state'][:,:,0])**2).mean(1))));worst.append({'world':world,'episode_index':idx})
 t=np.arange(CFG['steps'])*DT
 for ax,key,dim,ylabel in [(axes[0,j],'next_state',0,'Position [m]'),(axes[1,j],'normal',None,'Normalkraft [N]')]:
  values=truth[key][idx,:,dim] if dim is not None else truth[key][idx];ax.plot(t,values,'k',lw=2,label='Referenz')
  for m in ['nominal','analytic','moe','modular','joint']:
   dat=np.load(f'results/evaluation/{world}_{m}_trajectories.npz');values=dat[key][idx,:,dim] if dim is not None else dat[key][idx];ax.plot(t,values,label=labels[m],color=colors[m],alpha=.85)
  ax.set_xlabel('Zeit [s]');ax.set_ylabel(ylabel);ax.grid(alpha=.2)
 axes[0,j].set_title(f'{world}: schlechtester modularer Verlauf, Seed 0, Episode {idx}')
axes[1,1].legend(fontsize=8,bbox_to_anchor=(1.03,1));fig4.tight_layout();fig4.savefig(fig/'failure_cases.png');plt.close(fig4);dump('results/worst_cases.json',worst)
fig5,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
ns=summary(native[(native.budget==24)&(native.noise==0)],'position_rmse',['world','method']);ns.to_csv('results/summary_native.csv',index=False)
for ax,world in zip(axes,CFG['worlds']):
 for i,m in enumerate(['nominal','identified','analytic','moe','modular','joint']):
  r=base[(base.world==world)&(base.method==m)].iloc[0];r2=ns[(ns.world==world)&(ns.method==m)].iloc[0]
  ax.plot([0,1],[r['mean']*1000,r2['mean']*1000],marker='o',color=colors[m],label=labels[m])
 ax.set_xticks([0,1],['PhysX / regularisiert','MuJoCo / native Reibung']);ax.set_title(world);ax.grid(alpha=.2);ax.set_yscale('log')
axes[0].set_ylabel('Positions-RMSE [mm]');axes[1].legend(fontsize=8,bbox_to_anchor=(1.02,1));fig5.tight_layout();fig5.savefig(fig/'native_stress.png');plt.close(fig5)

# Independent implementation and timestep sensitivity: same initial conditions/actions.
convergence=[]
for world in CFG['worlds']:
 truth=np.load(f'results/data/{world}_test.npz')['next_state'];same=np.load(f'results/mujoco/{world}_test.npz')['next_state'];half=np.load(f'results/mujoco_half/{world}_test.npz')['next_state'];quarter=np.load(f'results/mujoco_quarter/{world}_test.npz')['next_state']
 convergence.append(dict(world=world,physx_vs_mujoco_x_rmse=float(np.sqrt(np.mean((truth[:,:,0]-same[:,:,0])**2))),dt_vs_half_x_rmse=float(np.sqrt(np.mean((same[:,:,0]-half[:,:,0])**2))),half_vs_quarter_x_rmse=float(np.sqrt(np.mean((half[:,:,0]-quarter[:,:,0])**2)))))
dump('results/convergence.json',convergence)
rows=[]
for world in CFG['worlds']:
 for m in methods:
  r=base[(base.world==world)&(base.method==m)].iloc[0];cc=cs[(cs.world==world)&(cs.method==m)].iloc[0];g=full[(full.world==world)&(full.method==m)]
  rows.append({'Welt':world,'Methode':labels[m],'Position mm (95%-CI)':f'{1000*r["mean"]:.2f} ± {1000*r.ci95:.2f}','Geschwindigkeit m/s':f'{g.velocity_rmse.mean():.4f}','Kraft N':f'{g.force_rmse.mean():.3f}','Aufgabenerfolg %':f'{100*cc["mean"]:.1f} ± {100*cc.ci95:.1f}'})
paired=[]
for world in CFG['worlds']:
 ss=full[full.world==world].groupby(['seed','method']).position_rmse.mean().unstack()
 for m in ['nominal','identified','analytic','joint','moe','no_history','decoupled','privileged']:
  mean,ci=interval((ss.modular-ss[m])*1000);paired.append({'Welt':world,'Modular minus':labels[m],'Differenz mm':f'{mean:.2f}','95%-CI mm':f'[{mean-ci:.2f}, {mean+ci:.2f}]'})
metadata=[json.loads(f.read_text()) for f in Path('results/models').glob('*/b*/s*/*/metadata.json')];md=pd.DataFrame(metadata)
cost=md.groupby('method').agg(parameters=('parameters','first'),seconds_mean=('seconds','mean'),seconds_total=('seconds','sum')).reset_index();cost.to_csv('results/model_cost.csv',index=False)
quality=json.loads(Path('results/data_quality.json').read_text());bad=int(full.unstable.sum());control_bad=int(c.unstable.sum());over=int((full.max_compression>.06).sum());positive=float(full.positive_friction_work.max())
noise=summary(p[(p.budget==24)&(p.noise==1)],'position_rmse',['world','method']);noisec=[]
for world in CFG['worlds']:
 for method in ['analytic','modular','joint','moe']:
  r=noise[(noise.world==world)&(noise.method==method)].iloc[0];n=ns[(ns.world==world)&(ns.method==method)].iloc[0];noisec.append({'Welt':world,'Methode':labels[method],'Initialrauschen RMSE mm':f'{r["mean"]*1000:.2f}','Native Reibung RMSE mm':f'{n["mean"]*1000:.2f}'})
text=f'''# Abschlussbericht — null-physics-transfer

Erstellt aus den Ergebnisdateien am {time.strftime('%Y-%m-%d %H:%M %Z')}.

## Urteil

**Negativer Befund für den stärkeren Anspruch einer überlegenen modularen Lernmethode.** Der ausführbare Prototyp komponiert separat gelernte Funktionen ohne Zielkombinationsdaten und verbessert im geprüften Aufbau nominelle Vorhersagen. Er übertrifft jedoch weder das ausdrucksstarke analytische Modell noch die implementierte MoE-Adaption. Ein Vorrang gegenüber etablierten strukturierten Verfahren ist damit nicht gestützt. Ein größerer Paper-Versuch mit dieser unveränderten Architektur ist derzeit nicht gerechtfertigt.

Dies ist ein abgeschlossener begrenzter Simulationsprototyp, kein Realtransfernachweis. Es wurden keine echten G1-Daten erfasst oder verwendet und keine Roboterschnittstellen gestartet. Native PhysX-Kontaktkräfte werden nicht gelernt: programmierte Materialkräfte wirken auf eine von PhysX integrierte geführte Masse. Der zusätzliche MuJoCo-Versuch prüft eine unabhängige Implementierung und native Gelenkreibung.

## Versuchsaufbau, Methode und Fairness

Eine bekannte Nockengeometrie koppelt die Schlittenbewegung an die Anpressung: δ=d+s x; die Beschleunigung folgt m v̇=u−sN+F_t bei m=2 kg. Zwei wahre Materialvarianten besitzen unterschiedliche Relaxations- und Reibungsfunktionen. In A wird die Normallast kontrolliert und Reibung gemessen; B beobachtet Nachgiebigkeit bei deaktivierter tangentialer Reibung. Kraftsensoren sind ideale Messungen der programmierten Lasten, Zustände ideal beobachtet. Die Aufnahme verwendet die bestehende Isaac-Lab-Installation und den nativen PrismaticJoint. Native Kollisionen dieses Materialpaares sind aus, damit keine Kraft doppelt auftritt.

Die zwei vorgeschlagenen MLPs lernen lokale nichtlineare Korrekturen aus Kompression/Geschwindigkeit und kausaler Filterhistorie. Sie werden vor der Testkombination eingefroren. Eine fest vorgegebene mechanische Kopplung berechnet aus den Vorhersagen die Kraft auf den Schlitten. Das gemeinsame neuronale Modell verwendet vergleichbare Kapazität und dieselben gültigen A/B-Supervisionsmasken. Es ist zugleich die Kontrolle ohne getrennte Module. Die bekannte Kennzeichnung von Kalibriereffekten wird **allen** Methoden gegeben. Das analytische Modell erhält ein breites mechanisch passendes Wörterbuch; dieser starke Strukturvorteil ist bewusst sichtbar. Seine Koeffizienten werden aus denselben Messdaten identifiziert, nicht aus Referenzparametern kopiert.

Die MoE-Kontrolle ist eine **reduzierte PRISM-WM-Adaption**, keine vollständige Reproduktion: Experten, Gate und Gram-Schmidt-Zwischenrepräsentation für direkt beobachtete Kraftdynamik. Latenter Encoder, Reward-/Policy-Lernen und Originalbenchmarks fehlen. Sie besitzt mehr Parameter als die Hauptmethode; der Vergleich ist deshalb kein strikter Beweis einer architekturunabhängigen Überlegenheit. Dagegen ist das wesentlich kleinere analytische Modell bereits ein ausreichender Gegenbefund zum stärkeren Lernanspruch. Vollständige Abgrenzung und geprüfte Primärquellen: [Neuheitsmatrix](novelty_matrix.md).

## Daten und Einfrieren

{len(metadata)} Modellläufe: zwei Referenzwelten × fünf Trainingsseeds × zwei Budgets × acht trainierte/identifizierte Bedingungen. Die Entkopplungsablation verwendet dieselben eingefrorenen modularen Gewichte. Je Budget 6 oder 24 vollständige Trajektorien **pro Effekt**, 800 Schritte bei dt=0,0025 s. Volles Budget entspricht 48 Trajektorien / 96 s Kalibrierung pro Referenzwelt; Validierung zusätzlich 16 Trajektorien / 32 s. Privilegiert erhält weitere 12 Kombinationsverläufe / 24 s. Diese Zusatzdaten sind der Hauptmethode gesperrt.

Vier zurückgehaltene Nockensteigungen und je sechs neue Verläufe ergeben 24 Testepisoden pro Referenzwelt. Aufgabenbewertung separat: 12 neue Episoden pro Welt/Seed/Methode; insgesamt {len(c)} Controller-Episoden. Ergebnisse desselben trainierten Modells sind keine unabhängigen Trainingswiederholungen. Die statistische Einheit unten ist der Mittelwert pro Trainingsseed, n=5; 95%-t-Intervalle mit 4 Freiheitsgraden. Deterministische Baselines haben beim vollen Datenbudget identische Ergebnisse in allen Seeds; ein Nullintervall bedeutet **keine** Nullunsicherheit über andere Aufgaben, Referenzgesetze oder Hardware.

Vor der Testgenerierung wurden Code, Konfiguration und Modellstände in `results/protocol_lock.json` per SHA-256 eingefroren. Modelltraining und Skalierung lesen keine Testdaten. Kein Nachtraining nach Einsicht dieser Auswertung. [Vorabprotokoll](experiment_protocol.md), [Entscheidungs-/Fehlerprotokoll](decisions.md).

## Gemessene Hauptresultate

Positions-/Geschwindigkeitsfehler stammen aus offenen 2-s-Rollouts mit identischen Aktionen. Kraft-RMSE umfasst normale und tangentiale Komponente. Der Controller verwendet für alle Modelle dieselbe Kandidatensuche, Zielkosten und Kraftgrenzen; Erfolg erfordert final <2 cm Positionsfehler und <0,06 m/s Geschwindigkeit.

{markdown_table(pd.DataFrame(rows))}

![Offene Vorhersagen](figures/prediction.png)

![Aufgabenerfolg](figures/control.png)

Gepaarte Positionsdifferenzen auf denselben Trainingsseeds; negativ bedeutet geringeren Fehler der modularen Methode:

{markdown_table(pd.DataFrame(paired))}

Bessere Vorhersagen allein begründen keinen allgemeinen Robotiknutzen. Die Controllerresultate zeigen den Nutzen von Modellkorrektur im gewählten einfachen Prüfstand, aber keinen exklusiven Vorteil der modularen Methode gegenüber starken Kontrollen.

## Lernkurven, Historie und Fehlerfälle

![Datenbudget](figures/learning_curve.png)

Nur zwei vorab gewählte Datenbudgets wurden gerechnet; daraus folgt keine belastbare asymptotische Skalierungskurve oder allgemeine Datenersparnis. Fünf Initialisierungen quantifizieren lokale Trainingsvarianz, keine erschöpfende Hyperparametersensitivität. Das privilegierte Modell kann trotz Zusatzdaten schwächer sein: gemeinsame Architektur, festes Trainingsbudget und A/B-Validierung bleiben unverändert. Dies ist kein Nachweis, dass Zielkombinationsdaten nutzlos sind.

![Schlechteste modulare Verläufe bei Seed 0](figures/failure_cases.png)

Die dargestellten Fehlerfälle sind automatisch die jeweils schlechtesten modularen Episoden von Seed 0; die Auswahlregel und Indizes stehen in `results/worst_cases.json`. Die modularen Netze extrapolieren lokale Kraftgesetze unter neuen gekoppelten Zustandsverläufen. Bei solchen Abweichungen kann korrekte mechanische Kopplung alleine keine richtigen Materialkräfte sicherstellen. Die Ablationen ohne Geschichte und ohne x-abhängige Kompression prüfen beide Bestandteile, ohne Modelländerungen am Haupttest vorzunehmen.

## Unabhängige Referenz, Rauschen und Numerik

MuJoCo 3.6.0 implementiert die beiden Referenzgesetze unabhängig, mit eigenem nativen Massenintegrator. Die gemeinsame Formel-Familie ist dabei beabsichtigt; dies ist keine Unabhängigkeit der ursprünglichen Materialannahmen. Der weitere Stress-Test ersetzt die regularisierte Reibung durch native `dof_frictionloss`, ohne neue Kalibrierung. Ein Leistungsabfall ist dort erwartbare Modellklassenverschiebung, kein fairer In-Distribution-Vergleich neuer Simulatortrainings.

{markdown_table(pd.DataFrame(noisec))}

![Native Reibung](figures/native_stress.png)

Die Rauschbedingung stört anfängliche Positions-, Geschwindigkeits- und Kompressionshistorien. Zukünftige kommandierte Anpressung und Aktionen bleiben bekannt. Sie ersetzt keinen Zustandsbeobachter und keine Untersuchung verrauschter Kraftkalibrierung.

Unabhängige Implementierung und Zeitschrittstudie (Positions-RMSE über vollständige Testtrajektorien, m):

{markdown_table(pd.DataFrame(convergence).round(9))}

Im Haupttest: {bad} numerisch instabile Vorhersageepisoden (Budget 24, ohne Initialrauschen), {control_bad} instabile Controller-Episoden; {over} Vorhersageepisoden überschritten 6 cm Kompression. Maximale positive Reibungsarbeit pro Vorhersageepisode: {positive:.3g} J. Die Reibungsrichtung ist konstruktiv beschränkt. Das ist keine allgemeine Passivitäts-/Energieerhaltungsgarantie: der bewegte Anpressaktor liefert Arbeit und die Nachgiebigkeit kann gespeicherte Energie abgeben. Normalkompression ist beabsichtigte Deformation, nicht die Durchdringung zweier nativer Kollisionsgeometrien. Solche Durchdringungen sind in diesem deaktivierten Kontaktpaar nicht definiert.

Der Kraftbilanztest prüft m Δv/dt gegen die tatsächlich an PhysX angelegte Kraft. Die klassischen k/c-Parameter sind im gewählten Kalibrierversuch linear identifizierbar (Rang 2); Spalten des breiten analytischen Wörterbuchs sind korreliert, seine Einzelkoeffizienten daher nicht eindeutig als reale Materialparameter interpretierbar. Details: `results/data_quality.json`.

## Kosten und Reproduzierbarkeit

{markdown_table(cost.assign(seconds_mean=cost.seconds_mean.round(3),seconds_total=cost.seconds_total.round(3)).rename(columns={'method':'Methode','parameters':'Parameter','seconds_mean':'Training s/Lauf','seconds_total':'Training s gesamt'}))}

Summe gemessener Trainings-/Identifikationszeiten: {md.seconds.sum():.2f} s. CPU-Training in der vorhandenen Torch-Umgebung, zwei Torch-Threads; kein GPU-Speicher für Modelltraining. Laufzeiten sind lokale Wallclock-Messungen und beinhalten Scheduling-Effekte gleichzeitig laufender eigener Aufgaben, keine Energie-/Eurokostenmessung. Backend-, Paket- und Git-Stände: [Umgebung](environment.md), `results/environment.json`, `configs/requirements-observed.txt`. GPU-Belegung während Rendering wird in `results/gpu_samples.csv` als Stichprobe protokolliert, **nicht** als vollständiger Peak aller Läufe.

Checkpoints, Hyperparameter, Quellcode-Sperrliste, Rohdaten, CSV-Endpunkte und alle fehlgeschlagenen Infrastrukturversuche bleiben erhalten. [README](../README.md) enthält die tatsächlich ausgeführten Befehle. [STATUS](../STATUS.md) enthält Fortsetzungsbefehle und Prozessstatus. Die Videos sind als Isaac-Sim-Rendering gespeicherter Trajektorien gekennzeichnet; sie erzeugen keine neuen Messergebnisse.

## Grenzen und nächster Schritt

1. Der stärkere Lernanspruch ist nicht gestützt: die passende analytische Identifikation genügt hier besser, außerdem ist die strukturierte MoE-Kontrolle stärker als die vorgeschlagene modulare Methode.
2. Der mechanische Prüfstand und die ideale Sensorik sind stark vereinfacht. Zwei synthetische Gesetze, vier Steigungen, zwei Datenbudgets und fünf Trainingsseeds erlauben nur einen begrenzten Simulationsbefund.
3. Keine Behauptung eines neuen PhysX-Kontaktlösers, von realer Datenersparnis, Ganzkörperstabilität, Herstellercontroller-Reproduktion oder Sim-to-Real-Transfer. Eine vollständige Originalimplementierung der Literaturmethoden bleibt ausstehend.
4. Als nächsten Schritt einen eng abgegrenzten Materialfall wählen, den ein starkes analytisches Modell nachweislich nicht erfasst, und zunächst Identifizierbarkeit und Messhistorie klären. Erst danach eine neue Architektur begründen und mit **frisch zurückgehaltenen** Kombinationen testen. Ein unverändertes Hochskalieren dieses Prototyps ist wissenschaftlich nicht sinnvoll.
'''
if Path('results/g1_fixed_demo.json').exists():
 g1=json.loads(Path('results/g1_fixed_demo.json').read_text())
 gd=np.load('results/g1_fixed_demo.npz');gf,ga=plt.subplots(1,2,figsize=(11,4));tt=gd['observations'][:,0]
 for j,name in enumerate(g1['used_joints']):
  ga[0].plot(tt,gd['joint_position'][:,j],label=name);ga[0].plot(tt,gd['joint_target'][:,j],'--',alpha=.6)
 ga[0].set(xlabel='Zeit [s]',ylabel='Gelenkposition [rad]',title='Fixierter G1: Ist / Soll (gestrichelt)');ga[0].legend(fontsize=7)
 ga[1].plot(tt,gd['observations'][:,4],'k',label='Referenz')
 for m in ['analytic','moe','modular','joint']:ga[1].plot(tt,gd['prediction_'+m][:,0],label=labels[m],color=colors[m])
 ga[1].set(xlabel='Zeit [s]',ylabel='Normalkraft [N]',title='Ein-Schritt-Lastvorhersage, gemeinsame Messhistorie');ga[1].legend(fontsize=7)
 gf.tight_layout();gf.savefig(fig/'g1_fixed.png');plt.close(gf)
 text+='\n\n## Ergänzender fixierter G1-Versuch (explorativ)\n\n'
 text+='Nach Abschluss der eingefrorenen Hauptauswertung wurde das vorhandene Unitree-G1-29-DOF/Dex3-Asset mit fest montierter Basis und bestehenden impliziten Aktuatorparametern ausgeführt. Zwei linke Armgelenke verfolgen vorgegebene Sollpositionen; die lokale Materiallast greift am linken Handgelenk an. Die Basisbewegung beträgt '+f"{g1['max_root_displacement_m']:.3g}"+' m. Kontakte sind für dieses Lastmodell deaktiviert; die programmierten Materialkräfte sind die einzige Wechselwirkung mit der Vorrichtung.\n\n'
 text+='Alle Modelle sagen aus derselben beobachteten Armtrajektorie Ein-Schritt-Lasten voraus; keine Modellanpassung und kein Austausch des Controllers zwischen Modellen. Nur Seed 0, ein 4-s-Verlauf: keine statistische Transferbestätigung. Kraft-RMSE über beide Komponenten: \n\n'
 text+=markdown_table(pd.DataFrame([{'Modell':labels[m],'Kraft-RMSE N':f'{e:.4f}'} for m,e in g1['force_rmse_N'].items()]))
 text+='\n\nGelenkverfolgungs-RMSE (rad): '+', '.join(f'{e:.4f}' for e in g1['joint_tracking_rmse_rad'])+'. Auch dieser Versuch stützt keinen Vorteil der modularen Methode gegenüber der analytischen Kontrolle. Er bestätigt die technische Wiederverwendung eingefrorener lokaler Modelle auf einer anderen simulierten Mechanik, bei idealen beobachteten Kontaktgrößen.\n\n'
 text+='![G1-Zusatzversuch](figures/g1_fixed.png)\n\nAsset-/Konfigurations-Hashes, Gelenkreihenfolge und Aktuatorparameter stehen in `results/g1_fixed_demo.json`; das aufgezeichnete Video in `videos/g1_fixed_load_demo.mp4`. Der fixierte G1 ist ausdrücklich kein Nachweis von Ganzkörperstabilität, Herstellerfirmware oder realem Transfer. Es wurde kein SDK importiert und kein echter G1 benutzt.\n'

support=[]
for world in CFG['worlds']:
 tr=np.load(f'results/data/{world}_train.npz');te=np.load(f'results/data/{world}_test.npz');rows_tr=json.loads(Path(f'results/data/{world}_train.json').read_text())['rows'];ia=[i for i,r in enumerate(rows_tr) if r['mode']=='A'];ib=[i for i,r in enumerate(rows_tr) if r['mode']=='B']
 dmin,dmax=float(tr['compression'][ib].min()),float(tr['compression'][ib].max());vmin,vmax=float(tr['state'][ia,:,1].min()),float(tr['state'][ia,:,1].max());outside=(te['compression']<dmin)|(te['compression']>dmax)|(te['state'][:,:,1]<vmin)|(te['state'][:,:,1]>vmax)
 support.append({'Welt':world,'Kalibrierung δ m':f'[{dmin:.4f}, {dmax:.4f}]','Test δ m':f"[{te['compression'].min():.4f}, {te['compression'].max():.4f}]",'Außerhalb lokaler δ/v-Bereiche %':f'{100*outside.mean():.1f}'})
text+='\n\n## Lokale Abdeckung und zusätzliche Zustandsmetriken\n\n'
text+=markdown_table(pd.DataFrame(support))
text+='\n\nDie Haupttests enthalten auch Kontaktöffnung und lokale Zustandsbereiche außerhalb der Kalibrierhülle. Sie isolieren deshalb Komposition nicht vollständig von lokaler Extrapolation. Diese nachträgliche Abdeckungsprüfung ändert weder Testauswahl noch Modell; der negative Befund wird dadurch nicht zu einem positiven Kompositionsnachweis umgedeutet. Ein Folgeexperiment sollte lokale Zustands- und Historienabdeckung vorab gezielt abgleichen.\n\n'
state_table=full.groupby(['world','method'])[['contact_accuracy','stick_accuracy']].mean().reset_index();state_table[['contact_accuracy','stick_accuracy']]=(100*state_table[['contact_accuracy','stick_accuracy']]).round(2)
text+=markdown_table(state_table.rename(columns={'world':'Welt','method':'Methode','contact_accuracy':'Kontaktzustand % korrekt','stick_accuracy':'Niedriggeschwindigkeit % korrekt'}))
text+='\n\nNiedriggeschwindigkeit bedeutet hier |v|<0,01 m/s und ist bei regularisierter Reibung kein Nachweis echter Haftung. Native Stick-Slip-Dynamik wird separat im MuJoCo-Stress-Test geprüft.\n'

(out/'final_report.md').write_text(text)
print('REPORT_WRITTEN',out/'final_report.md',flush=True)
