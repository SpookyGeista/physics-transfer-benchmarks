"""Exploratory task-relevance check; no learned predictor, selection or fresh test.

Read-only imports from study_v2. Native execution is an explicit --stage run;
--stage self-test does not start Isaac Sim. All scientific settings are fixed.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT.parent / "study_v2"
sys.path.insert(0, str(STUDY / "src"))
from decision_study.common import DT, MASS, cam  # noqa: E402
from decision_study.reference import Reference  # noqa: E402

OUT = ROOT / "results/preflight"
WORLDS = ("maxwell", "boucwen")
METHODS = ("nominal", "sensor_persistence_mu035", "oracle")
FORCES = np.linspace(-12.0, 12.0, 41)
HORIZON = 100
REPLAN = 20
WARM_STEPS = round(4.0 / DT)
STEPS = round(1.2 / DT)
SWITCH = round(0.6 / DT)
TAIL = round(0.1 / DT)
CHECKPOINTS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
PROTOCOL = """# Controller-Preflight: Aufgabenrelevanz vor neuer Bestätigung

Dieser **explorative Vorversuch** untersucht, ob genauere Materialvorhersagen
bei einer anspruchsvolleren, festgelegten Stellaufgabe überhaupt nutzbar sind.
Er verwendet keine gelernten Modelle und keine neuen Bestätigungsdaten. Die
alten Studien, Gewichte und Auswertungsregeln bleiben unverändert. Ein
privilegierter Oracle ist eine Diagnose der Materialmodellierung innerhalb
derselben beschränkten Planungsstrategie, kein global optimaler Controller.
Der echte G1 und sämtliche Roboter- oder DDS-Schnittstellen bleiben ungenutzt.

## Vorab festgelegter Aufbau

Die vorhandenen Maxwell- und Bouc-Wen-Referenzen treiben denselben nativen
Isaac-Lab/PhysX-Schlitten (CPU, Masse 2 kg, Zeitschritt 2,5 ms). Je Welt werden
alle 16 Kombinationen aus Vorbereitungsvorzeichen ±1, Cam-Amplitude ±0,005 m,
Wellenphase 0/π und erstem Zielvorzeichen ±1 ausgewertet. Die Reihenfolge steht
in `results/preflight/declaration.json`; keine Kombination wird ausgewählt.

Vier Sekunden Vorbereitung halten den Rahmen bei x=v=0 in einer expliziten
idealen Einspannvorrichtung: δ=0,02±0,008 sin²(πt) m während der ersten Sekunde,
danach drei Sekunden δ=0,02 m. Die wahren Materialhistorien werden bei jedem
Schritt aktualisiert. Die Einspannung trägt die resultierende Kraft; der
native Schlitten erhält Nettokraft null. Dies ist keine actuierte Aufgabe und
geht nicht in deren Kosten ein. Die Vorbereitungszustände und Kräfte werden
gespeichert; Drift der Einspannung über 10⁻⁹ m bzw. m/s bricht den Lauf ab.

Die anschließende freie Bewegung dauert 1,2 s. Der allen Planern bekannte
Kompressionsbefehl lautet d(t)=0,02+0,003 sin(2π·0,8t+φ) m. Die tatsächliche
Kompression ist δ=d+g(x), mit g(x)=a tanh(x/0,04) und δ̇=ḋ+g′(x)v. Der Startwert
d(0)=0,02 m ist für beide Phasen identisch. Das Ziel springt anfangs auf ±0,03 m
und nach 0,6 s auf das jeweilige Gegenzeichen. Der bekannte Zielwechsel wird
auch innerhalb des Planungshorizonts berücksichtigt. Die am Ende eines
Integrationsschritts bewertete Zielvorgabe ist diejenige, die am Beginn dieses
Schritts gilt: Der Checkpoint 0,6 s bewertet das erste Ziel vor dem Wechsel.

Alle drei Controller planen alle 20 Schritte (50 ms) über 100 Schritte
(250 ms). Genau 41 konstante Kraftkandidaten von −12 bis +12 N werden mit
derselben bekannten Geometrie und derselben semiimpliziten Integration
bewertet. Stufenkosten sind (x−Ziel)²+0,002v²+10⁻⁶u²; aufsummiert wird mit dt.
Der gewählte Kandidat bleibt bis zur nächsten Planung konstant. Gleichstände
werden durch die feste aufsteigende Kandidatenreihenfolge aufgelöst.

* **Nominal:** N=max(500δ,0) N, Ft=−0,35N tanh(v/0,04).
* **Sensor-Persistenz μ=0,35:** Die ideale aktuelle Normalkraft wird über den
  Planungshorizont konstant gehalten; Ft=−0,35Naktuell tanh(v/0,04). Es gibt
  weder eine nachträglich identifizierte Reibzahl noch Materialhistorien.
* **Oracle:** Exakte Referenzgleichungen und vollständige private Material-
  zustände; nur dieser Planer hat diesen ausdrücklich privilegierten Zugriff.
  Ein separater Klon pro Kandidat beginnt mit dem Zustand **vor** dem aktuellen
  Kraft-/Historienupdate. Die aktuelle Kraftmessung erfolgt an einem weiteren
  Wegwerfklon. Dadurch rücken weder Messung noch Planung die reale Historie vor.
  Die reale Referenz wird pro realem Schritt genau einmal aktualisiert.

Alle Planer erhalten ideale aktuelle Position, Geschwindigkeit und Normalkraft
und die identischen bekannten Kompressions- und Zielbefehle. Der Oracle nutzt
zusätzliche geheime Zustände; sein möglicher Vorteil ist daher kein Nachweis
eines für ein lernendes Modell erreichbaren Vorteils. Die tatsächliche Kraft
u−g′(x)N+Ft treibt PhysX; es gibt während der Bewegung keine Solltrajektorie oder
inverse Dynamik. Die Vorhersage erfolgt bei allen Planern in derselben
semiimpliziten numerischen Näherung an diesen nativen Schlitten.

## Festgelegte Auswertung und Grenzen

Alle Checkpoints **0,2; 0,4; 0,6; 0,8; 1,0; 1,2 s** bleiben sichtbar. Erfolg
bedeutet wie in V2: mittlerer absoluter Positionsfehler der unmittelbar letzten
100 ms <4 mm und mittlerer absoluter Betrag der Geschwindigkeit <0,04 m/s.
Zusätzlich werden Momentanfehler und Momentanerfolg getrennt gespeichert.
Ein nachträglich günstiger ausgewählter Zeitpunkt ist nicht zulässig.

Gespeichert werden zeitkontinuierliche Stufenkosten, Integralgesamtkosten,
Zustände, Aktionen, reale Kräfte, Befehle, Planzeiten und alle Kandidatenkosten.
Das erste vollständig erfolgreiche 100-ms-Fenster wird je Zielhälfte als
früheste Ankunft diagnostiziert; es garantiert kein anschließendes Verbleiben.
Für ein strengeres Einschwingen wird zusätzlich der erste Beginn eines
mindestens 100 ms langen Restintervalls gespeichert, in dem **alle** einzelnen
Schritte bis zum Ende der Zielhälfte innerhalb beider Toleranzen bleiben.
Nicht erreichte Ankünfte/Einschwingzeiten stehen als NaN in NPZ und leer in CSV.

Vergleiche sind gepaart über die identischen 16 festgelegten Bedingungen.
Kostenverhältnisse, alle Checkpoint-Erfolgsraten und Kostenunterschiede werden
deskriptiv berichtet; diese 16 Bedingungen sind keine unabhängigen Material-
replikate, deshalb gibt es weder konfirmatorische p-Werte noch Seed-Konfidenz-
intervalle. Ein klarer Oracle-Kostenvorteil bzw. frühere/effizientere Ankunft
motiviert weitere Modellforschung. Liegt Sensor-Persistenz ähnlich gut wie der
Oracle, spricht dieser konkrete Vorversuch gegen einen großen zusätzlichen
Nutzen gelernter Materialmodelle. Weder Schluss erlaubt eine allgemeine Aussage
zur Robotik, Skalierbarkeit oder Gleichwertigkeit. Der Vorversuch darf eine
spätere Aufgabenwahl begründen, zählt selbst aber nicht als deren Bestätigung.

## Reproduktion und Status

```bash
OPENBLAS_NUM_THREADS=1 /home/loya2/miniconda3/envs/unitree_sim_env/bin/python research_next/scripts/controller_preflight.py --stage self-test
OPENBLAS_NUM_THREADS=1 /home/loya2/miniconda3/envs/unitree_sim_env/bin/python research_next/scripts/controller_preflight.py --stage declare
# Native Ausführung nach Review durch den Hauptagenten:
OPENBLAS_NUM_THREADS=1 /home/loya2/miniconda3/envs/unitree_sim_env/bin/python research_next/scripts/controller_preflight.py --stage run --headless --device cpu
OPENBLAS_NUM_THREADS=1 /home/loya2/miniconda3/envs/unitree_sim_env/bin/python research_next/scripts/controller_preflight.py --stage report
```

Läufe können mit `--worlds maxwell` bzw. `--worlds boucwen` getrennt werden.
Bereits vollständige Ergebnisse werden nur bei identischem Quell-/Protokollhash
wiederverwendet. Teildateien werden nicht überschrieben. Ein Manifest erfasst
die Quellen von Referenz, Geometrie, IsaacRig und diesem Skript sowie alle
festen Parameter. Ergebnisse liegen ausschließlich unter `research_next`.

"""


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def conditions():
    return [dict(episode=i, preparation_sign=p, amplitude=a, phase=phase,
                 target_sign=s)
            for i, (p, a, phase, s) in enumerate(itertools.product(
                (-1, 1), (-0.005, 0.005), (0.0, np.pi), (-1, 1)))]


def preparation(t, signs):
    t = np.asarray(t, dtype=float)
    signs = np.asarray(signs, dtype=float)
    return (0.02 + signs * 0.008 * np.where(t < 1, np.sin(np.pi*t)**2, 0),
            signs * 0.008 * np.where(t < 1, np.pi*np.sin(2*np.pi*t), 0))


def commands(rows):
    t = np.arange(STEPS + HORIZON) * DT
    phase = np.array([r["phase"] for r in rows])[:, None]
    w = 2*np.pi*0.8
    d = 0.02 + 0.003*np.sin(w*t + phase)
    rate = 0.003*w*np.cos(w*t + phase)
    target = (np.array([r["target_sign"] for r in rows])[:, None] * 0.03
              * np.where(np.arange(len(t)) < SWITCH, 1, -1))
    return d, rate, target


def clone_reference(ref, repeats=1):
    """Explicit oracle-only expansion of each independent hidden material state."""
    clone = copy.deepcopy(ref)
    if repeats != 1:
        clone._n = ref._n * repeats
        clone._normal_history = np.repeat(ref._normal_history, repeats, axis=0)
        clone._friction_history = np.repeat(ref._friction_history, repeats, axis=0)
    return clone


def observe_force(ref, delta, rate, velocity):
    """A sensor read cannot advance the physical/reference hidden state."""
    return clone_reference(ref).force(delta, rate, velocity, dt=DT)


def stage_cost(state, u, target):
    return (state[:, 0]-target)**2 + 0.002*state[:, 1]**2 + 1e-6*u**2


def predicted_step(method, state, u, d, rate, amplitude, normal, oracle=None):
    x, v = state.T
    g, slope = cam(x, amplitude)
    delta = d + g
    if method == "oracle":
        if oracle is None:
            raise ValueError("Oracle reference state is required")
        n, ft = oracle.force(delta, rate+slope*v, v, dt=DT)
    elif method == "nominal":
        n = np.maximum(500.0*delta, 0)
        ft = -0.35*n*np.tanh(v/0.04)
    elif method == "sensor_persistence_mu035":
        n = normal
        ft = -0.35*n*np.tanh(v/0.04)
    else:
        raise ValueError(method)
    nv = v + DT*(u-slope*n+ft)/MASS
    ns = np.column_stack((x+DT*nv, nv))
    if not np.isfinite(ns).all() or np.max(np.abs(ns)) > 20:
        raise FloatingPointError("Preflight predicted trajectory left the fixed bounds")
    return ns, n, ft


def plan(method, state, normal, d, rate, target, amplitude, preupdate_ref):
    batch = len(state)
    k = len(FORCES)
    ss = np.repeat(state, k, axis=0)
    nn = np.repeat(normal, k)
    aa = np.repeat(amplitude, k)
    u = np.tile(FORCES, batch)
    rr = clone_reference(preupdate_ref, k) if method == "oracle" else None
    total = np.zeros(batch*k)
    for step in range(d.shape[1]):
        ss, _, _ = predicted_step(method, ss, u, np.repeat(d[:, step], k),
                                 np.repeat(rate[:, step], k), aa, nn, rr)
        total += DT*stage_cost(ss, u, np.repeat(target[:, step], k))
    scores = total.reshape(batch, k)
    return FORCES[np.argmin(scores, axis=1)], scores


def manifest():
    sources = [Path(__file__), STUDY/"src/decision_study/reference.py",
               STUDY/"src/decision_study/common.py", STUDY/"src/decision_study/isaac_rig.py",
               STUDY/"configs/experiment.json"]
    body = dict(exploratory=True, worlds=list(WORLDS), methods=list(METHODS),
                dt=DT, mass=MASS, forces=FORCES.tolist(), horizon=HORIZON,
                replan=REPLAN, warm_steps=WARM_STEPS, motion_steps=STEPS,
                target_switch_step=SWITCH, checkpoints_s=list(CHECKPOINTS),
                tail_steps=TAIL, position_tolerance=0.004, velocity_tolerance=0.04,
                rows=conditions(), source_sha256={str(p.relative_to(ROOT.parent)):
                hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                protocol_sha256=hashlib.sha256(PROTOCOL.encode()).hexdigest())
    body["manifest_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return body


def declare(out):
    out.mkdir(parents=True, exist_ok=True)
    doc = manifest()
    path = out/"declaration.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if prior["manifest_sha256"] != doc["manifest_sha256"]:
            raise RuntimeError("Existing declaration differs; review required, no overwrite")
    else:
        doc["declared_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(path, doc)
    return doc


def arrival_metrics(ns, target):
    """Return per-half first successful tail-end and stricter settling start."""
    arrival = np.full((len(ns), 2), np.nan)
    settle = np.full_like(arrival, np.nan)
    for h, (begin, end) in enumerate(((0, SWITCH), (SWITCH, STEPS))):
        error = np.abs(ns[:, begin:end, 0]-target[:, begin:end])
        speed = np.abs(ns[:, begin:end, 1])
        for i in range(len(ns)):
            for j in range(TAIL-1, end-begin):
                if error[i, j-TAIL+1:j+1].mean() < 0.004 and speed[i, j-TAIL+1:j+1].mean() < 0.04:
                    arrival[i, h] = (j+1)*DT
                    break
            inside = (error[i] < 0.004) & (speed[i] < 0.04)
            suffix = np.logical_and.accumulate(inside[::-1])[::-1]
            starts = np.flatnonzero(suffix[:len(suffix)-TAIL+1])
            if len(starts):
                settle[i, h] = (starts[0]+1)*DT
    return arrival, settle


def native_run(args, out, doc):
    # Import order is required by the installed Isaac Lab; CPU only here.
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    try:
        import torch
        from decision_study.isaac_rig import IsaacRig
        torch.set_num_threads(1)
        rows = conditions()
        n = len(rows)
        amplitude = np.array([r["amplitude"] for r in rows])
        signs = np.array([r["preparation_sign"] for r in rows])
        d, rate, target = commands(rows)
        rig = IsaacRig(n, "cpu", dt=DT)
        for world in args.worlds:
            for method in args.methods:
                path = out/f"{world}_{method}.npz"
                meta = path.with_suffix(".json")
                if path.exists() and meta.exists():
                    old = json.loads(meta.read_text())
                    if old["manifest_sha256"] != doc["manifest_sha256"]:
                        raise RuntimeError("Saved result has a different declaration")
                    if old["npz_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                        raise RuntimeError("Saved trajectory hash differs")
                    print("SKIP", path.name, flush=True)
                    continue
                if path.exists() or meta.exists():
                    raise RuntimeError(f"Incomplete result requires review: {path}")
                tic = time.perf_counter()
                rig.reset()
                ref = Reference(world, n, 0.02)
                warm = {k: [] for k in ("delta", "rate", "normal", "friction", "state")}
                for step in range(WARM_STEPS):
                    dd, rr = preparation(step*DT, signs)
                    nn, ff = ref.force(dd, rr, np.zeros(n), dt=DT)
                    ss = rig.advance(np.zeros(n))  # Ideal fixed-frame clamp bears material load.
                    if np.max(np.abs(ss)) > 1e-9:
                        raise RuntimeError("Clamped preparation drifted from x=v=0")
                    for key, value in (("delta", dd), ("rate", rr), ("normal", nn),
                                       ("friction", ff), ("state", ss)):
                        warm[key].append(value.copy())
                warm_seconds = time.perf_counter()-tic
                rec = {k: [] for k in ("state", "next_state", "normal", "friction", "u",
                                       "delta", "compression_rate", "net_force", "stage_cost")}
                plans, plan_seconds, measurement_residuals = [], [], []
                plan_clock = time.perf_counter()
                u = np.zeros(n)
                for step in range(STEPS):
                    state = rig.state()
                    g, slope = cam(state[:, 0], amplitude)
                    delta = d[:, step] + g
                    dr = rate[:, step] + slope*state[:, 1]
                    if step % REPLAN == 0:
                        observed_n, observed_ft = observe_force(ref, delta, dr, state[:, 1])
                        t0 = time.perf_counter()
                        u, costs = plan(method, state, observed_n, d[:, step:step+HORIZON],
                                        rate[:, step:step+HORIZON], target[:, step:step+HORIZON],
                                        amplitude, ref)
                        plan_seconds.append(time.perf_counter()-t0)
                        plans.append(costs)
                    normal, ft = ref.force(delta, dr, state[:, 1], dt=DT)
                    if step % REPLAN == 0:
                        residual = max(np.max(np.abs(normal-observed_n)), np.max(np.abs(ft-observed_ft)))
                        measurement_residuals.append(residual)
                        if residual > 1e-12:
                            raise AssertionError("Sensor/planner advanced the real reference state")
                    net = u-slope*normal+ft
                    ns = rig.advance(net)
                    if not np.isfinite(ns).all() or np.max(np.abs(ns)) > 20:
                        raise FloatingPointError("Native trajectory exceeded fixed numerical bounds")
                    for key, value in (("state", state), ("next_state", ns), ("normal", normal),
                        ("friction", ft), ("u", u), ("delta", delta), ("compression_rate", dr),
                        ("net_force", net), ("stage_cost", stage_cost(ns, u, target[:, step]))):
                        rec[key].append(value.copy())
                motion_seconds = time.perf_counter()-plan_clock
                arrays = {k: np.stack(v, axis=1) for k, v in rec.items()}
                arrays.update({"warm_"+k: np.stack(v, axis=1) for k, v in warm.items()})
                arrays.update(d=d[:, :STEPS], rate=rate[:, :STEPS], target=target[:, :STEPS],
                              amplitude=amplitude, time=(np.arange(STEPS)+1)*DT,
                              candidate_cost=np.stack(plans, axis=1),
                              plan_seconds=np.array(plan_seconds),
                              measurement_force_residual=np.array(measurement_residuals))
                arrays["integral_cost"] = DT*np.sum(arrays["stage_cost"], axis=1)
                arrays["arrival_s"], arrays["settling_s"] = arrival_metrics(arrays["next_state"], arrays["target"])
                np.savez_compressed(path, **arrays)
                write_json(meta, dict(world=world, method=method, rows=rows,
                    manifest_sha256=doc["manifest_sha256"],
                    npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    warm_seconds=warm_seconds, motion_seconds=motion_seconds,
                    planner_seconds=float(sum(plan_seconds)), total_seconds=time.perf_counter()-tic,
                    backend="native CPU PhysX, installed IsaacRig", oracle_privileged=method == "oracle",
                    clamped_preparation=True, learned_model_used=False, actual_reference_updates=WARM_STEPS+STEPS))
                print("DONE", path.name, f"{time.perf_counter()-tic:.2f}s", flush=True)
    finally:
        app.close(wait_for_replicator=False, skip_cleanup=True)


def csv_write(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def optional_number(x):
    return float(x) if np.isfinite(x) else None


def report(out, doc):
    episodes, checkpoint_rows, datasets, runtime = [], [], {}, []
    for world, method in itertools.product(WORLDS, METHODS):
        path = out/f"{world}_{method}.npz"
        if not path.exists() or not path.with_suffix(".json").exists():
            continue
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["manifest_sha256"] != doc["manifest_sha256"] or meta["npz_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise RuntimeError("Result declaration/content changed")
        with np.load(path) as z:
            data = {k: z[k] for k in z.files}
        datasets[world, method] = data
        runtime.append(dict(world=world, method=method, **{k: meta[k] for k in
                       ("warm_seconds", "motion_seconds", "planner_seconds", "total_seconds")}))
        for row in meta["rows"]:
            i = row["episode"]
            base = dict(world=world, method=method, **row)
            episodes.append(dict(**base, integral_cost=float(data["integral_cost"][i]),
                tracking_rmse_m=float(np.sqrt(np.mean((data["next_state"][i, :, 0]-data["target"][i])**2))),
                force_squared_integral=float(DT*np.sum(data["u"][i]**2)),
                arrival_first_s=optional_number(data["arrival_s"][i, 0]),
                arrival_second_s=optional_number(data["arrival_s"][i, 1]),
                settling_first_s=optional_number(data["settling_s"][i, 0]),
                settling_second_s=optional_number(data["settling_s"][i, 1])))
            for deadline in CHECKPOINTS:
                end = round(deadline/DT)
                tail = data["next_state"][i, end-TAIL:end]
                target = data["target"][i, end-1]
                error = float(np.mean(np.abs(tail[:, 0]-target)))
                speed = float(np.mean(np.abs(tail[:, 1])))
                checkpoint_rows.append(dict(**base, checkpoint_s=deadline,
                    target_m=float(target), mean_position_error_m=error, mean_abs_velocity_m_s=speed,
                    success=bool(error < 0.004 and speed < 0.04),
                    instantaneous_position_error_m=float(abs(tail[-1, 0]-target)),
                    instantaneous_abs_velocity_m_s=float(abs(tail[-1, 1])),
                    instantaneous_success=bool(abs(tail[-1, 0]-target) < 0.004 and abs(tail[-1, 1]) < 0.04),
                    cumulative_cost=float(DT*np.sum(data["stage_cost"][i, :end]))))
    csv_write(out/"episodes.csv", episodes)
    csv_write(out/"checkpoints.csv", checkpoint_rows)
    csv_write(out/"runtime.csv", runtime)
    paired = []
    for world in WORLDS:
        for comparator in METHODS[:-1]:
            if (world, "oracle") not in datasets or (world, comparator) not in datasets:
                continue
            oracle, other = datasets[world, "oracle"], datasets[world, comparator]
            for row in conditions():
                i = row["episode"]
                a, b = float(oracle["integral_cost"][i]), float(other["integral_cost"][i])
                paired.append(dict(world=world, comparator=comparator, **row,
                    oracle_cost=a, comparator_cost=b, oracle_minus_comparator=a-b,
                    oracle_cost_ratio=a/b))
    csv_write(out/"paired_costs.csv", paired)
    summary = dict(manifest_sha256=doc["manifest_sha256"], complete=len(datasets) == 6,
                   episode_count=len(episodes), checkpoint_count=len(checkpoint_rows),
                   runs=[])
    for world, method in datasets:
        er = [r for r in episodes if r["world"] == world and r["method"] == method]
        cr = [r for r in checkpoint_rows if r["world"] == world and r["method"] == method]
        summary["runs"].append(dict(world=world, method=method,
            mean_cost=float(np.mean([r["integral_cost"] for r in er])),
            checkpoint_success_fraction={str(t): float(np.mean([r["success"] for r in cr if r["checkpoint_s"] == t])) for t in CHECKPOINTS},
            arrival_first_count=sum(r["arrival_first_s"] is not None for r in er),
            arrival_second_count=sum(r["arrival_second_s"] is not None for r in er),
            settling_first_count=sum(r["settling_first_s"] is not None for r in er),
            settling_second_count=sum(r["settling_second_s"] is not None for r in er)))
    summary["paired"] = []
    for world, comparator in itertools.product(WORLDS, METHODS[:-1]):
        pr = [r for r in paired if r["world"] == world and r["comparator"] == comparator]
        if pr:
            a, b = sum(r["oracle_cost"] for r in pr), sum(r["comparator_cost"] for r in pr)
            summary["paired"].append(dict(world=world, comparator=comparator,
                mean_cost_difference=float(np.mean([r["oracle_minus_comparator"] for r in pr])),
                ratio_of_mean_costs=a/b,
                oracle_lower_cost_count=sum(r["oracle_minus_comparator"] < 0 for r in pr)))
    write_json(out/"summary.json", summary)
    text = PROTOCOL
    if not episodes:
        text += "Native Ergebnisse stehen aus. Der native Lauf wird erst nach Code-Review durch den Hauptagenten gestartet.\n"
    else:
        text += "## Ergebnisse des explorativen Vorversuchs\n\n"
        text += f"{'Vollständig' if summary['complete'] else 'Teilstand'}: {len(episodes)} Episoden, {len(checkpoint_rows)} festgelegte Checkpoint-Zeilen.\n\n"
        text += "| Welt | Controller | Mittlere Kosten | Erfolg 0,2 / 0,4 / 0,6 / 0,8 / 1,0 / 1,2 s | Ankunft erste / zweite Hälfte |\n|---|---|---:|---|---:|\n"
        for r in summary["runs"]:
            successes = " / ".join(f"{100*v:.0f}%" for v in r["checkpoint_success_fraction"].values())
            text += f"| {r['world']} | {r['method']} | {r['mean_cost']:.7f} | {successes} | {r['arrival_first_count']}/16 / {r['arrival_second_count']}/16 |\n"
        text += "\nDie Paare verwenden dieselben Bedingungen, aber unter Feedback unterschiedliche reale Trajektorien. Die folgenden Kostenverhältnisse vergleichen die Mittelwerte; sie sind keine statistischen Konfidenzgrenzen.\n\n"
        for r in summary["paired"]:
            text += f"* {r['world']}, Oracle gegenüber {r['comparator']}: Kostenverhältnis {r['ratio_of_mean_costs']:.3f}; mittlere Differenz {r['mean_cost_difference']:+.7f}; geringere Oracle-Kosten in {r['oracle_lower_cost_count']}/16 Bedingungen.\n"
        text += "\nDie Deutung erfolgt anhand beider Welten, aller Deadlines und der Sensor-Persistenz gemeinsam. Ergebnisse dieses Vorversuchs sind ausschließlich Entwicklungsbefunde und ersetzen keine frische Bestätigung.\n\n"
        text += "| Welt | Controller | Vorbereitung s | Bewegung inkl. Planer s | Planer s |\n|---|---|---:|---:|---:|\n"
        for r in runtime:
            text += f"| {r['world']} | {r['method']} | {r['warm_seconds']:.2f} | {r['motion_seconds']:.2f} | {r['planner_seconds']:.2f} |\n"
        text += "\nRohdaten und Volltabellen: [results/preflight](../results/preflight/), insbesondere `episodes.csv`, `checkpoints.csv`, `paired_costs.csv`, `runtime.csv` und `summary.json`. Jede NPZ enthält sämtliche Schritte, Vorbereitungsdaten und 41 Kandidatenkosten je Planung.\n"
    (ROOT/"reports/controller_preflight.md").write_text(text)
    return summary


def self_test(out):
    tests = []
    rows = conditions()
    assert len(rows) == 16 and len({tuple(r[k] for k in ("preparation_sign", "amplitude", "phase", "target_sign")) for r in rows}) == 16
    tests.append("full_16_condition_cross")
    tt = np.arange(WARM_STEPS+1)*DT
    dd, rr = preparation(tt[None, :], np.array([-1, 1])[:, None])
    np.testing.assert_allclose(dd[:, [0, 400, -1]], 0.02, atol=1e-16)
    np.testing.assert_allclose(dd[:, 200], [0.012, 0.028], atol=1e-16)
    np.testing.assert_allclose(dd[:, 400:], 0.02, atol=1e-16)
    np.testing.assert_allclose(rr[:, 400:], 0, atol=1e-16)
    tests.append("preparation_1s_pulse_3s_hold")
    d, rate, target = commands(rows)
    np.testing.assert_allclose(d[:, 0], 0.02, atol=1e-16)
    np.testing.assert_array_equal(target[:, SWITCH-1], [0.03*r["target_sign"] for r in rows])
    np.testing.assert_array_equal(target[:, SWITCH], [-0.03*r["target_sign"] for r in rows])
    assert tuple(round(t/DT) for t in CHECKPOINTS) == (80, 160, 240, 320, 400, 480)
    tests.append("fixed_waveforms_target_switch_deadlines")
    for world in WORLDS:
        ref = Reference(world, 2, [0.016, 0.024])
        for j in range(13):
            ref.force([0.018+j*1e-4, 0.023-j*1e-4], [0.04, -0.04], [0.05, -0.03])
        saved = clone_reference(ref)
        n, ft = observe_force(ref, [0.023, 0.017], [0.012, -0.009], [0.04, -0.06])
        np.testing.assert_array_equal(ref._normal_history, saved._normal_history)
        np.testing.assert_array_equal(ref._friction_history, saved._friction_history)
        expanded = clone_reference(ref, 3)
        xn, xf = expanded.force(np.repeat([0.023, 0.017], 3), np.repeat([0.012, -0.009], 3), np.repeat([0.04, -0.06], 3))
        np.testing.assert_allclose(xn, np.repeat(n, 3), atol=1e-12)
        np.testing.assert_allclose(xf, np.repeat(ft, 3), atol=1e-12)
        np.testing.assert_array_equal(ref._normal_history, saved._normal_history)
        np.testing.assert_array_equal(ref._friction_history, saved._friction_history)
        # Candidate rollout must start at this exact pre-update state, not at
        # the state returned by a side-effecting current-force measurement.
        state = np.array([[0.003, 0.08], [-0.002, -0.06]])
        aa = np.array([0.005, -0.005])
        g, slope = cam(state[:, 0], aa)
        observed = observe_force(ref, d[:2, 0]+g, rate[:2, 0]+slope*state[:, 1], state[:, 1])[0]
        u, scores = plan("oracle", state, observed, d[:2, :3], rate[:2, :3], target[:2, :3], aa, ref)
        expected = np.zeros((2, len(FORCES)))
        for k, force in enumerate(FORCES):
            one = clone_reference(ref)
            ss = state.copy()
            for step in range(3):
                ss, _, _ = predicted_step("oracle", ss, np.full(2, force), d[:2, step], rate[:2, step], aa, observed, one)
                expected[:, k] += DT*stage_cost(ss, np.full(2, force), target[:2, step])
        np.testing.assert_allclose(scores, expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(u, FORCES[np.argmin(expected, axis=1)])
        np.testing.assert_array_equal(ref._normal_history, saved._normal_history)
        np.testing.assert_array_equal(ref._friction_history, saved._friction_history)
        tests.append(world+"_nonmutating_observation_batch_clone_and_exact_time_index")
    ns = np.zeros((1, STEPS, 2))
    targets = np.zeros((1, STEPS))
    arrival, settle = arrival_metrics(ns, targets)
    np.testing.assert_allclose(arrival, 0.1)
    np.testing.assert_allclose(settle, DT)
    ns[:, :, 0] = 1
    arrival, settle = arrival_metrics(ns, targets)
    assert np.isnan(arrival).all() and np.isnan(settle).all()
    tests.append("arrival_and_settling_success_and_nonarrival")
    result = dict(passed=True, tests=tests, count=len(tests), native_simulation_started=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  utc=datetime.now(timezone.utc).isoformat())
    write_json(out/"self_test.json", result)
    print(json.dumps(result, indent=2))


def main():
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--stage", choices=("self-test", "declare", "run", "report"), default="self-test")
    stage, _ = preliminary.parse_known_args()
    parser = argparse.ArgumentParser(description=__doc__, parents=[preliminary])
    parser.add_argument("--worlds", nargs="+", choices=WORLDS, default=list(WORLDS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    if stage.stage == "run":
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.stage == "self-test":
        self_test(OUT)
        if not (ROOT/"reports/controller_preflight.md").exists():
            (ROOT/"reports/controller_preflight.md").write_text(PROTOCOL+"Native Ergebnisse stehen aus; nur synthetische Prüfungen wurden ausgeführt.\n")
        return
    if args.stage == "run" and getattr(args, "device", "cpu") != "cpu":
        raise ValueError("This preflight is fixed to CPU; pass --device cpu")
    doc = declare(OUT)
    if args.stage == "run":
        native_run(args, OUT, doc)
    report(OUT, doc)
    print("PREFLIGHT_"+args.stage.upper()+"_COMPLETE", doc["manifest_sha256"], flush=True)


if __name__ == "__main__":
    main()
