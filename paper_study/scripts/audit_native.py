"""Independent read-only audit of saved native development batches.

No simulator is imported or launched. Truth arrays enter ONLY this evaluator;
the observer and its forecasting inputs are neither imported nor modified.
The constitutive clock check replays saved preparation commands and saved
mechanical inputs through the existing reference transition, without Isaac.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / "study_v2" / "src"))
from decision_study.reference import Reference


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def maximum_abs(a):
    return float(np.max(np.abs(a))) if np.size(a) else 0.


def stats(a):
    a = np.asarray(a, dtype=float)
    return dict(mean=float(a.mean()), min=float(a.min()), max=float(a.max()))


def cam(x, amplitude):
    # Preserve stored native float32 kinematics in this generator-equivalence
    # check. The independent float64 geometry residual is reported separately.
    z = np.tanh(x / .04)
    return amplitude*z, amplitude*(1-z*z)/.04


def segment_metrics(z, first, last, dt, mass, end_d):
    """Report both left-end powers and force times recorded displacement.

    The first is the declared convention used in earlier studies. The second
    shows finite-step work along the actual saved path; it is not a continuum
    exactness claim. Normal command work is included as its own actuator channel.
    """
    old = z["state"][:, first:last].astype(float)
    new = z["next_state"][:, first:last].astype(float)
    n, ft, u = (z[key][:, first:last] for key in ("normal", "friction", "u"))
    amp = z["amplitude"][:, None]
    g, slope = cam(old[..., 0], amp)
    gn, _ = cam(new[..., 0], amp)
    v = old[..., 1]
    rates = z["d_dot"][:, first:last]
    powers = dict(tangential_actuator=u*v, compression_actuator=n*rates,
                  friction=ft*v, cam_reaction=-n*slope*v,
                  normal_material=n*z["compression_rate"][:, first:last])
    dnext = np.column_stack((z["d"][:, 1:], np.full(len(old), end_d)))[:, first:last]
    dd = dnext-z["d"][:, first:last]
    dx = new[..., 0]-old[..., 0]
    path_works = dict(tangential_actuator=u*dx, compression_actuator=n*dd,
                      friction=ft*dx, cam_reaction=-n*(gn-g),
                      normal_material=n*(dd+gn-g))
    rows = []
    for e in range(len(old)):
        row = dict(episode=e, steps=last-first, duration_s=(last-first)*dt,
                   displacement_m=float(dx[e].sum()),
                   path_length_m=float(np.abs(dx[e]).sum()),
                   slip_integral_m=float(dt*np.abs(v[e]).sum()),
                   max_abs_position_m=maximum_abs(np.concatenate((old[e, :, 0], new[e, :, 0]))),
                   max_abs_velocity_m_s=maximum_abs(np.concatenate((old[e, :, 1], new[e, :, 1]))),
                   force_squared_integral_N2_s=float(dt*np.sum(u[e]**2)),
                   guard_steps=int(z["guard"][e, first:last].sum()))
        for key, values in powers.items():
            row[key+"_signed_left_J"] = float(dt*values[e].sum())
            row[key+"_absolute_left_J"] = float(dt*np.abs(values[e]).sum())
            row[key+"_signed_path_J"] = float(path_works[key][e].sum())
            row[key+"_absolute_path_J"] = float(np.abs(path_works[key][e]).sum())
        row["total_actuator_signed_left_J"] = row["tangential_actuator_signed_left_J"]+row["compression_actuator_signed_left_J"]
        row["total_actuator_absolute_left_J"] = row["tangential_actuator_absolute_left_J"]+row["compression_actuator_absolute_left_J"]
        row["total_actuator_absolute_path_J"] = row["tangential_actuator_absolute_path_J"]+row["compression_actuator_absolute_path_J"]
        if last > first:
            dv = new[e, :, 1]-old[e, :, 1]
            kinetic = .5*mass*(new[e, -1, 1]**2-old[e, 0, 1]**2)
            net_left = dt*np.sum(z["net_force"][e, first:last]*old[e, :, 1])
            row["kinetic_energy_change_J"] = float(kinetic)
            row["semiimplicit_velocity_energy_correction_J"] = float(.5*mass*np.sum(dv**2))
            row["kinetic_left_balance_residual_J"] = float(kinetic-net_left-.5*mass*np.sum(dv**2))
        else:
            row.update(kinetic_energy_change_J=0., semiimplicit_velocity_energy_correction_J=0.,
                       kinetic_left_balance_residual_J=0.)
        rows.append(row)
    return rows


def prepared_pair(metadata, cfg, cache):
    preparation = metadata["prepared_reference_audit"]
    fingerprint = hashlib.sha256(json.dumps(preparation, sort_keys=True).encode()).hexdigest()
    if fingerprint not in cache:
        commands = {key: np.asarray(value) for key, value in preparation["commands"].items()}
        n = metadata["count"]
        normal = Reference(metadata["normal_world"], n, cfg["initial_compression_m"])
        friction = Reference(metadata["friction_world"], n, cfg["initial_compression_m"])
        assert commands["v"].shape == (n, 1601)
        assert maximum_abs(commands["time"]-np.arange(1601)*cfg["dt"]) < 1e-14
        for k in range(1600):
            friction.force(commands["delta"][:, k], commands["rate"][:, k],
                           commands["v"][:, k], normal_override=commands["normal"][:, k], dt=cfg["dt"])
        truth = metadata["initial_truth_audit"]
        assert truth["assembly_updates"] == 0
        np.testing.assert_array_equal(normal._normal_history, truth["normal_history"])
        np.testing.assert_array_equal(friction._friction_history, truth["friction_history"])
        cache[fingerprint] = normal, friction
    normal, friction = cache[fingerprint]
    truth = metadata["initial_truth_audit"]
    assert truth["normal_world"] == metadata["normal_world"]
    assert truth["friction_world"] == metadata["friction_world"]
    assert truth["assembly_updates"] == 0
    np.testing.assert_array_equal(normal._normal_history, truth["normal_history"])
    np.testing.assert_array_equal(friction._friction_history, truth["friction_history"])
    for index, row in enumerate(preparation["rows"]):
        assert row["index"] == index and row["seed"] == cfg["preparation_seed"] and row["split"] == cfg["split"]
    return copy.deepcopy(normal), copy.deepcopy(friction), fingerprint


def constitutive_replay(z, meta, cfg, cache):
    normal, friction, fingerprint = prepared_pair(meta, cfg, cache)
    e, steps = z["normal"].shape
    errors_n, errors_ft = [], []
    for k in range(steps):
        n, _ = normal.force(z["delta"][:, k], z["compression_rate"][:, k], 0.,
                             friction_enabled=False, dt=cfg["dt"])
        _, ft = friction.force(z["delta"][:, k], z["compression_rate"][:, k], z["state"][:, k, 1],
                                normal_override=n, dt=cfg["dt"])
        errors_n.append(maximum_abs(n-z["normal"][:, k]))
        errors_ft.append(maximum_abs(ft-z["friction"][:, k]))
    current = z["next_state"][:, -1]
    g, slope = cam(current[:, 0], z["amplitude"])
    terminal_d = cfg["initial_compression_m"] + cfg["future_compression_amplitude_m"]*np.sin(2*np.pi)
    terminal_rate = cfg["future_compression_amplitude_m"]*2*np.pi/cfg["future_seconds"]*np.cos(2*np.pi)
    terminal_normal, _ = normal.force(terminal_d+g, terminal_rate+slope*current[:, 1], 0.,
                                      friction_enabled=False, dt=cfg["dt"])
    return dict(normal_max_error_N=max(errors_n), friction_max_error_N=max(errors_ft),
                preparation_sha256=fingerprint, force_updates_per_episode=steps,
                terminal_normal=terminal_normal)


def audit_batch(path, cfg, cache):
    meta = json.loads(path.with_suffix(".json").read_text())
    assert path.stem == meta["sensor"] and path.parent.name == meta["probe"]["name"]
    assert path.parent.parent.name == f"{meta['normal_world']}_{meta['friction_world']}"
    assert sha(path) == meta["npz_sha256"], f"NPZ hash mismatch: {path}"
    with np.load(path, allow_pickle=False) as archive:
        z = {key: archive[key] for key in archive.files}
    for key, value in z.items():
        assert np.isfinite(value).all(), (path, key)
    dt, mass = cfg["dt"], cfg["mass_kg"]
    e = cfg["episodes_per_batch"]
    probe = meta["probe"]
    p = round(probe["duration_s"]/dt)
    t = p+round(cfg["future_seconds"]/dt)
    assert meta["count"] == e and meta["dt"] == dt and meta["observation_period_s"] == cfg["observation_steps"]*dt
    assert int(z["probe_steps"]) == p
    assert z["state"].shape == z["next_state"].shape == (e, t, 2)
    for key in ("normal", "friction", "delta", "compression_rate", "u", "net_force", "d", "d_dot", "guard", "phase"):
        assert z[key].shape == (e, t), (path, key, z[key].shape)
    obs_steps = np.array(sorted(set(range(0, t+1, cfg["observation_steps"])) | {p, t}))
    np.testing.assert_array_equal(z["observation_steps"], obs_steps)
    assert z["observations"].shape == (e, len(obs_steps), 3)
    np.testing.assert_array_equal(z["state"][:, 0], np.zeros((e, 2)))
    np.testing.assert_array_equal(z["next_state"][:, :-1], z["state"][:, 1:])
    np.testing.assert_array_equal(z["phase"], np.broadcast_to(np.arange(t) >= p, (e, t)))
    np.testing.assert_array_equal(z["amplitude"], np.resize(cfg["cam_amplitudes_m"], e))
    assert meta["sensor_parameters"] == cfg["sensor_conditions"][meta["sensor"]]
    assert [meta["normal_world"], meta["friction_world"]] in cfg["compositions"]
    assert probe in cfg["probes"]
    assert meta["noise_seed"] == cfg["sensor_noise_seed"]+100*cfg["compositions"].index([meta["normal_world"], meta["friction_world"]])

    old, new = z["state"].astype(float), z["next_state"].astype(float)
    g, slope = cam(old[..., 0], z["amplitude"][:, None])
    net_error = z["net_force"]-(z["u"]-slope*z["normal"]+z["friction"])
    newton = mass*(new[..., 1]-old[..., 1])/dt-z["net_force"]
    xpos = new[..., 0]-old[..., 0]-dt*new[..., 1]
    limits = cfg["guard"]
    guards = (~np.isfinite(new).all(axis=2)) | (np.abs(new[..., 0]) > limits["max_abs_position_m"]) | (np.abs(new[..., 1]) > limits["max_abs_velocity_m_s"]) | (z["normal"] < limits["min_normal_N"])
    np.testing.assert_array_equal(z["guard"], guards)
    assert int(guards.sum()) == meta["guards"]
    replay = constitutive_replay(z, meta, cfg, cache)
    fullstate = np.concatenate((z["state"], z["next_state"][:, -1:]), axis=1)
    fullnormal = np.column_stack((z["normal"], replay.pop("terminal_normal")))
    trueobs = np.concatenate((fullstate[:, obs_steps], fullnormal[:, obs_steps, None]), axis=2)
    max_total = round((max(q["duration_s"] for q in cfg["probes"])+cfg["future_seconds"])/dt)
    standard_noise = np.stack([np.random.default_rng(np.random.SeedSequence([meta["noise_seed"], i])).normal(size=(max_total+1, 3)) for i in range(e)])
    sensor = meta["sensor_parameters"]
    sigma = np.array([sensor["sigma_x_m"], sensor["sigma_v_m_s"], sensor["sigma_normal_N"]])
    observation_error = z["observations"]-trueobs-standard_noise[:, obs_steps]*sigma

    # Reconstruct commands from past/current public sensor frames only.
    expected_u = np.empty((e, t)); expected_d = np.empty(t); expected_rate = np.empty(t)
    signs = np.where(np.arange(e) % 2 == 0, 1., -1.)
    obs_at = {int(step): i for i, step in enumerate(obs_steps)}
    boundary = z["observations"][:, obs_at[p]]
    _, boundary_slope = cam(boundary[:, 0], z["amplitude"])
    bias = boundary_slope*boundary[:, 2]
    control = cfg["probe_controller"]
    hold = np.zeros(e)
    raw_excitation_values = []
    for k in range(t):
        if k < p:
            expected_d[k], expected_rate[k] = cfg["initial_compression_m"], 0.
            if k in obs_at:
                observed = z["observations"][:, obs_at[k]]
                _, oslope = cam(observed[:, 0], z["amplitude"])
                excitation = probe["excitation_N"]*np.sin(2*np.pi*k*dt/probe["duration_s"])
                raw_excitation_values.append(excitation)
                hold = np.clip(oslope*observed[:, 2]-control["kp_N_m"]*observed[:, 0]-control["kd_N_s_m"]*observed[:, 1]+signs*excitation, -control["max_force_N"], control["max_force_N"])
            expected_u[:, k] = hold
        else:
            phase = 2*np.pi*((k-p)*dt)/cfg["future_seconds"]
            expected_d[k] = cfg["initial_compression_m"]+cfg["future_compression_amplitude_m"]*np.sin(phase)
            expected_rate[k] = cfg["future_compression_amplitude_m"]*2*np.pi/cfg["future_seconds"]*np.cos(phase)
            expected_u[:, k] = np.clip(bias+signs*cfg["future_excitation_N"]*np.sin(phase), -control["max_force_N"], control["max_force_N"])
    result = dict(normal_world=meta["normal_world"], friction_world=meta["friction_world"],
        sensor=meta["sensor"], probe=probe["name"], duration_s=probe["duration_s"],
        file=str(path.relative_to(REPO)), npz_sha256=sha(path), metadata_sha256=sha(path.with_suffix(".json")),
        episodes=e, steps=t, observation_frames=len(obs_steps), native_batch_seconds=meta["seconds"],
        guard_steps=int(guards.sum()), clipped_action_steps=int(np.sum(np.abs(z["u"]) >= control["max_force_N"])),
        minimum_normal_N=float(z["normal"].min()), maximum_friction_power_W=float(np.max(z["friction"]*old[..., 1])),
        newton_max_residual_N=maximum_abs(newton), newton_rms_residual_N=float(np.sqrt(np.mean(newton**2))),
        semiimplicit_position_max_residual_m=maximum_abs(xpos),
        net_force_geometry_float64_max_residual_N=maximum_abs(net_error),
        delta_geometry_float64_max_residual_m=maximum_abs(z["delta"]-z["d"]-g),
        rate_geometry_float64_max_residual_m_s=maximum_abs(z["compression_rate"]-z["d_dot"]-slope*old[..., 1]),
        observation_reconstruction_max_error=maximum_abs(observation_error),
        command_max_error_N=maximum_abs(z["u"]-expected_u),
        d_command_max_error_m=maximum_abs(z["d"]-expected_d),
        d_dot_command_max_error_m_s=maximum_abs(z["d_dot"]-expected_rate),
        boundary_bias_max_error_N=maximum_abs(z["future_bias"]-bias),
        actual_sampled_excitation_max_N=maximum_abs(raw_excitation_values),
        constitutive_replay=replay,
        segments={name: segment_metrics(z, a, b, dt, mass, cfg["initial_compression_m"])
                  for name, a, b in (("probe", 0, p), ("future", p, t), ("all", 0, t))})
    return result


def degeneracy_audit(folder, cfg):
    result = []
    for nw, fw in cfg["compositions"]:
        for sensor in cfg["sensor_conditions"]:
            a = folder/f"{nw}_{fw}/hold_050/{sensor}.npz"
            b = folder/f"{nw}_{fw}/gentle_050/{sensor}.npz"
            with np.load(a, allow_pickle=False) as za, np.load(b, allow_pickle=False) as zb:
                differences = {key: maximum_abs(za[key].astype(float)-zb[key].astype(float)) for key in za.files}
                exact = {key: bool(np.array_equal(za[key], zb[key])) for key in za.files}
            result.append(dict(normal_world=nw, friction_world=fw, sensor=sensor,
                               maximum_array_differences=differences, exactly_equal_arrays=exact,
                               mechanically_identical=exact["state"] and exact["next_state"],
                               mechanically_equal_within_1e_12=max(differences["state"], differences["next_state"]) < 1e-12,
                               actions_equal_within_1e_12=differences["u"] < 1e-12))
    return result


def write_report(output, result):
    checks = result["checks"]
    batches = result["batches"]
    table = []
    for sensor in result["config"]["sensor_conditions"]:
        for probe in result["config"]["probes"]:
            chosen = [b for b in batches if b["sensor"] == sensor and b["probe"] == probe["name"]]
            p = [e for b in chosen for e in b["segments"]["probe"]]
            f = [e for b in chosen for e in b["segments"]["future"]]
            def mean(rows, key): return np.mean([e[key] for e in rows])
            table.append(f"| {sensor} | {probe['name']} | {mean(p,'path_length_m')*1000:.4f} | {mean(p,'total_actuator_absolute_left_J')*1000:.4f} | {mean(p,'friction_signed_left_J')*1000:.4f} | {mean(f,'total_actuator_absolute_left_J')*1000:.3f} | {mean(f,'compression_actuator_absolute_left_J')*1000:.3f} |")
    markdown = f"""# Unabhängiger Audit der nativen Entwicklungsdaten

**{len(batches)} vollständige Batches, {result['episode_count']} Episoden, {checks['guard_steps']} Guards; alle Integritäts- und Reproduktionsprüfungen bestanden: {str(result['passed']).lower()}.** Die Quelle wurde nicht verändert und kein Simulator gestartet. Es handelt sich um explorative Entwicklung mit zwölf wiederverwendeten Vorgeschichten je Materialpaar, nicht um 768 unabhängige Kalibrierungen oder eine konfirmatorische Stichprobe.

Die nominelle 50-ms-Sinussonde ist **degeneriert**: Bei 25-ms-Aktionsaktualisierung werden nur sin(0) und sin(π) benutzt. Der größte tatsächlich abgetastete Zusatz beträgt {checks['gentle_050_sampled_excitation_max_N']:.3g} N. Alle acht Paarungen mit hold_050 besitzen bis auf arithmetische Rundung dieselben gespeicherten mechanischen Trajektorien: verrauscht bitidentisch, ideal maximal {checks['degenerate_mechanics_max_difference']:.3g} in x/v; größte Aktionsabweichung {checks['degenerate_action_max_difference_N']:.3g} N. Gemeinsame Rückkopplungsstörungen machen auch die verrauschten Paarungen gleich. Diese zwei Namen bilden keinen wirksamen Vergleich verschiedener physischer Anregungen und dürfen nicht als zwei unabhängige Sondenalternativen interpretiert werden.

## Integrität, Zeit und Mechanik

- Alle sechs Hashes der [Erzeugungssperre](../results/native_development/generation_lock.json), alle 64 NPZ-Hashes und alle Metadatenidentitäten stimmen. Der Prozess meldet complete nach {result['process']['seconds']:.3f} s. Metadaten- und Rohdatenhashes vor/nach dem Audit sind identisch.
- Die zusätzlich zur direkten Sperrliste geladene alte Konfiguration `study_v2/configs/experiment.json` ist durch die alte v2-Protokollsperre bestätigt; Masse 2 kg, Schritt 2.5 ms und Cam-Länge 40 mm stimmen mit der neuen Konfiguration überein. Software-/Treiberbinärdateien sind hier nicht neu gehasht.
- {result['native_step_count']} native Episodenschritte; durchgängige gespeicherte Zustandsketten, Start x=v=0, vollständige Messraster einschließlich Sonden-/Zukunftsgrenze und Terminalmessung. Kein zusätzliches Materialupdate durch eine Sensormessung.
- Unabhängig nachgerechnet: max. |2·Δv/dt−F_net| = {checks['newton_max_residual_N']:.3g} N; max. |Δx−dt·v_next| = {checks['semiimplicit_position_max_residual_m']:.3g} m. Dies ist eine Konsistenzprüfung des festen PhysX-Zeitschritts, kein Konvergenznachweis für dt→0.
- Ein rein konstitutiver Offline-Replay der gespeicherten Vorbereitung und jeder gespeicherten lokalen Kinematik ergibt max. Normal-/Reibungskraftabweichungen {checks['normal_replay_max_error_N']:.3g}/{checks['friction_replay_max_error_N']:.3g} N. Er benutzt ausschließlich Evaluator-Truth und wird nie in die Schätzpipeline importiert.
- Die reproduzierten x/v/N-Sensormessungen einschließlich ihrer vorab bestimmten Rauschfolgen weichen maximal {checks['observation_reconstruction_max_error']:.3g} in den jeweiligen SI-Einheiten ab. Aktionen lassen sich ausschließlich aus vergangenen/aktuellen Sensormessungen und bekannten Befehlen rekonstruieren; max. Abweichung {checks['command_max_error_N']:.3g} N. Die Zukunftsaktion nutzt nur den kausalen Grenzzeitpunkt und den vorbestimmten Sinus.
- Kleinste Normalkraft {checks['minimum_normal_N']:.6f} N; größte Reibungsleistung F_t·v = {checks['maximum_friction_power_W']:.3g} W. Tangentiale Dissipation besteht für diese gespeicherten Zustände. Daraus folgt keine allgemeine Passivität des gesamten Material-/Aktuatorsystems.

## Tatsächliche Bewegung und Arbeitsbilanz

Die JSON-Datei enthält **für jede der 64 Bedingungen und jede der zwölf Episoden** getrennte Werte für Sonde, anschließende 250 ms und Gesamtverlauf: Weglänge, Schlupfintegral, Geschwindigkeits-/Positionsmaxima, vorzeichenbehaftete und absolute Arbeit beider Aktuatoren, Reibung, Cam-Reaktion und normaler Materialeingang. Null-Sonden werden als Dauer/Arbeit null geführt. Die 4-s-Vorgeschichte ist eine externe Präparation und nicht in diesen aktiven Episodenkosten enthalten.

Die linke Riemannsumme verwendet die tatsächlich gespeicherten Vorschrittgrößen: W_u=Σu·v·dt, W_d=ΣN·d_dot·dt, W_f=ΣF_t·v·dt. Gesamtaufwand ist Σ(|u·v|+|N·d_dot|)dt; Vorzeichen heben Aufwand nicht gegenseitig auf. **Der bewegte Kompressionskanal darf in der Zukunft nicht kostenlos bleiben.** Sein Sondenanteil ist null, weil d dort konstant ist; ein haltender Regler kann trotzdem tangential Arbeit verrichten. Zusätzlich stehen Kraft×gespeicherte Wegänderung sowie eine Korrektur der diskreten kinetischen Energiebilanz in JSON. Diese beiden Quadraturen dokumentieren die endliche Zeitauflösung und sind keine Messung elektrischer Energie oder thermischer Verluste.

Die folgende Übersicht mittelt deskriptiv gleichgewichtet über die vier Materialpaare und deren zwölf Vorgeschichten; sie ersetzt keine gepaarte Wirksamkeitsanalyse. Alle Arbeiten in mJ, Wege in mm.

| Sensor | Sonde | Sondenweglänge | Sondenaufwand absolut | Sondenreibungsarbeit | Zukunftsaufwand absolut | Davon Kompression |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

## Faire kausale Normalzustands-Baseline

Beim eingefrorenen Bouc-Wen-Normalgesetz gilt bei aktivem, nicht abgeschnittenem Kontakt N=k·δ+c·δ_dot+F·z. Damit ist **z_meas=(N_obs−k·δ_obs−c·δ_dot_obs)/F** eine kausal zulässige aktuelle Messinformation. δ_obs=d+g(x_obs) und δ_dot_obs=d_dot+g′(x_obs)v_obs benötigen weder wahre Reibungskraft noch Beschleunigung oder zukünftige Daten. Die Koeffizienten müssen aus dem unveränderten lokal identifizierten Modell stammen; ideale Sensoren machen ein fehlerhaft identifiziertes Gesetz nicht wahr.

Empfehlung: Vor dem nächsten Vergleich eine einfache Vorhersage-/Messfusion für `normal__z` mit deklarierter Kovarianz als starke Baseline festlegen. x/v/N-Rauschen, ihre gemeinsame Verwendung und Modellbias müssen bei der Unsicherheitsdeutung berücksichtigt werden. Bei N=0, geöffnetem Kontakt oder abgeschnittener Normalkraft ist die algebraische Inversion nicht eindeutig; dort bleibt die Zustandsvorhersage mit expliziter Messbehandlung. Die aktuelle Messung korrigiert den aktuellen Zustand; danach erfolgt genau ein normales Zeitupdate. Kein zukünftiges N darf als Lehrer dienen.

Für Maxwell misst ein einzelnes N nur die gewichtete Summe der drei Relaxationszustände: y=(k_eq+Σk_j)δ+c·δ_dot−N=Σk_j·h_j. Mit h_j,next=a_j h_j+(1−a_j)δ und verschiedenen bekannten Zeitkonstanten ist ein klassischer linearer zeitlicher Zustandsfilter eine faire Option; eine Einzelmessung rekonstruiert nicht eindeutig alle drei Zustände. Die bekannte frische Normalinitialisierung darf verwendet werden.

**Beide Friktionsvarianten müssen denselben kausalen Normalbeobachter und dieselben aktuellen Messwerte erhalten.** Ein zusätzlicher Gewinn aus korrigiertem Normalzustand darf nicht als Gewinn der unbekannten Reibungszustandsschätzung ausgegeben werden. Der bisherige Normalkraft-Hold ist ein klarer bedingter Pilot, aber der bewusste Verzicht auf diese Normalzustandsinformation wäre keine ausreichend starke Hauptbaseline. Aus dem konditionierten Gridposterior folgt keine kalibrierte Unsicherheitsdeckung.

## Reproduktion

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/loya2/miniconda3/envs/unitree_sim_env/bin/python paper_study/scripts/audit_native.py`

[Maschinenlesbarer Audit](native_development_audit.json), [Generator](../scripts/generate_native.py), [unveränderte Entwicklungskonfiguration](../configs/development.json), [Auditcode](../scripts/audit_native.py). Der JSON-Audit enthält seine eigene Quellprüfsumme, alle Eingangsprüfsummen, Grenzwerte der numerischen Konsistenzprüfung und vollständige Einzelwerte. Die tolerierten float32-Rundungen sind keine nachträglich festgelegten Wirksamkeitsgrenzen.
"""
    output.with_suffix(".md").write_text(markdown)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"results/native_development")
    parser.add_argument("--out", type=Path, default=ROOT/"reports/native_development_audit")
    args = parser.parse_args()
    started = time.perf_counter()
    folder = args.data.resolve()
    lock = json.loads((folder/"generation_lock.json").read_text())
    cfg = lock["config"]
    for path, expected in lock["sources_sha256"].items():
        assert sha(path) == expected, f"source hash mismatch: {path}"
    old_cfg_path = REPO/"study_v2/configs/experiment.json"
    old_lock_path = REPO/"study_v2/results/protocol_lock.json"
    old_expected = json.loads(old_lock_path.read_text())["files"][str(old_cfg_path)]
    assert sha(old_cfg_path) == old_expected
    old_cfg = json.loads(old_cfg_path.read_text())
    assert old_cfg["mass"] == cfg["mass_kg"] == 2.
    assert old_cfg["dt"] == cfg["dt"] == .0025 and old_cfg["cam_length"] == .04
    expected = {folder/f"{nw}_{fw}/{probe['name']}/{sensor}.npz" for nw, fw in cfg["compositions"] for probe in cfg["probes"] for sensor in cfg["sensor_conditions"]}
    assert set(folder.rglob("*.npz")) == expected
    assert set(folder.rglob("*.json")) == {p.with_suffix(".json") for p in expected} | {folder/"generation_lock.json", folder/"process.json"}
    process = json.loads((folder/"process.json").read_text())
    assert process["status"] == "complete" and process["batches"] == len(expected)
    assert process["episodes"] == len(expected)*cfg["episodes_per_batch"]
    input_paths = list(lock["sources_sha256"])+[str(p) for p in folder.rglob("*") if p.is_file()]+[str(old_cfg_path), str(old_lock_path)]
    before = {path: sha(path) for path in input_paths}
    cache = {}
    batches = [audit_batch(p, cfg, cache) for p in sorted(expected)]
    degeneracy = degeneracy_audit(folder, cfg)
    checks = dict(guard_steps=sum(b["guard_steps"] for b in batches),
                  clipped_action_steps=sum(b["clipped_action_steps"] for b in batches),
                  minimum_normal_N=min(b["minimum_normal_N"] for b in batches),
                  maximum_friction_power_W=max(b["maximum_friction_power_W"] for b in batches),
                  normal_replay_max_error_N=max(b["constitutive_replay"]["normal_max_error_N"] for b in batches),
                  friction_replay_max_error_N=max(b["constitutive_replay"]["friction_max_error_N"] for b in batches),
                  gentle_050_sampled_excitation_max_N=max(b["actual_sampled_excitation_max_N"] for b in batches if b["probe"] == "gentle_050"),
                  degenerate_action_max_difference_N=max(d["maximum_array_differences"]["u"] for d in degeneracy),
                  degenerate_mechanics_max_difference=max(max(d["maximum_array_differences"]["state"], d["maximum_array_differences"]["next_state"]) for d in degeneracy))
    for key in ("newton_max_residual_N", "semiimplicit_position_max_residual_m", "net_force_geometry_float64_max_residual_N", "delta_geometry_float64_max_residual_m", "rate_geometry_float64_max_residual_m_s", "observation_reconstruction_max_error", "command_max_error_N", "d_command_max_error_m", "d_dot_command_max_error_m_s", "boundary_bias_max_error_N"):
        checks[key] = max(b[key] for b in batches)
    # Engineering consistency limits, not inferential/effect thresholds.
    tolerances = dict(newton_max_residual_N=1e-4, semiimplicit_position_max_residual_m=1e-7,
                      net_force_geometry_float64_max_residual_N=1e-5,
                      delta_geometry_float64_max_residual_m=1e-8,
                      rate_geometry_float64_max_residual_m_s=1e-7,
                      observation_reconstruction_max_error=1e-10,
                      command_max_error_N=1e-10, d_command_max_error_m=1e-12,
                      d_dot_command_max_error_m_s=1e-12, boundary_bias_max_error_N=1e-12,
                      normal_replay_max_error_N=1e-12, friction_replay_max_error_N=1e-12,
                      maximum_friction_power_W=1e-12)
    passes = {key: checks[key] <= limit for key, limit in tolerances.items()}
    passes["no_guards"] = checks["guard_steps"] == 0
    passes["positive_normal_contact"] = checks["minimum_normal_N"] > cfg["guard"]["min_normal_N"]
    passes["degenerate_probe_confirmed"] = all(d["mechanically_equal_within_1e_12"] and d["actions_equal_within_1e_12"] for d in degeneracy)
    assert before == {path: sha(path) for path in input_paths}, "input changed during audit"
    result = dict(role="Independent exploratory native data audit; no simulation or estimator run", passed=all(passes.values()),
                  checks=checks, tolerances=tolerances, check_pass=passes,
                  config=cfg, process=process, batch_count=len(batches), episode_count=process["episodes"],
                  native_step_count=sum(b["episodes"]*b["steps"] for b in batches),
                  unique_preparation_cache_entries=len(cache), input_hashes=before,
                  audit_source_sha256=sha(__file__), audit_seconds=time.perf_counter()-started,
                  definitions=dict(power_quadrature="left-end recorded force*velocity*dt",
                      path_quadrature="recorded force times saved coordinate increment",
                      total_absolute_actuator="sum(abs(u*v)+abs(N*d_dot))*dt",
                      initial_preparation_cost="not included in active trial, fixed externally prepared history",
                      posterior_access="audit-only truth; no estimator imported"),
                  degeneracy=degeneracy, batches=batches)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.out, result)
    print(json.dumps(dict(passed=result["passed"], checks=checks, seconds=result["audit_seconds"],
                          outputs=[str(args.out.with_suffix(".json")), str(args.out.with_suffix(".md"))]), indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
