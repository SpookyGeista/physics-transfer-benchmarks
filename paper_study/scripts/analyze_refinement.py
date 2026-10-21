"""Independent descriptive audit of frozen native/observer timestep runs.

Never modifies existing experiments. Corrected privileged forecasts are new
diagnostics: preparation dt is read from its own design, coupled dt from the run.
Only r1/clean, ideal sensors, filtered_particles, and the declared 16 common
cells are scored. Complete provenance maps are verified, including unscored
artifacts listed in their manifests. No fitting or new native execution occurs.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models import load_model, model_paths, motion_step

LEVELS = {
    "h": ("native_development_v2", "observer_development_v2", 1),
    "h2": ("native_refinement_h2", "observer_refinement_h2", 2),
    "h4": ("native_refinement_h4", "observer_refinement_h4", 4),
}
WORLDS = ("maxwell", "boucwen")
PROBES = ("none", "hold_300", "gentle_150", "strong_300")
METRICS = ("position_m", "velocity_m_s", "normal_N", "friction_N")
FORECAST_KEYS = {
    "prior": "filtered_particles_prior_propagation_forecast",
    "post": "filtered_particles_trajectory_assimilation_forecast",
    "stored_privileged": "filtered_particles_privileged_forecast",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Provenance:
    def __init__(self):
        self.hashes = {}
        self.assertions = 0

    def remember(self, path):
        path = str(Path(path).resolve())
        value = digest(path)
        if path in self.hashes and self.hashes[path] != value:
            raise ValueError(f"Input changed during audit: {path}")
        self.hashes[path] = value
        return value

    def json(self, path):
        self.remember(path)
        return json.loads(Path(path).read_text())

    def verify(self, mapping):
        for path, expected in mapping.items():
            if self.remember(path) != expected:
                raise ValueError(f"Provenance hash mismatch: {path}")
            self.assertions += 1

    def unchanged(self):
        for path, expected in self.hashes.items():
            if digest(path) != expected:
                raise ValueError(f"Audited source/input changed: {path}")


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def summaries(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite numerical audit values")
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)), maximum=float(values.max()), minimum=float(values.min()))


def rmse(predicted, target):
    if predicted.shape != target.shape:
        raise ValueError(f"Unaligned shape: {predicted.shape} vs {target.shape}")
    return np.sqrt(np.mean((predicted-target)**2, axis=1))


def common_grid(values, divisor):
    """State is post-step, forces pre-step. Never slice all four alike."""
    return np.concatenate((values[:, divisor-1::divisor, :2], values[:, ::divisor, 2:]), axis=2)


def truth_tensor(raw, future_only=True):
    start = int(raw["probe_steps"]) if future_only else 0
    return np.concatenate((raw["next_state"][:, start:], raw["normal"][:, start:, None],
                           raw["friction"][:, start:, None]), axis=2)


def load_npz(path):
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def verified_level(level, provenance):
    native_name, observer_name, divisor = LEVELS[level]
    native, observer = ROOT / "results" / native_name, ROOT / "results" / observer_name
    generation = provenance.json(native / "generation_lock.json")
    evaluation = provenance.json(observer / "evaluation_lock.json")
    artifact = provenance.json(observer / "artifact_manifest.json")
    native_process, observer_process = [provenance.json(p / "process.json") for p in (native, observer)]
    if native_process["status"] != "complete" or observer_process["status"] != "complete":
        raise ValueError("Only complete native and observer runs may be audited")
    for mapping in (generation["sources_sha256"], evaluation["sources_sha256"],
                    evaluation["input_sha256"], evaluation["generation_sources_sha256"],
                    artifact["outputs_sha256"]):
        provenance.verify(mapping)
    if provenance.remember(observer / "evaluation_lock.json") != artifact["evaluation_lock_sha256"]:
        raise ValueError("Observer manifest does not identify this evaluation lock")
    if generation["config"]["dt"] != .0025/divisor:
        raise ValueError("Unexpected declared coupled step")
    # Existing CSV is an independently checked output, not the source of scores.
    csv_path = observer / "episodes.csv"
    provenance.remember(csv_path)
    selected_csv = {}
    with csv_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if (row["sensor"] == "ideal" and row["model_seed"] == "1" and row["calibration_noise"] == "clean"
                    and row["forecast_start"] == "filtered_particles" and row["probe"] in PROBES
                    and row["filter_kind"] == "joint_sensor_likelihood"):
                key = (row["normal_world"], row["friction_world"], row["probe"], int(row["episode"]), row["estimator"])
                if key in selected_csv:
                    raise ValueError("Duplicate selected CSV row")
                selected_csv[key] = row
    if len(selected_csv) != 16*48*3:
        raise ValueError(f"Expected 2304 selected CSV rows at {level}, found {len(selected_csv)}")
    return dict(native=native, observer=observer, divisor=divisor, generation=generation,
                evaluation=evaluation, csv=selected_csv,
                native_process=native_process, observer_process=observer_process)


def clone_state(state):
    return {key: np.asarray(value).copy() for key, value in state.items()}


def corrected_privileged(model, raw, metadata, preparation_cache):
    commands = metadata["prepared_reference_audit"]["commands"]
    prep_dt = float(metadata["prepared_reference_audit"]["design"]["dt"])
    if prep_dt != .0025:
        raise ValueError("Unexpected preparation-grid convention")
    d, rate, velocity, normal = [np.asarray(commands[key]) for key in ("delta", "rate", "v", "normal")]
    if velocity.shape[1] != 1601 or float(commands["time"][-1]) != 4.:
        raise ValueError("The unchanged fixture must still contain 1600 steps over 4 seconds")
    h = hashlib.sha256()
    for array in (d, rate, velocity, normal):
        h.update(np.ascontiguousarray(array).tobytes())
    cache_key = (metadata["normal_world"], metadata["friction_world"], h.hexdigest(), prep_dt)
    if cache_key not in preparation_cache:
        friction = model.friction_model
        state_f = friction.initial(d[:, 0], velocity[:, 0], normal[:, 0])
        for step in range(velocity.shape[1]-1):
            _, _, state_f = friction.step(d[:, step], rate[:, step], velocity[:, step], state_f,
                                          normal_override=normal[:, step], dt=prep_dt)
        preparation_cache[cache_key] = clone_state(state_f)
    count, end, dt = raw["u"].shape[0], int(raw["probe_steps"]), metadata["dt"]
    state = model.initial(np.full(count, .02), np.zeros(count), raw["observations"][:, 0, 2])
    state.update({"friction__"+key: value for key, value in clone_state(preparation_cache[cache_key]).items()})
    xv = np.zeros((count, 2))
    prediction = []
    for step in range(raw["u"].shape[1]):
        xv, state, n, ft = motion_step(model, xv, state,
            raw["u"][:, step], raw["d"][:, step], raw["d_dot"][:, step], raw["amplitude"], dt=dt)
        if step >= end:
            prediction.append(np.column_stack((xv, n, ft)))
    result = np.stack(prediction, axis=1)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite corrected fixed-model diagnostic")
    return result, prep_dt


def cell_audit(nw, fw, probe, levels, provenance, output, models, preparation_cache):
    relative = Path("p31001") / f"{nw}_{fw}" / probe / "ideal.npz"
    identifier = f"p31001/{nw}_{fw}/{probe}/ideal"
    raw_by_level, meta_by_level, observer_by_level, predictions, truth, results = {}, {}, {}, {}, {}, {}
    corrected_outputs = []
    max_csv_disagreement = 0.
    for level, info in levels.items():
        native_path = info["native"] / relative
        metadata = provenance.json(native_path.with_suffix(".json"))
        if metadata["status"] != "complete" or metadata["sensor"] != "ideal" or metadata["count"] != 48:
            raise ValueError("Unexpected selected native cell")
        native_hash = provenance.remember(native_path)
        if native_hash != metadata["npz_sha256"]:
            raise ValueError("Native sidecar hash mismatch")
        raw = load_npz(native_path)
        if raw["u"].shape[1] != metadata["expected_steps"] or raw["observation_steps"][-1] != metadata["expected_steps"]:
            raise ValueError("Incomplete selected horizon")
        for key in ("state", "next_state", "normal", "friction", "u", "d", "d_dot", "observations"):
            if not np.isfinite(raw[key]).all():
                raise FloatingPointError(f"Nonfinite selected raw data: {key}")
        pred_path = info["observer"] / "r1/clean" / relative.with_suffix("") / "joint_sensor_likelihood.npz"
        pred_meta = provenance.json(pred_path.with_suffix(".json"))
        pred_hash = provenance.remember(pred_path)
        if (pred_hash != pred_meta["npz_sha256"] or pred_meta["input_npz_sha256"] != native_hash
                or pred_meta["input_metadata_sha256"] != provenance.remember(native_path.with_suffix(".json"))):
            raise ValueError("Prediction provenance does not match its native input")
        observer = load_npz(pred_path)
        original = {name: observer[key] for name, key in FORECAST_KEYS.items()}
        corrected, prep_dt = corrected_privileged(models[(nw, fw)], raw, metadata, preparation_cache)
        if level == "h" and not np.array_equal(corrected, original["stored_privileged"]):
            raise ValueError("Corrected replay must reproduce the unaffected base privileged output exactly")
        original["corrected_privileged"] = corrected
        destination = output / "corrected_privileged" / level / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            np.savez_compressed(stream, forecast=corrected, preparation_dt=prep_dt, coupled_dt=metadata["dt"])
        correction_meta = dict(cell_id=identifier, level=level, preparation_dt=prep_dt,
            preparation_steps=1600, preparation_duration_s=4., coupled_dt=metadata["dt"],
            original_stored_model_preparation_duration_s=1600*metadata["dt"],
            original_fine_privileged_invalid_for_refinement=level != "h",
            input_npz_sha256=native_hash, input_metadata_sha256=provenance.remember(native_path.with_suffix(".json")),
            original_observer_npz_sha256=pred_hash, npz_sha256=digest(destination),
            model_seed=1, calibration_noise="clean", forecast_start="filtered_particles",
            model_sha256={str(p.resolve()): provenance.remember(p) for p in model_paths(nw, fw, 1, "clean")},
            scope="New diagnostic replay only; fixed fitted laws and full hidden local history, no true coupled state initialization or coefficient refit")
        write_json(destination.with_suffix(".json"), correction_meta)
        corrected_outputs.extend([destination, destination.with_suffix(".json")])
        target = truth_tensor(raw)
        aligned_target = common_grid(target, info["divisor"])
        score_record = {}
        for name, prediction in original.items():
            native_error, aligned_error = rmse(prediction, target), rmse(common_grid(prediction, info["divisor"]), aligned_target)
            score_record[name] = dict(own_grid={metric: summaries(native_error[:, j]) for j, metric in enumerate(METRICS)},
                common_grid={metric: summaries(aligned_error[:, j]) for j, metric in enumerate(METRICS)},
                common_grid_episode_rmse=aligned_error.tolist())
            if name != "corrected_privileged":
                estimator = {"prior": "prior_propagation", "post": "trajectory_assimilation",
                             "stored_privileged": "privileged_full_history"}[name]
                for episode in range(48):
                    row = info["csv"][(nw, fw, probe, episode, estimator)]
                    stored = np.array([float(row[key]) for key in ("future_position_rmse_m", "future_velocity_rmse_m_s",
                                                                    "future_normal_rmse_N", "future_friction_rmse_N")])
                    disagreement = float(abs(stored-native_error[episode]).max())
                    max_csv_disagreement = max(max_csv_disagreement, disagreement)
                    if disagreement > 1e-12:
                        raise ValueError("Independently calculated native-grid score differs from CSV")
        gains = (np.asarray(score_record["prior"]["common_grid_episode_rmse"])
                 - np.asarray(score_record["post"]["common_grid_episode_rmse"]))
        results[level] = dict(coupled_dt=metadata["dt"], preparation_dt=prep_dt, scores=score_record,
            gain_prior_minus_post={metric: summaries(gains[:, j]) for j, metric in enumerate(METRICS)},
            gain_episode=gains.tolist(), native_guard_steps=int(raw["guard"].sum()),
            posterior_finite=bool(np.isfinite(observer["weights"]).all()),
            original_stored_privileged_valid=(level == "h"))
        raw_by_level[level], meta_by_level[level], observer_by_level[level] = raw, metadata, observer
        predictions[level] = {name: common_grid(value, info["divisor"]) for name, value in original.items()}
        truth[level] = aligned_target
    base_raw, base_meta, base_observer = raw_by_level["h"], meta_by_level["h"], observer_by_level["h"]
    for level in ("h2", "h4"):
        fine, meta, observer = raw_by_level[level], meta_by_level[level], observer_by_level[level]
        divisor = levels[level]["divisor"]
        for key in ("amplitude", "excitation_sign", "preparation_seed", "preparation_index", "episode_id"):
            if not np.array_equal(fine[key], base_raw[key]):
                raise ValueError(f"Factor mapping differs: {key}")
        if meta["prepared_reference_audit"] != base_meta["prepared_reference_audit"] or meta["initial_truth_audit"] != base_meta["initial_truth_audit"]:
            raise ValueError("The fixture preparation/truth differs across coupled timesteps")
        if not np.allclose(fine["observation_steps"]*meta["dt"], base_raw["observation_steps"]*base_meta["dt"], rtol=0, atol=1e-15):
            raise ValueError("Physical observation clocks differ")
        for key in ("grid", "prior"):
            if not np.array_equal(observer[key], base_observer[key]):
                raise ValueError("Fixed model prior/grid differs across timesteps")
        if int(fine["probe_steps"]) != int(base_raw["probe_steps"])*divisor:
            raise ValueError("Probe boundary alignment differs")
        for key in ("d", "d_dot"):
            if not np.allclose(base_raw[key], fine[key][:, ::divisor], rtol=0, atol=1e-14):
                raise ValueError(f"Exogenous command differs at common physical times: {key}")
    pair_results = {}
    for left, right in (("h", "h2"), ("h2", "h4"), ("h", "h4")):
        differences = rmse(truth[left], truth[right])
        full_left = common_grid(truth_tensor(raw_by_level[left], False), levels[left]["divisor"])
        full_right = common_grid(truth_tensor(raw_by_level[right], False), levels[right]["divisor"])
        full_differences = rmse(full_left, full_right)
        left_u = raw_by_level[left]["u"][:, ::levels[left]["divisor"]]
        right_u = raw_by_level[right]["u"][:, ::levels[right]["divisor"]]
        tv = .5*np.sum(abs(observer_by_level[left]["weights"]-observer_by_level[right]["weights"]), axis=1)
        pair_results[f"{left}_{right}"] = dict(
            native_future_drift={metric: summaries(differences[:, j]) for j, metric in enumerate(METRICS)},
            native_future_episode_drift=differences.tolist(),
            native_complete_run_drift={metric: summaries(full_differences[:, j]) for j, metric in enumerate(METRICS)},
            native_future_max_absolute={metric: float(abs(truth[left][:, :, j]-truth[right][:, :, j]).max()) for j, metric in enumerate(METRICS)},
            policy_command_difference_N=summaries(np.sqrt(np.mean((left_u-right_u)**2, axis=1))),
            max_policy_command_difference_N=float(abs(left_u-right_u).max()),
            future_bias_difference_N=summaries(abs(raw_by_level[left]["future_bias"]-raw_by_level[right]["future_bias"])),
            posterior_total_variation=summaries(tv),
            model_forecast_drift={name: {metric: summaries(rmse(predictions[left][name], predictions[right][name])[:, j])
                                        for j, metric in enumerate(METRICS)} for name in ("prior", "post", "corrected_privileged")})
    ratios = {}
    for metric in METRICS:
        first = pair_results["h_h2"]["native_future_drift"][metric]["mean"]
        second = pair_results["h2_h4"]["native_future_drift"][metric]["mean"]
        ratios[metric] = None if first == 0 else second/first
    drift = pair_results["h_h4"]["native_future_drift"]["position_m"]["mean"]
    gain_h = results["h"]["gain_prior_minus_post"]["position_m"]["mean"]
    gain_h4 = results["h4"]["gain_prior_minus_post"]["position_m"]["mean"]
    return dict(cell_id=identifier, normal_world=nw, friction_world=fw, probe=probe,
        episode_ids=base_raw["episode_id"].tolist(), levels=results, paired_runs=pair_results,
        convergence_ratio_D24_over_D12=ratios, max_csv_absolute_disagreement=max_csv_disagreement,
        coarse_native_drift_exceeds_coarse_information_gain=bool(drift > gain_h),
        gain_change_h_to_h4_m=gain_h4-gain_h,
        corrected_artifacts={str(p): digest(p) for p in corrected_outputs})


def aggregate(cells):
    result = {}
    for group in ("all", "maxwell", "boucwen"):
        selected = cells if group == "all" else [c for c in cells if c["friction_world"] == group]
        result[group] = {}
        for level in LEVELS:
            methods = {}
            for method in ("prior", "post", "stored_privileged", "corrected_privileged"):
                errors = np.concatenate([np.asarray(c["levels"][level]["scores"][method]["common_grid_episode_rmse"]) for c in selected])
                methods[method] = {metric: summaries(errors[:, j]) for j, metric in enumerate(METRICS)}
            gains = np.concatenate([np.asarray(c["levels"][level]["gain_episode"]) for c in selected])
            result[group][level] = dict(methods=methods, gain_prior_minus_post={m: summaries(gains[:, j]) for j,m in enumerate(METRICS)})
    return result


def report_text(result, figure):
    cells = result["cells"]
    lines = ["# Unabhängige numerische Verfeinerung", "", "15.09.2026. Deskriptive Nachrechnung abgeschlossener eingefrorener Entwicklungsdaten; keine neue native Simulation und kein Parameterschätzen.", "",
        "**Befund:** Die Prior-/Assimilationskontraste werden auf drei Schrittweiten separat reproduziert. Die ursprünglichen feineren privilegierten Vergleiche sind wegen eines Präfixzeitfehlers für diesen Vergleich ungültig; ausschließlich neue Auditdateien korrigieren sie. Die tatsächlichen nativen Trajektorien ändern sich wesentlich stärker als die sehr kleinen Fehler mancher jeweils gleich diskretisierter Prädiktoren. Daraus folgt kein Nachweis einer entsprechend kleinen kontinuierlichen physikalischen Vorhersageabweichung.", "",
        "## Umfang und Provenienz", "",
        "Gemeinsam geprüft: 4 Materialkompositionen × {none, hold_300, gentle_150, strong_300} × 48 gekreuzte Episoden = 16 Batches/768 Episoden je Schrittweite. Modell r1/clean, ideale Sensoren, `joint_sensor_likelihood`, `filtered_particles`. Kopplungsschritte: h=2,5 ms, h/2=1,25 ms, h/4=0,625 ms. Messperiode bleibt 25 ms. Externe Präparation bleibt identisch: 1600 Schritte mit 2,5 ms über vier Sekunden, einschließlich exakt gleicher gespeicherter Präfixkommandos und Anfangswahrheit.", "",
        f"Alle sechs Prozessstände sind vollständig. {result['provenance']['assertions']} deklarierte Hashzuordnungen wurden verifiziert; {result['provenance']['unique_files']} unterschiedliche Quellen-/Modell-/Prior-/Roh-/Metadaten-/Outputdateien wurden zusätzlich am Ende auf Unverändertheit geprüft. Vollständige Provenienzlisten in der JSON-Datei. Die Manifestprüfung umfasst auch ungescorte Artefakte; Zahlenanalyse ausschließlich für die genannten 16 idealen Zellen.", "",
        "Das vorhandene Episoden-CSV wurde nur als Gegenkontrolle benutzt. Alle 6912 selektierten Originalscores (3 Raster × 768 Episoden × 3 Schätzer) wurden aus den gespeicherten Forecast-/Wahrheitsarrays neu berechnet. Maximale Abweichung über alle vier SI-Messgrößen: " + f"{max(c['max_csv_absolute_disagreement'] for c in cells):.3e}. Kein Guard in den selektierten nativen Daten: " + str(all(c['levels'][l]['native_guard_steps'] == 0 for c in cells for l in LEVELS)) + ".", "",
        "## Zeitangleichung und Fehler im privilegierten Präfix", "",
        "`next_state` und die ersten zwei Forecastkomponenten sind Nachschrittzustände. Deshalb wird im feinen Raster `(divisor−1)::divisor` gewählt. N und Ft sind Vorschrittkräfte; sie werden mit `::divisor` abgeglichen. Beides in einer gemeinsamen Slice zu behandeln wäre um einen Mikrostep falsch. Zusätzlich wurden Original-RMSE auf jedem eigenen Raster und RMSE auf dem gemeinsamen 100-Punkte-Zukunftsraster berechnet. Die Tabellen unten verwenden das gemeinsame Raster und jeweils den Mittelwert der 48 Episoden-RMSE, keinen gepoolten Punkt-RMSE.", "",
        "Im eingefrorenen `evaluate_native_v2.py`, Funktion `privileged_forecasts`, wird `metadata['dt']` sowohl für die gekoppelte Bewegung als auch für die unverändert 1600 Präfixkommandos verwendet. Damit erhält nur der privilegierte feinere Modellpfad irrtümlich zwei Sekunden bzw. eine Sekunde statt vier Sekunden Präparation. Native Wahrheit, fixe Populationsprior und reguläre Prior-/Posteriorprognosen sind davon nicht betroffen. Der grobe Lauf mit dt=2,5 ms ist ebenfalls unbeeinflusst.", "",
        "Neue `corrected_privileged`-Outputs integrieren denselben vollständigen Präfix mit `metadata.prepared_reference_audit.design.dt=.0025`, anschließend Probe und Zukunft mit dem jeweiligen gekoppelten dt. Keine Koeffizientenänderung, keine wahre gekoppelte Zustandsinitialisierung. Die grobe Nachrechnung ist bitgenau identisch zur gespeicherten groben privilegierten Prognose. Jede neue Datei nennt Eingabe-/Modellhashes und beide Schrittweiten. Ursprüngliche feinere privilegierte Werte bleiben als ungültige Diagnosewerte erhalten; sie dürfen insbesondere keinen vermeintlichen Sieg über die Vollhistory-Baseline begründen.", "",
        "## Gemeinsame Zukunfts-Zustandsfehler", "", "Position in µm. P=Prior, A=Assimilation, V=korrigierte privilegierte Vollhistory.", "",
        "| Normal/Reibung | Probe | P/A/V h | P/A/V h/2 | P/A/V h/4 | native Drift h→h/4 (µm) | D24/D12 |", "|---|---|---:|---:|---:|---:|---:|"]
    for c in cells:
        cols=[]
        for level in LEVELS:
            cols.append(" / ".join(f"{c['levels'][level]['scores'][m]['common_grid']['position_m']['mean']*1e6:.3f}" for m in ("prior","post","corrected_privileged")))
        drift=c['paired_runs']['h_h4']['native_future_drift']['position_m']['mean']*1e6
        ratio=c['convergence_ratio_D24_over_D12']['position_m']
        lines.append(f"| {c['normal_world']}/{c['friction_world']} | {c['probe']} | {' | '.join(cols)} | {drift:.3f} | {ratio:.4f} |")
    lines.extend(["", f"![Schrittweitenvergleich]({figure.name})", "", "## Numerische Drift gegenüber Informationsnutzen", ""])
    active=[c for c in cells if c['probe'] in ('gentle_150','strong_300')]
    low=[c for c in active if c['coarse_native_drift_exceeds_coarse_information_gain']]
    drifts=[c['paired_runs']['h_h4']['native_future_drift']['position_m']['mean']*1e6 for c in cells]
    changes=[abs(c['gain_change_h_to_h4_m'])*1e6 for c in cells]
    ratios=[c['convergence_ratio_D24_over_D12']['position_m'] for c in cells]
    lines.append(f"Die mittlere native Positionsänderung h→h/4 liegt je Zelle zwischen {min(drifts):.3f} und {max(drifts):.3f} µm. Sie übersteigt in {len(low)} der 8 aktiv angeregten Zellen den groben Positionsvorteil der Assimilation: " + ", ".join(f"{c['normal_world']}/{c['friction_world']} {c['probe']}" for c in low) + ". Bei keiner Probe/Ruhe ist der Prior-/Posteriorunterschied erwartungsgemäß praktisch null, während die folgende native Bewegung weiterhin schrittweitenabhängig ist.")
    lines.extend(["", f"Die jeweiligen gepaarten Positionsvorteile auf dem passenden Wahrheitsraster ändern sich h→h/4 um höchstens {max(changes):.3f} µm. Das native D24/D12-Verhältnis liegt zwischen {min(ratios):.4f} und {max(ratios):.4f}, was in diesen drei Rasterschritten einen ungefähr linearen Verfeinerungstrend zeigt. Es ist kein Konvergenzbeweis oder eine obere Fehlergrenze gegen eine kontinuierliche Lösung.", "",
        "Beides ist gleichzeitig möglich: ein stabiler Unterschied zwischen zwei Schätzern innerhalb derselben Diskretisierung und eine wesentlich größere Änderung der physikalischen Trajektorie beim Verfeinern. Geteilte Diskretisierung von Wahrheit und Modell kann Fehler kompensieren. Nanometer-/Submikrometerfehler gegenüber derselben diskreten Wahrheit dürfen daher nicht als absolute Genauigkeit gegenüber der kontinuierlichen Dynamik dargestellt werden. Die JSON-Datei enthält zusätzlich Geschwindigkeits- und Kraft-RMSE, vollständige Maximalabweichungen, Forecastdrift und Posterior-Totalvariation.", "",
        "Die Sinus-/Kompressionsprogramme stimmen zu gemeinsamen physikalischen Zeitpunkten überein. Die gehaltenen Feedbackkräfte und der Zukunftsbias werden jedoch in jedem nativen Lauf aus dessen eigener aktueller idealer Messung berechnet. Daher variieren die tatsächlich angelegten u-Werte geringfügig mit der Schrittweite. Dies ist Verfeinerung derselben geschlossenen Strategie mit gleicher Messperiode, kein Replay identischer gespeicherter offener Kräfte. Die entsprechenden u-/Biasunterschiede stehen gesondert im JSON.", "",
        "## Grenzen und Reproduktion", "",
        "Nur ein Modellseed und ein Präparationsseed, zwei feste Reibungsfamilien, vier Kompositionen, vier Proben und ideale Sensoren. Die 48 Episoden einer Zelle kreuzen sechs Basispräparationen mit Kopplung/Vorzeichen; sie sind keine 48 unabhängigen Materialzüge. Es werden keine p-Werte, Konfidenzintervalle, neuen Erfolgsgrenzen oder konfirmatorischen Entscheidungen aus diesen Daten erzeugt. Verrauschte Konditionen wurden ausgeschlossen, weil eine feinere Zeitindexfolge sonst andere realisierte Zufallszüge erzeugt. Der unveränderte 2,5-ms-Präfix wird gerade nicht numerisch verfeinert. Parameter-/Modellfehler, Likelihoodkalibrierung und globale Passivität werden nicht durch diese Rechnung nachgewiesen.", "",
        "```bash", "PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\", "/home/loya2/miniconda3/envs/unitree_sim_env/bin/python \\", "paper_study/scripts/analyze_refinement.py \\", "--out /home/loya2/physics-transfer/paper_study/results/refinement_independent_audit \\", "--report /home/loya2/physics-transfer/paper_study/reports/numerical_refinement", "```", "",
        "Die Ausgabe ist exklusiv: für eine erneute Ausführung neue `--out`- und `--report`-Pfade verwenden. Eingefrorene Ausgangsquellen und Ergebnisse werden ausschließlich gelesen."])
    return "\n".join(lines)+"\n"


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
    for ax, family in zip(axes, WORLDS):
        selected=[c for c in result['cells'] if c['friction_world']==family and c['probe'] in ('gentle_150','strong_300')]
        for method, label, color in (("prior", "Prior", "#8d6e63"), ("post", "Assimilation", "#007f86"),
                                     ("corrected_privileged", "Vollhistory korrigiert", "#7766aa")):
            y=[np.mean([c['levels'][l]['scores'][method]['common_grid']['position_m']['mean'] for c in selected])*1e6 for l in LEVELS]
            ax.plot([2.5,1.25,.625], y, marker='o', color=color, label=label)
        drift=np.mean([c['paired_runs']['h_h4']['native_future_drift']['position_m']['mean'] for c in selected])*1e6
        ax.axhline(drift, color='#c45535', ls='--', label='Native Drift h→h/4')
        ax.set(xlabel='Kopplungsschritt (ms)', ylabel='Position (µm; logarithmisch)', title=f'{family}: nur aktive Proben', yscale='log')
        ax.set_xticks([.625,1.25,2.5]); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Gepaarten Informationsnutzen und absolute Schrittweitensensitivität getrennt beurteilen', fontsize=11)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/refinement_independent_audit')
    parser.add_argument('--report', type=Path, default=ROOT/'reports/numerical_refinement')
    args=parser.parse_args()
    output, report=args.out.resolve(), args.report.resolve()
    if not output.is_relative_to(ROOT/'results') or not report.is_relative_to(ROOT/'reports'):
        raise ValueError('Audit output/report must stay in new paper_study result/report paths')
    if output.exists() or any(report.with_suffix(s).exists() for s in ('.md','.json','.png')):
        raise FileExistsError('Existing artifacts retained; select fresh output/report paths')
    output.mkdir(parents=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    start=time.perf_counter()
    provenance=Provenance()
    provenance.remember(__file__)
    levels={level: verified_level(level, provenance) for level in LEVELS}
    models={(nw,fw):load_model(nw,fw,1,'clean') for nw in WORLDS for fw in WORLDS}
    cache, cells={},[]
    for nw in WORLDS:
        for fw in WORLDS:
            for probe in PROBES:
                cells.append(cell_audit(nw,fw,probe,levels,provenance,output,models,cache))
                print(f'AUDITED {len(cells)}/16 {nw}/{fw} {probe}',flush=True)
    result=dict(status='complete', role='independent descriptive numerical audit; no confirmatory decision',
        command=sys.argv, seconds=time.perf_counter()-start, cells=cells, aggregate=aggregate(cells),
        provenance=dict(assertions=provenance.assertions, unique_files=len(provenance.hashes),
                        input_and_source_sha256=provenance.hashes),
        correction=dict(original_fine_privileged_invalid=True, original_base_unchanged=True,
                        preparation_dt=.0025, preparation_duration_s=4., coefficients_refitted=False))
    provenance.unchanged()
    figure=report.with_suffix('.png')
    plot(result,figure)
    report.with_suffix('.md').write_text(report_text(result,figure))
    result['report_artifacts_sha256']={str(p):digest(p) for p in (figure,report.with_suffix('.md'))}
    write_json(report.with_suffix('.json'),result)
    products={str(p):digest(p) for p in output.rglob('*') if p.is_file()}
    products.update({str(report.with_suffix(ext)):digest(report.with_suffix(ext)) for ext in ('.md','.json','.png')})
    write_json(output/'artifact_manifest.json',dict(outputs_sha256=products,analysis_source_sha256=digest(__file__)))
    print('COMPLETE',report.with_suffix('.md'),flush=True)


if __name__=='__main__':
    main()
