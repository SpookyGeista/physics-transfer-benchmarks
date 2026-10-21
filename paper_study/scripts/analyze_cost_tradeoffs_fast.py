"""Descriptive error/time/work tradeoffs for a completed native v2 evaluation.

No simulator, estimator update, scalar success weights, p values, or data-driven
exclusions. The retrospective error front is not an online decision policy.
Optional intervals use crossed resampling of model-bank and preparation-seed
blocks only when BOTH have at least eight independent levels. That threshold
is an engineering minimum, not a guarantee of asymptotic coverage.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
BASE = ["preparation_seed", "normal_world", "friction_world", "probe", "sensor", "episode"]
FACTORS = ["normal_world", "friction_world", "sensor", "calibration_noise", "filter_kind", "forecast_start", "estimator"]
IDENTITY = BASE+["model_seed", "calibration_noise", "filter_kind", "forecast_start", "estimator"]
METRICS = ["future_position_mse_m2", "future_friction_mse_N2", "probe_time_s",
           "total_absolute_actuator_work_J", "probe_absolute_actuator_work_J"]
ESTIMATORS = ("prior_propagation", "trajectory_assimilation", "privileged_full_history")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def verify_hash_map(hashes):
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError(f"hash mismatch: {path}")


def physical_costs(arrays, dt):
    """Actual saved pre-step forces/velocities; both physical actuation ports."""
    v = np.asarray(arrays["state"], dtype=float)[..., 1]
    n = np.asarray(arrays["normal"], dtype=float)
    u = np.asarray(arrays["u"], dtype=float)
    rate = np.asarray(arrays["d_dot"], dtype=float)
    if not (v.shape == n.shape == u.shape == rate.shape) or not np.isfinite([v, n, u, rate]).all():
        raise ValueError("invalid physical cost arrays")
    p = int(arrays["probe_steps"])
    if not 0 <= p < v.shape[1] or dt <= 0:
        raise ValueError("invalid probe/horizon")
    tangent, normal = u*v, n*rate
    absolute = np.abs(tangent)+np.abs(normal)
    result = dict(total_absolute_actuator_work_J=dt*absolute.sum(axis=1),
                  probe_absolute_actuator_work_J=dt*absolute[:, :p].sum(axis=1),
                  total_tangential_absolute_work_J=dt*np.abs(tangent).sum(axis=1),
                  total_compression_absolute_work_J=dt*np.abs(normal).sum(axis=1),
                  total_signed_actuator_work_J=dt*(tangent+normal).sum(axis=1),
                  probe_signed_actuator_work_J=dt*(tangent[:, :p]+normal[:, :p]).sum(axis=1),
                  probe_time_s=np.full(len(v), p*dt),
                  future_time_s=np.full(len(v), (v.shape[1]-p)*dt),
                  probe_path_length_m=np.abs(np.asarray(arrays["next_state"], dtype=float)[:, :p, 0]
                                             -np.asarray(arrays["state"], dtype=float)[:, :p, 0]).sum(axis=1),
                  probe_friction_work_J=dt*(arrays["friction"][:, :p]*v[:, :p]).sum(axis=1))
    return result


def nondominated(points):
    """Exact finite descriptive dominance: minimize all coordinates, ties kept."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or not np.isfinite(points).all():
        raise ValueError("finite 2D point matrix required")
    dominated = np.any(np.all(points[:, None, :] <= points[None, :, :], axis=2)
                       & np.any(points[:, None, :] < points[None, :, :], axis=2), axis=0)
    return ~dominated


def crossed_bootstrap(frame, columns, *, draws=2000, seed=9152026, minimum=8):
    """Crossed bank x preparation-seed cluster bootstrap, all cells retained.

    First average all fixed recipe/cam/sign replicates inside each crossed cell.
    Independently resample whole bank and whole preparation-seed blocks, with
    their multiplicities crossed. The same draw weights apply to every column.
    No interval is produced for too few independent blocks.
    """
    cell = frame.groupby(["model_seed", "preparation_seed"], sort=True)[columns].mean()
    banks = sorted(frame.model_seed.unique()); preparations = sorted(frame.preparation_seed.unique())
    info = dict(model_banks=len(banks), preparation_seed_blocks=len(preparations),
                interval_available=False, resamples=0, minimum_clusters_each=minimum)
    if len(banks) < minimum or len(preparations) < minimum:
        info["reason"] = "insufficient independent model banks and/or preparation-seed blocks"
        return {}, info
    expected = pd.MultiIndex.from_product([banks, preparations], names=cell.index.names)
    if set(cell.index) != set(expected):
        raise ValueError("crossed bootstrap requires all bank/preparation cells")
    matrix = cell.reindex(expected).to_numpy().reshape(len(banks), len(preparations), len(columns))
    rng = np.random.default_rng(seed)
    wb = rng.multinomial(len(banks), np.full(len(banks), 1/len(banks)), size=draws)/len(banks)
    wp = rng.multinomial(len(preparations), np.full(len(preparations), 1/len(preparations)), size=draws)/len(preparations)
    estimates = np.einsum("db,dp,bpk->dk", wb, wp, matrix)
    intervals = {key: tuple(map(float, np.quantile(estimates[:, j], [.025, .975]))) for j, key in enumerate(columns)}
    info.update(interval_available=True, reason="descriptive crossed-cluster percentile interval", resamples=draws)
    return intervals, info


def load_cost_rows(native, cfg):
    rows, mapping, hashes = [], {}, {}
    files = sorted(native.rglob("*.npz"))
    expected = {native/f"p{seed}/{nw}_{fw}/{probe['name']}/{sensor}.npz"
                for seed in cfg["preparation_seeds"] for nw, fw in cfg["compositions"]
                for probe in cfg["probes"] for sensor in cfg["sensor_conditions"]}
    if set(files) != expected:
        raise ValueError("native v2 file grid is incomplete or has extra files")
    for path in files:
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("status") != "complete" or metadata["npz_sha256"] != sha(path):
            raise ValueError(f"failed/hash-mismatched native batch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in ("state", "next_state", "normal", "friction", "u", "d_dot", "probe_steps", "guard")}
        factors = metadata["factor_rows"]
        if len(factors) != cfg["episodes_per_batch"] or len(arrays["u"]) != len(factors):
            raise ValueError("native factors/episode count mismatch")
        observed = {(r["preparation_index"], r["cam_index"], r["excitation_sign"]) for r in factors}
        expected_factors = set(itertools.product(cfg["preparation_indices"], range(len(cfg["cam_amplitudes_m"])), cfg["excitation_signs"]))
        if observed != expected_factors or len(observed) != len(factors):
            raise ValueError("native factor crossing incomplete/duplicated")
        costs = physical_costs(arrays, metadata["dt"])
        for episode, factor in enumerate(factors):
            row = {key: metadata[key] for key in ("preparation_seed", "normal_world", "friction_world", "sensor")}
            row.update(probe=metadata["probe"]["name"], episode=episode,
                       preparation_index=factor["preparation_index"], preparation_recipe=factor["preparation_recipe"],
                       cam_index=factor["cam_index"], amplitude_m=factor["amplitude"], excitation_sign=factor["excitation_sign"],
                       native_guard_count=int(arrays["guard"][episode].sum()),
                       native_npz=str(path), native_metadata=str(metadata_path))
            row.update({k: float(v[episode]) for k, v in costs.items()})
            rows.append(row)
        key = (metadata["preparation_seed"], metadata["normal_world"], metadata["friction_world"], metadata["probe"]["name"], metadata["sensor"])
        if key in mapping:
            raise ValueError("duplicate native condition")
        mapping[key] = path
        hashes.update({str(path): sha(path), str(metadata_path): sha(metadata_path)})
    return pd.DataFrame(rows), mapping, hashes


def verify_and_join_metrics(evaluation, native, cfg, cost, mapping):
    metrics = pd.read_csv(evaluation/"episodes.csv")
    if metrics.duplicated(IDENTITY).any():
        raise ValueError("duplicate prediction identity")
    methods = list(itertools.product(cfg["model_seeds"], cfg["calibration_noise"], cfg["filters"], cfg["forecast_starts"], ESTIMATORS))
    expected = {tuple(r[k] for k in BASE)+method for r in cost.to_dict("records") for method in methods}
    actual = set(metrics[IDENTITY].itertuples(index=False, name=None))
    if expected != actual:
        raise ValueError(f"prediction factorial incomplete/extra: expected {len(expected)}, got {len(actual)}")
    rmse_columns = ["future_position_rmse_m", "future_velocity_rmse_m_s", "future_normal_rmse_N", "future_friction_rmse_N"]
    if not np.isfinite(metrics[rmse_columns].to_numpy()).all() or (metrics[rmse_columns] < 0).any().any():
        raise ValueError("invalid prediction metric")
    lookup = metrics.set_index(IDENTITY)
    max_error = 0.
    for condition, path in mapping.items():
        prep, nw, fw, probe, sensor = condition
        with np.load(path, allow_pickle=False) as data:
            end = int(data["probe_steps"])
            target = np.concatenate((data["next_state"][:, end:], data["normal"][:, end:, None], data["friction"][:, end:, None]), axis=2)
        for model_seed, fitnoise, filter_kind in itertools.product(cfg["model_seeds"], cfg["calibration_noise"], cfg["filters"]):
            output = evaluation/f"r{model_seed}/{fitnoise}"/path.relative_to(native).with_suffix("")/f"{filter_kind}.npz"
            meta = json.loads(output.with_suffix(".json").read_text())
            if (meta["input_npz_sha256"] != sha(path) or meta["input_metadata_sha256"] != sha(path.with_suffix(".json"))
                    or meta["npz_sha256"] != sha(output)):
                raise ValueError("forecast/native provenance mismatch")
            with np.load(output, allow_pickle=False) as predictions:
                for start, estimator in itertools.product(cfg["forecast_starts"], ESTIMATORS):
                    key = f"{start}_privileged_forecast" if estimator == "privileged_full_history" else f"{start}_{estimator}_forecast"
                    prediction = predictions[key]
                    if prediction.shape != target.shape:
                        raise ValueError("forecast shape mismatch")
                    rmse = np.sqrt(np.mean((prediction-target)**2, axis=1))
                    # Batch the exact same keyed rows; no metric or aggregation changes.
                    identities = [(prep,nw,fw,probe,sensor,e,model_seed,fitnoise,filter_kind,start,estimator)
                                  for e in range(len(target))]
                    reported = lookup.loc[pd.MultiIndex.from_tuples(identities,names=IDENTITY),rmse_columns].to_numpy(dtype=float)
                    max_error = max(max_error,float(np.max(np.abs(reported-rmse))))
    if max_error > 1e-12:
        raise ValueError(f"NPZ/CSV metric discrepancy {max_error}")
    merged = metrics.merge(cost, on=BASE, validate="many_to_one", suffixes=("", "_physical"))
    if not np.array_equal(merged.native_guard_count, merged.native_guard_count_physical):
        raise ValueError("guard count mismatch")
    merged["future_position_mse_m2"] = merged.future_position_rmse_m**2
    merged["future_friction_mse_N2"] = merged.future_friction_rmse_N**2
    return merged, max_error


def summaries(frame, bootstrap_draws):
    rows, interval_status = [], []
    for keys, group in frame.groupby(FACTORS+["probe"], sort=True):
        row = dict(zip(FACTORS+["probe"], keys))
        # Full crossing is verified above. Macro-average clusters explicitly so
        # future additions cannot silently weight one bank/seed more heavily.
        cells = group.groupby(["model_seed", "preparation_seed"])[METRICS].mean()
        row.update({key: float(cells[key].mean()) for key in METRICS})
        row["pooled_future_position_rmse_m"] = float(np.sqrt(row["future_position_mse_m2"]))
        row["pooled_future_friction_rmse_N"] = float(np.sqrt(row["future_friction_mse_N2"]))
        row["mean_episode_position_rmse_m"] = float(group.future_position_rmse_m.mean())
        row["model_banks"] = int(group.model_seed.nunique())
        row["preparation_seed_blocks"] = int(group.preparation_seed.nunique())
        row["distinct_base_histories"] = len(group[["preparation_seed", "preparation_index"]].drop_duplicates())
        row["descriptive_prediction_rows"] = len(group)
        row["native_guard_steps"] = int(group.native_guard_count.sum())
        intervals, status = crossed_bootstrap(group, METRICS, draws=bootstrap_draws)
        row["interval_available"] = status["interval_available"]
        for metric, interval in intervals.items():
            row[metric+"_ci_low"], row[metric+"_ci_high"] = interval
        interval_status.append({**dict(zip(FACTORS+["probe"], keys)), **status})
        rows.append(row)
    summary = pd.DataFrame(rows)
    for _, group in summary.groupby(FACTORS, sort=True):
        for metric, name in (("future_position_mse_m2", "position_pareto"), ("future_friction_mse_N2", "friction_pareto")):
            flags = nondominated(group[[metric, "probe_time_s", "total_absolute_actuator_work_J"]].to_numpy())
            summary.loc[group.index, name] = flags
    return summary, interval_status


def paired_gains(frame, bootstrap_draws):
    pair_keys = BASE+["model_seed", "calibration_noise", "filter_kind", "forecast_start"]
    prior = frame[frame.estimator == "prior_propagation"].set_index(pair_keys)
    post = frame[frame.estimator == "trajectory_assimilation"].set_index(pair_keys)
    if set(prior.index) != set(post.index):
        raise ValueError("unpaired prior/assimilation rows")
    post = post.reindex(prior.index)
    for key in ("total_absolute_actuator_work_J", "probe_absolute_actuator_work_J", "probe_time_s"):
        np.testing.assert_array_equal(prior[key], post[key])
    paired = prior.reset_index()[pair_keys+["preparation_index"]].copy()
    gain_columns = []
    for metric in ("future_position_mse_m2", "future_friction_mse_N2"):
        key = "prior_minus_assimilated_"+metric
        paired[key] = prior[metric].to_numpy()-post[metric].to_numpy()
        gain_columns.append(key)
    rows = []
    factors = [key for key in FACTORS if key != "estimator"]+["probe"]
    for keys, group in paired.groupby(factors, sort=True):
        row = dict(zip(factors, keys))
        cells = group.groupby(["model_seed", "preparation_seed"])[gain_columns].mean()
        row.update({key: float(cells[key].mean()) for key in gain_columns})
        intervals, info = crossed_bootstrap(group, gain_columns, draws=bootstrap_draws)
        row.update(interval_available=info["interval_available"], model_banks=info["model_banks"], preparation_seed_blocks=info["preparation_seed_blocks"])
        for key, interval in intervals.items():
            row[key+"_ci_low"], row[key+"_ci_high"] = interval
        rows.append(row)
    return paired, pd.DataFrame(rows)


def place_labels(fig):
    """Move only text, never points or numerical values; leaders identify them."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    offsets = [(5, 5), (5, -15), (-28, 5), (-28, -15), (5, 18), (5, -28),
               (-28, 18), (-28, -28), (18, 30), (-42, 30), (18, -40), (-42, -40),
               (32, 42), (-56, 42), (32, -52), (-56, -52)]
    for ax in fig.axes:
        occupied = []
        for label in ax.texts:
            if label.get_bbox_patch() is None:
                continue
            candidates = []
            for offset in offsets:
                label.set_position(offset)
                label.update_positions(renderer)
                label.update_bbox_position_size(renderer)
                box = label.get_bbox_patch().get_window_extent(renderer).expanded(1.12, 1.15)
                inside = (ax.bbox.x0+2 <= box.x0 and box.x1 <= ax.bbox.x1-2
                          and ax.bbox.y0+2 <= box.y0 and box.y1 <= ax.bbox.y1-2)
                intersections = sum(box.overlaps(other) for other in occupied)
                candidates.append((not inside, intersections, offset, box))
                if inside and intersections == 0:
                    break
            _, _, offset, box = min(candidates, key=lambda item: (item[0], item[1]))
            label.set_position(offset)
            occupied.append(box)


def plot_tradeoffs(summary, out):
    """Time and work projections of a THREE-objective front; no 2D-only claim."""
    out.mkdir(parents=True, exist_ok=True)
    figures = []
    short = {"none": "N", "hold_050": "H50", "hold_150": "H150", "hold_300": "H300",
             "gentle_050": "G50", "gentle_150": "G150", "strong_150": "S150", "strong_300": "S300"}
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    for (fitnoise, start, filter_kind), group in summary.groupby(["calibration_noise", "forecast_start", "filter_kind"]):
        comps = sorted(set(zip(group.normal_world, group.friction_world)))
        sensors = sorted(group.sensor.unique())
        fig, axes = plt.subplots(len(sensors)*2, len(comps), figsize=(4.2*len(comps), 3.05*len(sensors)*2), squeeze=False)
        for si, sensor in enumerate(sensors):
            for ci, (nw, fw) in enumerate(comps):
                condition = group[(group.sensor == sensor)&(group.normal_world == nw)&(group.friction_world == fw)]
                main = condition[condition.estimator == "trajectory_assimilation"].sort_values(["probe_time_s", "probe"])
                for projection, xkey, xscale, xlabel in ((0, "probe_time_s", 1000, "Sondenzeit [ms]"),
                                                       (1, "total_absolute_actuator_work_J", 1000, "Gesamte absolute Aktuatorarbeit [mJ]")):
                    ax = axes[2*si+projection, ci]
                    for estimator, marker in (("prior_propagation", "x"), ("privileged_full_history", "^")):
                        part = condition[condition.estimator == estimator].set_index("probe").reindex(main.probe)
                        ax.scatter(part[xkey]*xscale, part.pooled_future_position_rmse_m*1000, s=26,
                                   marker=marker, color="#89949c", linewidths=.8, alpha=.8, zorder=1)
                    for index, row in enumerate(main.itertuples()):
                        color = plt.get_cmap("viridis")(.15+.7*row.probe_time_s/max(main.probe_time_s.max(), 1e-9))
                        x, y = getattr(row, xkey)*xscale, row.pooled_future_position_rmse_m*1000
                        ax.scatter(x, y, s=52, facecolor=color if row.position_pareto else "white",
                                   edgecolor=color, linewidths=1.5, zorder=3)
                        ax.annotate(short.get(row.probe, row.probe), (x, y), xytext=(5, 5),
                                    textcoords="offset points", fontsize=7.5, color="#202b32",
                                    bbox=dict(boxstyle="square,pad=.08", fc="white", ec="none", alpha=.85),
                                    arrowprops=dict(arrowstyle="-", color="#6a757d", lw=.5, shrinkA=2, shrinkB=3))
                    ax.set_title(f"N: {nw} / Reibung: {fw} · Sensor: {sensor}")
                    ax.set_xlabel(xlabel); ax.set_ylabel("Zukunftsposition: gepoolte RMSE [mm]")
                    ax.grid(alpha=.2); ax.margins(x=.18, y=.24)
                    values = condition.pooled_future_position_rmse_m.to_numpy()*1000
                    if np.min(values) > 0 and np.max(values)/np.min(values) > 30:
                        ax.set_yscale("log")
                        ax.set_ylim(np.min(values)/3, np.max(values)*3)
                        ax.set_ylabel("Positions-RMSE [mm] · logarithmisch")
                    else:
                        ax.set_ylim(0, np.max(values)*1.4)
        title = f"Fehler–Zeit–Arbeit · Kalibrierung {fitnoise} · {start}\n{filter_kind}; deskriptive Mittelwerte, keine Unsicherheitsintervalle abgebildet"
        fig.suptitle(title, fontsize=13, y=.997)
        handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor="#32688e", markeredgecolor="#32688e", label="Assimilation: 3D-Pareto-Punkt"),
                   Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="#32688e", label="Assimilation: dominiert"),
                   Line2D([], [], marker="x", linestyle="none", color="#89949c", label="Feste Populationsprior"),
                   Line2D([], [], marker="^", linestyle="none", color="#89949c", label="Privilegierte Vollhistorie")]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.002), fontsize=9)
        fig.tight_layout(rect=(0, .028, 1, .965), h_pad=2, w_pad=1.5)
        place_labels(fig)
        name = f"tradeoffs_{fitnoise}_{start}_{filter_kind}"
        for extension in ("png", "pdf"):
            path = out/f"{name}.{extension}"
            fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
            figures.append(path)
        plt.close(fig)
    return figures


def report_text(summary, gains, cost, diagnostic, cfg, native_cfg, out, figures):
    no_ci = not summary.interval_available.any()
    lines = ["# Fehler, Sondenzeit und physische Arbeit – Entwicklung v2", "",
        "**Die Paretofronten sind deskriptive Abwägungen aus gespeicherten Ergebnissen. Es wird kein universeller Gewinner gewählt.** Fehler, Vorbereitungszeit und beide Aktuatorkanäle bleiben getrennt; ein günstiger Punkt im Nachhinein ist keine kausal ausführbare Online-Auswahlregel.", "",
        f"Geprüft: {len(cost)} physische Episoden, {diagnostic['prediction_rows']} Prognosezeilen; {diagnostic['model_banks']} Modellbank(en), {diagnostic['preparation_seed_blocks']} unabhängige Präparations-Seedblöcke, {diagnostic['distinct_base_histories']} unterschiedliche Basisvorgeschichten. Die Kreuzungen von Materialpaar, Cam, Vorzeichen, Sensorrauschen, Fitnoise und Auswertung vermehren die unabhängigen Kalibrierungen oder Vorgeschichten nicht. {diagnostic['prior_description']} Das ist kein validiertes Modell beliebiger realer Vorgeschichten.", "",
        ("**Keine Konfidenzintervalle und keine p-Werte:** Es fehlen unabhängige Modell-/Präparationsblöcke. Aus r1 und sechs Basisprofilen wird insbesondere kein unabhängiges N=3072." if no_ci else "Deskriptive 95%-Perzentilintervalle werden nur für Gruppen mit mindestens acht Modellbanken und acht unabhängigen Präparations-Seedblöcken ausgegeben; keine p-Werte oder konfirmatorischen Signifikanzentscheidungen."), "",
        "## Definitionen und Integrität", "",
        f"Positions-MSE und Reibungs-MSE entstehen aus allen Schritten der {native_cfg['future_seconds']*1000:.0f}-ms-Zukunft. Die gepoolte RMSE ist sqrt(Mittel der Episoden-MSE), nicht das Mittel der Episoden-RMSE; letzteres steht zusätzlich in CSV. Zuerst werden die vollständigen festen Cam-/Vorzeichen-/Rezeptzellen, danach Modellbank und Präparations-Seed gleichgewichtet. Alle Nullsonden, Guards und Methoden bleiben erhalten.", "",
        "Absolute Arbeit = Σ(|u·v|+|N·d_dot|)·dt über Sonde plus Zukunft; Sondenarbeit wird zusätzlich separat ausgewiesen. Vorschrittkräfte und Vorschrittgeschwindigkeiten stammen aus dem nativen Log. Das sind physische Port-Arbeitsquadraturen, keine elektrische Energie. Die gemeinsame externe 4-s-Materialpräparation ist nicht Teil der aktiven Sonden-/Zukunftskosten. Zukünftige Kompressionsarbeit wird mitgezählt und nicht gegen negative Arbeit anderer Kanäle verrechnet.", "",
        f"Alle Rohdaten-/Metadaten-, Modell-/Quellen-/Prior- und Ausgabesperren wurden geprüft. Sämtliche vier Forecast-RMSEs wurden unabhängig aus NPZ-Trajektorien gegen native Labels nachgerechnet: größte CSV-Abweichung {diagnostic['maximum_metric_recomputation_error']:.3g}. Native Guards insgesamt {int(cost.native_guard_count.sum())}; keine Episode wurde herausgefiltert. Eingangsdateien waren vor/nach der Analyse unverändert.", "",
        "## Physischer Aufwand", "",
        "Mittel über die vier Materialpaare, alle festen Cam-/Vorzeichen-/Vorgeschichtszellen; Sensortypen getrennt. Dieselbe physische Episode wird bei mehreren Modellen/Schätzern nur einmal gezählt. Arbeiten in mJ, Wege in mm.", "",
        "| Sensor | Sonde | Zeit [ms] | Sondenarbeit absolut | Gesamtarbeit absolut | Kompression insgesamt | Sondenweg |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for keys, group in cost.groupby(["sensor", "probe"], sort=True):
        mean = group.groupby("preparation_seed")[["probe_time_s", "probe_absolute_actuator_work_J", "total_absolute_actuator_work_J", "total_compression_absolute_work_J", "probe_path_length_m"]].mean().mean()
        lines.append(f"| {keys[0]} | {keys[1]} | {mean.probe_time_s*1000:.0f} | {mean.probe_absolute_actuator_work_J*1000:.3f} | {mean.total_absolute_actuator_work_J*1000:.3f} | {mean.total_compression_absolute_work_J*1000:.3f} | {mean.probe_path_length_m*1000:.3f} |")
    lines += ["", "## Paretofronten und Messnutzen", "",
        "Eine Sonde ist dominiert, wenn eine andere bei Fehler, Sondenzeit und absoluter Gesamtarbeit nicht schlechter und bei mindestens einer Größe besser ist. Die Front wird separat für Positions-MSE und Reibungs-MSE gebildet, jeweils innerhalb derselben Material-/Sensor-/Fitnoise-/Start-/Schätzerbedingung. Gleichstände bleiben erhalten. Die Markierungen in den Diagrammen beziehen sich auf alle drei Ziele; eine zweidimensionale Projektion allein genügt nicht zur Dominanzentscheidung. Ein sehr kleiner numerischer Unterschied kann die exakte Frontzugehörigkeit ändern und ist kein praktisch relevanter Effektbeleg.", "",
        "Die folgende kompakte Übersicht nennt die **Positionsfront der Assimilation** bei verrauschten Sensoren und weitergetragenen Partikeln. Alle übrigen Fronten einschließlich gemessenem Start, idealen Sensoren, Prior und privilegiertem Verlauf stehen vollständig in den CSV-Dateien.", "",
        "| Normal / Reibung | Fitnoise | Nichtdominierte Sonden |",
        "|---|---|---|"]
    primary = summary[(summary.sensor == "noisy")&(summary.forecast_start == "filtered_particles")&(summary.estimator == "trajectory_assimilation")]
    for keys, group in primary.groupby(["normal_world", "friction_world", "calibration_noise"], sort=True):
        probes = ", ".join(group.loc[group.position_pareto.astype(bool), "probe"])
        lines.append(f"| {keys[0]} / {keys[1]} | {keys[2]} | {probes} |")
    lines += ["", "Die nächste Tabelle enthält alle Sonden bei verrauschten Sensoren und weitergetragenen Partikeln, **gleichgewichtet über die beiden Normalkraftfamilien**. Positionswerte sind gepoolte RMSE in µm, die letzte Spalte ist Reibungs-MSE in N². Materialpaarweise Werte und alle anderen Bedingungen bleiben in CSV/Abbildungen getrennt.", "",
              "| Reibungsfamilie | Fitnoise | Sonde | Position Prior | Position Assimilation | Position Vollhistorie | Reibungs-MSE Assimilation |",
              "|---|---|---|---:|---:|---:|---:|"]
    selected = summary[(summary.sensor == "noisy")&(summary.forecast_start == "filtered_particles")]
    for keys, group in selected.groupby(["friction_world", "calibration_noise", "probe"], sort=True):
        values = group.groupby("estimator")[["future_position_mse_m2", "future_friction_mse_N2"]].mean()
        positions = [np.sqrt(values.loc[est, "future_position_mse_m2"])*1e6 for est in ESTIMATORS]
        lines.append(f"| {keys[0]} | {keys[1]} | {keys[2]} | {positions[0]:.2f} | {positions[1]:.2f} | {positions[2]:.2f} | {values.loc['trajectory_assimilation','future_friction_mse_N2']:.6g} |")
    lines += ["", "Der gepaarte Messnutzen wird als Prior-MSE minus assimilierte MSE **nach derselben physischen Sonde** ausgewiesen. Positive Werte bedeuten kleineren Fehler durch Assimilation, negative einen Nachteil. Die Kosten dieser beiden Schätzvarianten stimmen pro Episode exakt überein. Dieser Wert trennt zusätzlichen Messnutzen von blinder physischer Zustandsänderung; er begründet keine neue Lernmethode oder allgemein nützliche Robotikstrategie.", "",
        "## Grenzen und Nachrechnen", "",
        "Weitergetragene x/v-Partikel und der Start von einer gemeinsamen rohen Grenzmessung sind eigene Faktoren. Der privilegierte Vergleich kennt die lokale Vorgeschichte, trägt aber ein fehlerbehaftetes eingefrorenes Modell fort und ist keine wahre Zustandsreferenz oder garantierte Fehleruntergrenze. Die Grenz-N-Messung enthält bereits den aufgerufenen Zukunfts-Kompressionsratenbefehl; sie ist nicht vor einer vorherigen Stoppen-/Fortfahren-Entscheidung verfügbar.", "",
        "Assimilation kann trotz unbekannter Vorgeschichte die privilegierte Vollhistorien-Prognose schlagen: Bei fehlerhaften eingefrorenen Parametern kann ein veränderter geschätzter Zustand Modellbias kompensieren. Kleinere Prognosefehler beweisen daher keine besonders genaue Rekonstruktion des wahren physischen Materialzustands. Eine starke Sonde kann zudem den physischen mechanischen Endzustand und den späteren Positionsfehler verändern. Aus Positionsfehlern allein folgt keine reine Aussage über Kontraktion des latenten Materialgedächtnisses.", "",
        "Die im Modell gespeicherte Posteriorstreuung ist keine automatisch kalibrierte Fehlerrisikometrik. Parameter-/Strukturfehler und eine verschobene Vorgeschichtspopulation werden dadurch nicht vollständig erfasst. Die hier verwendeten Fehler stammen aus nachträglichen nativen Labels und dürfen nicht als online verfügbare Bayes-Risiken ausgegeben werden.", "",
        "Für umfangreichere vollständig gekreuzte Datensätze resampelt der Code Modellbanken und Präparations-Seeds unabhängig; jede Kombination ihrer Bootstrap-Multiplizitäten wird gleichgewichtet. Alle Rezepte/Cams/Vorzeichen eines Präparations-Seeds bleiben im Block. Dieselben Ziehungsgewichte gelten für gepaarte Kontraste. Mindestens acht Blöcke auf jeder Achse sind eine konservative Mindestanforderung, kein Deckungsbeweis. Frontmitgliedschaft erhält keine Signifikanzmarkierung.", "",
        "Reproduktion (CPU, keine Isaac-Instanz):", "", "```bash",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/paper-cost-mpl /home/loya2/miniconda3/envs/unitree_sim_env/bin/python paper_study/scripts/analyze_cost_tradeoffs.py --out paper_study/results/cost_tradeoffs_v2_reproduced",
        "```", "", f"Tabellen: `{out}`. `physical_costs.csv` enthält jede native Episode einmal; `joined_episode_metrics.csv` die expliziten vollständigen Schlüssel; `condition_summary.csv` alle Mittelwerte/Fronten; `paired_assimilation_gains.csv` und `paired_gain_summary.csv` alle Messnutzenwerte. `analysis_manifest.json` dokumentiert Quellen und Artefakte."]
    lines += ["", "## Abbildungen", "",
              "N: sofort fortfahren; H: halten; G: sanfte Sonde; S: starke Sonde. Zahlen geben Millisekunden an. Farbe kodiert Sondendauer. Bei stark unterschiedlichen Fehlerskalen ist die y-Achse ausdrücklich logarithmisch; Punkte und Pareto-Regeln bleiben unverändert.", ""]
    for figure in figures:
        if figure.suffix == ".png":
            lines.append(f"![{figure.stem}]({figure.resolve()})")
            lines.append(f"[PDF]({figure.with_suffix('.pdf').resolve()})\n")
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=ROOT/"results/native_development_v2")
    parser.add_argument("--evaluation", type=Path, default=ROOT/"results/observer_development_v2")
    parser.add_argument("--out", type=Path, default=ROOT/"results/cost_tradeoffs_v2")
    parser.add_argument("--report", type=Path, default=ROOT/"reports/cost_tradeoffs_v2.md")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    native, evaluation, out = args.native.resolve(), args.evaluation.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError("Preserve existing analysis; choose a new --out directory")
    for folder in (native, evaluation):
        if json.loads((folder/"process.json").read_text())["status"] != "complete":
            raise ValueError(f"input process incomplete: {folder}")
    started = time.perf_counter()
    generation = json.loads((native/"generation_lock.json").read_text())
    locked = json.loads((evaluation/"evaluation_lock.json").read_text())
    artifacts = json.loads((evaluation/"artifact_manifest.json").read_text())
    if artifacts["evaluation_lock_sha256"] != sha(evaluation/"evaluation_lock.json"):
        raise ValueError("evaluation lock hash mismatch")
    before = {str(Path(__file__).resolve()): sha(__file__)}
    for mapping in (generation["sources_sha256"], locked["sources_sha256"], locked["input_sha256"], locked["generation_sources_sha256"], artifacts["outputs_sha256"]):
        verify_hash_map(mapping); before.update(mapping)
    for path in (native/"generation_lock.json", native/"process.json", evaluation/"evaluation_lock.json", evaluation/"process.json", evaluation/"artifact_manifest.json"):
        before[str(path)] = sha(path)
    cfg, native_cfg = locked["config"], generation["config"]
    cost, mapping, native_hashes = load_cost_rows(native, native_cfg)
    before.update(native_hashes)
    frame, max_error = verify_and_join_metrics(evaluation, native, cfg, cost, mapping)
    summary, intervals = summaries(frame, args.bootstrap_draws)
    paired, gains = paired_gains(frame, args.bootstrap_draws)
    out.mkdir(parents=True)
    for name, table in (("physical_costs", cost), ("joined_episode_metrics", frame), ("condition_summary", summary),
                        ("paired_assimilation_gains", paired), ("paired_gain_summary", gains)):
        table.to_csv(out/f"{name}.csv", index=False)
    figures = plot_tradeoffs(summary, out/"figures")
    prior_description = "Die konfigurierten breiten Gitterpriors wurden unverändert verwendet."
    if cfg.get("prior_bank_directory"):
        bank = Path(cfg["prior_bank_directory"])
        if not bank.is_absolute(): bank = ROOT/bank
        prior_cfg = json.loads((bank/"prior_lock.json").read_text())["config"]
        prior_description = f"Die eingefrorene Priorbank modelliert die öffentliche Präparationspopulation mit {100*prior_cfg['broad_prior_fraction']:g} % breitem Anteil."
    diagnostic = dict(prediction_rows=len(frame), physical_episodes=len(cost), model_banks=int(frame.model_seed.nunique()),
        preparation_seed_blocks=int(frame.preparation_seed.nunique()),
        distinct_base_histories=len(cost[["preparation_seed", "preparation_index"]].drop_duplicates()),
        maximum_metric_recomputation_error=max_error, posterior_spread_is_calibrated_risk=False,
        any_interval_available=bool(summary.interval_available.any()), prior_description=prior_description, bootstrap=intervals,
        seconds=time.perf_counter()-started)
    write_json(out/"summary.json", diagnostic)
    report = report_text(summary, gains, cost, diagnostic, cfg, native_cfg, out, figures)
    (out/"report.md").write_text(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    verify_hash_map(before)
    products = {str(p): sha(p) for p in out.rglob("*") if p.is_file()}
    products[str(args.report.resolve())] = sha(args.report)
    write_json(out/"analysis_manifest.json", dict(role="exploratory error-time-work analysis; no success scalarization",
        inputs_sha256=before, outputs_sha256=products, command=vars(args) | {"native": str(native), "evaluation": str(evaluation), "out": str(out), "report": str(args.report)},
        source_sha256=sha(__file__), statistics="No intervals below 8 banks AND 8 preparation-seed blocks; otherwise paired crossed-cluster percentile bootstrap"))
    print(json.dumps({key: value for key, value in diagnostic.items() if key != "bootstrap"}, indent=2))


if __name__ == "__main__":
    main()
