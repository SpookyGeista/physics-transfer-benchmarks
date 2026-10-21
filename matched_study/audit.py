#!/usr/bin/env python3
"""Independent audit of the matched-state study.

Pure statistical kernels and self-tests are written before result inspection.
The full artifact schema and inventory audit will be added after the runner API
is stable, still before reading any results. No simulator or model fitting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
CHANNELS = ["position_m", "velocity_m_s", "normal_N", "friction_N"]
# Comparison tolerances for independently reordered float64 arithmetic. These
# are integrity tolerances, not scientific effect sizes or confidence limits.
AUDIT_ATOL = 1e-10
AUDIT_RTOL = 1e-8


def finite_array(value, name, ndim=None):
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name}: expected {ndim} axes, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: nonfinite values")
    return array


def normalized_prior(prior, particles):
    prior = finite_array(prior, "prior", 1)
    if prior.shape != (particles,) or np.any(prior < 0) or prior.sum() <= 0:
        raise ValueError("Invalid prior shape/mass")
    return prior / prior.sum()


def likelihood_weights(observed, particle_past, effective_sigma, prior=None,
                       frame_mask=None, episode_chunk=8):
    """Gaussian trajectory likelihood with each included sensor frame once.

    observed: E,S,C; particle_past: P,S,C. This shared particle trajectory is
    valid only for a common, predetermined command sequence. All sigmas must
    be strictly positive; any ideal-sensor numerical floor must be declared
    by the experiment rather than invented by this audit.
    """
    y = finite_array(observed, "observed", 3)
    q = finite_array(particle_past, "particle_past", 3)
    sigma = finite_array(effective_sigma, "effective_sigma", 1)
    if q.shape[1:] != y.shape[1:] or sigma.shape != (y.shape[2],):
        raise ValueError("Sensor/particle axes disagree")
    if np.any(sigma <= 0):
        raise ValueError("Explicit positive effective sensor sigma required")
    if y.shape[1] < 1 or q.shape[0] < 1 or y.shape[0] < 1:
        raise ValueError("Empty observation/particle axis")
    if frame_mask is None:
        mask = np.ones(y.shape[1], dtype=bool)
    else:
        mask = np.asarray(frame_mask)
        if mask.dtype != bool or mask.shape != (y.shape[1],):
            raise ValueError("frame_mask must be one boolean per observed time")
    p = normalized_prior(np.ones(q.shape[0]) if prior is None else prior, q.shape[0])
    log_prior = np.full_like(p, -np.inf)
    np.log(p, out=log_prior, where=p > 0)
    squared_residual = np.empty((len(y), len(q)))
    weights = np.empty_like(squared_residual)
    for first in range(0, len(y), episode_chunk):
        delta = (y[first:first+episode_chunk, None, mask, :] - q[None, :, mask, :]) / sigma
        cost = np.sum(delta * delta, axis=(2, 3), dtype=np.float64)
        if not np.isfinite(cost).all():
            raise ValueError("Likelihood overflow; no silent truncation")
        log_mass = -.5 * cost + log_prior
        log_mass -= np.max(log_mass, axis=1, keepdims=True)
        mass = np.exp(log_mass)
        weights[first:first+episode_chunk] = mass / mass.sum(axis=1, keepdims=True)
        squared_residual[first:first+episode_chunk] = cost
    return weights, squared_residual


def prediction_spread(particle_future, weights, truth_future, prior=None):
    """Per-channel empirical spread and independently observed prediction MSE.

    Time samples receive equal weight. No channels with unlike units are added.
    Direct centered differences avoid E[Q^2]-E[Q]^2 cancellation. The posterior
    expected squared error to a realized truth equals posterior spread plus
    posterior-mean MSE; this algebra is not an uncertainty-calibration claim.
    """
    q = finite_array(particle_future, "particle_future", 3)
    w = finite_array(weights, "weights", 2)
    truth = finite_array(truth_future, "truth_future", 3)
    if w.shape != (len(truth), len(q)) or truth.shape[1:] != q.shape[1:]:
        raise ValueError("Future/weight axes disagree")
    if np.any(w < 0) or not np.allclose(w.sum(axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("Weights must be nonnegative and sum to one")
    p = normalized_prior(np.ones(len(q)) if prior is None else prior, len(q))
    prior_mean = np.einsum("p,ptc->tc", p, q)
    prior_variance_by_time = np.einsum("p,ptc->tc", p, (q-prior_mean)**2)
    prior_variance = prior_variance_by_time.mean(axis=0)
    means = np.einsum("ep,ptc->etc", w, q)
    conditional_variance = np.empty((len(truth), q.shape[2]))
    conditional_variance_by_time = np.empty_like(truth)
    weighted_truth_error = np.empty_like(conditional_variance)
    for episode in range(len(truth)):
        conditional_variance_by_time[episode] = np.einsum("p,ptc->tc", w[episode], (q-means[episode])**2)
        conditional_variance[episode] = conditional_variance_by_time[episode].mean(axis=0)
        weighted_truth_error[episode] = np.einsum("p,ptc->tc", w[episode], (q-truth[episode])**2).mean(axis=0)
    mse = ((means-truth)**2).mean(axis=1)
    return dict(
        prior_mean=prior_mean, prior_variance=prior_variance,
        prior_variance_by_time=prior_variance_by_time,
        conditioned_mean=means, conditioned_variance=conditional_variance,
        conditioned_variance_by_time=conditional_variance_by_time,
        mean_prediction_mse=mse,
        prior_mean_prediction_mse=((truth-prior_mean)**2).mean(axis=1),
        posterior_expected_truth_mse=weighted_truth_error,
        decomposition_residual=weighted_truth_error-conditional_variance-mse,
    )


def audit_arrays(ensemble, batch, methods, sensor_order, future_order,
                 declared_sensor_sigma):
    """Independently reconstruct one F/P/E/N/M batch; no file IO or future fit.

    Returns per-episode/channel losses and spread plus numerical discrepancies.
    The final CLI will bind these arrays to the frozen manifest and inventory.
    """
    past = finite_array(ensemble["particle_past"], "particle_past", 3)
    q = finite_array(ensemble["particle_future"], "particle_future", 4)
    true_past = finite_array(batch["truth_past"], "truth_past", 3)
    truth = finite_array(batch["truth_future"], "truth_future", 4)
    observed = finite_array(batch["observed"], "observed", 4)
    standard_noise = finite_array(batch["noise_standard"], "noise_standard", 3)
    sigma = finite_array(batch["effective_sigma"], "effective_sigma", 2)
    weights = finite_array(batch["weights"], "weights", 3)
    predictions = finite_array(batch["predictions"], "predictions", 6)
    saved_var = finite_array(batch["posterior_variance"], "posterior_variance", 5)
    saved_prior_var = finite_array(batch["prior_variance"], "prior_variance", 3)
    N, F, E, M, T, C = predictions.shape
    P, S, SC = past.shape
    expected_shapes = [
        (q.shape, (F, P, T, C)), (truth.shape, (F, E, T, C)),
        (true_past.shape, (E, S, SC)), (observed.shape, (N, E, S, SC)),
        (standard_noise.shape, (E, S, SC)), (sigma.shape, (N, SC)),
        (weights.shape, (N, E, P)), (saved_var.shape, (N, F, E, T, C)),
        (saved_prior_var.shape, (F, T, C)),
    ]
    if any(actual != wanted for actual, wanted in expected_shapes):
        raise ValueError(f"Incorrect matched batch axes: {expected_shapes}")
    if SC != 3 or C != 4 or (N, F, M) != (len(sensor_order), len(future_order), len(methods)):
        raise ValueError("Missing channel/factor labels")
    if len(set(methods)) != M or not {"prior", "posterior", "privileged_history"}.issubset(methods):
        raise ValueError("Method labels missing, repeated or inconsistent")
    if set(methods) not in ({"prior", "posterior", "privileged_history"}, {"prior", "posterior", "privileged_history", "mle_start1", "mle_two_start"}):
        raise ValueError("Unexpected method inventory")
    particle_branches = finite_array(ensemble["particle_past_branches"], "particle_past_branches", 4)
    truth_branches = finite_array(batch["truth_past_branches"], "truth_past_branches", 4)
    np.testing.assert_array_equal(particle_branches, np.broadcast_to(past, (F, P, S, SC)))
    np.testing.assert_array_equal(truth_branches, np.broadcast_to(true_past, (F, E, S, SC)))
    differences = dict(noise=0., weights=0., prior_mean=0., posterior_mean=0.,
                       prior_variance=0., posterior_variance=0., decomposition=0., mle_cost=0.)
    rows = []
    for noise_index, sensor in enumerate(sensor_order):
        sensor_sigma = finite_array(declared_sensor_sigma[sensor], "declared_sensor_sigma", 1)
        if sensor_sigma.shape != (SC,) or np.any(sensor_sigma < 0):
            raise ValueError("Invalid physical sensor noise")
        expected_observed = true_past + standard_noise * sensor_sigma
        differences["noise"] = max(differences["noise"], float(np.max(np.abs(observed[noise_index]-expected_observed))))
        w, _ = likelihood_weights(observed[noise_index], past, sigma[noise_index])
        differences["weights"] = max(differences["weights"], float(np.max(np.abs(weights[noise_index]-w))))
        for future_index, future_amplitude in enumerate(future_order):
            s = prediction_spread(q[future_index], w, truth[future_index])
            direct = predictions[noise_index, future_index]
            checks = {
                "prior_mean": direct[:, methods.index("prior")]-s["prior_mean"],
                "posterior_mean": direct[:, methods.index("posterior")]-s["conditioned_mean"],
                "prior_variance": saved_prior_var[future_index]-s["prior_variance_by_time"],
                "posterior_variance": saved_var[noise_index, future_index]-s["conditioned_variance_by_time"],
                "decomposition": s["decomposition_residual"],
            }
            for key, residual in checks.items():
                differences[key] = max(differences[key], float(np.max(np.abs(residual))))
            np.testing.assert_allclose(direct[:, methods.index("prior")], np.broadcast_to(s["prior_mean"], truth[future_index].shape), rtol=AUDIT_RTOL, atol=AUDIT_ATOL)
            np.testing.assert_allclose(direct[:, methods.index("posterior")], s["conditioned_mean"], rtol=AUDIT_RTOL, atol=AUDIT_ATOL)
            np.testing.assert_allclose(saved_prior_var[future_index], s["prior_variance_by_time"], rtol=AUDIT_RTOL, atol=AUDIT_ATOL)
            np.testing.assert_allclose(saved_var[noise_index, future_index], s["conditioned_variance_by_time"], rtol=AUDIT_RTOL, atol=AUDIT_ATOL)
            np.testing.assert_allclose(s["posterior_expected_truth_mse"], s["conditioned_variance"]+s["mean_prediction_mse"], rtol=AUDIT_RTOL, atol=AUDIT_ATOL)
            for episode in range(E):
                for method_index, method in enumerate(methods):
                    residual = direct[episode, method_index]-truth[future_index, episode]
                    for channel_index, channel in enumerate(CHANNELS):
                        mse = float(np.mean(residual[:, channel_index]**2))
                        rows.append(dict(sensor=sensor, future_amplitude_N=future_amplitude,
                            episode=episode, method=method, channel=channel,
                            mse=mse, rmse=float(np.sqrt(mse)), bias=float(np.mean(residual[:, channel_index])),
                            prior_variance=float(s["prior_variance"][channel_index]),
                            posterior_variance=float(s["conditioned_variance"][episode, channel_index]),
                            posterior_expected_truth_mse=float(s["posterior_expected_truth_mse"][episode, channel_index])))
        np.testing.assert_array_equal(observed[noise_index], expected_observed)
        np.testing.assert_allclose(weights[noise_index], w, rtol=AUDIT_RTOL, atol=AUDIT_ATOL)
    # The matched full-history control is deliberately a repeat of the same law.
    np.testing.assert_array_equal(predictions[:, :, :, methods.index("privileged_history")], np.broadcast_to(truth, (N, F, E, T, C)))
    if "mle_start1" in methods:
        costs = finite_array(batch["mle_cost"], "mle_cost", 3)
        choices = np.asarray(batch["mle_chosen"])
        initials = finite_array(batch["mle_initial"], "mle_initial", 4)
        if costs.shape != (N, E, 2) or choices.shape != (N, E) or initials.shape != (N, E, 2, 3):
            raise ValueError("MLE axes disagree")
        if not np.array_equal(choices, np.argmin(costs, axis=2)):
            raise ValueError("MLE selection differs from minimum past objective")
        candidate_past = finite_array(batch["mle_past"], "mle_past", 5)
        if candidate_past.shape != (N, E, 2, S, SC):
            raise ValueError("MLE candidate past shape mismatch")
        recomputed_cost = np.sum(((candidate_past-observed[:,:,None,:,:])/sigma[:,None,None,None,:])**2, axis=(3,4))
        differences["mle_cost"] = float(np.max(np.abs(costs-recomputed_cost)))
        np.testing.assert_allclose(costs, recomputed_cost, rtol=AUDIT_RTOL, atol=AUDIT_ATOL)
        candidate_future = finite_array(batch["mle_future"], "mle_future", 6)
        if candidate_future.shape != (N, F, E, 2, T, C):
            raise ValueError("MLE candidate future shape mismatch")
        np.testing.assert_array_equal(predictions[:,:,:,methods.index("mle_start1")], candidate_future[:,:,:,0])
        for n in range(N):
            for e in range(E):
                np.testing.assert_array_equal(predictions[n,:,e,methods.index("mle_two_start")], candidate_future[n,:,e,int(choices[n,e])])
            unchanged = choices[n] == 0
            np.testing.assert_allclose(predictions[n, :, unchanged, methods.index("mle_start1")],
                                       predictions[n, :, unchanged, methods.index("mle_two_start")],
                                       rtol=0, atol=0)
    return rows, differences


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_npz(path):
    with np.load(path, allow_pickle=False) as stored:
        data = {key: stored[key] for key in stored.files if key != "metadata_json"}
        meta = json.loads(str(stored["metadata_json"].item()))
    for key, array in data.items():
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError(f"Nonfinite raw array: {path}/{key}")
    return data, meta


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Empty audit table: {path}")
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def expected_inventory(directory, cfg):
    files, ensembles, batches = set(), [], []
    for bank, normal, case in itertools.product(cfg["banks"], cfg["normals"], cfg["cases"]):
        model = f"bank{bank}_{normal}_{case}"
        files.add(directory / "preparations" / f"{model}_prior.npz")
        for history in cfg["history_seeds"]:
            files.add(directory / "preparations" / f"{model}_h{history}.npz")
        for sign, duration, strength in itertools.product([-1, 1], cfg["durations"], cfg["strengths_N"]):
            indices = np.flatnonzero(np.asarray(cfg["sign_assignment"]) == sign)
            if not len(indices):
                continue
            arm = f"s{sign:+d}_d{round(duration*1000)}_a{strength:g}"
            path = directory / "ensembles" / model / (arm+".npz")
            identity = dict(model_id=model, bank=bank, normal=normal, case=case, sign=sign, duration=duration, strength=strength)
            ensembles.append((path, identity, indices))
            files.add(path)
            for history in cfg["history_seeds"]:
                path = directory / "batches" / model / f"h{history}" / (arm+".npz")
                batches.append((path, dict(identity, history=history), indices))
                files.add(path)
    return files, ensembles, batches


def audit_run(directory, prefix):
    directory, prefix = directory.resolve(), prefix.resolve()
    if prefix.parent != ROOT / "reports" or not prefix.name.startswith("audit."):
        raise ValueError("Audit output prefix must be matched_study/reports/audit.<label>")
    suffixes = [".json", ".md", ".metrics.csv", ".checks.csv", ".hashes.csv", ".ensembles.csv"]
    if any(Path(str(prefix)+suffix).exists() for suffix in suffixes):
        raise FileExistsError("Use a new audit output prefix")
    # No source NPZ is read before this strict completion/provenance gate.
    process = read_json(directory / "process.json")
    if process["status"] != "complete":
        raise ValueError("Raw audit requires a complete run; partial results remain unopened")
    started = time.perf_counter()
    lock_path, manifest_path = directory / "lock.json", directory / "manifest.json"
    lock, manifest = read_json(lock_path), read_json(manifest_path)
    lock_hash = sha(lock_path)
    if process["lock_sha256"] != lock_hash or manifest["lock_sha256"] != lock_hash:
        raise ValueError("Completion/manifest do not bind the same lock")
    cfg = lock["config"]
    if len(cfg["sign_assignment"]) != cfg["profiles_per_history"]:
        raise ValueError("Profile/sign assignment incomplete")
    if len(set(cfg["prior_seeds"]) & set(cfg["history_seeds"])):
        raise ValueError("Prior and truth history seed populations overlap")
    if len(cfg["prior_seeds"]) < len(cfg["banks"]):
        raise ValueError("No independently declared prior seed for each bank")
    files, ensemble_plan, batch_plan = expected_inventory(directory, cfg)
    if set(manifest["outputs_sha256"]) != {str(p) for p in files}:
        raise ValueError("Manifest differs from exact factorial inventory")
    if {p.resolve() for p in directory.rglob("*.npz")} != files:
        raise ValueError("Missing or extra NPZ outside the manifest")
    if process["batches"] != len(batch_plan) or process["files"] != len(files):
        raise ValueError("Completion counters disagree with full inventory")
    source_hashes = lock["source_sha256"]
    audit_prefrozen = source_hashes.get(str(Path(__file__).resolve())) == sha(__file__)
    if not audit_prefrozen and cfg["kind"] != "development":
        raise ValueError("This independent audit version was not frozen with the run")
    bindings = []
    artifact_bindings = dict(manifest["outputs_sha256"])
    control_bindings = {str(p): sha(p) for p in [lock_path, manifest_path, directory / "process.json", Path(__file__).resolve()]}
    def verify(stage):
        for kind, mapping in [("source", source_hashes), ("raw", artifact_bindings), ("control", control_bindings)]:
            for path, expected in mapping.items():
                actual = sha(path)
                if actual != expected:
                    raise ValueError(f"Changed {kind} artifact: {path}")
                bindings.append(dict(stage=stage, kind=kind, path=path, sha256=actual))
    verify("before")
    methods = ["prior", "posterior"] + (["mle_start1", "mle_two_start"] if cfg["mle_enabled"] else []) + ["privileged_history"]
    sensors, futures = list(cfg["sensor_sigma"]), cfg["future_amplitudes_N"]
    expected_effective_sigma = np.array([np.hypot(cfg["sensor_sigma"][s], cfg["numerical_floor"]) for s in sensors])
    metric_rows, check_rows, physical_rows = [], [], []
    numerical_max = dict(noise=0., weights=0., prior_mean=0., posterior_mean=0., prior_variance=0., posterior_variance=0., decomposition=0., mle_cost=0.)
    ensemble_lookup = {str(path): (identity, indices) for path, identity, indices in ensemble_plan}
    mle_counts = dict(starts=0, nonconverged=0, budget_flags=0, residual_calls=0, seconds=0.)
    prepared = {}
    def prep(model, label, seed, count):
        key = model, label
        if key not in prepared:
            path = directory / "preparations" / f"{model}_{label}.npz"
            a, m = read_npz(path)
            if m["seed"] != seed or m["count"] != count or m["normal_reset_after_preparation"] is not False:
                raise ValueError("Preparation seed/count/history contract mismatch")
            np.testing.assert_array_equal(a["initial_xv"], np.zeros((count, 2)))
            if a["initial_pi"].shape != (count, 3) or a["normal0"].shape != (3,):
                raise ValueError("Invalid preparation state shape")
            prepared[key] = (a, m)
        return prepared[key]
    # Grouping keeps one ensemble in RAM while auditing all its truth blocks.
    by_ensemble = {}
    for path, identity, indices in batch_plan:
        arm = path.name
        ep = directory / "ensembles" / identity["model_id"] / arm
        by_ensemble.setdefault(ep, []).append((path, identity, indices))
    for ens_path, group in by_ensemble.items():
        ensemble, ensmeta = read_npz(ens_path)
        identity, _ = ensemble_lookup[str(ens_path)]
        for key, expected in identity.items():
            if ensmeta[key] != expected:
                raise ValueError(f"Ensemble identity mismatch: {key}")
        for key, expected in [("dt", cfg["dt"]), ("particles", cfg["particles"]), ("futures", futures), ("initial_compression_m", .02)]:
            if ensmeta[key] != expected:
                raise ValueError(f"Ensemble config mismatch: {key}")
        model = identity["model_id"]
        prior_seed = cfg["prior_seeds"][cfg["banks"].index(identity["bank"])]
        prior, prior_meta = prep(model, "prior", prior_seed, cfg["particles"])
        expected_prior_path = directory / "preparations" / f"{model}_prior.npz"
        if ensmeta["prior_preparation_path"] != str(expected_prior_path) or ensmeta["prior_preparation_sha256"] != artifact_bindings[str(expected_prior_path)]:
            raise ValueError("Unbound ensemble prior preparation")
        np.testing.assert_array_equal(ensemble["initial_pi"], prior["initial_pi"])
        np.testing.assert_array_equal(ensemble["normal0"], prior["normal0"])
        S, T = round(identity["duration"]/.025), round(cfg["future_duration"]/.0025)
        np.testing.assert_array_equal(ensemble["past_times"], np.arange(1, S+1)*.025)
        np.testing.assert_array_equal(ensemble["future_times"], np.arange(1, T+1)*.0025)
        if ensemble["particle_past"].shape != (cfg["particles"], S, 3) or ensemble["particle_future"].shape != (len(futures), cfg["particles"], T, 4):
            raise ValueError("Ensemble clock/axes mismatch")
        np.testing.assert_array_equal(ensemble["boundary_xv"], ensemble["particle_past"][:, -1, :2])
        for fi, future in enumerate(futures):
            q = ensemble["particle_future"][fi]
            v = np.mean((q-np.mean(q, axis=0))**2, axis=0)
            np.testing.assert_allclose(ensemble["prior_variance"][fi], v, rtol=AUDIT_RTOL, atol=AUDIT_ATOL)
            diag = ensmeta["diagnostics"][fi]
            if np.max(diag["max_solver_residual_N"]) > cfg["maximum_solver_residual_N"]:
                raise ValueError("Recorded ensemble solver tolerance violated")
            for channel_index, channel in enumerate(CHANNELS):
                physical_rows.append(dict(**identity, future_amplitude_N=future, channel=channel,
                    particles=cfg["particles"], prior_variance=float(v[:,channel_index].mean()),
                    probe_absolute_tangential_work_J=float(np.mean(diag["prep_absolute_tangential_work_J"])),
                    future_absolute_tangential_work_J=float(np.mean(diag["future_absolute_tangential_work_J"])),
                    total_absolute_tangential_work_J=float(np.mean(diag["total_absolute_tangential_work_J"])),
                    maximum_solver_residual_N=float(np.max(diag["max_solver_residual_N"])),
                    ensemble_path=str(ens_path)))
        for path, identity, indices in group:
            batch, meta = read_npz(path)
            for key, expected in identity.items():
                if meta[key] != expected:
                    raise ValueError(f"Batch identity mismatch: {path}/{key}")
            for key, expected in [("model", model), ("dt", cfg["dt"]), ("particles", cfg["particles"]), ("methods", methods), ("sensors", sensors), ("futures", futures), ("channel_order", CHANNELS), ("initial_compression_m", .02)]:
                if meta[key] != expected:
                    raise ValueError(f"Batch config mismatch: {path}/{key}")
            if meta["ensemble_path"] != str(ens_path) or meta["ensemble_sha256"] != artifact_bindings[str(ens_path)]:
                raise ValueError("Batch ensemble binding mismatch")
            np.testing.assert_array_equal(batch["profile_indices"], indices)
            truth_prep, _ = prep(model, f"h{identity['history']}", identity["history"], cfg["profiles_per_history"])
            np.testing.assert_array_equal(batch["truth_initial_pi"], truth_prep["initial_pi"][indices])
            np.testing.assert_array_equal(batch["normal0"], truth_prep["normal0"])
            np.testing.assert_array_equal(batch["normal0"], prior["normal0"])
            np.testing.assert_array_equal(batch["truth_boundary_xv"], batch["truth_past"][:, -1, :2])
            np.testing.assert_array_equal(batch["effective_sigma"], expected_effective_sigma)
            expected_noise = np.stack([np.random.default_rng(np.random.SeedSequence([cfg["noise_stream_key"], int(identity["history"]), int(i), int(identity["sign"])+1])).normal(size=(12,3))[:S] for i in indices])
            np.testing.assert_array_equal(batch["noise_standard"], expected_noise)
            rows, differences = audit_arrays(ensemble, batch, methods, sensors, futures, cfg["sensor_sigma"])
            for row in rows:
                row.update(identity)
                row["profile_index"] = int(indices[row["episode"]])
                row["batch_path"] = str(path)
            metric_rows.extend(rows)
            for key, value in differences.items():
                numerical_max[key] = max(numerical_max[key], value)
            check_rows.append(dict(path=str(path), episodes=len(indices), **differences))
            for diag in meta["diagnostics"]:
                if np.max(diag["max_solver_residual_N"]) > cfg["maximum_solver_residual_N"]:
                    raise ValueError("Recorded truth solver tolerance violated")
            if cfg["mle_enabled"]:
                if len(meta["mle_diagnostics"]) != len(sensors)*len(indices):
                    raise ValueError("MLE diagnostic inventory incomplete")
                ids = [(d["sensor"], d["profile_index"]) for d in meta["mle_diagnostics"]]
                if len(set(ids)) != len(ids) or set(ids) != set(itertools.product(sensors, map(int, indices))):
                    raise ValueError("Repeated or missing MLE profile/sensor identities")
                model_path = REPO / f"paper_study/multistate_learning/results/models/{identity['case']}/r{61000+identity['bank']}/noisy/model.json"
                if str(model_path) not in source_hashes:
                    raise ValueError("Fitted PI coefficients not source-bound")
                alpha = np.array(read_json(model_path)["yield_per_normal_m_N"])
                bound = alpha * prior_meta["diagnostics"]["initial_normal_N"]
                for d in meta["mle_diagnostics"]:
                    if [s["start"] for s in d["starts"]] != [0, 1]:
                        raise ValueError("Missing MLE start diagnostics")
                    ni, ei = sensors.index(d["sensor"]), list(indices).index(d["profile_index"])
                    for start in d["starts"]:
                        si = start["start"]
                        initial_q = np.zeros(3) if si == 1 else np.clip(prior["initial_pi"][np.argmax(batch["weights"][ni,ei])]/bound, -1., 1.)
                        np.testing.assert_allclose(start["initial_q"], initial_q, rtol=0, atol=1e-12)
                        final_q = finite_array(start["final_q"], "MLE normalized final state", 1)
                        if final_q.shape != (3,) or np.any(abs(final_q)>1.+1e-12):
                            raise ValueError("MLE final state outside fixed box")
                        np.testing.assert_allclose(batch["mle_initial"][ni,ei,si], final_q*bound, rtol=0, atol=1e-12)
                        if start["residual_calls"] > cfg["mle"]["max_residual_calls"] or (start["optimizer_nfev"] is not None and start["optimizer_nfev"] > cfg["mle"]["max_nfev"]):
                            raise ValueError("MLE exceeds fixed evaluation budget")
                        if start["max_solver_residual_N_all_trials"] > cfg["maximum_solver_residual_N"] or start["wall_budget_used"]:
                            raise ValueError("MLE solver/time budget contract mismatch")
                        mle_counts["starts"] += 1
                        mle_counts["nonconverged"] += int(not start["solver_success"])
                        mle_counts["budget_flags"] += int(start["budget_exhausted"])
                        mle_counts["residual_calls"] += int(start["residual_calls"])
                        mle_counts["seconds"] += float(start["seconds"])
            if len(check_rows) % 48 == 0:
                print("AUDIT", len(check_rows), "/", len(batch_plan), "batches", flush=True)
    expected_rows = sum(len(indices) for _, _, indices in batch_plan)*len(sensors)*len(futures)*len(methods)*4
    if len(metric_rows) != expected_rows:
        raise ValueError("Final audit metric row count incomplete")
    verify("after")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {suffix: Path(str(prefix)+suffix) for suffix in suffixes}
    for suffix, rows in [(".metrics.csv", metric_rows), (".checks.csv", check_rows), (".hashes.csv", bindings), (".ensembles.csv", physical_rows)]:
        write_csv(outputs[suffix], rows)
    summary = dict(status="passed", kind=cfg["kind"], source_directory=str(directory),
        data_files=len(files), ensembles=len(ensemble_plan), batches=len(batch_plan),
        truth_episodes=sum(len(indices) for _, _, indices in batch_plan),
        noise_future_episode_conditions=expected_rows//(len(methods)*4),
        episode_method_rows=expected_rows//4, channel_metric_rows=expected_rows,
        hash_binding_checks=len(bindings), methods=methods, sensors=sensors,
        future_amplitudes_N=futures, independent_banks=len(cfg["banks"]), history_blocks=len(cfg["history_seeds"]),
        maximum_recompute_differences=numerical_max,
        mle_accounting=mle_counts,
        arithmetic_comparison=dict(rtol=AUDIT_RTOL, atol=AUDIT_ATOL),
        seconds=time.perf_counter()-started, lock_sha256=lock_hash,
        audit_source_sha256=sha(__file__),
        audit_source_in_generation_lock=audit_prefrozen,
        scope="Controlled matched-law synthetic experiment, not independent native physics or robot validation",
        limits=["Finite particle posterior variance is not automatically calibrated forecast error", "No pointwise posterior variance reduction assumed", "Truth and prior have independent history seeds but the same fitted material laws", "Recorded solver diagnostics checked; this raw audit does not rerun any physical integrator"])
    outputs[".md"].write_text(f"""# Independent matched-study raw audit

**Passed:** {summary['ensembles']} shared ensembles, {summary['batches']} complete batches,
{summary['noise_future_episode_conditions']} noise/future/episode conditions and
{summary['episode_method_rows']} episode/method rows ({summary['channel_metric_rows']} channel rows).
Every expected factor, method and adverse finite prediction is retained.

The audit independently reconstructed Gaussian likelihood weights, prior/posterior
forecast means, directly centered variance at every future time, and every
episode's four-channel MSE, RMSE and signed bias. It verified the exact empirical
posterior error = variance + mean-prediction-error identity, declared effective
noise floors, deterministic shared noise draws, initial-state preparation binds,
MLE candidate past costs, minimum-past-objective selection and recorded
evaluation/solver budgets. Separately saved past prefixes from both future
excitation branches were bit-identical for every prior particle and truth.
All {mle_counts['starts']} MLE starts are retained, including {mle_counts['nonconverged']}
nonconverged and {mle_counts['budget_flags']} budget-flagged starts. Initial best-prior
and zero starts, normalized box bounds and final-state scaling were verified.
The privileged matched-history prediction equals the synthetic truth exactly by
design; it is not an independent model-validation result.

There were {len(bindings)} hash-binding checks before/after analysis. Sources,
models, configuration, prior/truth preparations, raw arrays and completion
records remained unchanged. The source code used here was itself included in
the generation lock: **{audit_prefrozen}** (development audits may be added after
generation; their own source hash is recorded here). All recalculation discrepancies are in
`{outputs['.checks.csv'].name}`; per-channel raw metrics are in
`{outputs['.metrics.csv'].name}`. Uniform physical-ensemble variance is deduplicated
by ensemble/future/channel in `{outputs['.ensembles.csv'].name}`, without multiplying
it by truth history, sensor noise or estimator count.

The experiment uses {summary['independent_banks']} pre-existing fitted banks and
{summary['history_blocks']} separately seeded truth-history blocks. This audit
supplies no inferential p-values or posterior-calibration guarantee. Physical
state evolution and conditioning use identical propagated candidate trajectories;
only the weights differ. A change of unconditioned spread across probes is not
itself information gain, and individual posterior variances need not decrease.
No simulator, optimizer or coefficient fit was rerun by the audit. Recorded
solver residuals were checked against the declared bound. There is no new native
Isaac or physical robot validation in this matched-law control.
""")
    summary["outputs_sha256"] = {str(path): sha(path) for suffix, path in outputs.items() if suffix != ".json"}
    with outputs[".json"].open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


def self_test():
    # One scalar observation with two candidates gives odds exp(-1/2).
    q = np.array([[[0.]], [[1.]]])
    w, cost = likelihood_weights(np.array([[[0.]]]), q, [1.])
    np.testing.assert_allclose(w[0, 1]/w[0, 0], np.exp(-.5), rtol=1e-14)
    np.testing.assert_allclose(cost, [[0., 1.]])
    # Each additional independent observation enters exactly once.
    two, _ = likelihood_weights(np.zeros((1, 2, 1)), np.repeat(q, 2, axis=1), [1.])
    np.testing.assert_allclose(two[0, 1]/two[0, 0], np.exp(-1), rtol=1e-14)
    excluded, _ = likelihood_weights(np.zeros((1, 2, 1)), np.repeat(q, 2, axis=1), [1.], frame_mask=np.array([True, False]))
    np.testing.assert_array_equal(excluded, w)
    # Future samples are absent from likelihood API; ignored frames cannot leak.
    changed = np.array([[[0.], [1e9]]])
    causal, _ = likelihood_weights(changed, np.repeat(q, 2, axis=1), [1.], frame_mask=np.array([True, False]))
    np.testing.assert_array_equal(causal, w)
    # Stable arithmetic in a strongly concentrated likelihood.
    extreme, _ = likelihood_weights(np.array([[[100.]]]), np.array([[[0.]], [[100.]]]), [.001])
    np.testing.assert_array_equal(extreme, [[0., 1.]])
    # Identical candidates retain the exact prior, including zero-mass members.
    prior = np.array([.2, 0, .8])
    retained, _ = likelihood_weights(np.ones((2, 3, 2)), np.zeros((3, 3, 2)), [1., 2.], prior)
    np.testing.assert_allclose(retained, np.broadcast_to(prior, retained.shape))
    # Proper E/P/T/C axes and exact posterior bias-variance decomposition.
    rng = np.random.default_rng(3107)
    future = rng.normal(size=(11, 7, 4))
    truth = rng.normal(size=(5, 7, 4))
    mass = rng.uniform(size=(5, 11)); mass /= mass.sum(axis=1, keepdims=True)
    s = prediction_spread(future, mass, truth)
    np.testing.assert_allclose(s["decomposition_residual"], 0, atol=2e-15)
    np.testing.assert_allclose(s["conditioned_mean"][2, 3, 1], sum(mass[2, p]*future[p, 3, 1] for p in range(11)))
    # Physical contraction changes the unconditioned spread without observations.
    contracted = prediction_spread(.3*future, mass, .3*truth)
    np.testing.assert_allclose(contracted["prior_variance"], .09*s["prior_variance"])
    # Conditioning need not reduce variance for each single observation.
    spread = prediction_spread(np.array([[[-1.]], [[0.]], [[1.]]]), np.array([[.5, 0, .5]]), np.zeros((1, 1, 1)), [.1, .8, .1])
    assert spread["conditioned_variance"][0, 0] > spread["prior_variance"][0]
    # Invalid noise specifications must not obtain an undeclared floor.
    try:
        likelihood_weights(np.zeros((1, 1, 1)), q, [0.])
    except ValueError:
        pass
    else:
        raise AssertionError("Missing ideal-sensor numerical floor accepted")
    # Full fabricated N/F/E/M/T/C schema. Fixture likelihood is computed using
    # explicit scalar loops, independently of the vectorized audit kernel.
    P, E, N, F, S, T, C = 2, 3, 2, 2, 2, 3, 4
    fake_past = np.broadcast_to(np.arange(P)[:, None, None], (P, S, 3)).copy()
    fake_future = np.empty((F, P, T, C))
    for f in range(F):
        fake_future[f] = np.broadcast_to((f+1)*np.arange(P)[:, None, None], (P, T, C))
    fake_truth_past = np.broadcast_to(np.arange(E)[:, None, None]/2, (E, S, 3)).copy()
    fake_truth = np.broadcast_to(np.arange(E)[None, :, None, None]/2, (F, E, T, C)).copy()
    fake_noise = np.arange(E*S*3).reshape(E, S, 3)/100
    physical_noise = {"clean": [0., 0., 0.], "noisy": [.1, .2, .3]}
    effective = np.array([[.5, .5, .5], [1., 1., 1.]])
    obs = np.stack([fake_truth_past + fake_noise*np.array(physical_noise[n]) for n in ["clean", "noisy"]])
    fake_weights = np.empty((N, E, P))
    for n in range(N):
        for e in range(E):
            log_lik = np.array([-.5*sum(((obs[n,e,s,c]-fake_past[p,s,c])/effective[n,c])**2 for s in range(S) for c in range(3)) for p in range(P)])
            probability = np.exp(log_lik-log_lik.max())
            fake_weights[n,e] = probability/probability.sum()
    methods = ["prior", "posterior", "mle_start1", "mle_two_start", "privileged_history"]
    pred = np.empty((N, F, E, 5, T, C))
    var = np.empty((N, F, E, T, C))
    cost = np.broadcast_to([[0., 1.], [2., 1.], [1., 1.]], (N, E, 2)).copy()
    choice = np.argmin(cost, axis=2)
    for n in range(N):
        for f in range(F):
            for e in range(E):
                pred[n,f,e,0] = np.mean(fake_future[f], axis=0)
                pred[n,f,e,1] = sum(fake_weights[n,e,p]*fake_future[f,p] for p in range(P))
                pred[n,f,e,2] = fake_truth[f,e]+.1
                pred[n,f,e,3] = pred[n,f,e,2] if choice[n,e] == 0 else fake_truth[f,e]-.2
                pred[n,f,e,4] = fake_truth[f,e]
                var[n,f,e] = sum(fake_weights[n,e,p]*(fake_future[f,p]-pred[n,f,e,1])**2 for p in range(P))
    ens = dict(particle_past=fake_past, particle_future=fake_future,
               particle_past_branches=np.broadcast_to(fake_past,(F,P,S,3)).copy())
    batch = dict(truth_past=fake_truth_past, truth_future=fake_truth, observed=obs,
                 noise_standard=fake_noise, effective_sigma=effective, weights=fake_weights,
                 predictions=pred, posterior_variance=var,
                 prior_variance=np.var(fake_future, axis=1), mle_cost=cost,
                 mle_chosen=choice, mle_initial=np.zeros((N,E,2,3)),
                 truth_past_branches=np.broadcast_to(fake_truth_past,(F,E,S,3)).copy(),
                 mle_past=obs[:,:,None,:,:]+np.sqrt(cost[:,:,:,None,None]/(S*3))*effective[:,None,None,None,:])
    fake_mle_future = np.stack((pred[:,:,:,2], np.broadcast_to(fake_truth,(N,F,E,T,C))-.2), axis=3)
    batch["mle_future"] = fake_mle_future
    rows, differences = audit_arrays(ens, batch, methods, ["clean", "noisy"], [2., 10.], physical_noise)
    assert len(rows) == N*F*E*5*C
    assert max(differences.values()) < 1e-12, differences
    assert all(row["mse"] == 0 for row in rows if row["method"] == "privileged_history")
    # A duplicate label or bad selection must be rejected before scoring.
    for bad in [dict(batch, mle_chosen=np.ones_like(choice)),
                dict(batch, predictions=pred[:, :, :2])]:
        try:
            audit_arrays(ens, bad, methods, ["clean", "noisy"], [2., 10.], physical_noise)
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupt full-schema fixture accepted")
    return dict(status="passed", checks=14,
                data_used="synthetic only; no study results read")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["self-test", "run"])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path, help="Output prefix matched_study/reports/audit.<label>")
    args = parser.parse_args()
    if args.command == "self-test":
        print(json.dumps(self_test(), indent=2))
    else:
        if args.data is None or args.out is None:
            parser.error("run requires --data and --out")
        print(json.dumps(audit_run(args.data, args.out), indent=2))
