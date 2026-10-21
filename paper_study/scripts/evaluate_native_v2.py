"""Separate exploratory evaluator with propagated mechanical hypotheses.

The primary forecast starts from each filtered mechanical/material hypothesis.
Starting every hypothesis from the last raw x/v measurement remains a named
ablation. Prior and posterior use exactly the same particles in either condition.
The default filter scores the full independent x/v/N measurement sequence.
Future truth and hidden preparation are excluded by an explicit input whitelist.
The full-history comparator is separately privileged and never supplies its
history or modeled state to the ordinary estimators. All law weights stay fixed.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/paper-study-numba")
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
from observer import GridObserver
from joint_observer import JointGridObserver

FILTERS = ("joint_sensor_likelihood", "conditional_N_hold")
STARTS = ("filtered_particles", "measured_start")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def clone_state(state):
    return {key: np.array(value, copy=True) for key, value in state.items()}


def grid_for(family, cfg):
    spec = cfg["prior"][family]
    if spec["spacing"] not in ("linear", "log"):
        raise ValueError("prior spacing must be linear or log")
    spacing = np.linspace if spec["spacing"] == "linear" else np.geomspace
    return spacing(*spec["bounds"], cfg["grid_size"])


def prior_path(cfg, normal_world, friction_world, model_seed, calibration_noise):
    directory = cfg.get("prior_bank_directory")
    if directory is None:
        return None
    directory = Path(directory)
    if not directory.is_absolute():
        directory = ROOT/directory
    return directory.resolve()/f"{normal_world}_{friction_world}/r{model_seed}/{calibration_noise}.npz"


def load_prior_bank(path, cfg):
    """Population prior only; no sampled history, row identity or truth read."""
    if path is None:
        return None, None
    with np.load(path, allow_pickle=False) as bank:
        grid, weights = bank["grid"].copy(), bank["weights"].copy()
    if grid.ndim != 1 or weights.shape != grid.shape or len(grid) != cfg["grid_size"]:
        raise ValueError("prior bank grid/weight dimensions must match configured grid_size")
    if (not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0) or
            not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("invalid fixed population prior")
    return grid, weights


def observable_inputs(arrays):
    """Only public commands, geometry and sensor samples through the boundary.

    Boundary N includes the invoked future d_dot. This supports a forecast made
    after that invocation, not an earlier stop/continue decision. Future commands
    are precommitted from that boundary sample; no future sensor or force labels
    are used. Native v2 hidden factor arrays are deliberately omitted.
    """
    end = int(arrays["probe_steps"])
    times = np.asarray(arrays["observation_steps"])
    mask = times <= end
    result = {**{k: arrays[k].copy() for k in ("u", "d", "d_dot", "amplitude")},
              "observation_steps": times[mask].copy(),
              "observations": arrays["observations"][:, mask].copy(), "probe_steps": end}
    n, total = result["u"].shape
    if not (0 <= end < total):
        raise ValueError("positive forecast horizon and valid probe boundary required")
    if (result["observation_steps"].ndim != 1 or len(result["observation_steps"]) == 0 or
            result["observation_steps"][0] != 0 or result["observation_steps"][-1] != end or
            np.any(np.diff(result["observation_steps"]) <= 0)):
        raise ValueError("unique ordered initial/boundary observations required")
    if result["observations"].shape != (n, len(result["observation_steps"]), 3):
        raise ValueError("observations must have shape episodes x frames x 3")
    for key in ("d", "d_dot"):
        if result[key].shape != (n, total):
            raise ValueError("command shapes differ")
    if result["amplitude"].shape != (n,) or not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError("invalid/nonfinite public input")
    return result


def propagate_forecast(model, xv, state, inputs, dt, grid_size=1):
    end = inputs["probe_steps"]
    count, total = inputs["u"].shape
    prediction = np.empty((count, grid_size, total-end, 4))
    amp = np.repeat(inputs["amplitude"], grid_size)
    xv, state = np.array(xv, copy=True), clone_state(state)
    for j, t in enumerate(range(end, total)):
        command = [np.repeat(inputs[key][:, t], grid_size) for key in ("u", "d", "d_dot")]
        xv, state, n, ft = motion_step(model, xv, state, *command, amp, dt=dt)
        prediction[:, :, j] = np.column_stack((xv, n, ft)).reshape(count, grid_size, 4)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("nonfinite forecast; retain run as failed")
    return prediction


def configured_starts(cfg):
    starts = tuple(cfg.get("forecast_starts", STARTS))
    if not starts or len(set(starts)) != len(starts) or any(s not in STARTS for s in starts):
        raise ValueError("forecast_starts must select unique declared conditions")
    return starts


def mechanical_start(grid_xv, measured_xv, condition):
    """The sole controlled difference between the two forecast-start factors."""
    if condition == "filtered_particles":
        return np.array(grid_xv, copy=True).reshape(-1, 2)
    if condition == "measured_start":
        return np.repeat(measured_xv, grid_xv.shape[1], axis=0)
    raise ValueError(condition)


def infer_and_forecast(model, inputs, meta, cfg, kind="joint_sensor_likelihood", *, prior=None, grid=None):
    if kind not in FILTERS:
        raise ValueError(kind)
    count = inputs["u"].shape[0]
    grid = grid_for(model.friction_family, cfg) if grid is None else np.array(grid, dtype=float, copy=True)
    # A bank supplies one population distribution for the whole batch, never
    # weights conditioned on an episode's hidden history/factor/recipe.
    if prior is not None and np.asarray(prior).shape != grid.shape:
        raise ValueError("population prior must be one shared (grid,) vector")
    sensor, floor = meta["sensor_parameters"], cfg["numerical_floor"]
    cls = GridObserver if kind == "conditional_N_hold" else JointGridObserver
    extra = {} if cls is GridObserver else dict(numerical_N=floor["normal_N"])
    observer = cls(model, grid, np.zeros((count, 2)), .02, inputs["observations"][:, 0, 2],
        dt=meta["dt"], sigma_x=sensor["sigma_x_m"], sigma_v=sensor["sigma_v_m_s"],
        sigma_N=sensor["sigma_normal_N"], numerical_x=floor["x_m"], numerical_v=floor["v_m_s"],
        prior=prior, **extra)
    indices = {int(t): i for i, t in enumerate(inputs["observation_steps"])}
    weight_trace = [observer.weights.copy()]
    for t in range(inputs["probe_steps"]):
        observer.advance(*(inputs[k][:, t] for k in ("u", "d", "d_dot")), inputs["amplitude"])
        if t+1 in indices:
            obs = inputs["observations"][:, indices[t+1]]
            # At the boundary, use the rate command that was active when the
            # boundary sensor sample was taken, not the previous probe rate.
            extra = {} if cls is GridObserver else dict(d=inputs["d"][:, t+1],
                d_dot=inputs["d_dot"][:, t+1], amplitude=inputs["amplitude"])
            observer.observe(obs[:, 0], obs[:, 1], obs[:, 2], **extra)
            weight_trace.append(observer.weights.copy())
    snapshot = observer.snapshot()
    details = dict(grid=grid.copy(), weights=snapshot["weights"], prior=snapshot["prior"],
        weight_trace=np.stack(weight_trace, axis=1), log_likelihood=snapshot["log_likelihood"],
        observation_steps=inputs["observation_steps"].copy(),
        filtered_grid_xv=snapshot["grid_xv"], boundary_observed_xv=inputs["observations"][:, -1, :2].copy())
    predictions = {}
    for start in configured_starts(cfg):
        xv = mechanical_start(snapshot["grid_xv"], inputs["observations"][:, -1, :2], start)
        particles = propagate_forecast(model, xv, snapshot["state"], inputs, meta["dt"], len(grid))
        details[start+"_initial_xv"] = xv.reshape(count, len(grid), 2)
        # Both estimates use this SAME single forecast tensor. Assimilation
        # changes weights, never physical preparation, candidates or commands.
        for estimator, weight in (("prior_propagation", observer.prior), ("trajectory_assimilation", observer.weights)):
            mean = np.einsum("eg,egtk->etk", weight, particles)
            variance = np.maximum(np.einsum("eg,egtk->etk", weight, particles**2)-mean**2, 0.)
            predictions[f"{kind}/{start}/{estimator}"] = mean
            details[f"{start}_{estimator}_forecast"] = mean
            details[f"{start}_{estimator}_variance"] = variance
    return predictions, details


def privileged_forecasts(model, inputs, metadata, cfg, kind="joint_sensor_likelihood"):
    """Full local preparation command history; no true active-task states.

    Its mechanically propagated forecast is also a fixed-model prediction,
    never a true-state oracle or a bound on achievable estimation error.
    """
    if kind not in FILTERS:
        raise ValueError(kind)
    commands = metadata["prepared_reference_audit"]["commands"]
    d, r, v, n = [np.asarray(commands[k]) for k in ("delta", "rate", "v", "normal")]
    count, dt = inputs["u"].shape[0], metadata["dt"]
    friction = model.friction_model
    state_f = friction.initial(d[:, 0], v[:, 0], n[:, 0])
    for t in range(v.shape[1]-1):
        _, _, state_f = friction.step(d[:, t], r[:, t], v[:, t], state_f, normal_override=n[:, t], dt=dt)
    state = model.initial(np.full(count, .02), np.zeros(count), inputs["observations"][:, 0, 2])
    state.update({"friction__"+k: value for k, value in state_f.items()})
    xv = np.zeros((count, 2))
    indices = {int(t): i for i, t in enumerate(inputs["observation_steps"])}
    held_n = inputs["observations"][:, 0, 2]
    for t in range(inputs["probe_steps"]):
        xv, state, _, _ = motion_step(model, xv, state,
            *(inputs[k][:, t] for k in ("u", "d", "d_dot")), inputs["amplitude"], dt=dt,
            normal_override=held_n if kind == "conditional_N_hold" else None)
        if t+1 in indices:
            held_n = inputs["observations"][:, indices[t+1], 2]
    result, details = {}, {"privileged_filtered_initial_xv": xv.copy()}
    for start in configured_starts(cfg):
        initial_xv = mechanical_start(xv[:, None, :], inputs["observations"][:, -1, :2], start)
        forecast = propagate_forecast(model, initial_xv, state, inputs, dt)[:, 0]
        result[f"{kind}/{start}/privileged_full_history"] = forecast
        details[start+"_privileged_forecast"] = forecast
    return result, details


def preparation_seed(metadata):
    if "preparation_seed" in metadata:
        return int(metadata["preparation_seed"])
    # Legacy v1 metadata compatibility; called ONLY in the scoring path.
    seeds = {int(row["seed"]) for row in metadata["prepared_reference_audit"]["rows"]}
    if len(seeds) != 1:
        raise ValueError("batch must have exactly one preparation_seed")
    return seeds.pop()


def score_forecasts(predictions, truth, metadata, model_seed, calibration_noise):
    """Evaluator-only labels are introduced only after all predictions exist."""
    end = int(truth["probe_steps"])
    target = np.concatenate((truth["next_state"][:, end:], truth["normal"][:, end:, None],
                              truth["friction"][:, end:, None]), axis=2)
    rows = []
    for method, prediction in predictions.items():
        filter_kind, start, estimator = method.split("/")
        if prediction.shape != target.shape:
            raise ValueError("forecast shape differs from scoring horizon")
        rmse = np.sqrt(np.mean((prediction-target)**2, axis=1))
        for i in range(len(target)):
            row = dict(normal_world=metadata["normal_world"], friction_world=metadata["friction_world"],
                preparation_seed=preparation_seed(metadata), probe=metadata["probe"]["name"],
                duration_s=metadata["probe"]["duration_s"], sensor=metadata["sensor"], episode=i,
                model_seed=model_seed, calibration_noise=calibration_noise,
                filter_kind=filter_kind, forecast_start=start, estimator=estimator, method=method,
                native_guard_count=int(truth["guard"][i].sum()))
            row.update(dict(zip(("future_position_rmse_m", "future_velocity_rmse_m_s", "future_normal_rmse_N",
                                 "future_friction_rmse_N"), map(float, rmse[i]))))
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/observer_development_v2.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--model-seeds", type=int, nargs="+")
    parser.add_argument("--calibration-noise", nargs="+")
    parser.add_argument("--filters", nargs="+", choices=FILTERS)
    parser.add_argument("--forecast-starts", nargs="+", choices=STARTS)
    args = parser.parse_args()
    out, data = args.out.resolve(), args.data.resolve()
    if out == ROOT/"results" or not out.is_relative_to(ROOT/"results"):
        raise ValueError("Output must be a fresh subdirectory of paper_study/results")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("Existing results retained; use fresh output")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    files = sorted(data.rglob("*.npz"))
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        raise ValueError("No native data")
    generation = json.loads((data/"generation_lock.json").read_text())
    for path, expected in generation["sources_sha256"].items():
        if sha(path) != expected:
            raise ValueError(f"generation source mismatch: {path}")
    if json.loads((data/"process.json").read_text())["status"] != "complete":
        raise ValueError("Native generation incomplete")
    cfg = json.loads(args.config.read_text())
    for key in ("grid_size", "model_seeds", "calibration_noise", "filters", "forecast_starts"):
        override = getattr(args, key)
        if override is not None:
            cfg[key] = override
    cfg.setdefault("filters", ["joint_sensor_likelihood"])
    cfg.setdefault("forecast_starts", list(STARTS))
    configured_starts(cfg)
    if not cfg["filters"] or len(set(cfg["filters"])) != len(cfg["filters"]) or any(k not in FILTERS for k in cfg["filters"]):
        raise ValueError("invalid filters")
    source_paths = [Path(__file__), args.config, ROOT/"src/models.py", ROOT/"src/observer.py",
                    ROOT/"src/joint_observer.py", ROOT.parent/"research_next/src/state_id.py"]
    input_paths = [data/"generation_lock.json", data/"process.json"]
    metadata_by_path = {}
    for path in files:
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata.get("status", "complete") != "complete" or sha(path) != metadata["npz_sha256"]:
            raise ValueError(f"failed or hash-mismatched native batch: {path}")
        metadata_by_path[path] = metadata
        input_paths.extend((path, path.with_suffix(".json")))
        for seed in cfg["model_seeds"]:
            for noise in cfg["calibration_noise"]:
                source_paths.extend(model_paths(metadata["normal_world"], metadata["friction_world"], seed, noise))
                bank = prior_path(cfg, metadata["normal_world"], metadata["friction_world"], seed, noise)
                if bank is not None:
                    load_prior_bank(bank, cfg)
                    source_paths.append(bank)
                    bank_root = bank.parents[2]
                    for extra in (bank_root/"prior_lock.json", bank_root/"process.json"):
                        if not extra.is_file():
                            raise ValueError(f"prior provenance file missing: {extra}")
                        input_paths.append(extra)
                    if json.loads((bank_root/"process.json").read_text())["status"] != "complete":
                        raise ValueError("prior bank generation incomplete")
                    bank_lock = json.loads((bank_root/"prior_lock.json").read_text())
                    for filename, expected in bank_lock["sources_sha256"].items():
                        if sha(filename) != expected:
                            raise ValueError(f"prior generation source mismatch: {filename}")
                        source_paths.append(Path(filename))
    hashes = {str(p.resolve()): sha(p) for p in source_paths}
    input_hashes = {str(p.resolve()): sha(p) for p in input_paths}
    generation_sources = generation["sources_sha256"]
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/"evaluation_lock.json", dict(version=2, config=cfg, sources_sha256=hashes,
        input_sha256=input_hashes, generation_sources_sha256=generation_sources,
        command=sys.argv, role="exploratory; source/model/prior/raw/metadata capture before evaluation",
        factor_definitions=dict(filtered_particles="carry each model hypothesis's own mechanical/material state",
            measured_start="reset every model hypothesis to common final noisy x/v",
            prior_propagation="same particles weighted by fixed population prior",
            trajectory_assimilation="same particles weighted by causal sequence likelihood",
            privileged_full_history="full hidden local command history, not true active-task states")))
    write_json(out/"process.json", dict(status="running", pid=os.getpid(), started_unix=time.time()))
    started = time.perf_counter()
    all_rows, output_hashes = [], {}
    try:
        for batch_number, path in enumerate(files):
            metadata = metadata_by_path[path]
            with np.load(path, allow_pickle=False) as stored:
                truth = {k: stored[k] for k in stored.files}
            expected = metadata.get("expected_steps", round((metadata["probe"]["duration_s"]+generation["config"]["future_seconds"])/metadata["dt"]))
            if truth["u"].shape[1] != expected or truth["observation_steps"][-1] != expected:
                raise ValueError("incomplete native horizon")
            inputs = observable_inputs(truth)
            for seed in cfg["model_seeds"]:
                for noise in cfg["calibration_noise"]:
                    model = load_model(metadata["normal_world"], metadata["friction_world"], seed, noise)
                    bank = prior_path(cfg, metadata["normal_world"], metadata["friction_world"], seed, noise)
                    grid, prior = load_prior_bank(bank, cfg)
                    predictions = {}
                    for kind in cfg["filters"]:
                        tic = time.perf_counter()
                        inferred, detail = infer_and_forecast(model, inputs, metadata, cfg, kind, grid=grid, prior=prior)
                        inference_forecast_seconds = time.perf_counter()-tic
                        privileged, privileged_detail = privileged_forecasts(model, inputs, metadata, cfg, kind)
                        predictions.update(inferred); predictions.update(privileged)
                        destination = out/f"r{seed}/{noise}"/path.relative_to(data).with_suffix("")/f"{kind}.npz"
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with destination.open("xb") as stream:
                            np.savez_compressed(stream, **detail, **privileged_detail)
                        write_json(destination.with_suffix(".json"), dict(filter_kind=kind,
                            forecast_starts=cfg["forecast_starts"], preparation_seed=preparation_seed(metadata),
                            model_seed=seed, calibration_noise=noise, input_npz_sha256=sha(path),
                            input_metadata_sha256=sha(path.with_suffix(".json")),
                            prior_bank=None if bank is None else str(bank),
                            prior_bank_sha256=None if bank is None else sha(bank),
                            npz_sha256=sha(destination), inference_and_forecast_seconds=inference_forecast_seconds,
                            scope="Same forecast particles for prior/assimilation; no future truth/sensors in predictor"))
                        for product in (destination, destination.with_suffix(".json")):
                            output_hashes[str(product)] = sha(product)
                    all_rows.extend(score_forecasts(predictions, truth, metadata, seed, noise))
            print(f"EVALUATED v2 {batch_number+1}/{len(files)} {path.relative_to(data)}", flush=True)
        with (out/"episodes.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
            writer.writeheader(); writer.writerows(all_rows)
        if hashes != {p: sha(p) for p in hashes} or input_hashes != {p: sha(p) for p in input_hashes}:
            raise RuntimeError("Evaluation source/model/prior/input changed during run")
        if generation_sources != {p: sha(p) for p in generation_sources}:
            raise RuntimeError("Generation source changed during evaluation")
        output_hashes[str(out/"episodes.csv")] = sha(out/"episodes.csv")
        write_json(out/"artifact_manifest.json", dict(outputs_sha256=output_hashes,
                   evaluation_lock_sha256=sha(out/"evaluation_lock.json")))
        write_json(out/"process.json", dict(status="complete", seconds=time.perf_counter()-started,
            batches=len(files), rows=len(all_rows), ended_unix=time.time()))
    except BaseException as exc:
        write_json(out/"process.json", dict(status="failed", error=repr(exc), seconds=time.perf_counter()-started))
        raise


if __name__ == "__main__":
    main()
