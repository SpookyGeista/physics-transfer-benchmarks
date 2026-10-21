"""Replay only causal sensor/command inputs, then evaluate frozen forecasts.

Truth arrays are passed to a separate scoring function after prediction. Hidden
preparation commands are used only by an explicitly privileged known-history
comparator. No coefficient fitting or hyperparameter selection occurs here.
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def grid_for(family, cfg):
    prior = cfg["prior"][family]
    spacing = np.linspace if prior["spacing"] == "linear" else np.geomspace
    return spacing(*prior["bounds"], cfg["grid_size"])


def observable_inputs(arrays):
    """Whitelist; future sensor samples and physical labels never leave here."""
    end = int(arrays["probe_steps"])
    mask = arrays["observation_steps"] <= end
    return {**{k: arrays[k].copy() for k in ("u", "d", "d_dot", "amplitude")},
            "observation_steps": arrays["observation_steps"][mask].copy(),
            "observations": arrays["observations"][:, mask].copy(), "probe_steps": end}


def propagate_forecast(model, xv, state, inputs, dt, grid_size=1):
    end = inputs["probe_steps"]
    count, total = inputs["u"].shape
    prediction = np.empty((count, grid_size, total-end, 4))
    amp = np.repeat(inputs["amplitude"], grid_size)
    for j, t in enumerate(range(end, total)):
        command = [np.repeat(inputs[key][:, t], grid_size) for key in ("u", "d", "d_dot")]
        xv, state, n, ft = motion_step(model, xv, state, *command, amp, dt=dt)
        prediction[:, :, j] = np.column_stack((xv, n, ft)).reshape(count, grid_size, 4)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("nonfinite forecast; retain run as failed")
    return prediction


def infer_and_forecast(model, inputs, meta, cfg, kind):
    count = inputs["u"].shape[0]
    grid = grid_for(model.friction_family, cfg)
    sensor, floor = meta["sensor_parameters"], cfg["numerical_floor"]
    cls = GridObserver if kind == "conditional_N_hold" else JointGridObserver
    extra = {} if cls is GridObserver else dict(numerical_N=floor["normal_N"])
    observer = cls(model, grid, np.zeros((count, 2)), .02, inputs["observations"][:, 0, 2],
        dt=meta["dt"], sigma_x=sensor["sigma_x_m"], sigma_v=sensor["sigma_v_m_s"],
        sigma_N=sensor["sigma_normal_N"], numerical_x=floor["x_m"], numerical_v=floor["v_m_s"], **extra)
    indices = {int(t): i for i, t in enumerate(inputs["observation_steps"])}
    weight_trace = [observer.weights.copy()]
    for t in range(inputs["probe_steps"]):
        observer.advance(*(inputs[k][:, t] for k in ("u", "d", "d_dot")), inputs["amplitude"])
        if t+1 in indices:
            obs = inputs["observations"][:, indices[t+1]]
            extra = {} if cls is GridObserver else dict(d=inputs["d"][:, t+1],
                d_dot=inputs["d_dot"][:, t+1], amplitude=inputs["amplitude"])
            observer.observe(obs[:, 0], obs[:, 1], obs[:, 2], **extra)
            weight_trace.append(observer.weights.copy())
    start = observer.forecast_start(inputs["observations"][:, -1, :2])
    particles = propagate_forecast(model, start["xv"], start["state"], inputs, meta["dt"], len(grid))
    prior_mean = np.einsum("eg,egtk->etk", observer.prior, particles)
    post_mean = np.einsum("eg,egtk->etk", observer.weights, particles)
    post_variance = np.maximum(np.einsum("eg,egtk->etk", observer.weights, particles**2)-post_mean**2, 0.)
    prior_variance = np.maximum(np.einsum("eg,egtk->etk", observer.prior, particles**2)-prior_mean**2, 0.)
    details = dict(grid=grid, weights=observer.weights, prior=observer.prior,
        weight_trace=np.stack(weight_trace, axis=1), log_likelihood=observer.log_likelihood,
        prior_forecast=prior_mean, assimilated_forecast=post_mean,
        prior_variance=prior_variance, assimilated_variance=post_variance)
    return {kind+"/prior_propagation": prior_mean, kind+"/trajectory_assimilation": post_mean}, details


def privileged_forecast(model, inputs, metadata, kind):
    """Full hidden command history, explicitly separate information privilege."""
    commands = metadata["prepared_reference_audit"]["commands"]
    d, r, v, n = [np.asarray(commands[k]) for k in ("delta", "rate", "v", "normal")]
    count = inputs["u"].shape[0]
    friction = model.friction_model
    state_f = friction.initial(d[:, 0], v[:, 0], n[:, 0])
    dt = metadata["dt"]
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
    xv = inputs["observations"][:, -1, :2].copy()
    return propagate_forecast(model, xv, state, inputs, dt)[:, 0]


def score_forecasts(predictions, truth, metadata, model_seed, calibration_noise):
    end = int(truth["probe_steps"])
    target = np.concatenate((truth["next_state"][:, end:], truth["normal"][:, end:, None],
                              truth["friction"][:, end:, None]), axis=2)
    rows = []
    for method, prediction in predictions.items():
        errors = prediction-target
        rmse = np.sqrt(np.mean(errors**2, axis=1))
        for i in range(len(target)):
            row = dict(normal_world=metadata["normal_world"], friction_world=metadata["friction_world"],
                probe=metadata["probe"]["name"], duration_s=metadata["probe"]["duration_s"],
                sensor=metadata["sensor"], episode=i, model_seed=model_seed, calibration_noise=calibration_noise,
                method=method, native_guard_count=int(truth["guard"][i].sum()))
            row.update(dict(zip(("future_position_rmse_m", "future_velocity_rmse_m_s", "future_normal_rmse_N",
                                 "future_friction_rmse_N"), map(float, rmse[i]))))
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/observer_development.json")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    out, data = args.out.resolve(), args.data.resolve()
    if not out.is_relative_to(ROOT/"results"):
        raise ValueError("Output must stay under paper_study/results")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Existing results retained; use fresh output")
    files = sorted(data.rglob("*.npz"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise ValueError("No native data")
    generation = json.loads((data/"generation_lock.json").read_text())
    for path, expected in generation["sources_sha256"].items():
        assert sha(path) == expected, path
    if json.loads((data/"process.json").read_text())["status"] != "complete":
        raise ValueError("Native generation incomplete")
    cfg = json.loads(args.config.read_text())
    source_paths = [Path(__file__), args.config, ROOT/"src/models.py", ROOT/"src/observer.py",
                    ROOT/"src/joint_observer.py", ROOT.parent/"research_next/src/state_id.py"]
    for path in files:
        metadata = json.loads(path.with_suffix(".json").read_text())
        for seed in cfg["model_seeds"]:
            for noise in cfg["calibration_noise"]:
                source_paths.extend(model_paths(metadata["normal_world"], metadata["friction_world"], seed, noise))
    hashes = {str(p.resolve()): sha(p) for p in source_paths}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/"evaluation_lock.json", dict(config=cfg, sources_sha256=hashes,
        input_sha256={str(p): sha(p) for p in files}, native_lock_sha256=sha(data/"generation_lock.json"),
        command=sys.argv, role="exploratory; capture before evaluation"))
    write_json(out/"process.json", dict(status="running", pid=os.getpid(), started_unix=time.time()))
    started = time.perf_counter()
    all_rows = []
    try:
        for batch_number, path in enumerate(files):
            metadata = json.loads(path.with_suffix(".json").read_text())
            assert sha(path) == metadata["npz_sha256"], path
            with np.load(path) as stored:
                truth = {k: stored[k] for k in stored.files}
            total = round((metadata["probe"]["duration_s"]+.25)/metadata["dt"])
            assert truth["u"].shape[1] == total
            assert truth["observation_steps"][-1] == total
            inputs = observable_inputs(truth)
            for seed in cfg["model_seeds"]:
                for noise in cfg["calibration_noise"]:
                    model = load_model(metadata["normal_world"], metadata["friction_world"], seed, noise)
                    predictions = {}
                    for kind in cfg["filters"]:
                        inferred, detail = infer_and_forecast(model, inputs, metadata, cfg, kind)
                        predictions.update(inferred)
                        predictions[kind+"/privileged_full_history"] = privileged_forecast(model, inputs, metadata, kind)
                        destination = out/f"r{seed}/{noise}"/path.relative_to(data).with_suffix("")/f"{kind}.npz"
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with destination.open("xb") as stream:
                            np.savez_compressed(stream, **detail, privileged_forecast=predictions[kind+"/privileged_full_history"])
                    all_rows.extend(score_forecasts(predictions, truth, metadata, seed, noise))
            print(f"EVALUATED {batch_number+1}/{len(files)} {path.relative_to(data)}", flush=True)
        with (out/"episodes.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
            writer.writeheader(); writer.writerows(all_rows)
        assert hashes == {p: sha(p) for p in hashes}, "Evaluation source changed during run"
        write_json(out/"process.json", dict(status="complete", seconds=time.perf_counter()-started,
            batches=len(files), rows=len(all_rows), ended_unix=time.time()))
    except BaseException as exc:
        write_json(out/"process.json", dict(status="failed", error=repr(exc), seconds=time.perf_counter()-started))
        raise


if __name__ == "__main__":
    main()
