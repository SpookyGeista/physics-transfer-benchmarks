"""Fit old_flex/path_ridge on fresh exploratory isolated development data."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from feature_baselines import DT, TAUS, LAMBDAS, VREG, FeatureBaseline, feature_sequence, fit_head


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def load_data(root, world, seed, split, noise="clean"):
    path = root / world / f"r{seed}" / f"{split}.npz"
    meta = json.loads(path.with_suffix(".json").read_text())
    with np.load(path) as arr:
        data = {key: np.asarray(arr[key], dtype=np.float64).copy() for key in ("delta", "rate", "state", "normal", "friction")}
    if meta["steps"] != 2400 or not np.isclose(meta["dt"], DT):
        raise ValueError("These fits require the complete fresh 6 s development profiles")
    if meta.get("role") != "exploratory_development_only":
        raise ValueError("Input is not the new exploratory development dataset")
    rows = meta["rows"]
    if not all(row["mode"] in ["A", "B"] for row in rows):
        raise ValueError("Only isolated A/B rows allowed")
    index = {mode: [i for i, row in enumerate(rows) if row["mode"] == mode] for mode in ["B", "A"]}
    if [rows[i]["profile"] for i in index["B"]] != [rows[i]["profile"] for i in index["A"]]:
        raise ValueError("A/B source profile pairing differs")
    if not np.array_equal(data["normal"][index["B"]], data["normal"][index["A"]]):
        raise ValueError("Measured A/B normal-load replay differs")
    if noise == "noisy":
        # Exact StateID draw order, magnitudes, and full arrays BEFORE A/B masks.
        rng = np.random.default_rng(7200000 + seed*100 + {"train": 1, "val": 2}[split])
        data["delta"] += rng.normal(0, .00005, data["delta"].shape)
        data["rate"] += rng.normal(0, .001, data["rate"].shape)
        data["state"][..., 1] += rng.normal(0, .002, data["state"][..., 1].shape)
        data["normal"] += rng.normal(0, .15, data["normal"].shape)
        data["friction"] += rng.normal(0, .10, data["friction"].shape)
    elif noise != "clean":
        raise ValueError(noise)
    return data, rows, index, {str(path): sha(path), str(path.with_suffix('.json')): sha(path.with_suffix('.json'))}


def local_features(data, rows, index, kind):
    nf, ff = feature_sequence(data, kind)
    ni, fi = index["B"], index["A"]
    return dict(normal=nf[ni].reshape(-1, nf.shape[-1]),
                friction=ff[fi].reshape(-1, ff.shape[-1]),
                target_n=data["normal"][ni].ravel(), target_f=data["friction"][fi].ravel(),
                factor=(-data["normal"][fi]*np.tanh(data["state"][fi, :, 1]/VREG)).ravel())


def predict_batched(model, values, head, block=4096):
    return np.concatenate([model.predict_head(values[i:i+block], head) for i in range(0, len(values), block)])


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true)**2)))


def evaluate(model, data, rows, index, local):
    ni, fi = index["B"], index["A"]
    count, steps = len(ni), data["delta"].shape[1]
    pn = 20*np.maximum(predict_batched(model, local["normal"], "normal"), 0)
    pf = local["factor"]*np.maximum(predict_batched(model, local["friction"], "friction"), 0)
    d, r, v = data["delta"][ni], data["rate"][ni], data["state"][fi, :, 1]
    true_n, true_f = data["normal"][ni], data["friction"][fi]
    state = model.initial(d[:, 0], v[:, 0], true_n[:, 0])
    cn, cf = np.empty((count, steps)), np.empty((count, steps))
    for t in range(steps):
        cn[:, t], cf[:, t], state = model.step(d[:, t], r[:, t], v[:, t], state)
    # Normal prediction must not change when coupled to the tangential history.
    if not np.allclose(cn, pn.reshape(count, steps), atol=1e-8, rtol=1e-9):
        raise AssertionError("Normal model differs between isolated and recombined inputs")
    hold = np.zeros((count, steps), dtype=bool)
    times = np.arange(steps)*DT
    for j, i in enumerate(ni):
        for start, end in rows[i]["normal_hold_intervals"]:
            hold[j] |= (times >= start) & (times < end)
    if not hold.any():
        raise ValueError("No declared normal holds in full development data")
    metrics = dict(normal_rmse=rmse(pn, local["target_n"]),
                   friction_conditional_rmse=rmse(pf, local["target_f"]),
                   normal_hold_rmse=rmse(cn[hold], true_n[hold]), normal_hold_samples=int(hold.sum()),
                   recombined_normal_rmse=rmse(cn, true_n), recombined_friction_rmse=rmse(cf, true_f),
                   recombined_friction_power_max=float(np.max(cf*v)),
                   normal_profile_rmse=np.sqrt(np.mean((cn-true_n)**2, axis=1)).tolist(),
                   friction_conditional_profile_rmse=np.sqrt(np.mean((pf.reshape(count, steps)-true_f)**2, axis=1)).tolist(),
                   recombined_friction_profile_rmse=np.sqrt(np.mean((cf-true_f)**2, axis=1)).tolist())
    arrays = dict(normal=pn.reshape(count, steps), friction_conditional=pf.reshape(count, steps),
                  recombined_normal=cn, recombined_friction=cf, true_normal=true_n, true_friction=true_f,
                  hold_mask=hold, delta=d, rate=r, velocity=v)
    return metrics, arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "results/development")
    p.add_argument("--out", type=Path, default=ROOT / "results/feature_development")
    p.add_argument("--worlds", nargs="+", choices=["maxwell", "boucwen"], default=["maxwell", "boucwen"])
    p.add_argument("--methods", nargs="+", choices=list(TAUS), default=list(TAUS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise", nargs="+", choices=["clean", "noisy"], default=["clean", "noisy"])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for world, noise in itertools.product(args.worlds, args.noise):
        data, rows, index, hashes = {}, {}, {}, {}
        for split in ["train", "val"]:
            data[split], rows[split], index[split], hs = load_data(args.data, world, args.seed, split, noise)
            hashes.update(hs)
        if not set(row["id"] for row in rows["train"]).isdisjoint(row["id"] for row in rows["val"]):
            raise ValueError("Training and validation ids overlap")
        clean_validation = load_data(args.data, world, args.seed, "val", "clean") if noise == "noisy" else None
        for kind in args.methods:
            dest = args.out / world / f"r{args.seed}" / noise / kind
            dest.mkdir(parents=True, exist_ok=True)
            if any((dest/name).exists() for name in ["model.json", "metrics.json", "predictions.npz", "declaration.json"]):
                raise FileExistsError(f"Existing fit retained: {dest}")
            tic = time.perf_counter()
            declaration = dict(world=world, replicate=args.seed, method=kind, noise=noise,
                scope="exploratory fresh isolated development only; validation used for ridge selection",
                started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), python=sys.executable,
                argv=sys.argv, input_sha256=hashes,
                source_sha256={str(path): sha(path) for path in [Path(__file__).resolve(), ROOT/"src/feature_baselines.py"]},
                history_taus=TAUS[kind], ridge_grid=LAMBDAS[kind], dt=DT,
                calibration_noise={"seed": 7200000 + args.seed*100,
                    "split_offset": {"train": 1, "val": 2}, "draw_order": ["delta", "rate", "velocity", "normal", "friction"],
                    "standard_deviations": [.00005, .001, .002, .15, .10], "draws_before_masks": True},
                normal_force_labels_as_inputs=False, reference_parameters_used=False,
                force_validation="N: isolated B; Ft conditional: A observed full N history; recombined: B d/rate + A v, only initial N observed",
                recombined_scope="prescribed local inputs, not free mechanical rollout or controller evaluation")
            write_json(dest/"declaration.json", declaration)
            print("FIT_START", world, noise, kind, flush=True)
            local = {split: local_features(data[split], rows[split], index[split], kind) for split in ["train", "val"]}
            fits = {}
            for head, label in [("normal", "target_n"), ("friction", "target_f")]:
                fits[head] = fit_head(kind, head, local["train"][head], local["train"][label],
                    local["val"][head], local["val"][label], local["train"]["factor"], local["val"]["factor"])
                print("HEAD_FITTED", world, noise, kind, head, fits[head]["selected_lambda"], flush=True)
            model = FeatureBaseline(kind, fits["normal"], fits["friction"], metadata=declaration)
            metrics, arrays = {}, {}
            for split in ["train", "val"]:
                metrics[split], prediction = evaluate(model, data[split], rows[split], index[split], local[split])
                arrays.update({split+"_"+key: value for key, value in prediction.items()})
            if clean_validation is not None:
                cd, cr, ci, _ = clean_validation
                cl = local_features(cd, cr, ci, kind)
                metrics["clean_validation"], prediction = evaluate(model, cd, cr, ci, cl)
                arrays.update({"clean_validation_"+key: value for key, value in prediction.items()})
            usage = resource.getrusage(resource.RUSAGE_SELF)
            report = dict(**declaration, parameters=model.parameter_count(),
                stored_scaling_values=2*model.parameter_count(),
                online_state_values_per_episode=len(TAUS[kind])*4+6,
                normal_inputs=local["train"]["normal"].shape[1], friction_inputs=local["train"]["friction"].shape[1],
                selected_ridge={head: fits[head]["selected_lambda"] for head in fits},
                seconds=time.perf_counter()-tic, cpu_user_seconds=usage.ru_utime,
                cpu_system_seconds=usage.ru_stime, peak_rss_kib=usage.ru_maxrss, **metrics)
            for path, expected in hashes.items():
                if sha(path) != expected:
                    raise AssertionError(f"Input changed during fit: {path}")
            for path, expected in declaration["source_sha256"].items():
                if sha(path) != expected:
                    raise AssertionError(f"Source changed during fit: {path}")
            model.save(dest/"model.json")
            write_json(dest/"metrics.json", report)
            with (dest/"predictions.npz").open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            print("FIT_DONE", world, noise, kind, json.dumps({"val": metrics["val"], "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
