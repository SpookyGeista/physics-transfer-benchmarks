"""Fresh isolated EXPLORATORY material data in existing native CPU PhysX.

No controller, target composition, real robot, or reference parameter access.
Full defaults: 24 train + 8 validation profiles, A and B each, 6 s, replicate 0.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = RESEARCH_ROOT.parent / "study_v2"
sys.path.insert(0, str(STUDY_ROOT / "src"))
sys.path.insert(0, str(RESEARCH_ROOT / "src"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", nargs="+", choices=["maxwell", "boucwen"], default=["maxwell", "boucwen"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--out", type=Path, default=RESEARCH_ROOT / "results/development")
    parser.add_argument("--count", type=int, help="Smoke cap on BOTH split profile counts")
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--train-source-base", type=int, default=7_100_000)
    parser.add_argument("--val-source-base", type=int, default=7_200_000)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.device != "cpu":
        parser.error("This development generator requires --device cpu")
    if not 1 <= args.steps <= 2400 or (args.count is not None and not 1 <= args.count <= 24):
        parser.error("Require 1<=steps<=2400 and optional 1<=count<=24")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("Replicates must be unique nonnegative integers")
    if len(set(args.worlds)) != len(args.worlds):
        parser.error("Worlds must be unique")
    out = args.out if args.out.is_absolute() else RESEARCH_ROOT / args.out
    out = out.resolve()
    if not out.is_relative_to((RESEARCH_ROOT / "results").resolve()):
        parser.error("Output must remain under research_next/results")
    counts = {"train": min(args.count or 24, 24), "val": min(args.count or 8, 8)}
    targets = [(world, seed, split, out / world / f"r{seed}" / f"{split}.npz")
               for world in args.worlds for seed in args.seeds for split in counts]
    for _, _, _, path in targets:
        if path.exists() or path.with_suffix(".json").exists():
            raise FileExistsError(f"Refusing to overwrite existing development data: {path}")
    # Verify old sources/data before startup; this does not amend the old lock.
    from decision_study.locking import verify_lock
    original_lock_sha = verify_lock()
    from stimuli import profiles, DT, DURATION
    from decision_study.common import MASS, features, history_init, history_update
    import numpy as np

    # Material-independent sample generation and seed-collision checks happen first.
    stimuli = {}
    used_sources = set()
    for seed in args.seeds:
        for split, count in counts.items():
            base = args.train_source_base if split == "train" else args.val_source_base
            pr, rows = profiles(split, seed, count, args.steps, source_base=base)
            sources = {row["stimulus_seed"] for row in rows}
            if used_sources.intersection(sources):
                raise ValueError("Source seed collision across requested splits/replicates")
            used_sources.update(sources)
            stimuli[seed, split] = pr, rows
    out.mkdir(parents=True, exist_ok=True)
    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-pid{os.getpid()}"
    event_path = out / f"{run_id}.jsonl"
    process_path = out / f"{run_id}.json"
    started = time.perf_counter()
    source_files = [Path(__file__).resolve(), RESEARCH_ROOT / "src/stimuli.py",
                    STUDY_ROOT / "src/decision_study/reference.py",
                    STUDY_ROOT / "src/decision_study/isaac_rig.py",
                    STUDY_ROOT / "src/decision_study/common.py"]
    process = {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
               "python": sys.executable, "python_version": sys.version, "argv": sys.argv,
               "cwd": str(Path.cwd()), "output": str(out), "role": "exploratory_development_only",
               "original_lock_sha256": original_lock_sha,
               "source_sha256": {str(path): sha256(path) for path in source_files},
               "threads": 2, "dt": DT, "steps": args.steps, "full_duration_seconds": DURATION,
               "counts": counts, "worlds": args.worlds, "replicates": args.seeds}
    write_json(process_path, process)

    def event(status, **values):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        item = {"status": status, "pid": os.getpid(),
                "utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "cpu_user_seconds": usage.ru_utime, "cpu_system_seconds": usage.ru_stime,
                "peak_rss_kib": usage.ru_maxrss, **values}
        with event_path.open("a") as stream:
            stream.write(json.dumps(item, allow_nan=False) + "\n")
        print(json.dumps(item, allow_nan=False), flush=True)

    app = None
    event("starting")
    try:
        app = AppLauncher(args).app
        import torch
        from decision_study.reference import Reference
        from decision_study.isaac_rig import IsaacRig
        torch.set_num_threads(2)
        nmax = max(counts.values())
        rig = IsaacRig(nmax, "cpu", dt=DT)
        for world, seed, split, path in targets:
            tic = time.perf_counter()
            pr, source_rows = stimuli[seed, split]
            n = len(source_rows)
            batches, rows = [], []
            normal_load = None
            for mode in ["B", "A"]:
                rig.reset()
                d0 = pr["delta"][:, 0] if mode == "B" else np.full(n, .02)
                reference = Reference(world, n, d0)
                records = {key: [] for key in ("features", "normal", "friction", "state", "next_state",
                                                "delta", "rate", "u", "d", "force", "velocity_target")}
                history = None
                for t in range(args.steps):
                    state = rig.state()[:n]
                    velocity = state[:, 1]
                    delta = pr["delta"][:, t] if mode == "B" else np.full(n, .02)
                    rate = pr["rate"][:, t] if mode == "B" else np.zeros(n)
                    normal, friction = reference.force(delta, rate, velocity,
                        normal_override=None if mode == "B" else normal_load[:, t],
                        friction_enabled=mode == "A", dt=DT)
                    if history is None:
                        history = history_init(delta, velocity, normal)
                    target_v = np.zeros(n) if mode == "B" else pr["v"][:, t + 1]
                    u = np.zeros(n) if mode == "B" else MASS * (target_v - velocity) / DT - friction
                    force = u + friction
                    full_force = np.zeros(nmax)
                    full_force[:n] = force
                    next_state = rig.advance(full_force)[:n]
                    values = dict(features=features(delta, rate, velocity, history, normal), normal=normal,
                                  friction=friction, state=state, next_state=next_state, delta=delta,
                                  rate=rate, u=u, d=delta, force=force, velocity_target=target_v)
                    for key, value in values.items():
                        records[key].append(np.array(value, copy=True))
                    history = history_update(history, delta, velocity, normal, dt=DT)
                arrays = {key: np.stack(value, axis=1) for key, value in records.items()}
                if not all(np.isfinite(value).all() for value in arrays.values()):
                    raise FloatingPointError(f"Nonfinite development output: {world}/{split}/{mode}")
                if mode == "B":
                    normal_load = arrays["normal"].copy()
                    if np.max(np.abs(arrays["state"])) > 1e-12 or np.any(arrays["friction"] != 0):
                        raise AssertionError("Isolated normal fixture must remain exactly stationary")
                batches.append(arrays)
                rows.extend([{**row, "mode": mode, "id": row["id"] + "-" + mode,
                              "amplitude": 0.} for row in source_rows])
                event("mode_complete", world=world, replicate=seed, split=split, mode=mode)
            merged = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
            if not np.array_equal(batches[0]["normal"], batches[1]["normal"]):
                raise AssertionError("A load clamp must replay B measured normal exactly")
            stats = {"normal_min": float(merged["normal"].min()), "normal_max": float(merged["normal"].max()),
                     "friction_power_max": float((merged["friction"] * merged["state"][..., 1]).max()),
                     "velocity_tracking_max": float(np.max(np.abs(merged["next_state"][..., 1] - merged["velocity_target"]))),
                     "B_stationary_max": float(np.max(np.abs(batches[0]["state"]))),
                     "A_B_normal_clamp_max": 0.}
            path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation prevents accidental or concurrent replacement.
            with path.open("xb") as stream:
                np.savez_compressed(stream, **merged)
            write_json(path.with_suffix(".json"), {**process, "world": world, "replicate": seed,
                       "split": split, "rows": rows, "seconds": time.perf_counter() - tic,
                       "backend": "Isaac Lab / native CPU PhysX", "schema": "v2 isolated calibration plus velocity_target",
                       "cascade": "B measured normal-force history imposed as independent clamp load in A",
                       "oracle_stimulus": True, "oracle_scope": "one-time isolated A velocity-tracking command construction",
                       "reference_parameters_read": False, "oracle_target_data": False,
                       "confirmed_main_test": False, "measured": stats, "data_sha256": sha256(path)})
            event("saved", path=str(path), rows=len(rows), measured=stats)
        if verify_lock() != original_lock_sha:
            raise AssertionError("Original protocol lock changed during generation")
        for path, expected in process["source_sha256"].items():
            if sha256(path) != expected:
                raise AssertionError(f"Source changed during generation: {path}")
        event("complete", original_lock_verified_after=True)
    except BaseException as exc:
        event("failed", exception=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if app is not None:
            app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
