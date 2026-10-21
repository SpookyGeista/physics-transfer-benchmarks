"""Generate prospectively locked round-1 reference command logs in native CPU PhysX.

No fitting or test-model prediction occurs here. Code AND model locks are
required before any reserved test stimulus is generated or Isaac is started.
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

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT.parent/"study_v2"
sys.path[:0] = [str(ROOT/"src"), str(STUDY/"src")]
import numpy as np
from round1_lock import verify_code_lock, verify_model_lock
from round1_reference import CompositeReference
from round1_stimuli import conditions, preparation, commands, velocity_profile, load_config


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_locks():
    return dict(code_lock_sha256=verify_code_lock(), model_lock_sha256=verify_model_lock())


def inverse_action(next_velocity, state, normal, friction, slope, dt, mass):
    """Privileged one-time stimulus construction, using actual current velocity."""
    return mass*(next_velocity-state[:, 1])/dt + slope*normal - friction


def source_hashes():
    files = [Path(__file__).resolve(), ROOT/"src/round1_reference.py", ROOT/"src/round1_stimuli.py",
             ROOT/"src/round1_lock.py", ROOT/"configs/round1.json",
             STUDY/"src/decision_study/reference.py", STUDY/"src/decision_study/isaac_rig.py",
             STUDY/"src/decision_study/common.py", STUDY/"configs/experiment.json"]
    return {str(path): digest(path) for path in files}


def native_run(args, locked):
    # Recheck immediately before native setup, including every fitted checkpoint.
    if verify_locks() != locked:
        raise RuntimeError("Locks changed between preflight and native startup")
    cfg = load_config()
    dt, mass = cfg["dt"], cfg["mass"]
    steps, warm_steps = round(cfg["motion_seconds"]/dt), round(cfg["warmup_seconds"]/dt)
    from decision_study.common import cam, DT, MASS
    if dt != DT or mass != MASS:
        raise ValueError("Round configuration differs from unchanged IsaacRig constants")
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    start = time.perf_counter()
    try:
        import torch
        from decision_study.isaac_rig import IsaacRig
        torch.set_num_threads(2)
        rig = IsaacRig(cfg["profiles_per_replicate"]["evaluation"], "cpu", dt=dt)
        hashes = source_hashes()
        mapping = {f"{normal}_{friction}": (normal, friction) for normal, friction in cfg["compositions"]}
        for composition in args.compositions:
            normal_world, friction_world = mapping[composition]
            for seed in args.seeds:
                path = args.out/composition/f"r{seed}.npz"
                metadata_path = path.with_suffix(".json")
                if path.exists() and metadata_path.exists():
                    old = json.loads(metadata_path.read_text())
                    if old["source_sha256"] != hashes or old["locks"] != locked or old["npz_sha256"] != digest(path):
                        raise RuntimeError(f"Existing test provenance differs: {path}")
                    print("SKIP_VERIFIED", path, flush=True)
                    continue
                if path.exists() or metadata_path.exists():
                    raise FileExistsError(f"Incomplete existing test retained: {path}")
                tic = time.perf_counter()
                rows = conditions(seed, cfg)
                count = len(rows)
                amplitude = np.asarray([row["amplitude"] for row in rows])
                d, rate, target = commands(rows, cfg, extra_steps=0)
                target_velocity = velocity_profile(rows, cfg)
                rig.reset()
                reference = CompositeReference(normal_world, friction_world, count, .02)
                warm = {key: [] for key in ["delta", "rate", "velocity", "normal", "friction", "state", "next_state"]}
                print("TEST_START", composition, seed, "pid", os.getpid(), flush=True)
                for t in range(warm_steps):
                    state = rig.state()
                    dd, rr = preparation(t*dt, rows)
                    nn, ff = reference.force(dd, rr, state[:, 1], dt=dt)
                    ns = rig.advance(np.zeros(count))  # Ideal frame clamp supports material reaction.
                    values = dict(delta=dd, rate=rr, velocity=state[:, 1], normal=nn,
                                  friction=ff, state=state, next_state=ns)
                    for key, value in values.items():
                        warm[key].append(np.array(value, copy=True))
                records = {key: [] for key in ["state", "next_state", "u", "delta", "compression_rate",
                                               "normal", "friction", "force", "slope"]}
                for t in range(steps):
                    state = rig.state()
                    gap, slope = cam(state[:, 0], amplitude)
                    delta = d[:, t] + gap
                    compression_rate = rate[:, t] + slope*state[:, 1]
                    normal, friction = reference.force(delta, compression_rate, state[:, 1], dt=dt)
                    u = inverse_action(target_velocity[:, t+1], state, normal, friction, slope, dt, mass)
                    force = u - slope*normal + friction
                    next_state = rig.advance(force)
                    values = dict(state=state, next_state=next_state, u=u, delta=delta,
                                  compression_rate=compression_rate, normal=normal, friction=friction,
                                  force=force, slope=slope)
                    for key, value in values.items():
                        records[key].append(np.array(value, copy=True))
                arrays = {key: np.stack(value, 1) for key, value in records.items()}
                arrays.update({"warm_"+key: np.stack(value, 1) for key, value in warm.items()})
                arrays.update(d=d, rate=rate, amplitude=amplitude, target=target,
                              target_velocity=target_velocity, time=(np.arange(steps)+1)*dt,
                              warm_time=np.arange(warm_steps)*dt)
                # Preserve all native diagnostics; a failed numerical gate does
                # not drop cases or silently change their commands/tolerances.
                all_finite = all(np.isfinite(value).all() for value in arrays.values())
                tracking = np.abs(arrays["next_state"][..., 1] - target_velocity[:, 1:])
                position_residual = np.abs(arrays["next_state"][..., 0] - arrays["state"][..., 0] - dt*arrays["next_state"][..., 1])
                velocity_residual = np.abs(arrays["next_state"][..., 1] - arrays["state"][..., 1] - dt*arrays["force"]/mass)
                arrays.update(velocity_tracking_error=tracking, position_step_residual=position_residual,
                              velocity_step_residual=velocity_residual)
                numeric = dict(all_finite=all_finite,
                    native_velocity_tracking_max=float(np.max(tracking)) if all_finite else None,
                    native_step_position_residual_max=float(np.max(position_residual)) if all_finite else None,
                    native_step_velocity_residual_max=float(np.max(velocity_residual)) if all_finite else None,
                    warm_clamp_state_max=float(np.max(np.abs(arrays["warm_next_state"]))) if all_finite else None,
                    friction_power_max=float(np.max(arrays["friction"]*arrays["state"][..., 1])) if all_finite else None,
                    normal_min=float(np.min(arrays["normal"])) if all_finite else None,
                    normal_max=float(np.max(arrays["normal"])) if all_finite else None,
                    velocity_min=float(np.min(arrays["state"][..., 1])) if all_finite else None,
                    velocity_max=float(np.max(arrays["state"][..., 1])) if all_finite else None,
                    delta_min=float(np.min(arrays["delta"])) if all_finite else None,
                    delta_max=float(np.max(arrays["delta"])) if all_finite else None)
                limits = cfg["numerical_validity"]
                numeric["native_gate_pass"] = bool(all_finite and
                    numeric["native_velocity_tracking_max"] <= limits["native_velocity_tracking_max"] and
                    numeric["native_step_position_residual_max"] <= limits["native_step_position_residual_max"] and
                    numeric["warm_clamp_state_max"] <= 1e-9)
                if reference.updates != warm_steps + steps:
                    raise AssertionError("Reference timestep was duplicated or skipped")
                if verify_locks() != locked or source_hashes() != hashes:
                    raise RuntimeError("Locked inputs changed during reference generation")
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                usage = resource.getrusage(resource.RUSAGE_SELF)
                metadata = dict(composition=composition, normal_world=normal_world, friction_world=friction_world,
                    replicate=seed, rows=rows, dt=dt, mass=mass, steps=steps, warm_steps=warm_steps,
                    source_sha256=hashes, locks=locked, **locked, npz_sha256=digest(path), numerical=numeric,
                    backend="existing IsaacRig / native CPU PhysX", pid=os.getpid(), python=sys.executable,
                    argv=sys.argv, generated_utc=datetime.now(timezone.utc).isoformat(),
                    seconds=time.perf_counter()-tic, total_process_seconds=time.perf_counter()-start,
                    cpu_user_seconds=usage.ru_utime, cpu_system_seconds=usage.ru_stime, peak_rss_kib=usage.ru_maxrss,
                    oracle_commands="one-time actual-velocity inverse dynamics; saved u is immutable for predictors",
                    compression_command="known d sinusoid; actual delta=d+cam(actual x), no geometry cancellation",
                    reference_updates=reference.updates, hidden_reference_states_saved=False,
                    preparation="observed full prefix in ideal fixed frame; friction state advances even at zero v",
                    scope="prospectively locked round1 held-out reference commands; no model evaluation")
                with metadata_path.open("x") as stream:
                    json.dump(metadata, stream, indent=2, allow_nan=False)
                print("TEST_SAVED", composition, seed, json.dumps(numeric), flush=True)
        if verify_locks() != locked:
            raise RuntimeError("Final code/model lock verification failed")
        print("ROUND1_REFERENCE_GENERATION_COMPLETE", flush=True)
    finally:
        app.close(wait_for_replicator=False, skip_cleanup=True)


def main():
    locked = verify_locks()  # MUST precede conditions() and AppLauncher startup.
    cfg = load_config()
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", choices=cfg["replicates"], default=cfg["replicates"])
    labels = [f"{n}_{f}" for n, f in cfg["compositions"]]
    parser.add_argument("--compositions", nargs="+", choices=labels, default=labels)
    parser.add_argument("--out", type=Path, default=ROOT/"results/round1/test")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.device != "cpu":
        parser.error("Native generation requires --device cpu")
    args.out = args.out.resolve()
    if not args.out.is_relative_to((ROOT/"results/round1").resolve()):
        parser.error("Round-1 test output must stay under results/round1")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.compositions)) != len(args.compositions):
        parser.error("Duplicate seed/composition requests are not allowed")
    native_run(args, locked)


if __name__ == "__main__":
    main()
