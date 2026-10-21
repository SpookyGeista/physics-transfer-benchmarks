"""Crossed causal native trials: microstep feedforward and sensor-ZOH feedback.

Separate version; previous generator, sources and captured data remain intact.
Preparation is an ideal local material fixture, not force-limited native motion.
Importing this module does not import or launch Isaac. Only main starts the app.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/paper-study-numba")
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import hashlib
import json
import signal
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path[:0] = [str(ROOT / "src"), str(REPO / "study_v2/src")]
from preparation_v2 import (RECIPES, TEST_LOCK_ROLE, check_split_access, factorial_design,
                            referencepair_indices, truth_preparation, truth_snapshot)
from decision_study.common import cam


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value, *, exclusive=False):
    with Path(path).open("x" if exclusive else "w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def numeric_audit(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): numeric_audit(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [numeric_audit(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def probe_feedforward(t, probe, signs):
    """Known sinusoidal force evaluated every dt, including the 50-ms probe."""
    signs = np.asarray(signs, dtype=float)
    if probe["duration_s"] == 0:
        return np.zeros_like(signs)
    return signs * probe["excitation_N"] * np.sin(2 * np.pi * t / probe["duration_s"])


def probe_action(held_feedback, t, probe, signs, cfg):
    """Hold only feedback; add current feedforward and then saturate once."""
    limit = cfg["probe_controller"]["max_force_N"]
    return np.clip(held_feedback + probe_feedforward(t, probe, signs), -limit, limit)


def future_command(t, duration, bias, signs, cfg):
    phase = 2 * np.pi * t / duration
    d = cfg["initial_compression_m"] + cfg["future_compression_amplitude_m"] * np.sin(phase)
    rate = cfg["future_compression_amplitude_m"] * 2 * np.pi / duration * np.cos(phase)
    force = bias + signs * cfg["future_excitation_N"] * np.sin(phase)
    limit = cfg["probe_controller"]["max_force_N"]
    return d, rate, np.clip(force, -limit, limit)


def source_paths(config_path):
    return [Path(__file__).resolve(), Path(config_path).resolve(), ROOT / "src/preparation_v2.py",
            ROOT / "src/preparation.py", REPO / "study_v2/src/decision_study/reference.py",
            REPO / "study_v2/src/decision_study/isaac_rig.py",
            REPO / "study_v2/src/decision_study/common.py"]


def verify_test_lock(cfg, config_path, hashes, test_lock):
    """Reserved test generation requires an explicit matching authorization.

    JSON schema: schema_version=1, role=TEST_LOCK_ROLE, status='locked',
    split='test', preparation_seeds equal to config, config_sha256, and
    sources_sha256 containing every generation source plus any other frozen
    study sources. Every listed file hash is checked before and after the run.
    Merely creating an arbitrary or empty lock file does not authorize a test.
    """
    if cfg["split"] != "test":
        if test_lock is not None:
            raise ValueError("--test-lock is reserved for an explicitly locked test split")
        return None
    if test_lock is None:
        raise ValueError("Test split requires --test-lock before any reserved profile is drawn")
    locked = json.loads(Path(test_lock).read_text())
    if (locked.get("schema_version") != 1 or locked.get("role") != TEST_LOCK_ROLE
            or locked.get("status") != "locked" or locked.get("split") != "test"
            or locked.get("preparation_seeds") != cfg["preparation_seeds"]
            or locked.get("config_sha256") != sha(config_path)):
        raise ValueError("Test lock does not match the declared generation configuration")
    frozen = locked.get("sources_sha256", {})
    if not all(frozen.get(path) == digest for path, digest in hashes.items()):
        raise ValueError("Test lock is missing or mismatches a generation source")
    for path, digest in frozen.items():
        if sha(path) != digest:
            raise ValueError(f"Test authorization source mismatch: {path}")
    return sha(test_lock)


def validate_config(cfg, *, test_lock=None):
    """Reject unsupported fixture/native settings without drawing any profile."""
    def finite(value, name, positive=False, nonnegative=False):
        if (not np.isscalar(value) or isinstance(value, (bool, str)) or not np.isfinite(value)
                or (positive and value <= 0) or (nonnegative and value < 0)):
            raise ValueError(f"Invalid {name}: {value}")
    for key in ("dt", "future_seconds", "mass_kg", "max_wall_seconds"):
        finite(cfg[key], key, positive=True)
    if cfg["dt"] != .0025 or cfg["mass_kg"] != 2. or cfg["initial_compression_m"] != .02:
        raise ValueError("This version retains the existing dt=.0025, mass=2kg and fresh delta=.02m")
    obs_steps = cfg["observation_steps"]
    finite(obs_steps, "observation_steps", positive=True)
    if int(obs_steps) != obs_steps:
        raise ValueError("observation_steps must be an integer")
    seeds = cfg["preparation_seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("preparation_seeds must be nonempty and unique")
    for seed in seeds:
        check_split_access(seed, cfg["split"], test_lock)
    if cfg["preparation_indices"] != list(range(len(RECIPES))):
        raise ValueError("The fully crossed batch requires preparation_indices [0,1,2,3,4,5]")
    amps = np.asarray(cfg["cam_amplitudes_m"], dtype=float)
    if amps.shape != (4,) or not np.isfinite(amps).all() or len(np.unique(amps)) != 4:
        raise ValueError("The batch requires four distinct finite cam amplitudes")
    if len(cfg["excitation_signs"]) != 2 or set(cfg["excitation_signs"]) != {-1, 1}:
        raise ValueError("The batch requires both excitation signs exactly once")
    if cfg["episodes_per_batch"] != len(RECIPES) * len(amps) * 2:
        raise ValueError("episodes_per_batch must equal the full 48-cell crossing")
    compositions = cfg["compositions"]
    if (not compositions or len({tuple(c) for c in compositions}) != len(compositions)
            or any(len(c) != 2 or any(w not in ("maxwell", "boucwen") for w in c) for c in compositions)):
        raise ValueError("Invalid or duplicate material composition")
    names = []
    for probe in cfg["probes"]:
        name = probe["name"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("Probe names must be safe single path components")
        names.append(name)
        finite(probe["duration_s"], "probe duration", nonnegative=True)
        finite(probe["excitation_N"], "probe excitation", nonnegative=True)
        if probe["duration_s"] == 0 and probe["excitation_N"] != 0:
            raise ValueError("Zero-duration preparation cannot have excitation")
    if not names or len(set(names)) != len(names):
        raise ValueError("Probes must be nonempty and have unique names")
    for duration in [cfg["future_seconds"], *[p["duration_s"] for p in cfg["probes"]]]:
        if not np.isclose(duration / cfg["dt"], round(duration / cfg["dt"]), rtol=0, atol=1e-10):
            raise ValueError("Every duration must be an integer number of microsteps")
    for key in ("future_compression_amplitude_m", "future_excitation_N"):
        finite(cfg[key], key, nonnegative=True)
    for name, sensor in cfg["sensor_conditions"].items():
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("Sensor names must be safe single path components")
        for key in ("sigma_x_m", "sigma_v_m_s", "sigma_normal_N"):
            finite(sensor[key], key, nonnegative=True)
    if not cfg["sensor_conditions"]:
        raise ValueError("At least one sensor condition is required")
    for key in ("kp_N_m", "kd_N_s_m", "max_force_N"):
        finite(cfg["probe_controller"][key], key, nonnegative=True)
    if cfg["probe_controller"]["max_force_N"] <= 0:
        raise ValueError("max_force_N must be positive")
    for key in ("max_abs_position_m", "max_abs_velocity_m_s", "min_normal_N"):
        finite(cfg["guard"][key], key, positive=True)
    finite(cfg["sensor_noise_seed"], "sensor_noise_seed", nonnegative=True)
    if int(cfg["sensor_noise_seed"]) != cfg["sensor_noise_seed"]:
        raise ValueError("sensor_noise_seed must be an integer")


def run_batch(rig, nw, fw, probe, sensor_name, cfg, factors, destination, *, test_lock=None,
              deadline=None):
    """Record the unchanged observer schema plus explicit truth-only factors.

    Finite guard excursions remain recorded. Any nonfinite value, interruption
    or incomplete horizon retains a failed batch and raises to fail the run.
    """
    count, dt = len(factors), cfg["dt"]
    seeds = {row["preparation_seed"] for row in factors}
    if len(seeds) != 1:
        raise ValueError("One batch must contain exactly one preparation seed")
    preparation_seed = seeds.pop()
    indices = np.asarray([row["preparation_index"] for row in factors], dtype=int)
    amplitudes = np.asarray([row["amplitude"] for row in factors])
    signs = np.asarray([row["excitation_sign"] for row in factors], dtype=float)
    probe_steps = round(probe["duration_s"] / dt)
    future_steps = round(cfg["future_seconds"] / dt)
    total = probe_steps + future_steps
    ref = referencepair_indices(fw, indices, preparation_seed, cfg["split"],
                                normal_world=nw, test_lock=test_lock)
    prep_audit, initial_truth = truth_preparation(ref), truth_snapshot(ref)
    rig.reset()
    sensor = cfg["sensor_conditions"][sensor_name]
    noise_seed = cfg["sensor_noise_seed"] + 100 * cfg["compositions"].index([nw, fw])
    max_total = round((max(p["duration_s"] for p in cfg["probes"]) + cfg["future_seconds"]) / dt)
    # Stable factor-addressed streams shared across probes and sensor conditions.
    noise_keys = [[noise_seed, preparation_seed, row["preparation_index"], row["cam_index"],
                   int(row["excitation_sign"] > 0)] for row in factors]
    standard_noise = np.stack([np.random.default_rng(np.random.SeedSequence(key)).normal(
        size=(max_total + 1, 3)) for key in noise_keys])
    noise_scale = np.array([sensor["sigma_x_m"], sensor["sigma_v_m_s"], sensor["sigma_normal_N"]])
    held_feedback = np.zeros(count)
    future_bias = np.full(count, np.nan)  # failed batches may never reach this boundary
    obs_records = []
    records = {k: [] for k in ["state", "next_state", "normal", "friction", "delta", "compression_rate",
                               "u", "net_force", "d", "d_dot", "guard", "phase"]}
    start = time.perf_counter()
    error = None
    try:
        for step in range(total + 1):
            if deadline is not None and time.perf_counter() > deadline:
                raise TimeoutError("Declared generation wall-clock budget exceeded")
            current = rig.state()
            if current.shape != (count, 2) or not np.isfinite(current).all():
                raise FloatingPointError("Invalid or nonfinite current native state")
            g, slope = cam(current[:, 0], amplitudes)
            if step < probe_steps:
                d, rate = cfg["initial_compression_m"], 0.
            else:
                d, rate, _ = future_command((step - probe_steps) * dt, cfg["future_seconds"],
                                            np.zeros(count), signs, cfg)
            delta, compression_rate = d + g, rate + slope * current[:, 1]
            if step % cfg["observation_steps"] == 0 or step == probe_steps or step == total:
                # Boundary N is t=0+ after invoking the known future d_dot.
                # It is unavailable to a stop/continue decision made before that invocation.
                observed_normal = ref.observe(delta, compression_rate, current[:, 1], dt=dt)[0]
                noisy = np.column_stack((current, observed_normal)) + standard_noise[:, step] * noise_scale
                if not np.isfinite(noisy).all():
                    raise FloatingPointError("Nonfinite sensor observation")
                obs_records.append((step, noisy.copy()))
                if step < probe_steps:
                    observed_slope = cam(noisy[:, 0], amplitudes)[1]
                    control = cfg["probe_controller"]
                    held_feedback = observed_slope * noisy[:, 2] - control["kp_N_m"] * noisy[:, 0] \
                                    - control["kd_N_s_m"] * noisy[:, 1]
            if step == probe_steps:
                future_bias = cam(noisy[:, 0], amplitudes)[1] * noisy[:, 2]
            if step == total:
                break
            if step >= probe_steps:
                _, _, action = future_command((step - probe_steps) * dt, cfg["future_seconds"],
                                               future_bias, signs, cfg)
            else:
                action = probe_action(held_feedback, step * dt, probe, signs, cfg)
            normal, friction = ref.force(delta, compression_rate, current[:, 1], dt=dt)
            net = action - slope * normal + friction
            force_finite = np.isfinite(np.column_stack((normal, friction, net, action))).all(axis=1)
            # Never feed NaN/Inf into PhysX; retain the failed force and unchanged current state.
            next_state = rig.advance(net) if force_finite.all() else current.copy()
            limits = cfg["guard"]
            state_finite = np.isfinite(next_state).all(axis=1)
            guard = (~state_finite) | (~force_finite) \
                    | (np.abs(next_state[:, 0]) > limits["max_abs_position_m"]) \
                    | (np.abs(next_state[:, 1]) > limits["max_abs_velocity_m_s"]) \
                    | (normal < limits["min_normal_N"])
            for key, value in [("state", current), ("next_state", next_state), ("normal", normal),
                               ("friction", friction), ("delta", delta), ("compression_rate", compression_rate),
                               ("u", action), ("net_force", net), ("d", np.full(count, d)),
                               ("d_dot", np.full(count, rate)), ("guard", guard),
                               ("phase", np.full(count, int(step >= probe_steps)))]:
                records[key].append(value.copy())
            if not state_finite.all() or not force_finite.all():
                raise FloatingPointError("Nonfinite force/state; batch retained as failed")
        if len(records["u"]) != total or not obs_records or obs_records[-1][0] != total:
            raise RuntimeError("Native horizon or final observation incomplete")
    except BaseException as exc:
        error = exc
    arrays = {k: np.stack(v, axis=1) if v else np.empty((count, 0, 2) if k in ("state", "next_state")
                                                      else (count, 0)) for k, v in records.items()}
    arrays.update(amplitude=amplitudes, observation_steps=np.array([t for t, _ in obs_records], dtype=int),
                  observations=np.stack([v for _, v in obs_records], axis=1) if obs_records else np.empty((count, 0, 3)),
                  future_bias=future_bias, probe_steps=np.array(probe_steps), preparation_index=indices,
                  preparation_seed=np.full(count, preparation_seed), excitation_sign=signs,
                  episode_id=np.asarray([row["episode_id"] for row in factors]))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    write_json(destination.with_suffix(".json"), dict(normal_world=nw, friction_world=fw, probe=probe,
        sensor=sensor_name, sensor_parameters=sensor, count=count, dt=dt,
        observation_period_s=cfg["observation_steps"] * dt, seconds=time.perf_counter() - start,
        status="complete" if error is None else "failed", error=None if error is None else repr(error),
        expected_steps=total, recorded_steps=len(records["u"]), preparation_seed=preparation_seed,
        split=cfg["split"], factor_rows=factors, native_backend="Isaac/CPU PhysX, guided mass, explicit material forces",
        npz_sha256=sha(destination), guards=int(arrays["guard"].sum()), noise_seed=noise_seed,
        noise_stream_keys=noise_keys, prepared_reference_audit=numeric_audit(prep_audit),
        initial_truth_audit=numeric_audit(initial_truth),
        preparation_scope="Six unchanged ideal local velocity/load fixture profiles; normal fresh. Histories reached by force transitions, not injected. Not a claim of reachability by the later 12N-limited actuator.",
        factor_scope="Recipe, preparation indices/seed and episode IDs are evaluator-only labels, not observer inputs.",
        observation_scope="Only observations, commands, geometry and declared initial preparation convention belong to the observer. Truth force/state arrays and hidden preparation are evaluator-only.",
        probe_action_scope="Sensor-derived feedback is held for 25ms (configured observation period); known sinusoidal feedforward is evaluated every dt, then the total force is clipped once.",
        future_action_scope="Precommitted from causal boundary observation and known sinusoidal command; no future feedback or inverse-reference action construction.",
        boundary_observation_scope="N at future t=0+ includes the invoked compression-rate command; not available before making a decision to invoke that command."), exclusive=True)
    print(f"NATIVE v2 p{preparation_seed} {nw}/{fw} {probe['name']} {sensor_name}: "
          f"{count} episodes, status={'complete' if error is None else 'failed'}, guards={arrays['guard'].sum()}", flush=True)
    if error is not None:
        raise error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/development_v2.json")
    parser.add_argument("--smoke", action="store_true",
                        help="One seed, BW/BW, gentle_050, noisy; still all 48 factor cells")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test-lock", type=Path, help="Required explicit matching authorization for split=test")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.device != "cpu":
        parser.error("The existing native rig uses CPU PhysX")
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text())
    hashes = {str(p): sha(p) for p in source_paths(config_path)}
    test_lock_sha = verify_test_lock(cfg, config_path, hashes, args.test_lock)
    validate_config(cfg, test_lock=args.test_lock)
    if args.smoke and cfg["split"] == "test":
        parser.error("Smoke mode cannot draw a subset of reserved test profiles")
    if args.smoke and (["boucwen", "boucwen"] not in cfg["compositions"]
                       or "noisy" not in cfg["sensor_conditions"]
                       or "gentle_050" not in [p["name"] for p in cfg["probes"]]):
        parser.error("Smoke requires configured BW/BW, noisy and gentle_050")
    out = args.out.resolve()
    if out == ROOT / "results" or not out.is_relative_to(ROOT / "results"):
        parser.error("Output must be a fresh subdirectory under paper_study/results")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("Existing data retained; choose a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "generation_lock.json", dict(role="generation snapshot before execution",
        version=2, sources_sha256=hashes, config=cfg, test_lock_sha256=test_lock_sha,
        smoke=args.smoke), exclusive=True)
    started = time.perf_counter()
    started_unix = time.time()
    process = dict(pid=os.getpid(), started_unix=started_unix, command=sys.argv, status="starting")
    write_json(out / "process.json", process, exclusive=True)
    app, batches, episodes = None, 0, 0
    previous_signals = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    def interrupted(signum, _frame):
        raise SystemExit(f"Generation interrupted by signal {signum}")
    for signum in previous_signals:
        signal.signal(signum, interrupted)
    try:
        app = AppLauncher(args).app
        # Isaac installs its own handlers while starting. Override only the
        # ordinary user interrupts so failures are persisted before shutdown.
        for signum in previous_signals:
            signal.signal(signum, interrupted)
        from decision_study.isaac_rig import IsaacRig
        rig = IsaacRig(cfg["episodes_per_batch"], "cpu", dt=cfg["dt"])
        write_json(out / "process.json", {**process, "status": "running"})
        seeds = cfg["preparation_seeds"][:1] if args.smoke else cfg["preparation_seeds"]
        compositions = [["boucwen", "boucwen"]] if args.smoke else cfg["compositions"]
        probes = [p for p in cfg["probes"] if p["name"] == "gentle_050"] if args.smoke else cfg["probes"]
        sensors = ["noisy"] if args.smoke else list(cfg["sensor_conditions"])
        for seed in seeds:
            factors = factorial_design(seed, cfg["split"], cfg["cam_amplitudes_m"],
                                       cfg["excitation_signs"], test_lock=args.test_lock)
            for nw, fw in compositions:
                for probe in probes:
                    for sensor in sensors:
                        destination = out / f"p{seed}/{nw}_{fw}/{probe['name']}/{sensor}.npz"
                        run_batch(rig, nw, fw, probe, sensor, cfg, factors, destination,
                                  test_lock=args.test_lock, deadline=started + cfg["max_wall_seconds"])
                        batches += 1
                        episodes += len(factors)
        if hashes != {p: sha(p) for p in hashes}:
            raise RuntimeError("Source changed during generation")
        if test_lock_sha != verify_test_lock(cfg, config_path, hashes, args.test_lock):
            raise RuntimeError("Test authorization changed during generation")
        # Isaac's immediate framework shutdown may not return. Persist the
        # verified generation result before requesting shutdown in finally.
        write_json(out / "process.json", {**process, "status": "complete", "ended_unix": time.time(),
            "seconds": time.perf_counter() - started, "batches": batches, "episodes": episodes})
        print("COMPLETE", out, flush=True)
    except BaseException as exc:
        write_json(out / "process.json", {**process, "status": "failed", "error": repr(exc),
            "ended_unix": time.time(), "seconds": time.perf_counter() - started,
            "completed_batches": batches, "completed_episodes": episodes})
        raise
    finally:
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)
        if app is not None:
            app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
