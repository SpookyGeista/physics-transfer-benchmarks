"""Prospectively locked native PI held-out extension; no robot interfaces.

New study sources and outputs only. Import is CPU-only and does not start Isaac.
All generative inputs are fixed before execution; no fitting or score selection.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
sys.path[:0] = [str(HERE), str(ROOT / "src"), str(REPO / "study_v2/src")]
from geometry import cam, GEOMETRY_NAMES, LENGTH_M
from profiles import factorial_design
from plant import PreparedPlant, CouplingFailure


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value, exclusive=False):
    with Path(path).open("x" if exclusive else "w") as stream:
        stream.write(json.dumps(serializable(value), indent=2, allow_nan=False)+"\n")


def source_paths(config_path):
    return [Path(__file__).resolve(), Path(config_path).resolve(), HERE / "geometry.py", HERE / "profiles.py",
            HERE / "plant.py", HERE / "main2_protocol.md", ROOT / "src/implicit_stop_coupling.py",
            ROOT / "alternatives/multistate_friction.py", REPO / "study_v2/src/decision_study/reference.py",
            REPO / "study_v2/src/decision_study/isaac_rig.py", REPO / "study_v2/src/decision_study/common.py",
            REPO / "study_v2/configs/experiment.json"]


def authorization_source_paths(config_path):
    """The test draw binds both generation and all fixed evaluation choices.

    Importing this evaluator only loads CPU definitions and fitted-model paths;
    it does not read native data, construct a prior or draw a material history.
    """
    from evaluate_main2 import source_paths as evaluation_sources
    paths = source_paths(config_path) + evaluation_sources() + [
        ROOT / "tests/test_main2_generation.py", HERE / "test_main2_evaluation.py"]
    return sorted({Path(path).resolve() for path in paths})


def future_command(t, bias, signs, cfg):
    phase = 2*np.pi*t/cfg["future_seconds"]
    d = cfg["initial_compression_m"]+cfg["future_compression_amplitude_m"]*np.sin(phase)
    rate = cfg["future_compression_amplitude_m"]*2*np.pi/cfg["future_seconds"]*np.cos(phase)
    limit = cfg["probe_controller"]["max_force_N"]
    u = np.clip(bias+signs*cfg["future_excitation_N"]*np.sin(phase), -limit, limit)
    return d, rate, u


def probe_action(feedback, t, probe, signs, cfg):
    feedforward = 0. if probe["duration_s"] == 0 else probe["excitation_N"]*np.sin(2*np.pi*t/probe["duration_s"])
    limit = cfg["probe_controller"]["max_force_N"]
    return np.clip(feedback+signs*feedforward, -limit, limit)


def validate_config(cfg):
    """Validate fixed Main2 design without drawing a reserved history."""
    if cfg.get("split") != "test":
        raise ValueError("Main2 generator requires its locked test configuration")
    seeds = cfg["preparation_seeds"]
    if (not seeds or len(set(seeds)) != len(seeds)
            or seeds != [53201,53202,53203,53204]):
        raise ValueError("Main2 uses exactly reserved history blocks 53201 through 53204")
    fixed = dict(profiles_per_seed=6, episodes_per_batch=36, mass_kg=2., initial_compression_m=.02,
                 geometry_length_m=LENGTH_M, cam_amplitude_m=.003, preparation_seconds=4.,
                 preparation_waypoint_limit_m=.025, observation_steps=10)
    for key, expected in fixed.items():
        if cfg[key] != expected:
            raise ValueError(f"This version fixes {key}={expected}")
    if cfg["dt"] != .0025:
        raise ValueError("Initial native pilot uses dt=.0025; refinement needs a declared separate run")
    if cfg["geometry_ids"] != [0, 1, 2] or cfg["excitation_signs"] != [-1, 1]:
        raise ValueError("Use all three declared geometries and both declared signs")
    if cfg["normal_worlds"] != ["maxwell", "boucwen"] or cfg["friction_parameter_sets"] != [
            "balanced", "wide_scales", "compliant", "stiff_high_capacity"]:
        raise ValueError("All two normal types and four predeclared PI parameter sets are required")
    for key in ("max_wall_seconds", "max_solver_residual_N", "future_seconds"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if isinstance(cfg["solver_iterations"], bool) or not isinstance(cfg["solver_iterations"], int) or cfg["solver_iterations"] < 1:
        raise ValueError("solver_iterations must be a positive integer")
    if not isinstance(cfg["sensor_noise_seed"], int) or cfg["sensor_noise_seed"] < 0:
        raise ValueError("sensor_noise_seed must be a nonnegative integer")
    expected_probes = [("hold_150", .15, 0.), ("gentle_150", .15, 3.), ("strong_300", .3, 8.)]
    if [(p["name"], p["duration_s"], p["excitation_N"]) for p in cfg["probes"]] != expected_probes:
        raise ValueError("Use all three declared Main2 probes")
    if cfg["future_seconds"] != .4 or cfg["future_compression_amplitude_m"] != .002 or cfg["future_excitation_N"] != 10.:
        raise ValueError("Future command is fixed at .4s, 2mm compression sine and 10N force sine")
    if cfg["probe_controller"] != {"kp_N_m": 120., "kd_N_s_m": 20., "max_force_N": 12.}:
        raise ValueError("Use the declared 120/20 PD gains and +/-12N actuator limit")
    if cfg["sensor_conditions"] != {
            "noisy": {"sigma_x_m": .00005, "sigma_v_m_s": .002, "sigma_normal_N": .15}}:
        raise ValueError("Use the unchanged ideal/noisy sensor conditions")
    for key in ("max_abs_position_m", "max_abs_velocity_m_s", "min_normal_N"):
        if not np.isfinite(cfg["guard"][key]) or cfg["guard"][key] <= 0:
            raise ValueError(f"Invalid guard limit: {key}")


SCALAR_FIELDS = ("normal", "friction", "u", "d", "d_dot", "delta", "compression_rate", "slope", "net_force",
    "solver_velocity", "solver_residual_N", "solver_iterations", "solver_bracket_width_m_s", "friction_capacity",
    "normal_updates", "stop_energy_before_J", "actual_displacement", "friction_commit", "friction_commit_minus_applied",
    "native_velocity_minus_solver", "native_position_residual_m", "native_momentum_residual_N", "stop_energy_after_J",
    "stop_work_input_J", "stop_dissipation_J", "stop_load_release_J", "stop_slip_dissipation_J", "stop_material_residual_J",
    "external_work_J", "kinetic_change_J", "numerical_dissipation_J", "tangential_balance_residual_J",
    "balance_residual_decomposition_J", "energy_decomposition_error_J", "normal_drive_work_zoh_J",
    "normal_drive_work_discrete_J", "normal_geometry_work_J", "normal_slope_work_J", "normal_geometry_discretization_J",
    "normal_material_work_discrete_J", "actuator_work_J", "normal_reaction_work_J", "friction_applied_work_J")
RECORD_SHAPES = {**{key: () for key in SCALAR_FIELDS}, "state": (2,), "next_state": (2,),
                 "stop_state_before": (3,), "stop_state_after": (3,), "phase": (), "guard": ()}


def run_batch(rig, normal_world, parameter_set, probe, sensor_name, cfg, factors, destination, *,
              config_path=None, test_lock=None, deadline=None):
    # A direct API call has the same prospective guard as the CLI. This must
    # precede PreparedPlant, which is the first operation that draws a history.
    if config_path is None or test_lock is None:
        raise ValueError("Main2 run_batch requires prospective config_path and test_lock")
    validate_config(cfg)
    verify_main2_lock(cfg, config_path, test_lock, {})
    if (normal_world not in cfg["normal_worlds"] or parameter_set not in cfg["friction_parameter_sets"]
            or probe not in cfg["probes"] or sensor_name not in cfg["sensor_conditions"]
            or not factors or factors[0].get("preparation_seed") not in cfg["preparation_seeds"]):
        raise ValueError("Batch arguments are outside the locked Main2 design")
    expected_factors = factorial_design(factors[0]["preparation_seed"])
    for factor in expected_factors:
        factor["episode_id"] = factor["episode_id"].replace("dev-", "test-", 1)
    if factors != expected_factors:
        raise ValueError("Main2 requires the complete locked 36-factor batch before a history draw")
    count, dt = len(factors), cfg["dt"]
    probe_steps, future_steps = round(probe["duration_s"]/dt), round(cfg["future_seconds"]/dt)
    total = probe_steps+future_steps
    seed = factors[0]["preparation_seed"]
    amplitude = np.asarray([row["amplitude"] for row in factors])
    geometry = np.asarray([row["geometry"] for row in factors], dtype=int)
    signs = np.asarray([row["excitation_sign"] for row in factors], dtype=float)
    plant = PreparedPlant(normal_world, parameter_set, factors, cfg)
    prep_audit, initial_truth = plant.preparation_audit(), plant.preparation_friction_audit
    rig.reset()
    sensor = cfg["sensor_conditions"][sensor_name]
    scale = np.array([sensor["sigma_x_m"], sensor["sigma_v_m_s"], sensor["sigma_normal_N"]])
    world_index, parameter_index = cfg["normal_worlds"].index(normal_world), cfg["friction_parameter_sets"].index(parameter_set)
    keys = [[cfg["sensor_noise_seed"], seed, world_index, parameter_index, row["profile_index"],
             row["geometry"], int(row["excitation_sign"] > 0)] for row in factors]
    max_total = round((max(p["duration_s"] for p in cfg["probes"])+cfg["future_seconds"])/dt)
    noise = np.stack([np.random.default_rng(np.random.SeedSequence(key)).normal(size=(max_total+1, 3)) for key in keys])
    held_feedback, future_bias = np.zeros(count), np.full(count, np.nan)
    records = {key: [] for key in RECORD_SHAPES}
    obs_records, error = [], None
    started = time.perf_counter()
    def append_record(record, phase, failed=False):
        limits = cfg["guard"]
        state = record.get("next_state", np.full((count, 2), np.nan))
        normal = record.get("normal", np.full(count, np.nan))
        guards = (~np.isfinite(state).all(axis=1)) | (~np.isfinite(normal)) \
                 | (np.abs(state[:, 0]) > limits["max_abs_position_m"]) \
                 | (np.abs(state[:, 1]) > limits["max_abs_velocity_m_s"]) | (normal < limits["min_normal_N"])
        if failed:
            guards[:] = True
        record.update(phase=np.full(count, phase), guard=guards)
        for key, shape in RECORD_SHAPES.items():
            records[key].append(np.asarray(record.get(key, np.full((count,)+shape, np.nan))).copy())
    try:
        for step in range(total+1):
            if deadline is not None and time.perf_counter() > deadline:
                raise TimeoutError("Declared generation wall-clock budget exceeded")
            current = rig.state()
            if current.shape != (count, 2) or not np.isfinite(current).all():
                raise FloatingPointError("Invalid current native state")
            is_future = step >= probe_steps
            if is_future:
                d, d_dot, _ = future_command((step-probe_steps)*dt, np.zeros(count), signs, cfg)
            else:
                d, d_dot = cfg["initial_compression_m"], 0.
            g, slope = cam(current[:, 0], amplitude, geometry)
            if step % cfg["observation_steps"] == 0 or step in (probe_steps, total):
                observed_n = plant.observe_normal(d+g, d_dot+slope*current[:, 1])
                observed = np.column_stack((current, observed_n))+noise[:, step]*scale
                if not np.isfinite(observed).all():
                    raise FloatingPointError("Nonfinite noisy observation")
                obs_records.append((step, observed.copy()))
                if not is_future:
                    observed_slope = cam(observed[:, 0], amplitude, geometry)[1]
                    control = cfg["probe_controller"]
                    held_feedback = observed_slope*observed[:, 2]-control["kp_N_m"]*observed[:, 0] \
                                    -control["kd_N_s_m"]*observed[:, 1]
            if step == probe_steps:
                future_bias = cam(observed[:, 0], amplitude, geometry)[1]*observed[:, 2]
            if step == total:
                break
            if is_future:
                _, _, u = future_command((step-probe_steps)*dt, future_bias, signs, cfg)
            else:
                u = probe_action(held_feedback, step*dt, probe, signs, cfg)
            next_d = (future_command((step+1-probe_steps)*dt, np.zeros(count), signs, cfg)[0]
                      if step+1 >= probe_steps else cfg["initial_compression_m"])
            try:
                record = plant.step(rig, current, u, d, d_dot, next_d, amplitude, geometry, cfg)
            except CouplingFailure as exc:
                append_record(exc.record, int(is_future), failed=True)
                raise
            append_record(record, int(is_future))
        if len(records["u"]) != total or not obs_records or obs_records[-1][0] != total:
            raise RuntimeError("Incomplete horizon or boundary observations")
    except BaseException as exc:
        error = exc
    arrays = {key: np.stack(values, axis=1) if values else np.empty((count, 0)+RECORD_SHAPES[key])
              for key, values in records.items()}
    arrays.update(amplitude=amplitude, geometry=geometry, excitation_sign=signs,
        observation_steps=np.asarray([t for t, _ in obs_records], dtype=int),
        observations=np.stack([value for _, value in obs_records], axis=1) if obs_records else np.empty((count, 0, 3)),
        probe_steps=np.array(probe_steps), future_bias=future_bias,
        preparation_seed=np.full(count, seed), preparation_index=np.asarray([r["profile_index"] for r in factors]),
        episode_id=np.asarray([r["episode_id"] for r in factors]))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    numerical = {}
    for key in ("solver_residual_N", "native_velocity_minus_solver", "native_position_residual_m",
                "native_momentum_residual_N", "friction_commit_minus_applied", "stop_material_residual_J",
                "tangential_balance_residual_J", "energy_decomposition_error_J", "normal_geometry_discretization_J"):
        value = arrays[key]
        numerical[f"max_abs_{key}"] = float(abs(value).max()) if value.size and np.isfinite(value).all() else None
    write_json(destination.with_suffix(".json"), dict(status="complete" if error is None else "failed",
        error=None if error is None else repr(error), normal_world=normal_world, friction_world=parameter_set,
        friction_family="three_positive_stops", probe=probe, sensor=sensor_name, sensor_parameters=sensor,
        count=count, dt=dt, mass_kg=cfg["mass_kg"], initial_compression_m=cfg["initial_compression_m"],
        future_seconds=cfg["future_seconds"], observation_period_s=cfg["observation_steps"]*dt,
        expected_steps=total, recorded_steps=len(records["u"]), seconds=time.perf_counter()-started,
        split="test", preparation_seed=seed, factor_rows=factors, geometry_names=GEOMETRY_NAMES,
        npz_sha256=sha(destination), guards=int(arrays["guard"].sum()), numerical=numerical,
        noise_stream_keys=keys, prepared_reference_audit=prep_audit, initial_truth_audit=initial_truth,
        native_backend="Isaac/CPU PhysX guided 2kg mass, collision disabled, explicit implicit-solved material force",
        force_semantics="normal pre-update/current input; friction is applied step-average PI force. friction_commit uses actual native dx and may differ. No Newton method is used; solver_iterations counts bisection iterations.",
        preparation_scope="Ideal imposed C2 displacement at fresh normal initial load. Normal state freshly initialized after friction preparation. Actual force-limited actuator reachability is not claimed.",
        observation_scope="Only causal observations, u,d,d_dot,known geometry/amplitude and public preparation distribution belong to regular inference. Factor IDs/seed, true commands and all material diagnostics are evaluator-only.",
        future_action_scope="Open future force from boundary N/x and known sine; no future feedback. Boundary N is observed after invoking future d_dot at t=0+ and cannot inform an earlier stop/continue decision.",
        energy_scope="Tangential stop plus mass balance with frozen current N/s. Normal drive and nonlinear geometry terms are recorded separately; no whole-system passivity claim."), exclusive=True)
    print(f"PI NATIVE p{seed} {normal_world}/{parameter_set} {probe['name']} {sensor_name}: "
          f"n={count} status={'complete' if error is None else 'failed'} guards={arrays['guard'].sum()}", flush=True)
    if error is not None:
        raise error


def verify_main2_lock(cfg, config_path, lock_path, hashes):
    """Check every locked source before any reserved profile is generated."""
    validate_config(cfg)
    if cfg != json.loads(Path(config_path).read_text()):
        raise ValueError("In-memory Main2 configuration differs from the locked config file")
    lock=json.loads(Path(lock_path).read_text())
    if (lock.get('role')!='paper_study_main2_test_authorization' or lock.get('status')!='locked'
            or lock.get('schema_version')!=1 or lock.get('split')!='test'
            or lock.get('config_sha256')!=sha(config_path)
            or lock.get('preparation_seeds')!=cfg['preparation_seeds']):
        raise ValueError('Main2 authorization does not match fixed configuration')
    frozen=lock.get('sources_sha256',{})
    required = {str(path): sha(path) for path in authorization_source_paths(config_path)}
    required.update(hashes)
    if not all(frozen.get(path)==digest for path,digest in required.items()):
        raise ValueError('Main2 lock omits or mismatches generation/evaluation/model sources')
    for path,digest in frozen.items():
        if sha(path)!=digest:raise ValueError(f'Main2 locked source changed: {path}')
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "main2_native.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test-lock", type=Path, required=True, help="Prospective Main2 source/config/model lock")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.device != "cpu":
        parser.error("Only the existing CPU PhysX rig is permitted")
    config_path, out = args.config.resolve(), args.out.resolve()
    cfg = json.loads(config_path.read_text())
    validate_config(cfg)
    if out == ROOT / "results" or not out.is_relative_to(ROOT / "results"):
        parser.error("Use a fresh subdirectory of paper_study/results")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("Existing results retained; choose a fresh output directory")
    hashes = {str(path): sha(path) for path in source_paths(config_path)}
    locked = verify_main2_lock(cfg, config_path, args.test_lock, hashes)
    args.smoke = False
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "generation_lock.json", dict(role="Main2 locally locked held-out native extension",
        config=cfg, smoke=False, sources_sha256=hashes,
        test_lock_sha256=sha(args.test_lock), test_sources_sha256=locked["sources_sha256"]), exclusive=True)
    started = time.perf_counter()
    process = dict(pid=os.getpid(), command=sys.argv, started_unix=time.time(), status="starting")
    write_json(out / "process.json", process, exclusive=True)
    app, batches, episodes = None, 0, 0
    previous_signals = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    def interrupted(signum, _frame):
        raise SystemExit(f"Native generation interrupted by signal {signum}")
    for sig in previous_signals:
        signal.signal(sig, interrupted)
    try:
        app = AppLauncher(args).app
        for sig in previous_signals:
            signal.signal(sig, interrupted)
        from decision_study.isaac_rig import IsaacRig
        from decision_study.common import MASS
        if MASS != cfg["mass_kg"]:
            raise ValueError("Installed native rig mass does not match Main2 configuration")
        rig = IsaacRig(cfg["episodes_per_batch"], "cpu", dt=cfg["dt"])
        write_json(out / "process.json", {**process, "status": "running"})
        seeds = cfg["preparation_seeds"][:1] if args.smoke else cfg["preparation_seeds"]
        normals = ["boucwen"] if args.smoke else cfg["normal_worlds"]
        parameters = ["balanced"] if args.smoke else cfg["friction_parameter_sets"]
        probes = [p for p in cfg["probes"] if p["name"] == "gentle_050"] if args.smoke else cfg["probes"]
        sensors = ["noisy"] if args.smoke else list(cfg["sensor_conditions"])
        for seed in seeds:
            factors = factorial_design(seed)
            for factor in factors:
                factor["episode_id"] = factor["episode_id"].replace("dev-", "test-", 1)
            for normal_world in normals:
                for parameter_set in parameters:
                    for probe in probes:
                        for sensor in sensors:
                            path = out / f"p{seed}/{normal_world}_{parameter_set}/{probe['name']}/{sensor}.npz"
                            run_batch(rig, normal_world, parameter_set, probe, sensor, cfg, factors, path,
                                      config_path=config_path, test_lock=args.test_lock,
                                      deadline=started+cfg["max_wall_seconds"])
                            batches += 1
                            episodes += len(factors)
        if hashes != {path: sha(path) for path in hashes}:
            raise RuntimeError("A locked generation source changed during execution")
        verify_main2_lock(cfg, config_path, args.test_lock, hashes)
        # Immediate Isaac framework shutdown may not return; persist first.
        write_json(out / "process.json", {**process, "status": "complete", "ended_unix": time.time(),
            "seconds": time.perf_counter()-started, "batches": batches, "episodes": episodes})
        print("COMPLETE", out, flush=True)
    except BaseException as exc:
        write_json(out / "process.json", {**process, "status": "failed", "error": repr(exc),
            "ended_unix": time.time(), "seconds": time.perf_counter()-started,
            "completed_batches": batches, "completed_episodes": episodes})
        raise
    finally:
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
        if app is not None:
            app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
