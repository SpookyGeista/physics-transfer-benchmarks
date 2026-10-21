"""Generate causal noisy preparation trials in native CPU PhysX, without fitting."""
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
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path[:0] = [str(ROOT / "src"), str(REPO / "study_v2/src")]
from preparation import referencepair, truth_preparation, truth_snapshot
from decision_study.common import cam


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def future_command(t, duration, bias, signs, cfg):
    phase = 2 * np.pi * t / duration
    d = cfg["initial_compression_m"] + cfg["future_compression_amplitude_m"] * np.sin(phase)
    rate = cfg["future_compression_amplitude_m"] * 2 * np.pi / duration * np.cos(phase)
    force = bias + signs * cfg["future_excitation_N"] * np.sin(phase)
    return d, rate, np.clip(force, -cfg["probe_controller"]["max_force_N"],
                           cfg["probe_controller"]["max_force_N"])


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


def run_batch(rig, nw, fw, probe, sensor_name, cfg, count, destination):
    dt = cfg["dt"]
    probe_steps = round(probe["duration_s"] / dt)
    future_steps = round(cfg["future_seconds"] / dt)
    total = probe_steps + future_steps
    amplitudes = np.resize(np.asarray(cfg["cam_amplitudes_m"]), count)
    signs = np.where(np.arange(count) % 2 == 0, 1., -1.)
    ref = referencepair(fw, count, cfg["preparation_seed"], cfg["split"], normal_world=nw)
    prep_audit = truth_preparation(ref)
    initial_truth = truth_snapshot(ref)
    rig.reset()
    sensor = cfg["sensor_conditions"][sensor_name]
    noise_seed = cfg["sensor_noise_seed"] + 100 * cfg["compositions"].index([nw, fw])
    max_total = round((max(p["duration_s"] for p in cfg["probes"]) + cfg["future_seconds"]) / dt)
    standard_noise = np.stack([np.random.default_rng(np.random.SeedSequence([noise_seed, i])).normal(
        size=(max_total + 1, 3)) for i in range(count)])
    noise_scale = np.array([sensor["sigma_x_m"], sensor["sigma_v_m_s"], sensor["sigma_normal_N"]])
    held_action = np.zeros(count)
    future_bias = None
    obs_records = []
    records = {k: [] for k in ["state", "next_state", "normal", "friction", "delta", "compression_rate",
                               "u", "net_force", "d", "d_dot", "guard", "phase"]}
    last_obs = None
    start = time.perf_counter()
    for step in range(total + 1):
        current = rig.state()
        g, slope = cam(current[:, 0], amplitudes)
        if step < probe_steps:
            d, rate = cfg["initial_compression_m"], 0.
        else:
            t = (step - probe_steps) * dt
            d, rate, _ = future_command(t, cfg["future_seconds"], np.zeros(count), signs, cfg)
        # Boundary normal at t=0+ includes the declared future d_dot. It is an
        # actual available observation, not a future normal-force trajectory.
        delta = d + g
        compression_rate = rate + slope * current[:, 1]
        if step % cfg["observation_steps"] == 0 or step == probe_steps or step == total:
            observed_normal = ref.observe(delta, compression_rate, current[:, 1], dt=dt)[0]
            noisy = np.column_stack((current, observed_normal)) + standard_noise[:, step] * noise_scale
            last_obs = noisy
            obs_records.append((step, noisy.copy()))
            if step < probe_steps:
                observed_slope = amplitudes * (1 - np.tanh(noisy[:, 0] / .04) ** 2) / .04
                excitation = probe["excitation_N"] * np.sin(2 * np.pi * step * dt / probe["duration_s"])
                control = cfg["probe_controller"]
                held_action = observed_slope * noisy[:, 2] - control["kp_N_m"] * noisy[:, 0] \
                              - control["kd_N_s_m"] * noisy[:, 1] + signs * excitation
                held_action = np.clip(held_action, -control["max_force_N"], control["max_force_N"])
        if step == probe_steps:
            observed_slope = amplitudes * (1 - np.tanh(last_obs[:, 0] / .04) ** 2) / .04
            future_bias = observed_slope * last_obs[:, 2]
        if step == total:
            break
        if step >= probe_steps:
            _, _, action = future_command((step - probe_steps) * dt, cfg["future_seconds"], future_bias, signs, cfg)
        else:
            action = held_action
        normal, friction = ref.force(delta, compression_rate, current[:, 1], dt=dt)
        net = action - slope * normal + friction
        next_state = rig.advance(net)
        limits = cfg["guard"]
        guard = (~np.isfinite(next_state).all(axis=1)) | (np.abs(next_state[:, 0]) > limits["max_abs_position_m"]) \
                | (np.abs(next_state[:, 1]) > limits["max_abs_velocity_m_s"]) | (normal < limits["min_normal_N"])
        for key, value in [("state", current), ("next_state", next_state), ("normal", normal), ("friction", friction),
                           ("delta", delta), ("compression_rate", compression_rate), ("u", action), ("net_force", net),
                           ("d", np.full(count, d)), ("d_dot", np.full(count, rate)), ("guard", guard),
                           ("phase", np.full(count, int(step >= probe_steps)))]:
            records[key].append(value.copy())
        if not np.isfinite(next_state).all():
            break
    arrays = {k: np.stack(v, axis=1) for k, v in records.items()}
    arrays.update(amplitude=amplitudes, observation_steps=np.array([t for t, _ in obs_records]),
                  observations=np.stack([v for _, v in obs_records], axis=1),
                  future_bias=future_bias, probe_steps=np.array(probe_steps))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    write_json(destination.with_suffix(".json"), dict(normal_world=nw, friction_world=fw, probe=probe,
        sensor=sensor_name, sensor_parameters=sensor, count=count, dt=dt, observation_period_s=cfg["observation_steps"] * dt,
        seconds=time.perf_counter() - start, native_backend="Isaac/CPU PhysX, guided mass, explicit material forces",
        npz_sha256=sha(destination), guards=int(arrays["guard"].sum()), noise_seed=noise_seed,
        prepared_reference_audit=numeric_audit(prep_audit), initial_truth_audit=numeric_audit(initial_truth),
        observation_scope="Only observations, commands, geometry and declared initial preparation convention belong to the observer. Truth force/state arrays and hidden preparation are evaluator-only.",
        future_action_scope="Precommitted from causal boundary observation and known sinusoidal command; no future feedback or inverse-reference action construction."))
    print(f"NATIVE {nw}/{fw} {probe['name']} {sensor_name}: {count} episodes, guards={arrays['guard'].sum()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.device != "cpu":
        parser.error("The validated native rig uses CPU PhysX")
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / "results"):
        parser.error("Output must remain under paper_study/results")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Existing data retained; choose a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs/development.json"
    cfg = json.loads(config_path.read_text())
    sources = [Path(__file__), config_path, ROOT / "src/preparation.py", REPO / "study_v2/src/decision_study/reference.py",
               REPO / "study_v2/src/decision_study/isaac_rig.py", REPO / "study_v2/src/decision_study/common.py"]
    hashes = {str(p): sha(p) for p in sources}
    write_json(out / "generation_lock.json", dict(role="exploratory generation snapshot before execution", sources_sha256=hashes, config=cfg))
    write_json(out / "process.json", dict(pid=os.getpid(), started_unix=time.time(), command=sys.argv, status="starting"))
    app = AppLauncher(args).app
    started = time.perf_counter()
    try:
        from decision_study.isaac_rig import IsaacRig
        count = 2 if args.smoke else cfg["episodes_per_batch"]
        rig = IsaacRig(count, "cpu", dt=cfg["dt"])
        compositions = [["boucwen", "boucwen"]] if args.smoke else cfg["compositions"]
        probes = [p for p in cfg["probes"] if p["name"] == "gentle_150"] if args.smoke else cfg["probes"]
        sensors = ["noisy"] if args.smoke else list(cfg["sensor_conditions"])
        for nw, fw in compositions:
            for probe in probes:
                for sensor in sensors:
                    if time.perf_counter() - started > cfg["max_wall_seconds"]:
                        raise TimeoutError("Declared generation wall-clock budget exceeded")
                    run_batch(rig, nw, fw, probe, sensor, cfg, count, out / f"{nw}_{fw}/{probe['name']}/{sensor}.npz")
        assert hashes == {p: sha(p) for p in hashes}, "Source changed during generation"
        write_json(out / "process.json", dict(pid=os.getpid(), ended_unix=time.time(), command=sys.argv,
            status="complete", seconds=time.perf_counter() - started,
            batches=len(compositions) * len(probes) * len(sensors), episodes=count * len(compositions) * len(probes) * len(sensors)))
        print("COMPLETE", out, flush=True)
    finally:
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
