"""Replay frozen force commands in native MuJoCo with independently coded laws.

This script never imports decision_study.reference or a learned predictor.
The material families/parameters are deliberately the same known synthetic
worlds, independently expressed here. This checks implementation/backend
sensitivity; it is not an independent empirical source of ground truth.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decision_study.common import CFG, DT, MASS, cam


class IndependentMaterial:
    """SI-unit force law, evaluated before its single history advance per call.

    Maxwell states are branch FORCES after relaxation at the previous position,
    unlike the original implementation's relaxed displacement states. Friction
    lag is a dimensionless Stribeck fraction instead of a coefficient state.
    """

    def __init__(self, world, n, initial_delta):
        if world not in ("maxwell", "boucwen"):
            raise ValueError(world)
        self.world = world
        self.n = n
        self.previous_delta = np.broadcast_to(np.maximum(initial_delta, 0), (n,)).copy()
        self.branch_force = np.zeros((n, 3))
        self.stribeck_fraction = np.ones(n)
        self.z = np.tanh(self.previous_delta / .009)
        self.age = np.full(n, .008 / .12)

    def force(self, delta, rate, v, normal_override=None, friction_enabled=True, dt=DT):
        d, dr, vel = [np.broadcast_to(np.asarray(x, dtype=float), (self.n,)) for x in (delta, rate, v)]
        active = d > 0
        dplus = np.maximum(d, 0)
        dr = np.where(active, dr, 0)
        speed = np.abs(vel)
        if self.world == "maxwell":
            spring = np.array([210., 125., 80.])
            branch_now = self.branch_force + (dplus - self.previous_delta)[:, None] * spring
            raw_load = 630 * dplus + branch_now.sum(axis=1) + 2.2 * dr
        else:
            raw_load = 680 * dplus + 4.5 * self.z + 1.8 * dr
        applied_load = raw_load if normal_override is None else np.broadcast_to(normal_override, (self.n,))
        normal = np.where(active, np.maximum(applied_load, 0), 0)
        loaded = active & (normal > 0)

        if self.world == "maxwell":
            steady_fraction = np.exp(-np.power(speed / .11, 1.4))
            coefficient = (.29 + .24 * (.65 * steady_fraction + .35 * self.stribeck_fraction))
            coefficient *= np.exp(-.055 * np.log((normal + 3) / 18))
            visc = .35
        else:
            coefficient = .40 + .018 * np.log(np.sqrt(vel * vel + .002**2) / .12)
            coefficient += .026 * np.log(self.age / (.008 / .12))
            coefficient = np.maximum(coefficient, 0)
            visc = .40
        friction = np.where(
            loaded & np.broadcast_to(friction_enabled, (self.n,)),
            -(normal * coefficient * np.tanh(vel / .04) + visc * vel), 0,
        )

        if self.world == "maxwell":
            self.branch_force = branch_now * np.exp(-dt / np.array([.025, .17, .80]))
            self.previous_delta = dplus.copy()
            decay = np.exp(-dt / (.05 + .001 * normal))
            goal = np.where(loaded, steady_fraction, 1.)
            self.stribeck_fraction = goal + (self.stribeck_fraction - goal) * decay
        else:
            speed_scaled = dr / .009
            count = max(1, int(np.ceil(np.max(np.abs(speed_scaled)) * dt / .05)))
            h = dt / count

            def dz(z):
                # Algebraically equivalent n=2 Bouc-Wen equation, independently
                # expressed without a generic exponent or parameter object.
                return speed_scaled * (1 - .35 * z * z) - .65 * np.abs(speed_scaled) * np.abs(z) * z

            z = self.z.copy()
            for _ in range(count):
                a = dz(z) * h
                b = dz(z + a / 2) * h
                c = dz(z + b / 2) * h
                e = dz(z + c) * h
                z += (a + 2 * b + 2 * c + e) / 6
            self.z = np.where(active, z, self.z * np.exp(-dt / .12))
            inverse_time = speed / .008
            exp_decay = np.exp(-inverse_time * dt)
            increment = np.divide(-np.expm1(-inverse_time * dt), inverse_time,
                                  out=np.full(self.n, dt), where=inverse_time > 0)
            new_age = self.age * exp_decay + increment
            recovered = .008 / .12 + (self.age - .008 / .12) * np.exp(-dt / .20)
            self.age = np.where(loaded, new_age, recovered)
        return normal, friction


def make_model(dt):
    return mujoco.MjModel.from_xml_string(f"""
    <mujoco model="independent_fixture_replay">
      <option timestep="{dt:.17g}" gravity="0 0 0" integrator="Euler"
              solver="Newton" iterations="100" tolerance="1e-12"/>
      <worldbody><body name="slider">
        <joint name="horizontal_guide" type="slide" axis="1 0 0"
               damping="0" frictionloss="0" armature="0"/>
        <geom type="box" size=".06 .05 .04" mass="{MASS:.17g}"
              contype="0" conaffinity="0"/>
      </body></worldbody>
    </mujoco>""")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def replay(world, seed, substeps, input_root, output_root):
    data_path = input_root / world / f"r{seed}" / "replay.npz"
    metadata_path = data_path.with_suffix(".json")
    if not data_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Frozen replay inputs required: {data_path} and {metadata_path}")
    started = time.perf_counter()
    with np.load(data_path) as saved:
        recorded = {k: saved[k] for k in ("state", "next_state", "normal", "friction", "u", "d", "rate")}
    metadata = json.loads(metadata_path.read_text())
    rows = metadata["rows"]
    n, steps = recorded["normal"].shape
    if len(rows) != n or recorded["state"].shape != (n, steps, 2):
        raise ValueError("Episode metadata and recorded state shapes do not match")
    if not np.isclose(metadata["dt"], DT, rtol=0, atol=1e-15):
        raise ValueError("Recorded observation timestep differs from configured DT")
    for key, values in recorded.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite recorded input in {key}")

    step_dt = DT / substeps
    model = make_model(step_dt)
    sim = mujoco.MjData(model)
    output = {"state": np.empty((n, steps, 2)), "next_state": np.empty((n, steps, 2)),
              "normal": np.empty((n, steps)), "friction": np.empty((n, steps))}
    for episode, row in enumerate(rows):
        mujoco.mj_resetData(model, sim)
        sim.qpos[0], sim.qvel[0] = recorded["state"][episode, 0]
        mujoco.mj_forward(model, sim)
        amplitude = float(row["amplitude"])
        initial_delta = recorded["d"][episode, 0] + cam(sim.qpos[0], amplitude)[0]
        material = IndependentMaterial(world, 1, initial_delta)
        for t in range(steps):
            output["state"][episode, t] = sim.qpos[0], sim.qvel[0]
            u = recorded["u"][episode, t]  # Frozen log; never recompute inverse dynamics.
            rate_command = recorded["rate"][episode, t]
            for sub in range(substeps):
                x, velocity = float(sim.qpos[0]), float(sim.qvel[0])
                gap, slope = cam(x, amplitude)
                # First-order hold of prescribed compression consistent with the
                # logged local derivative; actuator force is held constant.
                command = recorded["d"][episode, t] + sub * step_dt * rate_command
                delta = command + gap
                compression_rate = rate_command + slope * velocity
                normal, friction = material.force(delta, compression_rate, velocity, dt=step_dt)
                if sub == 0:
                    output["normal"][episode, t] = normal[0]
                    output["friction"][episode, t] = friction[0]
                sim.qfrc_applied[0] = u - slope * normal[0] + friction[0]
                mujoco.mj_step(model, sim)
            output["next_state"][episode, t] = sim.qpos[0], sim.qvel[0]
    if not all(np.isfinite(value).all() for value in output.values()):
        raise FloatingPointError("MuJoCo replay produced a nonfinite state or force")
    errors = {}
    for key in ("state", "next_state", "normal", "friction"):
        difference = output[key] - recorded[key]
        if difference.ndim == 3:
            errors[key] = {
                "position_rmse_m": float(np.sqrt(np.mean(difference[..., 0] ** 2))),
                "velocity_rmse_m_s": float(np.sqrt(np.mean(difference[..., 1] ** 2))),
                "position_max_abs_m": float(np.max(np.abs(difference[..., 0]))),
                "velocity_max_abs_m_s": float(np.max(np.abs(difference[..., 1]))),
                "episode_position_rmse_m": np.sqrt(np.mean(difference[..., 0] ** 2, axis=1)).tolist(),
            }
        else:
            errors[key] = {"rmse_N": float(np.sqrt(np.mean(difference ** 2))),
                           "max_abs_N": float(np.max(np.abs(difference)))}
    summary = {
        "world": world, "replicate": seed, "backend": "MuJoCo native slide joint",
        "mujoco_version": mujoco.__version__, "numpy_version": np.__version__,
        "dt_observation": DT, "dt_integrator": step_dt, "substeps": substeps,
        "episodes": n, "steps_per_episode": steps, "rows": rows,
        "seconds": time.perf_counter() - started, "comparison_to_recorded_physx": errors,
        "contact_state_disagreements": int(np.sum((output["normal"] > 0) != (recorded["normal"] > 0))),
        "maximum_friction_power_W": float(np.max(output["friction"] * output["state"][..., 1])),
        "native_collisions": False, "native_material_friction": False,
        "force_commands": "identical recorded u; piecewise constant at observation interval",
        "compression_interpolation": "d(t_j)+substep*dt_integrator*rate(t_j)",
        "source": {"input_npz": str(data_path.resolve()), "input_sha256": file_hash(data_path),
                   "metadata_sha256": file_hash(metadata_path), "script_sha256": file_hash(__file__)},
        "limitation": "Known synthetic equations independently implemented; not independent empirical truth.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    stem = output_root / f"{world}_r{seed}_h{substeps}"
    np.savez_compressed(stem.with_suffix(".npz"), **output)
    write_json(stem.with_suffix(".json"), summary)
    print(json.dumps({"result": str(stem), "seconds": summary["seconds"],
                      "next_state_error": errors["next_state"]}), flush=True)
    return summary


def self_test():
    """Synthetic pre-test checks without opening any experiment data."""
    diagnostics = {}
    for world in ("maxwell", "boucwen"):
        material = IndependentMaterial(world, 4, [.01, .018, .025, .031])
        max_power = -float("inf")
        for t in np.arange(0, 1, DT):
            d = .02 + .009 * np.sin(2 * np.pi * t + np.arange(4))
            rate = .009 * 2 * np.pi * np.cos(2 * np.pi * t + np.arange(4))
            velocity = .4 * np.sin(3 * np.pi * t + np.arange(4))
            normal, friction = material.force(d, rate, velocity)
            assert np.isfinite(normal).all() and np.isfinite(friction).all()
            assert np.all(normal >= 0) and np.all(friction * velocity <= 0)
            max_power = max(max_power, float(np.max(friction * velocity)))
        normal, friction = material.force(-.01, 0, .2, normal_override=20.)
        assert np.all(normal == 0) and np.all(friction == 0)
        diagnostics[world] = {"synthetic_steps": 400, "maximum_friction_power_W": max_power}
    model = make_model(DT)
    data = mujoco.MjData(model)
    data.qfrc_applied[0] = 2.
    for _ in range(10):
        mujoco.mj_step(model, data)
    np.testing.assert_allclose(data.qvel[0], 2. / MASS * 10 * DT, rtol=1e-12)
    diagnostics["native_mujoco"] = {"version": mujoco.__version__, "known_force_velocity_m_s": float(data.qvel[0])}
    print(json.dumps(diagnostics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--worlds", nargs="+", choices=["maxwell", "boucwen"], default=CFG["worlds"])
    parser.add_argument("--substeps", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--input", type=Path, default=Path("results/data"))
    parser.add_argument("--out", type=Path, default=Path("results/mujoco"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    for world in args.worlds:
        for seed in args.seeds:
            replay(world, seed, args.substeps, args.input, args.out)


if __name__ == "__main__":
    main()
