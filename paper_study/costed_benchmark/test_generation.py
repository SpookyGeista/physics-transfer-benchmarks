"""Bounded CPU-only contracts; these tests never start Isaac or draw test data."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from geometry import cam
from profiles import profile_commands, factorial_design
from plant import PreparedPlant
from generate import validate_config, run_batch


def config():
    return json.loads((HERE / "development.json").read_text())


def test_factor_balance_and_profile_identity_without_material_knowledge():
    rows = factorial_design(33001)
    assert len(rows) == len({row["episode_id"] for row in rows}) == 36
    counts = Counter((row["profile_index"], row["geometry"], row["excitation_sign"]) for row in rows)
    assert set(counts) == {(p, g, s) for p in range(6) for g in range(3) for s in (-1, 1)}
    assert set(counts.values()) == {1}
    base = profile_commands(33001, list(range(6)))
    subset = profile_commands(33001, [3, 1, 3])
    np.testing.assert_array_equal(subset["x"], base["x"][[3, 1, 3]])
    np.testing.assert_array_equal(base["x"][:, [0, -1]], 0.)
    np.testing.assert_array_equal(base["continuous_velocity"][:, [0, -1]], 0.)
    assert np.max(abs(base["x"])) <= .025
    assert len({tuple(row["waypoints_m"]) for row in base["rows"]}) == 6
    for i, row in enumerate(base["rows"]):
        sample = np.rint(np.array(row["knot_times_s"])/.0025).astype(int)
        np.testing.assert_allclose(base["x"][i, sample], row["waypoints_m"], rtol=0, atol=1e-16)
        np.testing.assert_allclose(base["continuous_velocity"][i, sample], 0., rtol=0, atol=1e-14)
    refined = profile_commands(33001, list(range(6)), dt=.000625)
    np.testing.assert_allclose(base["x"], refined["x"][:, ::4], rtol=0, atol=1e-15)


def test_geometry_analytic_derivatives_and_distinct_functions():
    x = np.broadcast_to(np.linspace(-.03, .03, 41), (3, 41))
    geometry = np.broadcast_to(np.arange(3)[:, None], x.shape)
    value, derivative = cam(x, .003, geometry)
    numeric = (cam(x+1e-8, .003, geometry)[0]-cam(x-1e-8, .003, geometry)[0])/(2e-8)
    np.testing.assert_allclose(derivative, numeric, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(cam(np.zeros(3), .003, np.arange(3))[0], 0.)
    np.testing.assert_allclose(cam(np.zeros(3), .003, np.arange(3))[1], [.25, .25, 0.])
    assert not np.allclose(value[0], value[1])
    np.testing.assert_allclose(value[2], value[2, ::-1], rtol=0, atol=1e-17)


class MechanicalStub:
    def __init__(self, n, dt=.0025, fail=False):
        self.n, self.dt, self.fail = n, dt, fail
        self.reset()
    def reset(self):
        self.xv = np.zeros((self.n, 2))
    def state(self):
        return self.xv.copy()
    def advance(self, force):
        self.xv[:, 1] += self.dt*np.asarray(force)/2.
        self.xv[:, 0] += self.dt*self.xv[:, 1]
        if self.fail:
            self.xv[0, 0] = np.nan
        return self.state()


@pytest.mark.parametrize("normal_world", ["maxwell", "boucwen"])
@pytest.mark.parametrize("parameter_set", ["balanced", "wide_scales", "compliant", "stiff_high_capacity"])
def test_prepared_plant_one_normal_update_real_dx_commit_and_energy(normal_world, parameter_set):
    cfg = config()
    rows = factorial_design(33001)
    plant = PreparedPlant(normal_world, parameter_set, rows, cfg)
    stop = plant.friction.audit()
    for profile in range(6):
        np.testing.assert_array_equal(stop["deflection"][profile*6:profile*6+6],
                                      np.broadcast_to(stop["deflection"][profile*6], (6, 3)))
    normal_before = plant.normal._normal_history.copy()
    nobs = plant.observe_normal(np.full(36, .02), np.zeros(36))
    np.testing.assert_array_equal(plant.normal._normal_history, normal_before)
    assert plant.normal_updates == 0
    amplitude, geometry = np.full(36, .003), np.asarray([r["geometry"] for r in rows])
    rig = MechanicalStub(36)
    record = plant.step(rig, rig.state(), np.ones(36), .02, 0., .02, amplitude, geometry, cfg)
    assert plant.normal_updates == 1
    np.testing.assert_array_equal(record["normal"], nobs)
    assert plant.friction.updates == 1601
    np.testing.assert_allclose(record["native_velocity_minus_solver"], 0., rtol=0, atol=4e-14)
    np.testing.assert_allclose(record["tangential_balance_residual_J"], 0., rtol=0, atol=2e-15)
    np.testing.assert_allclose(record["energy_decomposition_error_J"], 0., rtol=0, atol=1e-17)
    np.testing.assert_array_equal(record["actual_displacement"], rig.state()[:, 0])
    np.testing.assert_array_equal(plant.friction.audit()["last_step"]["displacement"], record["actual_displacement"])


def test_generator_schema_causal_boundary_short_probe_and_failure_status(tmp_path):
    cfg = config()
    rows = factorial_design(33001)
    probe = next(p for p in cfg["probes"] if p["name"] == "gentle_050")
    destination = tmp_path / "batch.npz"
    run_batch(MechanicalStub(36), "boucwen", "balanced", probe, "noisy", cfg, rows, destination)
    meta = json.loads(destination.with_suffix(".json").read_text())
    assert meta["status"] == "complete"
    assert meta["recorded_steps"] == 180
    with np.load(destination) as data:
        assert data["observations"].shape == (36, 19, 3)
        assert data["state"].shape == (36, 180, 2)
        assert data["stop_state_after"].shape == (36, 180, 3)
        assert data["geometry"].dtype.kind in "iu"
        np.testing.assert_array_equal(data["normal_updates"], np.broadcast_to(np.arange(1, 181), (36, 180)))
        observed = data["observations"][:, 0]
        slope = cam(observed[:, 0], data["amplitude"], data["geometry"])[1]
        feedback = slope*observed[:, 2]-120*observed[:, 0]-20*observed[:, 1]
        expected = np.clip(feedback+data["excitation_sign"]*3*np.sin(2*np.pi/20), -12., 12.)
        np.testing.assert_allclose(data["u"][:, 1], expected, rtol=0, atol=1e-13)
        assert not np.allclose(data["u"][:, 1], data["u"][:, 0])
        np.testing.assert_array_equal(data["d_dot"][:, 19], 0.)
        np.testing.assert_allclose(data["d_dot"][:, 20], .002*2*np.pi/.4, rtol=0, atol=1e-15)
        assert not data["guard"].any()
    bad = tmp_path / "failed.npz"
    with pytest.raises(RuntimeError, match="Nonfinite native state"):
        run_batch(MechanicalStub(36, fail=True), "boucwen", "balanced", probe, "noisy", cfg, rows, bad)
    assert json.loads(bad.with_suffix(".json").read_text())["status"] == "failed"
    with np.load(bad) as data:
        assert data["guard"].all()
        assert np.isnan(data["next_state"][0, 0, 0])


def test_only_declared_development_generation_allowed():
    cfg = config()
    validate_config(cfg)
    for split in ("val", "test"):
        cfg["split"] = split
        with pytest.raises(ValueError, match="cannot generate"):
            validate_config(cfg)
    cfg["split"] = "dev"
    cfg["preparation_seeds"] = [43001]
    with pytest.raises(ValueError, match="Development preparation_seeds"):
        validate_config(cfg)
