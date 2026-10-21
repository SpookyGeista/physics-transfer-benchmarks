"""Reference/stimulus invariants; no reserved force logs or native test runs.

All stimulus exercises use a separate synthetic configuration and seed 101.
The generator schema test uses a minimal deterministic rigid-body stub.
"""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT.parent/"study_v2/src")]
from round1_reference import CompositeReference
from round1_stimuli import commands, conditions, load_config, preparation, velocity_profile
from decision_study.reference import Reference
from decision_study.common import cam


def synthetic_config():
    cfg = copy.deepcopy(load_config())
    cfg["replicates"] = [101]
    cfg["evaluation_source_base"] = 999000000
    return cfg


def generator_module():
    spec = importlib.util.spec_from_file_location("round1_generator_test", ROOT/"scripts/generate_round1_test.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_inputs(steps=120, count=3):
    t = np.arange(steps)*.0025
    phase = np.arange(count)[:, None]*.7
    d = .02 + .006*np.sin(2*np.pi*3*t + phase)
    rate = .006*2*np.pi*3*np.cos(2*np.pi*3*t + phase)
    v = .2*np.sin(2*np.pi*7*t + phase)
    # Include unloaded intervals, reopening and disabled friction in parity.
    d[0, 30:42] = -.001
    return d, rate, v


@pytest.mark.parametrize("world", ["maxwell", "boucwen"])
def test_same_family_is_exact_old_reference(world):
    delta, rate, velocity = synthetic_inputs()
    old = Reference(world, 3, .02)
    new = CompositeReference(world, world, 3, .02)
    for t in range(delta.shape[1]):
        enabled = t % 11 != 0
        expected = old.force(delta[:, t], rate[:, t], velocity[:, t], friction_enabled=enabled)
        actual = new.force(delta[:, t], rate[:, t], velocity[:, t], friction_enabled=enabled)
        for x, y in zip(actual, expected):
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(new._normal._normal_history, old._normal_history)
        np.testing.assert_array_equal(new._friction._friction_history, old._friction_history)
    assert new.updates == delta.shape[1]


@pytest.mark.parametrize("normal_world,friction_world", itertools.product(["maxwell", "boucwen"], repeat=2))
def test_composition_uses_selected_independent_laws(normal_world, friction_world):
    delta, rate, velocity = synthetic_inputs(50)
    normal_law = Reference(normal_world, 3, .02)
    friction_law = Reference(friction_world, 3, .02)
    composite = CompositeReference(normal_world, friction_world, 3, .02)
    for t in range(delta.shape[1]):
        n, _ = normal_law.force(delta[:, t], rate[:, t], np.zeros(3), friction_enabled=False)
        _, ft = friction_law.force(delta[:, t], rate[:, t], velocity[:, t], normal_override=n)
        actual_n, actual_ft = composite.force(delta[:, t], rate[:, t], velocity[:, t])
        np.testing.assert_array_equal(actual_n, n)
        np.testing.assert_array_equal(actual_ft, ft)
        assert np.all(actual_n >= 0)
        assert np.all(actual_ft*velocity[:, t] <= 0)
    # Competing heads never share storage for their private normal histories.
    assert not np.shares_memory(composite._normal._normal_history, composite._friction._normal_history)


@pytest.mark.parametrize("world", ["maxwell", "boucwen"])
def test_observation_and_candidate_clones_preserve_preupdate_state(world):
    ref = CompositeReference(world, world, 2, [.018, .022])
    ref.force([.021, .019], [.02, -.02], [.1, -.1])
    before = copy.deepcopy(ref)
    current = dict(delta=np.array([.023, .017]), rate=np.array([.03, -.03]), v=np.array([.16, -.11]))
    expected = before.clone().force(**current)
    for _ in range(3):
        for actual, target in zip(ref.observe(**current), expected):
            np.testing.assert_array_equal(actual, target)
    assert ref.updates == before.updates
    for name in ("_normal", "_friction"):
        for state in ("_normal_history", "_friction_history"):
            np.testing.assert_array_equal(getattr(getattr(ref, name), state), getattr(getattr(before, name), state))
    branch = ref.clone(repeats=3)
    expanded = {k: np.repeat(v, 3) for k, v in current.items()}
    actual = branch.force(**expanded)
    for x, y in zip(actual, expected):
        np.testing.assert_array_equal(x, np.repeat(y, 3))
    branch.force(np.full(6, .026), np.full(6, -.05), np.linspace(-.2, .2, 6))
    assert ref.updates == before.updates
    assert branch.updates == before.updates + 2
    for x, y in zip(ref.force(**current), expected):
        np.testing.assert_array_equal(x, y)
    with pytest.raises(ValueError):
        ref.clone(repeats=0)


def test_synthetic_stimuli_factor_cross_derivatives_and_bounds():
    cfg = synthetic_config()
    rows = conditions(101, cfg)
    assert rows == conditions(101, cfg)
    assert len({r["stimulus_seed"] for r in rows}) == 16
    expected = set(itertools.product((-1, 1), (-1, 1), (0., float(np.pi)), (-1, 1)))
    assert {(r["preparation_sign"], r["cam_sign"], r["phase"], r["target_sign"]) for r in rows} == expected
    t = np.linspace(0, 4, 1601)
    delta, rate = preparation(t, rows)
    assert delta.min() >= .012 and delta.max() <= .028
    np.testing.assert_array_equal(delta[:, -1], np.full(16, .02))
    np.testing.assert_array_equal(rate[:, [0, -1]], np.zeros((16, 2)))
    # Central finite differences away from the two piecewise boundaries.
    for time in [.2, .6, 2., 3.9]:
        eps = 1e-7
        plus, _ = preparation(time+eps, rows)
        minus, _ = preparation(time-eps, rows)
        _, derivative = preparation(time, rows)
        np.testing.assert_allclose((plus-minus)/(2*eps), derivative, atol=3e-11, rtol=1e-8)
    for row in rows:
        endpoint, endpoint_rate = preparation(row["preparation_duration"], [row])
        np.testing.assert_array_equal(endpoint, [.02])
        np.testing.assert_array_equal(endpoint_rate, [0.])
    d, analytic_rate, target = commands(rows, cfg)
    assert d.shape == analytic_rate.shape == target.shape == (16, 580)
    assert np.max(np.abs(d-.02)) <= .0035
    phase = np.array([r["phase"] for r in rows])[:, None]
    amplitude = np.array([r["wave_amplitude"] for r in rows])[:, None]
    omega = 2*np.pi*np.array([r["wave_frequency"] for r in rows])[:, None]
    time = np.arange(580)[None, :]*cfg["dt"]
    eps = 1e-7
    finite_difference = amplitude*(np.sin(omega*(time+eps)+phase)-np.sin(omega*(time-eps)+phase))/(2*eps)
    np.testing.assert_allclose(finite_difference, analytic_rate, atol=3e-11, rtol=1e-7)
    np.testing.assert_array_equal(target[:, 240], -target[:, 239])
    velocity = velocity_profile(rows, cfg)
    assert velocity.shape == (16, 481)
    np.testing.assert_array_equal(velocity[:, [0, 240, 480]], np.zeros((16, 3)))
    np.testing.assert_allclose(velocity[:, 1:240], -velocity[:, 241:480], rtol=1e-12, atol=1e-15)
    assert np.max(np.abs(velocity)) <= .16
    with pytest.raises(ValueError):
        conditions(1, cfg)


def test_declared_seed_spaces_do_not_overlap_without_generating_reserved_rows():
    cfg = load_config()
    spaces = []
    for key in ("train", "val"):
        # Calibration profile generator uses base + seed*1000 + profile index.
        spaces.append({cfg["calibration_source_bases"][key]+seed*1000+i
                       for seed in cfg["replicates"] for i in range(cfg["profiles_per_replicate"][key])})
    spaces.append({cfg["evaluation_source_base"]+seed*1000+i
                   for seed in cfg["replicates"] for i in range(cfg["profiles_per_replicate"]["evaluation"])})
    for a, b in itertools.combinations(spaces, 2):
        assert a.isdisjoint(b)
    assert all(x < 999000000 for values in spaces for x in values)


def test_inverse_action_tracks_velocity_and_geometry_uses_actual_state():
    generator = generator_module()
    rng = np.random.default_rng(999)
    state = rng.uniform([-.07, -.3], [.07, .3], size=(17, 2))
    amplitude = rng.uniform(-.006, .006, 17)
    gap, slope = cam(state[:, 0], amplitude)
    target_velocity = rng.uniform(-.2, .2, 17)
    n, ft = rng.uniform(2, 30, 17), rng.uniform(-8, 8, 17)
    dt, mass = .0025, 2.
    u = generator.inverse_action(target_velocity, state, n, ft, slope, dt, mass)
    net = u-slope*n+ft
    np.testing.assert_allclose(state[:, 1]+dt*net/mass, target_velocity, atol=1e-16)
    eps = 1e-7
    derivative = (cam(state[:, 0]+eps, amplitude)[0]-cam(state[:, 0]-eps, amplitude)[0])/(2*eps)
    np.testing.assert_allclose(slope, derivative, atol=1e-10, rtol=1e-8)
    external_rate = np.linspace(-.02, .02, 17)
    composed = external_rate + slope*state[:, 1]
    d1 = .02+eps*external_rate + cam(state[:, 0]+eps*state[:, 1], amplitude)[0]
    d0 = .02-eps*external_rate + cam(state[:, 0]-eps*state[:, 1], amplitude)[0]
    np.testing.assert_allclose(composed, (d1-d0)/(2*eps), atol=1e-10, rtol=1e-7)
    assert not np.allclose(gap, 0)


def test_main_requires_lock_before_config_or_isaac_startup(monkeypatch):
    generator = generator_module()
    def blocked():
        raise FileNotFoundError("synthetic absent model lock")
    def forbidden():
        raise AssertionError("configuration must not be used before lock")
    monkeypatch.setattr(generator, "verify_locks", blocked)
    monkeypatch.setattr(generator, "load_config", forbidden)
    with pytest.raises(FileNotFoundError, match="synthetic absent"):
        generator.main()


def test_short_synthetic_stub_generator_schema_and_provenance(tmp_path, monkeypatch):
    generator = generator_module()
    cfg = synthetic_config()
    cfg["motion_seconds"], cfg["warmup_seconds"] = .025, .02
    cfg["compositions"] = [["maxwell", "boucwen"]]
    locked = dict(code_lock_sha256="synthetic_code", model_lock_sha256="synthetic_model")
    checks = []
    def verify():
        checks.append(True)
        return locked.copy()
    monkeypatch.setattr(generator, "load_config", lambda: cfg)
    monkeypatch.setattr(generator, "verify_locks", verify)
    class StubRig:
        def __init__(self, n, device, dt):
            assert device == "cpu"
            self.values = np.zeros((n, 2)); self.dt = dt
        def reset(self):
            self.values[:] = 0
        def state(self):
            return self.values.copy()
        def advance(self, force):
            self.values[:, 1] += self.dt*np.asarray(force)/2
            self.values[:, 0] += self.dt*self.values[:, 1]
            return self.state()
    closed = []
    app_module = types.ModuleType("isaaclab.app")
    app_module.AppLauncher = lambda args: types.SimpleNamespace(app=types.SimpleNamespace(close=lambda **kwargs: closed.append(kwargs)))
    rig_module = types.ModuleType("decision_study.isaac_rig")
    rig_module.IsaacRig = StubRig
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    monkeypatch.setitem(sys.modules, "decision_study.isaac_rig", rig_module)
    args = types.SimpleNamespace(compositions=["maxwell_boucwen"], seeds=[101], out=tmp_path)
    generator.native_run(args, locked)
    path = tmp_path/"maxwell_boucwen/r101.npz"
    data = dict(np.load(path)); meta = json.loads(path.with_suffix(".json").read_text())
    for key in ("state", "next_state"):
        assert data[key].shape == (16, 10, 2)
    for key in ("u", "d", "rate", "delta", "compression_rate", "normal", "friction"):
        assert data[key].shape == (16, 10)
    for key in ("warm_delta", "warm_rate", "warm_velocity", "warm_normal", "warm_friction"):
        assert data[key].shape == (16, 8)
    assert data["amplitude"].shape == (16,)
    assert meta["reference_updates"] == 18
    assert meta["model_lock_sha256"] == locked["model_lock_sha256"]
    assert meta["locks"] == locked
    assert meta["npz_sha256"] == generator.digest(path)
    assert meta["numerical"]["native_gate_pass"]
    assert len(checks) >= 3 and closed
    assert "hidden_reference_states_saved" in meta and not meta["hidden_reference_states_saved"]
    for component in ("delta", "normal", "velocity"):
        assert component+"_min" in meta["numerical"] and component+"_max" in meta["numerical"]
    # Verified resume leaves saved bytes unchanged; mismatches remain visible.
    before = path.read_bytes()
    generator.native_run(args, locked)
    assert path.read_bytes() == before
    meta["npz_sha256"] = "wrong"
    path.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="provenance differs"):
        generator.native_run(args, locked)
