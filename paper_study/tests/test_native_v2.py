"""Independent CPU-only contracts; no native launch and no reserved test draw."""
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import preparation as original
import preparation_v2 as prep

spec = importlib.util.spec_from_file_location("native_v2_contract", ROOT / "scripts/generate_native_v2.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def config():
    return json.loads((ROOT / "configs/development_v2.json").read_text())


@pytest.mark.parametrize("seed", [31001, 31002, 31006])
def test_complete_crossing_has_unique_stable_ids_and_one_of_every_factor_cell(seed):
    cfg = config()
    rows = prep.factorial_design(seed, "dev", cfg["cam_amplitudes_m"])
    assert len(rows) == 48
    assert len({row["episode_id"] for row in rows}) == 48
    cells = Counter((row["preparation_recipe"], row["amplitude"], row["excitation_sign"]) for row in rows)
    expected = {(r, a, s) for r in original.RECIPES for a in cfg["cam_amplitudes_m"] for s in (-1, 1)}
    assert set(cells) == expected
    assert set(cells.values()) == {1}
    assert Counter(row["preparation_index"] for row in rows) == dict.fromkeys(range(6), 8)
    assert rows == prep.factorial_design(seed, "development", cfg["cam_amplitudes_m"])


def test_indices_expand_unchanged_base_commands_and_reject_invalid_inputs():
    indices = np.array([4, 1, 4, 0, 5])
    base = original.preparation_plan(6, 31002, "dev")
    plan = prep.indexed_plan(indices, 31002, "dev")
    for key in ("velocity", "normal_scale", "x"):
        np.testing.assert_array_equal(getattr(plan, key), getattr(base, key)[indices])
        assert not np.shares_memory(getattr(plan, key), getattr(base, key))
        assert not getattr(plan, key).flags.writeable
    assert [row["index"] for row in plan.rows] == indices.tolist()
    for invalid in ([6], [-1], [0.5], [True], [], [[0, 1]]):
        with pytest.raises(ValueError):
            prep.indexed_plan(invalid, 31001, "dev")


@pytest.mark.parametrize("world", ["maxwell", "boucwen"])
def test_reached_replicas_equal_original_and_continue_independently(world):
    indices = np.array([2, 2, 5, 0])
    base = original.referencepair(world, 6, 31001, "dev", normal_world="boucwen")
    expanded = prep.referencepair_indices(world, indices, 31001, "dev", normal_world="boucwen")
    expected = original.truth_snapshot(base)
    actual = prep.truth_snapshot(expanded)
    np.testing.assert_array_equal(actual["friction_history"], expected["friction_history"][indices])
    np.testing.assert_array_equal(actual["normal_history"], expected["normal_history"][indices])
    clone = expanded.clone()
    expanded.force(.02, 0., np.array([0., .2, 0., 0.]))
    after = prep.truth_snapshot(expanded)["friction_history"]
    assert after[0] != after[1]
    np.testing.assert_array_equal(prep.truth_snapshot(clone)["friction_history"], actual["friction_history"])


def test_short_sine_has_real_microstep_excitation_despite_zero_sensor_samples():
    cfg = config()
    probe = next(p for p in cfg["probes"] if p["name"] == "gentle_050")
    t = np.arange(20) * cfg["dt"]
    waveform = np.array([native.probe_feedforward(time, probe, np.array([-1., 1.])) for time in t])
    np.testing.assert_allclose(waveform[[0, 10]], 0., atol=1e-14)
    np.testing.assert_allclose(waveform[5], [-3., 3.], atol=1e-14)
    np.testing.assert_allclose(waveform[15], [3., -3.], atol=1e-14)
    assert np.count_nonzero(np.abs(waveform[:, 0]) > .1) == 18
    np.testing.assert_allclose(waveform.sum(axis=0), 0., atol=1e-13)
    held = np.array([2., -2.])
    command = native.probe_action(held, t[5], probe, np.array([-1., 1.]), cfg)
    np.testing.assert_allclose(command, [-1., 1.], atol=1e-14)
    np.testing.assert_array_equal(held, [2., -2.])
    np.testing.assert_array_equal(native.probe_action(np.array([20., -20.]), t[5], probe,
                                                     np.array([-1., 1.]), cfg), [12., -12.])
    np.testing.assert_array_equal(native.probe_feedforward(0., cfg["probes"][0], [-1, 1]), [0., 0.])


def test_config_and_test_authorization_fail_before_profile_draw(monkeypatch, tmp_path):
    cfg = config()
    native.validate_config(cfg)
    cfg["preparation_seeds"] = [31001, 31002]
    native.validate_config(cfg)
    def forbidden_draw(*args, **kwargs):
        raise AssertionError("Reserved profile must not be drawn")
    monkeypatch.setattr(original, "preparation_plan", forbidden_draw)
    with pytest.raises(ValueError, match="explicit test lock"):
        prep.factorial_design(51001, "test", cfg["cam_amplitudes_m"])
    with pytest.raises(ValueError, match="explicit test lock"):
        prep.referencepair_indices("maxwell", [0], 51001, "test")
    cfg.update(split="test", preparation_seeds=[51001])
    path = tmp_path / "test_config.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="--test-lock"):
        native.verify_test_lock(cfg, path, {}, None)
    lock = tmp_path / "lock.json"
    lock.write_text("{}")
    with pytest.raises(ValueError, match="does not match"):
        native.verify_test_lock(cfg, path, {}, lock)
    # Only validate authorization metadata, never generate a reserved profile.
    hashes = {str(path): native.sha(path)}
    payload = dict(schema_version=1, role=prep.TEST_LOCK_ROLE, status="locked", split="test",
                   preparation_seeds=[51001], config_sha256=native.sha(path), sources_sha256=hashes.copy())
    lock.write_text(json.dumps(payload))
    assert native.verify_test_lock(cfg, path, hashes, lock) == native.sha(lock)
    payload["sources_sha256"][str(path)] = "0" * 64
    lock.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="mismatches"):
        native.verify_test_lock(cfg, path, hashes, lock)


class MechanicalStub:
    """SI semi-implicit mechanical stub only; not a claimed native validation."""
    def __init__(self, n, dt, fail=False):
        self.n, self.dt, self.fail = n, dt, fail
        self.reset()

    def reset(self):
        self.xv = np.zeros((self.n, 2))

    def state(self):
        return self.xv.copy()

    def advance(self, force):
        self.xv[:, 1] += self.dt * np.asarray(force) / 2.
        self.xv[:, 0] += self.dt * self.xv[:, 1]
        if self.fail:
            self.xv[0, 1] = np.nan
        return self.state()


def test_recorded_commands_use_current_sine_and_sensor_held_feedback(tmp_path):
    cfg = config()
    rows = prep.factorial_design(31001, "dev", cfg["cam_amplitudes_m"])[:2]
    probe = next(p for p in cfg["probes"] if p["name"] == "gentle_050")
    destination = tmp_path / "success.npz"
    native.run_batch(MechanicalStub(2, cfg["dt"]), "boucwen", "boucwen", probe, "ideal", cfg, rows, destination)
    with np.load(destination) as data:
        assert data["u"].shape == (2, 120)
        np.testing.assert_array_equal(data["observation_steps"], np.arange(0, 121, 10))
        observed = data["observations"][:, 0]
        slope = cfg["cam_amplitudes_m"][0] / .04
        initial_feedback = slope * observed[:, 2]
        for t in range(10):
            expected = initial_feedback + np.array([-1., 1.]) * 3 * np.sin(2 * np.pi * t / 20)
            np.testing.assert_allclose(data["u"][:, t], expected, rtol=0, atol=1e-13)
        assert not np.allclose(data["u"][:, 1], data["u"][:, 0])
        assert data["d_dot"][0, 19] == 0.
        assert data["d_dot"][0, 20] == pytest.approx(.002 * 2 * np.pi / .25)
        assert not data["guard"].any()
        assert data["episode_id"].dtype.kind == "U"
    metadata = json.loads(destination.with_suffix(".json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["factor_rows"] == rows
    assert metadata["npz_sha256"] == native.sha(destination)
    assert metadata["noise_stream_keys"][0] != metadata["noise_stream_keys"][1]


def test_nonfinite_batch_is_preserved_with_failed_status(tmp_path):
    cfg = config()
    rows = prep.factorial_design(31001, "dev", cfg["cam_amplitudes_m"])[:1]
    probe = next(p for p in cfg["probes"] if p["name"] == "gentle_050")
    destination = tmp_path / "failed.npz"
    with pytest.raises(FloatingPointError, match="retained as failed"):
        native.run_batch(MechanicalStub(1, cfg["dt"], fail=True), "boucwen", "boucwen", probe,
                         "ideal", cfg, rows, destination)
    meta = json.loads(destination.with_suffix(".json").read_text())
    assert meta["status"] == "failed"
    assert meta["recorded_steps"] == 1 < meta["expected_steps"]
    with np.load(destination) as data:
        assert data["guard"][0, 0]
        assert np.isnan(data["next_state"][0, 0, 1])
    with pytest.raises(FileExistsError):
        native.run_batch(MechanicalStub(1, cfg["dt"]), "boucwen", "boucwen", probe,
                         "ideal", cfg, rows, destination)


@pytest.mark.parametrize("failure", [KeyboardInterrupt("interrupted"), FloatingPointError("nonfinite")])
def test_main_marks_interruptions_and_numeric_failures_as_failed_without_launching_isaac(monkeypatch, tmp_path, failure):
    cfg = config()
    root = tmp_path / "paper_study"
    root.mkdir()
    cfg_path = root / "development_v2.json"
    cfg_path.write_text(json.dumps(cfg))
    out = root / "results/smoke"
    class Launcher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--device", default="cpu")
        def __init__(self, _args):
            self.app = types.SimpleNamespace(close=lambda **kwargs: None)
    fake_isaac = types.ModuleType("isaaclab.app")
    fake_isaac.AppLauncher = Launcher
    fake_rig = types.ModuleType("decision_study.isaac_rig")
    fake_rig.IsaacRig = lambda n, device, dt: MechanicalStub(n, dt)
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_isaac)
    monkeypatch.setitem(sys.modules, "decision_study.isaac_rig", fake_rig)
    monkeypatch.setattr(native, "ROOT", root)
    monkeypatch.setattr(native, "source_paths", lambda path: [Path(path)])
    def failed_batch(*args, **kwargs):
        raise failure
    monkeypatch.setattr(native, "run_batch", failed_batch)
    monkeypatch.setattr(sys, "argv", ["generate_native_v2.py", "--config", str(cfg_path),
                                     "--out", str(out), "--smoke", "--device", "cpu"])
    with pytest.raises(type(failure)):
        native.main()
    process = json.loads((out / "process.json").read_text())
    assert process["status"] == "failed"
    assert process["completed_batches"] == 0
    assert repr(failure) == process["error"]
