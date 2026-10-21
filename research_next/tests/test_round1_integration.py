"""Round-1 integration using synthetic logs and existing development weights.

No reserved stimulus function or reference law is executed by these tests.
All filesystem layouts are temporary; r1..r8 names below carry fabricated zeros
solely to validate the full CSV key space and never represent experiment data.
"""
import copy
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
import round1_models
import round1_lock
from online import SensorPersistent, clone_state
from run_model_development import plan
from analyze_round1 import validate_metrics


def script(name):
    spec = importlib.util.spec_from_file_location("integration_"+name, ROOT/f"scripts/{name}.py")
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def config():
    return json.loads((ROOT/"configs/round1.json").read_text())


@pytest.fixture
def development_layout(tmp_path, monkeypatch):
    """Place old development weights under synthetic Round-1 loader paths."""
    root = tmp_path/"models"
    for world, noise in itertools.product(("maxwell", "boucwen"), ("clean", "noisy")):
        sources = {
            f"recurrent/r101/{noise}/{world}/model.pt": ROOT/f"results/recurrent_development_6000/{world}/model.pt",
            f"state_id/{world}/r101/{noise}/blind_selection/model.json": ROOT/f"results/state_id_development/{world}/r0/{noise}/blind_selection/model.json",
            **{f"features/{world}/r101/{noise}/{method}/model.json": ROOT/f"results/feature_development/{world}/r0/{noise}/{method}/model.json"
               for method in ("old_flex", "path_ridge")},
        }
        for relative, source in sources.items():
            assert source.is_file(), f"Existing development fixture missing: {source}"
            target = root/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
    monkeypatch.setattr(round1_models, "ROOT", root)
    return root


class Trace:
    def __init__(self, model):
        self.model = model
        self.calls = []
    def step(self, delta, rate, v, state, normal_override=None, dt=.0025):
        self.calls.append(None if normal_override is None else np.array(normal_override, copy=True))
        return self.model.step(delta, rate, v, state, normal_override=normal_override, dt=dt)


@pytest.mark.parametrize("method", config()["methods"])
@pytest.mark.parametrize("normal_world,friction_world", config()["compositions"])
def test_all_model_loaders_current_sensor_and_causal_mpc(development_layout, method, normal_world, friction_world):
    import torch
    torch.set_num_threads(2)
    model = round1_models.load(method, normal_world, friction_world, 101, "clean")
    d = np.array([.019, .021]); v = np.array([.05, -.07]); r = np.array([.01, -.02])
    n = np.array([31., 29.])
    h = model.initial(d, v, n)
    before = clone_state(h)
    _, _, observed = model.step(d, r, v, h, normal_override=n)
    for key in h:
        np.testing.assert_array_equal(h[key], before[key])
        assert not np.shares_memory(h[key], observed[key])
    current_n, _, future = model.step(d, r, v, h, normal_override=n)
    np.testing.assert_array_equal(current_n, n)
    predicted_n = model.step(d, r, v, future)[0]
    if method == "sensor_state_id":
        np.testing.assert_array_equal(predicted_n, n)
    else:
        assert not np.allclose(predicted_n, n), "Future must use model N, not persist a sensor silently"
    traced = Trace(model)
    state = np.column_stack([np.zeros(2), v])
    commands = np.repeat(d[:, None], 3, axis=1)
    plan(traced, state, h, n, commands, commands*0, commands*.5, np.array([.003, -.004]))
    assert len(traced.calls) == 3
    np.testing.assert_array_equal(traced.calls[0], np.repeat(n, 41))
    assert traced.calls[1:] == [None, None]
    for key in h:
        np.testing.assert_array_equal(h[key], before[key])


@pytest.mark.parametrize("method", ["nominal", "old_flex", "path_ridge", "recurrent", "blind_state_id", "state_reset"])
def test_normal_prediction_has_no_force_history_label_input(development_layout, method):
    model = round1_models.load(method, "boucwen", "maxwell", 101, "noisy")
    d, v = np.array([.02, .021]), np.array([.06, -.09])
    left = model.initial(d, v, np.array([10., 11.]))
    right = model.initial(d, v, np.array([35., 32.]))
    for t in range(4):
        rate = np.array([.01, -.01])
        _, _, left = model.step(d, rate, v, left, normal_override=np.array([10., 11.]))
        _, _, right = model.step(d, rate, v, right, normal_override=np.array([35., 32.]))
    nl = model.step(d, rate, v, left)[0]
    nr = model.step(d, rate, v, right)[0]
    np.testing.assert_array_equal(nl, nr)


class CountingModel:
    def __init__(self):
        self.starts = []
        self.calls = []
    def initial(self, d, v, n):
        self.starts.append((np.array(d), np.array(v), np.array(n)))
        return {"count": np.zeros(len(d)), "load": np.array(n)}
    def step(self, d, rate, v, state, normal_override=None, dt=.0025):
        self.calls.append((state["count"].copy(), None if normal_override is None else np.array(normal_override)))
        n = np.full(len(d), 12.) if normal_override is None else np.array(normal_override)
        return n, np.zeros(len(d)), {"count": state["count"]+1, "load": n.copy()}


def toy_data(n=2, steps=3, warm=4):
    scalar = np.zeros((n, steps))
    state = np.zeros((n, steps, 2))
    return dict(state=state.copy(), next_state=state.copy(), u=scalar.copy(), d=scalar+.02,
                rate=scalar.copy(), delta=scalar+.02, compression_rate=scalar.copy(),
                normal=scalar+15, friction=scalar.copy(), amplitude=np.full(n, .003),
                warm_delta=np.full((n, warm), .02), warm_rate=np.zeros((n, warm)),
                warm_velocity=np.zeros((n, warm)), warm_normal=10+np.tile(np.arange(warm), (n, 1)),
                warm_friction=np.zeros((n, warm)), target=scalar+.03,
                target_velocity=np.zeros((n, steps+1)))


def write_toy_test(root, module, data, seed=101, composition="maxwell_boucwen", lock="synthetic-model"):
    path = root/f"test/{composition}/r{seed}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    n = len(data["state"])
    nw, fw = composition.split("_")
    meta = dict(rows=[dict(episode=i, profile=i, id=f"synthetic-only-{seed}-{i}") for i in range(n)],
                normal_world=nw, friction_world=fw, replicate=seed, dt=.0025, mass=2.,
                steps=data["state"].shape[1], warm_steps=data["warm_delta"].shape[1],
                model_lock_sha256=lock, code_lock_sha256="synthetic-code",
                locks=dict(model_lock_sha256=lock, code_lock_sha256="synthetic-code"),
                source_sha256={}, npz_sha256=module.digest(path))
    path.with_suffix(".json").write_text(json.dumps(meta))
    return path, meta


def test_observed_prefix_once_and_state_reset_only_at_motion_start(tmp_path, monkeypatch):
    evaluate = script("evaluate_round1")
    monkeypatch.setattr(evaluate, "OUT", tmp_path)
    monkeypatch.setattr(evaluate, "verify_model_lock", lambda: "synthetic-model")
    cfg = config(); cfg.update(compositions=[["maxwell", "boucwen"]], replicates=[101], noise=["clean"])
    data = toy_data()
    write_toy_test(tmp_path, evaluate, data)
    models = {}
    def load(method, *unused):
        models[method] = CountingModel()
        return models[method]
    monkeypatch.setattr(evaluate, "load", load)
    args = types.SimpleNamespace(seeds=[101], methods=["blind_state_id", "state_reset"])
    evaluate.run_predictions(args, cfg, "synthetic-model")
    for method, model in models.items():
        assert len(model.calls) == 7
        assert len(model.starts) == (2 if method == "state_reset" else 1)
        np.testing.assert_array_equal(model.starts[0][2], data["warm_normal"][:, 0])
        for t in range(4):
            np.testing.assert_array_equal(model.calls[t][0], [t, t])
            np.testing.assert_array_equal(model.calls[t][1], data["warm_normal"][:, t])
        expected_start = 0 if method == "state_reset" else 4
        np.testing.assert_array_equal(model.calls[4][0], [expected_start, expected_start])
        np.testing.assert_array_equal(model.calls[4][1], data["normal"][:, 0])
        assert model.calls[5][1] is None and model.calls[6][1] is None
        assert evaluate.done(tmp_path/f"predictions/maxwell_boucwen/r101/clean/{method}.npz", "synthetic-model", evaluate.digest(tmp_path/"test/maxwell_boucwen/r101.npz"))


def test_corrupted_test_npz_rejected_before_model_load(tmp_path, monkeypatch):
    evaluate = script("evaluate_round1")
    monkeypatch.setattr(evaluate, "OUT", tmp_path)
    monkeypatch.setattr(evaluate, "verify_model_lock", lambda: "synthetic-model")
    path, meta = write_toy_test(tmp_path, evaluate, toy_data())
    meta["npz_sha256"] = "wrong"
    path.with_suffix(".json").write_text(json.dumps(meta))
    def forbidden(*args):
        raise AssertionError("Model loaded before test-data integrity validation")
    monkeypatch.setattr(evaluate, "load", forbidden)
    cfg = config(); cfg.update(compositions=[["maxwell", "boucwen"]], replicates=[101], noise=["clean"])
    with pytest.raises(RuntimeError):
        evaluate.run_predictions(types.SimpleNamespace(seeds=[101], methods=["nominal"]), cfg, "synthetic-model")


def test_full_synthetic_metric_schema_has_7168_distinct_rows(tmp_path, monkeypatch):
    evaluate = script("evaluate_round1")
    monkeypatch.setattr(evaluate, "OUT", tmp_path)
    monkeypatch.setattr(evaluate, "verify_model_lock", lambda: "synthetic-model")
    cfg = config(); n, steps, warm = 16, 480, 1600
    data = toy_data(n, steps, warm)
    prediction = dict(state=data["next_state"].copy(), normal=data["normal"].copy(),
                      friction=data["friction"].copy(), guard=np.zeros((n, steps), bool))
    controller = dict(next_state=data["next_state"].copy(), target=data["target"].copy(),
                      stage_cost=np.full((n, steps), .03**2), u=np.zeros((n, steps)),
                      candidate_guard=np.zeros((n, 24, 41), bool), plan_seconds=np.full(24, .016))
    templates = {}
    for label, arrays in [("prediction", prediction), ("controller", controller)]:
        path = tmp_path/f"synthetic_{label}_template.npz"
        np.savez_compressed(path, **arrays); templates[label] = path
    for nw, fw in cfg["compositions"]:
        for seed in cfg["replicates"]:
            testpath, meta = write_toy_test(tmp_path, evaluate, data, seed, f"{nw}_{fw}")
            testsha = evaluate.digest(testpath)
            for noise, method, stage in itertools.product(cfg["noise"], cfg["methods"], templates):
                dest = tmp_path/f"{stage}s/{nw}_{fw}/r{seed}/{noise}/{method}.npz"
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.link(templates[stage], dest)
                saved = dict(rows=meta["rows"], normal_world=nw, friction_world=fw, seed=seed,
                             noise=noise, method=method, model_lock_sha256="synthetic-model",
                             test_sha256=testsha, npz_sha256=evaluate.digest(dest))
                dest.with_suffix(".json").write_text(json.dumps(saved))
    evaluate.metrics(cfg, "synthetic-model")
    for stage in templates:
        frame = pd.read_csv(tmp_path/f"{stage}_metrics.csv")
        verified = validate_metrics(frame, stage, cfg)
        assert len(verified) == 7168
    assert pd.read_csv(tmp_path/"prediction_metrics.csv")["position_rmse"].max() == 0
    np.testing.assert_allclose(pd.read_csv(tmp_path/"controller_metrics.csv")["position_squared_integral"], 1.2*.03**2)


@pytest.mark.parametrize("stage", ["calibration", "neural", "classical", "lock-models"])
def test_preparation_requires_code_lock_before_stage_action(tmp_path, monkeypatch, stage):
    prepare = script("prepare_round1")
    (tmp_path/"configs").mkdir()
    (tmp_path/"configs/round1.json").write_text(json.dumps(config()))
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare, "OUT", tmp_path/"results")
    def no_lock():
        raise FileNotFoundError("Synthetic absent code lock")
    def forbidden(*args, **kwargs):
        raise AssertionError("Stage command reached without a code lock")
    monkeypatch.setattr(prepare, "verify_code_lock", no_lock)
    monkeypatch.setattr(prepare, "command", forbidden)
    monkeypatch.setattr(sys, "argv", ["prepare_round1.py", stage])
    with pytest.raises(FileNotFoundError, match="absent code lock"):
        prepare.main()


def test_temporary_code_and_model_lock_order_and_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(round1_lock, "ROOT", tmp_path)
    monkeypatch.setattr(round1_lock, "OUT", tmp_path/"locks")
    source = tmp_path/"source.py"; source.write_text("synthetic fixed algorithm")
    code = round1_lock.save_lock(tmp_path/"locks/code_lock.json", [source])
    with pytest.raises(FileNotFoundError):
        round1_lock.verify_model_lock()
    weights = tmp_path/"weights.json"; weights.write_text('{"synthetic": true}')
    model = round1_lock.save_lock(tmp_path/"locks/model_lock.json", [weights], code_lock_sha256=code)
    assert round1_lock.verify_model_lock() == model
    with pytest.raises(FileNotFoundError):
        round1_lock.verify_test_lock()
    fake_test = tmp_path/"fabricated_test.npz"; fake_test.write_bytes(b"fabricated zero data")
    testlock = round1_lock.save_lock(tmp_path/"locks/test_lock.json", [fake_test], model_lock_sha256=model)
    assert round1_lock.verify_test_lock() == testlock
    fake_test.write_bytes(b"changed test")
    with pytest.raises(RuntimeError):
        round1_lock.verify_test_lock()
    weights.write_text("changed")
    with pytest.raises(RuntimeError):
        round1_lock.verify_model_lock()


def test_partial_metrics_explicit_status_and_no_complete_csv(tmp_path, monkeypatch):
    evaluate = script("evaluate_round1")
    monkeypatch.setattr(evaluate, "OUT", tmp_path)
    with pytest.raises(ValueError):
        evaluate.metrics(config(), "synthetic-model")
    status = json.loads((tmp_path/"prediction_metrics_status.json").read_text())
    assert status["rows"] == 0 and status["expected_rows"] == 7168 and not status["complete"]
    assert not (tmp_path/"prediction_metrics.csv").exists()


def test_resume_artifact_manifest_detects_checkpoint_change(tmp_path):
    prepare = script("prepare_round1")
    weights = tmp_path/"model.json"; weights.write_text("synthetic fitted weights")
    with pytest.raises(RuntimeError, match="no artifact manifest"):
        prepare.artifact_manifest(tmp_path)
    prepare.artifact_manifest(tmp_path, create=True)
    prepare.artifact_manifest(tmp_path)
    weights.write_text("changed checkpoint")
    with pytest.raises(RuntimeError, match="artifact changed"):
        prepare.artifact_manifest(tmp_path)


def test_calibration_resume_checks_payload_source_and_seed_identity(tmp_path, monkeypatch):
    prepare = script("prepare_round1")
    monkeypatch.setattr(prepare, "OUT", tmp_path)
    cfg = config(); cfg["worlds"] = ["maxwell"]
    source = tmp_path/"source.py"; source.write_text("synthetic unchanged")
    paths = []
    for split in ("train", "val"):
        path = tmp_path/f"calibration/maxwell/r101/{split}.npz"
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"synthetic calibration")
        meta = dict(data_sha256=prepare.digest(path), source_sha256={str(source): prepare.digest(source)},
                    rows=[dict(source_base=cfg["calibration_source_bases"][split], replicate=101)])
        path.with_suffix(".json").write_text(json.dumps(meta)); paths.append(path)
    prepare.verify_calibration(cfg, [101])
    source.write_text("changed source")
    with pytest.raises(RuntimeError, match="source changed"):
        prepare.verify_calibration(cfg, [101])
    source.write_text("synthetic unchanged")
    first = paths[0]; meta = json.loads(first.with_suffix(".json").read_text())
    meta["rows"][0]["replicate"] = 102
    first.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="identity"):
        prepare.verify_calibration(cfg, [101])
    meta["rows"][0]["replicate"] = 101
    first.with_suffix(".json").write_text(json.dumps(meta)); first.write_bytes(b"changed data")
    with pytest.raises(RuntimeError, match="Calibration changed"):
        prepare.verify_calibration(cfg, [101])


def test_evaluation_main_requires_test_lock_before_model_load(tmp_path, monkeypatch):
    evaluate = script("evaluate_round1")
    def no_test_lock():
        raise FileNotFoundError("synthetic absent test lock")
    def forbidden(*args):
        raise AssertionError("Evaluation reached before test lock")
    monkeypatch.setattr(evaluate, "verify_test_lock", no_test_lock)
    monkeypatch.setattr(evaluate, "run_predictions", forbidden)
    monkeypatch.setattr(sys, "argv", ["evaluate_round1.py", "prediction"])
    with pytest.raises(FileNotFoundError, match="absent test lock"):
        evaluate.main()


@pytest.mark.parametrize("fail_native", [False, True])
def test_mock_controller_observer_once_per_real_step_and_failure_retention(tmp_path, monkeypatch, fail_native):
    evaluate = script("evaluate_round1")
    monkeypatch.setattr(evaluate, "OUT", tmp_path)
    monkeypatch.setattr(evaluate, "verify_model_lock", lambda: "synthetic-model")
    cfg = config(); cfg.update(compositions=[["maxwell", "boucwen"]], replicates=[101], noise=["clean"],
                               warmup_seconds=.01, motion_seconds=.02)
    cfg["profiles_per_replicate"]["evaluation"] = 2
    cfg["controller"].update(horizon_steps=3, replan_steps=2)
    data = toy_data(2, 8, 4); data["warm_normal"][:] = 15
    write_toy_test(tmp_path, evaluate, data)
    references, models, closed = [], {}, []
    class FakeReference:
        def __init__(self, *args):
            self.count = 0; self.observations = 0; references.append(self)
        def force(self, d, r, v, dt):
            self.count += 1
            return np.full(len(d), 15.), np.zeros(len(d))
        def observe(self, d, r, v, dt):
            self.observations += 1
            return np.full(len(d), 15.), np.zeros(len(d))
    class FakeRig:
        def __init__(self, n, device, dt):
            self.s = np.zeros((n, 2)); self.dt = dt; self.steps = 0
        def reset(self):
            self.s[:] = 0; self.steps = 0
        def state(self):
            return self.s.copy()
        def advance(self, force):
            self.s[:, 1] += self.dt*np.asarray(force)/2
            self.s[:, 0] += self.dt*self.s[:, 1]
            self.steps += 1
            if fail_native and self.steps == 7:
                self.s[0, 0] = 21.
            return self.state()
    app = types.ModuleType("isaaclab.app")
    app.AppLauncher = lambda args: types.SimpleNamespace(app=types.SimpleNamespace(close=lambda **kwargs: closed.append(kwargs)))
    rig = types.ModuleType("decision_study.isaac_rig"); rig.IsaacRig = FakeRig
    reference = types.ModuleType("round1_reference"); reference.CompositeReference = FakeReference
    stimulus = types.ModuleType("round1_stimuli")
    stimulus.preparation = lambda t, rows: (np.full(len(rows), .02), np.zeros(len(rows)))
    stimulus.commands = lambda rows, cfg: (np.full((len(rows), 11), .02), np.zeros((len(rows), 11)), np.full((len(rows), 11), .03))
    for name, module in [("isaaclab.app", app), ("decision_study.isaac_rig", rig), ("round1_reference", reference), ("round1_stimuli", stimulus)]:
        monkeypatch.setitem(sys.modules, name, module)
    def load(method, *unused):
        models[method] = CountingModel(); return models[method]
    monkeypatch.setattr(evaluate, "load", load)
    args = types.SimpleNamespace(seeds=[101], methods=["blind_state_id", "state_reset"])
    if fail_native:
        with pytest.raises(FloatingPointError, match="partial raw trajectory preserved"):
            evaluate.run_controllers(args, cfg, "synthetic-model")
        failed = tmp_path/"controllers/maxwell_boucwen/r101/clean/blind_state_id_failed.npz"
        saved = dict(np.load(failed))
        assert saved["failed_step"] == 2 and saved["next_state"].shape == (2, 2, 2)
        assert len(json.loads(failed.with_suffix(".json").read_text())["rows"]) == 2
    else:
        evaluate.run_controllers(args, cfg, "synthetic-model")
        for method, model in models.items():
            real = [item for item in model.calls if len(item[0]) == 2]
            candidates = [item for item in model.calls if len(item[0]) == 82]
            assert len(real) == 12 and len(candidates) == 12
            assert all(x[1] is not None for x in real)
            expected = list(range(4))+list(range(8)) if method == "state_reset" else list(range(12))
            np.testing.assert_array_equal([x[0][0] for x in real], expected)
            assert [x[1] is not None for x in candidates] == [True, False, False]*4
        assert [ref.count for ref in references] == [12, 12]
        assert [ref.observations for ref in references] == [4, 5]
    assert closed
