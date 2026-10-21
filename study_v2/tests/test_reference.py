"""Numerical and information-boundary checks for the independent references."""
import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from decision_study.reference import Reference


@pytest.mark.parametrize("world", ["maxwell", "boucwen"])
def test_force_sign_contact_gate_and_no_input_mutation(world):
    delta = np.linspace(-0.004, 0.035, 97)
    rate = np.linspace(-0.12, 0.12, len(delta))
    v = np.linspace(-0.5, 0.5, len(delta))
    original = [x.copy() for x in (delta, rate, v)]
    ref = Reference(world, len(delta), delta)
    for _ in range(25):
        normal, friction = ref.force(delta, rate, v, normal_override=20.0)
        assert np.all(np.isfinite(normal)) and np.all(np.isfinite(friction))
        assert np.all(normal >= 0.0)
        assert np.all(friction * v <= 0.0)
        assert np.all(normal[delta <= 0] == 0.0)
        assert np.all(friction[delta <= 0] == 0.0)
    for actual, expected in zip((delta, rate, v), original):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("world", ["maxwell", "boucwen"])
def test_separate_local_history_replay_is_exact(world):
    """The controlled-load A experiment reproduces combined-world friction."""
    n = 5
    phase = np.arange(n) * 0.25
    d0 = 0.020 + 0.008 * np.sin(phase)
    combined = Reference(world, n, d0)
    normal_only = Reference(world, n, d0)
    friction_only = Reference(world, n, np.full(n, 0.020))
    for t in np.arange(0, 1.5, 0.0025):
        d = 0.020 + 0.008 * np.sin(2 * np.pi * 1.3 * t + phase)
        rate = 0.008 * 2 * np.pi * 1.3 * np.cos(2 * np.pi * 1.3 * t + phase)
        v = 0.24 * np.sin(2 * np.pi * 0.9 * t + phase)
        nc, fc = combined.force(d, rate, v)
        nb, fb = normal_only.force(d, rate, 0.0, friction_enabled=False)
        na, fa = friction_only.force(0.020, 0.0, v, normal_override=nb)
        np.testing.assert_array_equal(nc, nb)
        np.testing.assert_array_equal(fb, np.zeros(n))
        np.testing.assert_array_equal(na, nc)
        np.testing.assert_array_equal(fa, fc)


def test_maxwell_relaxation_matches_closed_form_and_updates_once():
    ref = Reference("maxwell", 1, 0.01)
    stiffness = np.array([210.0, 125.0, 80.0])
    taus = np.array([0.025, 0.17, 0.80])
    dt = 0.0025
    for step in range(401):
        normal, friction = ref.force(0.02, 0.0, 0.0, dt=dt)
        expected = 630 * 0.02 + np.sum(stiffness * 0.01 * np.exp(-step * dt / taus))
        np.testing.assert_allclose(normal, expected, rtol=2e-14, atol=2e-13)
        assert friction[0] == 0.0


@pytest.mark.parametrize("world", ["maxwell", "boucwen"])
def test_open_contact_relaxes_normal_history(world):
    ref = Reference(world, 3, [0.009, 0.02, 0.03])
    for _ in range(30):
        ref.force(0.029, 0.05, 0.1)
    initial_norm = np.linalg.norm(ref._normal_history)
    for _ in range(800):
        normal, friction = ref.force(-0.005, -0.1, 0.3, normal_override=25.0)
        assert np.all(normal == 0.0) and np.all(friction == 0.0)
    assert np.linalg.norm(ref._normal_history) < 0.10 * initial_norm


@pytest.mark.parametrize("world", ["maxwell", "boucwen"])
def test_friction_mask_does_not_change_normal_evolution(world):
    yes, masked = Reference(world, 3, 0.02), Reference(world, 3, 0.02)
    for step in range(40):
        d = 0.02 + 0.0001 * step
        n1, f1 = yes.force(d, 0.04, [0.1, -0.1, 0.0])
        n2, f2 = masked.force(d, 0.04, [0.1, -0.1, 0.0], friction_enabled=[True, False, True])
        np.testing.assert_array_equal(n1, n2)
        np.testing.assert_array_equal(f2, [f1[0], 0.0, 0.0])


def test_boucwen_finite_under_envelope_and_hysteretic():
    ref = Reference("boucwen", 1, 0.02)
    normal_values = []
    history_values = []
    for t in np.arange(0, 8, 0.0025):
        omega = 2 * np.pi
        d = 0.02 + 0.01 * np.sin(omega * t)
        rate = 0.01 * omega * np.cos(omega * t)
        normal, friction = ref.force(d, rate, 0.3 * np.sin(1.4 * omega * t))
        normal_values.append(normal[0])
        history_values.append(ref._normal_history[0])
        assert np.isfinite(friction).all()
    assert min(normal_values) >= 0.0 and max(normal_values) < 40.0
    assert max(np.abs(history_values)) <= 1.0000001
    # At d=.02 in the settled final cycle, rising/falling branches differ.
    assert abs(normal_values[7 * 400] - normal_values[7 * 400 + 200]) > 1.0


@pytest.mark.parametrize("world", ["maxwell", "boucwen"])
def test_time_refinement_converges_on_smooth_inputs(world):
    def trace(dt):
        ref = Reference(world, 1, 0.02)
        out = []
        for t in np.arange(0, 1, dt):
            omega = 2 * np.pi * 1.1
            n, f = ref.force(
                0.02 + 0.008 * np.sin(omega * t),
                0.008 * omega * np.cos(omega * t),
                0.15 + 0.10 * np.sin(2 * np.pi * 0.7 * t), dt=dt,
            )
            out.append([n[0], f[0]])
        return np.asarray(out)
    a, b, c = [trace(dt) for dt in (0.0025, 0.00125, 0.000625)]
    err_ab = np.sqrt(np.mean((a - b[::2]) ** 2))
    err_bc = np.sqrt(np.mean((b - c[::2]) ** 2))
    assert err_ab < 0.1  # N, numerical diagnostic; not a method-performance target.
    assert err_bc < 0.60 * err_ab


@pytest.mark.parametrize("world", ["maxwell", "boucwen"])
def test_parameter_objects_are_immutable_and_invalid_input_is_rejected(world):
    ref = Reference(world, 2, 0.02)
    with pytest.raises(FrozenInstanceError):
        ref._params.equilibrium_stiffness = 1.0
    for kwargs in ({"dt": 0.0}, {"dt": np.nan}, {"normal_override": np.nan}):
        with pytest.raises(ValueError):
            ref.force(0.02, 0.0, 0.1, **kwargs)
    with pytest.raises(ValueError):
        ref.force([0.02, 0.03, 0.01], 0.0, 0.1)
    with pytest.raises(ValueError):
        ref.force(0.02, 0.0, np.inf)


def test_reference_has_no_predictor_imports():
    source = Path(__file__).parents[1] / "src" / "decision_study" / "reference.py"
    imports = []
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append((node.module or "").split(".")[0])
    assert set(imports) <= {"numpy", "dataclasses"}
