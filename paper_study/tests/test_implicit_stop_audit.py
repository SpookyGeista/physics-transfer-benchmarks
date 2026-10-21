"""Independent SI energy/causality checks of the proposed scalar PI coupling.

These are pure NumPy/constitutive tests, not evidence of native PhysX accuracy.
All initial material states are reached by force transitions, never injected.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "alternatives")]
from implicit_stop_coupling import solve_velocity
from multistate_friction import MultiStateFriction, PARAMETER_SETS


def pure_average(reference, normal, dt):
    return lambda dx: reference.clone().force(np.asarray(dx) / dt, normal, dt)


def balance(before, after, v, vn, u, slope, normal, dt, mass=2.):
    dx = dt * vn
    energy_change = after["stored_energy"] - before["stored_energy"]
    dissipation = after["dissipation"] - before["dissipation"]
    kinetic_change = .5 * mass * (vn**2 - v**2)
    numerical_dissipation = .5 * mass * (vn-v)**2
    return (u-slope*normal)*dx - kinetic_change - energy_change - dissipation - numerical_dissipation


@pytest.mark.parametrize("name", PARAMETER_SETS)
def test_all_elastic_exact_solution_and_reversible_energy_release(name):
    dt, mass, normal = .0025, 2., 15.
    reference = MultiStateFriction(name, n=3, initial_normal=normal)
    # Small reached deflection, all three elements strictly elastic.
    reference.force(np.array([.01, -.01, .01]), normal, dt)
    old = reference.audit()
    stiffness = sum(PARAMETER_SETS[name].stiffness)
    current_force = old["force_instantaneous"]
    v = np.zeros(3)
    u, slope = np.zeros(3), np.zeros(3)
    capacity = normal * sum(PARAMETER_SETS[name].friction_capacity)
    vn, force, residual = solve_velocity(v, u, slope, normal, capacity,
                                         pure_average(reference, normal, dt), dt=dt, mass=mass)
    # Entire interval remains elastic: Ft_mean=Ft(0)-0.5*sum(k)*dt*vn.
    expected = (v + dt * current_force / mass) / (1 + .5 * stiffness * dt**2 / mass)
    np.testing.assert_allclose(vn, expected, rtol=0, atol=2e-14)
    np.testing.assert_allclose(residual, 0., rtol=0, atol=1e-11)
    np.testing.assert_array_equal(reference.audit()["deflection"], old["deflection"])
    assert reference.updates == old["updates"]  # all bisection evaluations are pure
    committed_force = reference.force(vn, normal, dt)
    np.testing.assert_array_equal(force, committed_force)
    after = reference.audit()
    assert np.all(force * vn > 0.)  # stored energy legitimately accelerates the mass
    assert np.all(after["stored_energy"] < old["stored_energy"])
    np.testing.assert_array_equal(after["last_step"]["dissipation"], np.zeros(3))
    np.testing.assert_allclose(balance(old, after, v, vn, u, slope, normal, dt), 0., atol=1e-15, rtol=0)


@pytest.mark.parametrize("name", PARAMETER_SETS)
def test_monotone_bounded_average_bracket_existence_and_purity_at_load_changes(name):
    reference = MultiStateFriction(name, n=7, initial_normal=22.)
    reference.force(np.linspace(-.15, .15, 7), 22., .1)
    reference.force(np.linspace(.015, -.01, 7), 22., .1)
    before = reference.audit()
    normal = np.array([0., .001, 1., 7., 15., 22., 35.])
    capacity = normal * sum(PARAMETER_SETS[name].friction_capacity)
    fn = pure_average(reference, normal, .0025)
    displacements = np.linspace(-.1, .1, 401)
    forces = np.array([fn(dx) for dx in displacements])
    assert np.max(np.diff(forces, axis=0)) <= 1e-12
    assert np.all(abs(forces) <= capacity + 1e-12)
    np.testing.assert_array_equal(reference.audit()["deflection"], before["deflection"])
    assert reference.updates == before["updates"]
    v = np.linspace(-.1, .12, 7)
    u = np.linspace(-12., 12., 7)
    slope = np.linspace(-.15, .15, 7)
    free = v + .0025 * (u-slope*normal)/2.
    lower, upper = free-.0025*capacity/2., free+.0025*capacity/2.
    residual_lower = lower-free-.0025*fn(lower*.0025)/2.
    residual_upper = upper-free-.0025*fn(upper*.0025)/2.
    assert np.max(residual_lower) < 1e-14
    assert np.min(residual_upper) > -1e-14
    vn, _, residual = solve_velocity(v, u, slope, normal, capacity, fn)
    assert np.all(vn >= lower) and np.all(vn <= upper)
    np.testing.assert_allclose(residual, 0., rtol=0, atol=3e-11)


@pytest.mark.parametrize("name", PARAMETER_SETS)
def test_zero_load_and_reopening_account_for_load_release(name):
    dt = .0025
    reference = MultiStateFriction(name, n=4, initial_normal=20.)
    reference.force(np.array([.1, -.1, .02, -.02]), 20., .1)
    before = reference.audit()
    assert np.all(before["stored_energy"] > 0.)
    v, u, slope = np.array([0., .1, -.1, .02]), np.array([0., 3., -5., 2.]), .12
    vn, force, residual = solve_velocity(v, u, slope, 0., 0., pure_average(reference, 0., dt))
    np.testing.assert_array_equal(force, np.zeros(4))
    np.testing.assert_allclose(vn, v+dt*u/2., rtol=0, atol=1e-16)
    reference.force(vn, 0., dt)
    opened = reference.audit()
    np.testing.assert_array_equal(opened["stored_energy"], np.zeros(4))
    np.testing.assert_array_equal(opened["last_step"]["load_release"], before["stored_energy"])
    np.testing.assert_allclose(balance(before, opened, v, vn, u, slope, 0., dt), 0., rtol=0, atol=1e-15)
    # Restoring N at zero displacement cannot create energy in this adaptation.
    reference.force(0., 30., dt)
    np.testing.assert_array_equal(reference.audit()["stored_energy"], np.zeros(4))


def run_random_audit(name, steps=100, seed=9152026):
    """Deterministic diagnostic, reused by the report without storing trajectories."""
    rng = np.random.default_rng(seed)
    n, mass = 8, 2.
    reference = MultiStateFriction(name, n=n, initial_normal=15.)
    reference.force(rng.uniform(-.2, .2, n), 15., .1)
    v = rng.uniform(-.3, .3, n)
    maxima = dict(momentum_residual_N=0., energy_residual_J=0., force_commit_difference_N=0.,
                  max_positive_material_power_W=0., max_load_release_J=0.)
    minimum_dissipation = np.inf
    for step in range(steps):
        dt = (.0025, .00125, .000625)[step % 3]
        normal = rng.uniform(.1, 35., n)
        normal[0] = 0. if step % 5 == 0 else normal[0]
        u, slope = rng.uniform(-12., 12., n), rng.uniform(-.15, .15, n)
        before = reference.audit()
        capacity = normal * sum(PARAMETER_SETS[name].friction_capacity)
        vn, force, residual = solve_velocity(v, u, slope, normal, capacity,
                                             pure_average(reference, normal, dt), dt=dt, mass=mass)
        np.testing.assert_array_equal(reference.audit()["deflection"], before["deflection"])
        assert reference.updates == before["updates"]
        committed_force = reference.force(vn, normal, dt)
        after = reference.audit()
        energy_residual = balance(before, after, v, vn, u, slope, normal, dt, mass)
        maxima["momentum_residual_N"] = max(maxima["momentum_residual_N"], float(abs(residual).max()))
        maxima["energy_residual_J"] = max(maxima["energy_residual_J"], float(abs(energy_residual).max()))
        maxima["force_commit_difference_N"] = max(maxima["force_commit_difference_N"],
                                                  float(abs(force-committed_force).max()))
        maxima["max_positive_material_power_W"] = max(maxima["max_positive_material_power_W"],
                                                      float((force*vn).max()))
        maxima["max_load_release_J"] = max(maxima["max_load_release_J"],
                                            float(after["last_step"]["load_release"].max()))
        minimum_dissipation = min(minimum_dissipation, float(after["last_step"]["dissipation"].min()))
        v = vn
    return dict(parameter_set=name, steps=steps, lanes=n, seed=seed,
                min_dissipation_J=minimum_dissipation, **maxima)


@pytest.mark.parametrize("name", PARAMETER_SETS)
def test_random_signed_motion_load_changes_and_numerical_energy_balance(name):
    result = run_random_audit(name)
    assert result["momentum_residual_N"] < 4e-11
    assert result["energy_residual_J"] < 4e-14
    assert result["force_commit_difference_N"] == 0.
    assert result["min_dissipation_J"] >= 0.
    assert result["max_positive_material_power_W"] > 0.
    assert result["max_load_release_J"] > 0.


@pytest.mark.parametrize("name", PARAMETER_SETS)
def test_native_residual_decomposition_includes_commit_displacement(name):
    """Algebra check with synthetic float32 storage, explicitly NOT PhysX."""
    dt, mass, normal, u, slope = .0025, 2., 15., np.array([5.]), .08
    reference = MultiStateFriction(name)
    reference.force(.01, normal, .1)
    before = reference.audit()
    v = np.array([.013])
    vn, applied_force, _ = solve_velocity(v, u, slope, normal,
        normal * sum(PARAMETER_SETS[name].friction_capacity), pure_average(reference, normal, dt))
    actual_vn = vn.astype(np.float32).astype(float)
    actual_dx = (dt * actual_vn + 2e-9).astype(np.float32).astype(float)
    committed_force = reference.force(actual_dx/dt, normal, dt)
    after = reference.audit()
    external, net = u-slope*normal, u-slope*normal+applied_force
    momentum_error = mass*(actual_vn-v)/dt-net
    kinematic_error = actual_dx-dt*actual_vn
    energy_error = external*actual_dx - .5*mass*(actual_vn**2-v**2) \
                   - (after["stored_energy"]-before["stored_energy"]) \
                   - after["last_step"]["dissipation"] - .5*mass*(actual_vn-v)**2
    decomposed = (committed_force-applied_force)*actual_dx + net*kinematic_error \
                 - dt*momentum_error*actual_vn
    np.testing.assert_allclose(energy_error, decomposed, rtol=0, atol=1e-17)


def test_rejected_basic_invalid_inputs():
    for kwargs in (dict(normal=-1.), dict(capacity=-1.), dict(dt=0.), dict(mass=0.), dict(v=np.nan),
                   dict(iterations=0), dict(iterations=True)):
        arguments = dict(v=0., u=0., slope=0., normal=1., capacity=1., force_average=lambda dx: -np.asarray(dx))
        arguments.update(kwargs)
        with pytest.raises(ValueError):
            solve_velocity(**arguments)
