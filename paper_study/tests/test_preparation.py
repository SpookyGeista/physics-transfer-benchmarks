"""Small development-only checks. Reserved test histories are never generated."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import preparation as prep


def test_plan_reproducibility_prefix_and_disjoint_seed_contract():
    a = prep.preparation_plan(7, 31001, "dev")
    b = prep.preparation_plan(7, 31001, "development")
    small = prep.preparation_plan(3, 31001, "dev")
    other = prep.preparation_plan(7, 41001, "val")
    np.testing.assert_array_equal(a.velocity, b.velocity)
    np.testing.assert_array_equal(a.velocity[:3], small.velocity)
    assert a.rows == b.rows
    assert not np.array_equal(a.velocity, other.velocity)
    ranges = list(prep.SEED_RANGES.values())
    assert all(a[1] < b[0] or b[1] < a[0]
               for i, a in enumerate(ranges) for b in ranges[i+1:])
    # Validate partition errors without drawing any reserved test profile.
    with pytest.raises(ValueError):
        prep.preparation_plan(1, 31001, "test")


def test_end_kinematics_commands_and_readonly_plan():
    plan = prep.preparation_plan(18, 31002, "dev")
    assert plan.velocity.shape == (18, prep.STEPS + 1)
    np.testing.assert_array_equal(plan.velocity[:, [0, -1]], 0.)
    np.testing.assert_allclose(plan.x[:, -1], 0., rtol=0, atol=1e-14)
    np.testing.assert_array_equal(plan.normal_scale[:, [0, -1]], 1.)
    assert np.max(np.abs(plan.velocity)) <= .32
    assert np.min(plan.normal_scale) >= .7
    assert np.max(plan.normal_scale) <= 1.3
    assert all(row['duration'] == 4. for row in plan.rows)
    with pytest.raises(ValueError):
        plan.velocity[0, 0] = 2.
    # Each analytic ramp has zero boundary slope. Consecutive endpoint samples
    # need not be equal; finite differences are not the continuous derivative.
    for row, velocity in zip(plan.rows, plan.velocity):
        if row['recipe'] != 'rest':
            assert velocity[row['start_step']] == velocity[row['end_step']] == 0.


@pytest.mark.parametrize('world', prep.WORLDS)
def test_reachable_state_equals_full_reference_replay(world):
    pair = prep.referencepair(world, 6, 31001, 'dev')
    audit = prep.truth_preparation(pair)
    cmd = audit['commands']
    reference = prep.Reference(world, 6, prep.DELTA0)
    for t in range(prep.STEPS):
        reference.force(cmd['delta'][:, t], cmd['rate'][:, t], cmd['v'][:, t],
                        normal_override=cmd['normal'][:, t], dt=prep.DT)
    # No state assignment: both sides reached their states through force calls.
    np.testing.assert_array_equal(prep.truth_snapshot(pair)['friction_history'],
                                  reference._friction_history)
    assert np.ptp(prep.truth_snapshot(pair)['friction_history']) > 1e-4
    normal_replay = prep.Reference(world, 6, prep.DELTA0)
    for v in (.0, .03, -.08, .0):
        n, f = pair.force(prep.DELTA0, 0., v)
        expected_n, _ = normal_replay.force(prep.DELTA0, 0., 0., friction_enabled=False)
        np.testing.assert_array_equal(n, expected_n)
        _, truth_f = reference.force(prep.DELTA0, 0., v, normal_override=n)
        np.testing.assert_array_equal(f, truth_f)
    assert pair.updates == 4


@pytest.mark.parametrize('normal_world', prep.WORLDS)
def test_common_normal_initial_observation_and_no_history_payload(normal_world):
    pair = prep.referencepair('boucwen', 6, 31001, 'dev', normal_world=normal_world)
    obs = pair.initial_observation()
    assert set(obs) == {'x', 'v', 'delta', 'compression_rate', 'normal'}
    for value in obs.values():
        assert value.shape == (6,)
        assert np.isfinite(value).all()
    np.testing.assert_array_equal(obs['v'], 0.)
    np.testing.assert_array_equal(obs['delta'], .020)
    np.testing.assert_array_equal(obs['normal'], obs['normal'][0])
    np.testing.assert_allclose(obs['x'], 0., atol=1e-14)
    fresh = prep.Reference(normal_world, 6, .02)
    before = prep.truth_snapshot(pair)
    n, f = pair.observe()
    np.testing.assert_array_equal(n, fresh.force(.02, 0., 0.)[0])
    np.testing.assert_array_equal(f, 0.)
    np.testing.assert_array_equal(before['normal_history'], prep.truth_snapshot(pair)['normal_history'])
    np.testing.assert_array_equal(before['friction_history'], prep.truth_snapshot(pair)['friction_history'])
    assert pair.updates == 0
    obs['normal'][:] = 0.
    assert np.all(pair.initial_observation()['normal'] > 0.)


def test_clone_continuation_and_override_do_not_reset_history():
    pair = prep.referencepair('maxwell', 3, 31002, 'dev', normal_world='boucwen')
    clone = pair.clone()
    before = prep.truth_snapshot(pair)
    for velocity in (.01, -.02, .0):
        a = pair.force(.02, .001, velocity, normal_override=15.)
        b = clone.force(.02, .001, velocity, normal_override=np.full(3, 15.))
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(a[0], 15.)
    assert not np.array_equal(before['friction_history'], prep.truth_snapshot(pair)['friction_history'])
    untouched = prep.truth_snapshot(clone)['friction_history']
    pair.force(.02, 0., .2)
    np.testing.assert_array_equal(untouched, prep.truth_snapshot(clone)['friction_history'])


def test_preparation_transitions_only_once_per_timestep(monkeypatch):
    calls = []
    original = prep.Reference.force
    def traced(self, *args, **kwargs):
        calls.append((id(self), kwargs.get('normal_override') is not None))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(prep.Reference, 'force', traced)
    pair = prep.referencepair('maxwell', 1, 31002, 'dev')
    assert sum(has_override for _, has_override in calls) == prep.STEPS
    assert len(calls) == prep.STEPS + 1  # non-mutating initial N query on a clone
    assert pair.updates == 0
