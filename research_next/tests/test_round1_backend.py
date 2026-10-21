"""Synthetic, nonreserved parity checks; never opens held-out round-1 data."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT.parent/'study_v2/src')]
from round1_mujoco_reference import (
    IndependentMaterial, IndependentComposite, cam, preparation, compression,
    make_model, replay,
)
from round1_reference import CompositeReference
from round1_stimuli import preparation as primary_preparation, commands as primary_commands
sys.path.insert(0, str(ROOT/'scripts'))
from check_round1_backends import position_errors, refinement


def synthetic_rows():
    """Fixed arbitrary values; no reserved seed or condition generator is used."""
    return [dict(preparation_duration=.63+.11*i, preparation_sign=(-1)**i,
                 preparation_amplitude=.0031+.0006*i, amplitude=(-1)**i*.0037,
                 wave_amplitude=.0021+.0003*i, wave_frequency=.53+.12*i,
                 phase=.19+.7*i, target_sign=(-1)**i) for i in range(4)]


class MaterialTests(unittest.TestCase):
    def test_real_data_mode_checks_locks_before_reading_holdout(self):
        import check_round1_backends as audit
        with mock.patch.object(audit, 'verify_code_lock', return_value='code'), \
             mock.patch.object(audit, 'verify_model_lock', side_effect=RuntimeError('unlocked')), \
             mock.patch.object(Path, 'read_text', side_effect=AssertionError('read before lock')):
            with self.assertRaisesRegex(RuntimeError, 'unlocked'):
                audit.run(None)

    def test_real_data_mode_requires_test_lock_before_any_data_read(self):
        import check_round1_backends as audit
        with mock.patch.object(audit, 'verify_code_lock', return_value='code'), \
             mock.patch.object(audit, 'verify_model_lock', return_value='models'), \
             mock.patch.object(audit, 'verify_test_lock', side_effect=RuntimeError('test unlocked')), \
             mock.patch.object(Path, 'read_text', side_effect=AssertionError('read before test lock')), \
             mock.patch.object(np, 'load', side_effect=AssertionError('NPZ read before test lock')):
            with self.assertRaisesRegex(RuntimeError, 'test unlocked'):
                audit.run(None)

    def test_refinement_statistics_preserve_profiles_and_alignment(self):
        exact = np.zeros((4, 9, 2))
        a, b, c = exact.copy(), exact.copy(), exact.copy()
        a[..., 0] = np.arange(1, 5)[:, None]*.016
        b[..., 0] = np.arange(1, 5)[:, None]*.008
        c[..., 0] = np.arange(1, 5)[:, None]*.004
        value = refinement(a, b, c)
        np.testing.assert_allclose(value['D24_D12_episode'], .5, atol=1e-15)
        self.assertEqual(value['median_D24_D12'], .5)
        self.assertAlmostEqual(value['h1_h4_position_rmse_m'], np.sqrt(np.mean((np.arange(1, 5)*.012)**2)))
        self.assertEqual(refinement(exact, exact, exact)['median_D24_D12'], 0)
        self.assertEqual(refinement(exact, exact, a)['median_D24_D12'], 1e100)
        metrics = position_errors(a, exact)
        np.testing.assert_allclose(metrics['episode_position_rmse_m'], np.arange(1, 5)*.016)
        for function, arrays in [(position_errors, (a, exact[:1])), (refinement, (a, b[:1], c))]:
            with self.assertRaises(ValueError):
                function(*arrays)

    def test_no_primary_or_learner_imports_in_independent_production(self):
        tree = ast.parse((ROOT/'src/round1_mujoco_reference.py').read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or '')
        self.assertEqual(set(imports), {'time', 'numpy', 'mujoco'})

    def test_all_cartesian_laws_match_synthetic_primary_sequences(self):
        for normal in ('maxwell', 'boucwen'):
            for friction in ('maxwell', 'boucwen'):
                for dt in (.0025, .00125, .000625):
                    independent = IndependentComposite(normal, friction, 4, [.008, .016, .024, .032])
                    primary = CompositeReference(normal, friction, 4, [.008, .016, .024, .032])
                    for j in range(160):
                        t = j*dt
                        delta = .016 + .018*np.sin(11*t + np.arange(4))
                        rate = .198*np.cos(11*t + np.arange(4))
                        velocity = .23*np.sin(19*t + .3*np.arange(4))
                        actual = independent.force(delta, rate, velocity, dt=dt)
                        expected = primary.force(delta, rate, velocity, dt=dt)
                        for a, b in zip(actual, expected):
                            np.testing.assert_allclose(a, b, rtol=2e-13, atol=2e-12)
                        self.assertTrue(np.all(actual[0] >= 0))
                        self.assertTrue(np.all(actual[1]*velocity <= 0))
                    self.assertEqual(independent.updates, 160)

    def test_single_world_force_override_and_open_contact(self):
        from decision_study.reference import Reference
        for world in ('maxwell', 'boucwen'):
            independent, primary = IndependentMaterial(world, 2, .02), Reference(world, 2, .02)
            for delta, rate, v, override in [(.023, .025, .14, 6.),
                                            (-.01, -.2, .1, 30.),
                                            (.017, .004, -.13, 24.)]:
                for actual, expected in zip(independent.force(delta, rate, v, override),
                                            primary.force(delta, rate, v, override)):
                    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=1e-13)

    def test_state_output_uses_history_before_update(self):
        material = IndependentMaterial('boucwen', 1, .02)
        initial_z = np.tanh(.02/.009)
        n, _ = material.force(.021, .09, .1)
        self.assertAlmostEqual(n[0], 680*.021 + 4.5*initial_z + 1.8*.09, places=12)
        self.assertNotAlmostEqual(material.z[0], initial_z, places=6)

    def test_long_quiet_prefix_preserves_hysteresis_and_ages_friction(self):
        ref = IndependentComposite('boucwen', 'boucwen', 2)
        for j in range(1600):
            t = j*.0025
            bump = .007*np.sin(np.pi*t/.8)**2 if t < .8 else 0
            rate = .007*np.pi/.8*np.sin(2*np.pi*t/.8) if t < .8 else 0
            ref.force(.02 + np.array([bump, -bump]), np.array([rate, -rate]), 0)
        n, _ = ref.force(.02, 0, 0)
        self.assertGreater(abs(n[0]-n[1]), 2.)
        np.testing.assert_allclose(ref.friction.age, .008/.12 + 4.0025, atol=1e-12)

    def test_waveform_and_geometry_parity_without_reserved_rows(self):
        rows = synthetic_rows()
        for t in (0., .07125, .44, 1.2, 3.999375):
            for actual, expected in zip(preparation(t, rows), primary_preparation(t, rows)):
                np.testing.assert_allclose(actual, expected, atol=1e-16)
        cfg = dict(dt=.0025, motion_seconds=1.2,
                   controller=dict(horizon_steps=0, switch_seconds=.6, target_magnitude=.03))
        d, rate, _ = primary_commands(rows, cfg)
        for j in (0, 1, 59, 479):
            actual_d, actual_rate = compression(j*.0025, rows)
            np.testing.assert_allclose(actual_d, d[:, j], atol=1e-16)
            np.testing.assert_allclose(actual_rate, rate[:, j], atol=1e-16)
        x, amp = np.array([-.02, 0., .04]), np.array([-.004, .003, .005])
        g, derivative = cam(x, amp)
        numerical = (cam(x+1e-7, amp)[0]-cam(x-1e-7, amp)[0])/2e-7
        np.testing.assert_allclose(derivative, numerical, atol=1e-11)

    def test_input_validation(self):
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                IndependentMaterial('maxwell', bad, .02)
        material = IndependentMaterial('maxwell', 1, .02)
        for dt in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                material.force(.02, 0, 0, dt=dt)
        with self.assertRaises(ValueError):
            material.force(np.nan, 0, 0)


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'Native MuJoCo is in its existing separate environment')
class NativeTests(unittest.TestCase):
    def test_native_known_force_parallel_sliders_and_no_cross_talk(self):
        import mujoco
        for dt in (.0025, .00125, .000625):
            model = make_model(4, dt)
            sim = mujoco.MjData(model)
            force = np.array([2., -4., 0., 7.])
            for _ in range(16):
                sim.qfrc_applied[:] = force
                mujoco.mj_step(model, sim)
            np.testing.assert_allclose(sim.qvel, force/2*16*dt, atol=1e-14)
            np.testing.assert_allclose(sim.qpos, force/2*dt**2*16*17/2, atol=1e-14)

    def test_native_h1_coupled_states_match_independent_primary_euler(self):
        rows = synthetic_rows()
        commands = np.vstack([3*np.sin(np.arange(48)*.11+i) for i in range(4)])
        for normal in ('maxwell', 'boucwen'):
            for friction in ('maxwell', 'boucwen'):
                actual, meta = replay(normal, friction, rows, commands, warmup_seconds=.2)
                primary = CompositeReference(normal, friction, 4, .02)
                for j in range(80):
                    d, rate = primary_preparation(j*.0025, rows)
                    primary.force(d, rate, np.zeros(4))
                x, v = np.zeros(4), np.zeros(4)
                amp = np.array([r['amplitude'] for r in rows])
                for j in range(48):
                    d, rate = compression(j*.0025, rows)
                    gap, slope = cam(x, amp)
                    n, ft = primary.force(d+gap, rate+slope*v, v)
                    v = v+.0025*(commands[:, j]-slope*n+ft)/2
                    x = x+.0025*v
                    np.testing.assert_allclose(actual['next_state'][:, j], np.column_stack([x, v]), atol=3e-14)
                    np.testing.assert_allclose(actual['normal'][:, j], n, atol=3e-12)
                self.assertEqual(meta['warm_motion_max'], 0)
                self.assertEqual(meta['material_updates'], 128)

    def test_native_coarse_alignment_and_command_ownership(self):
        rows = synthetic_rows()
        force = np.tile(np.array([4., 4., -2., 0., 5., 1.]), (4, 1))
        old = force.copy()
        for h in (1, 2, 4):
            output, meta = replay('boucwen', 'maxwell', rows, force, substeps=h, warmup_seconds=.1)
            self.assertEqual(output['state'].shape, (4, 6, 2))
            self.assertEqual(output['warm_normal'].shape, (4, 40))
            self.assertEqual(meta['material_updates'], 46*h)
            np.testing.assert_array_equal(output['state'][:, 1:], output['next_state'][:, :-1])
            np.testing.assert_array_equal(force, old)
        with self.assertRaises(ValueError):
            replay('maxwell', 'maxwell', rows, force, substeps=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
