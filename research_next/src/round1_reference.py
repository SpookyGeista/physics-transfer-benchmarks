"""Evaluation-only Cartesian assembly of unchanged v2 material laws.

No fitting module may import this file. Each component owns independent hidden
history; forces are evaluated before exactly one update of each instance.
"""
import copy
from pathlib import Path
import sys

import numpy as np

STUDY_ROOT = Path(__file__).resolve().parents[2]/"study_v2"
sys.path.insert(0, str(STUDY_ROOT/"src"))
from decision_study.reference import Reference


class CompositeReference:
    def __init__(self, normal_world, friction_world, n, initial_delta):
        self.normal_world = normal_world
        self.friction_world = friction_world
        self._normal = Reference(normal_world, n, initial_delta)
        self._friction = Reference(friction_world, n, initial_delta)
        self.n = int(n)
        self.updates = 0

    def force(self, delta, rate, v, friction_enabled=True, dt=.0025):
        # The unused friction state in _normal and unused normal state in
        # _friction are isolated and never coupled back into the active laws.
        normal, _ = self._normal.force(delta, rate, np.zeros(self.n),
                                      friction_enabled=False, dt=dt)
        _, friction = self._friction.force(delta, rate, v, normal_override=normal,
                                          friction_enabled=friction_enabled, dt=dt)
        self.updates += 1
        return normal, friction

    def clone(self, repeats=1):
        """Oracle/sensor-only copy; repeated candidate histories are independent."""
        if isinstance(repeats, bool) or int(repeats) != repeats or repeats < 1:
            raise ValueError("Positive integer repeats required")
        cloned = copy.deepcopy(self)
        if repeats != 1:
            for component in (cloned._normal, cloned._friction):
                component._n *= repeats
                component._normal_history = np.repeat(component._normal_history, repeats, axis=0)
                component._friction_history = np.repeat(component._friction_history, repeats, axis=0)
            cloned.n *= repeats
        return cloned

    def observe(self, delta, rate, v, friction_enabled=True, dt=.0025):
        """Current ideal force observation without advancing physical history."""
        return self.clone().force(delta, rate, v, friction_enabled=friction_enabled, dt=dt)


def clone_reference(reference, repeats=1):
    return reference.clone(repeats)


def observe_force(reference, delta, rate, velocity, dt=.0025):
    return reference.observe(delta, rate, velocity, dt=dt)
