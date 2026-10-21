"""Truth-only fixture/evaluator. Never imported by the grid observer.

    Synthetic unknown initial states and imposed N/v are explicit privileges.
    Maxwell truth uses coefficient M; normalized mix is an evaluation gauge.
"""
import copy
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "study_v2/src"))
from decision_study.reference import Reference


def diagnostic_state(reference):
    if reference.world == "maxwell":
        p = reference._params
        return (reference._friction_history.copy() - p.mu_c) / (p.mu_s - p.mu_c)
    return reference._friction_history.copy()


def run_fixture(world, initial, probe_v, probe_n, future_v, future_n,
                duration_steps, delta=.02, dt=.0025):
    reference = Reference(world, len(initial), delta)
    if world == "maxwell":
        p = reference._params
        reference._friction_history[:] = p.mu_c + (p.mu_s - p.mu_c) * initial
    else:
        reference._friction_history[:] = initial
    saved = {0: copy.deepcopy(reference)}
    # A query on a copy proves the v=0 force has no initial-state information.
    _, at_rest = copy.deepcopy(reference).force(delta, 0., 0., normal_override=probe_n[0], dt=dt)
    force = np.empty((len(initial), len(probe_v)))
    for step, (v, n) in enumerate(zip(probe_v, probe_n)):
        _, force[:, step] = reference.force(delta, 0., v, normal_override=n, dt=dt)
        if step + 1 in duration_steps:
            saved[step + 1] = copy.deepcopy(reference)
    current, future = [], []
    for steps in duration_steps:
        reference = copy.deepcopy(saved[int(steps)])
        current.append(diagnostic_state(reference))
        ft = []
        for v, n in zip(future_v, future_n):
            _, f = reference.force(delta, 0., v, normal_override=n, dt=dt)
            ft.append(f)
        future.append(np.stack(ft, axis=1))
    first = saved[0]
    if world == "maxwell":
        rate = 1. / (first._params.friction_lag + first._params.friction_lag_per_load * probe_n[0])
    else:
        rate = abs(probe_v[0]) / first._params.characteristic_slip
    return dict(initial=np.asarray(initial), current=np.stack(current), force=force,
                future=np.stack(future), at_rest_force=at_rest,
                analytic_contraction=np.exp(-np.asarray(duration_steps) * dt * rate))
