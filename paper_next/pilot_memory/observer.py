"""Fixed learned-law grid observer. No reference or true-state inputs/imports."""
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "research_next/src"))
from state_id import StateID


def load_friction_model(path):
    selected = StateID.load(path)
    return selected.friction_model


def candidate_bank(model_path, grid, probe_v, probe_n, future_v, future_n,
                   duration_steps, delta=.02, dt=.0025):
    """Outputs precede updates; saved states are AFTER precisely k probe steps."""
    model = load_friction_model(model_path)
    state_key = "mix" if model.world == "maxwell" else "age"
    grid = np.asarray(grid, float)
    d = np.full(len(grid), delta)
    zero = np.zeros(len(grid))
    state = model.initial(d, zero, np.full(len(grid), probe_n[0]))
    state[state_key] = grid.copy()
    saved = {0: {k: v.copy() for k, v in state.items()}}
    force = np.empty((len(grid), len(probe_v)))
    for step, (v, n) in enumerate(zip(probe_v, probe_n)):
        _, force[:, step], state = model.step(d, zero, np.full(len(grid), v), state,
                                             normal_override=n, dt=dt)
        if step + 1 in duration_steps:
            saved[step + 1] = {k: x.copy() for k, x in state.items()}
    current, future = [], []
    for steps in duration_steps:
        state = {k: x.copy() for k, x in saved[int(steps)].items()}
        current.append(state[state_key].copy())
        trajectory = []
        for v, n in zip(future_v, future_n):
            _, ft, state = model.step(d, zero, np.full(len(grid), v), state,
                                      normal_override=n, dt=dt)
            trajectory.append(ft)
        future.append(np.stack(trajectory, axis=1))
    return dict(grid=grid, force=force, current=np.stack(current),
                future=np.stack(future), family=model.world)


def posterior_weights(force_candidates, measurements, sigma, batch_size=None):
    """Only force measurements from the completed probe enter the likelihood.

    measurements=None is physical probing with the force channel masked.
    Empty observed sequences likewise preserve the original discrete prior.
    """
    force_candidates = np.asarray(force_candidates, float)
    count, steps = force_candidates.shape
    if measurements is None:
        if batch_size is None:
            raise ValueError("A blinded batch needs its size, not hidden labels")
        return np.full((batch_size, count), 1. / count)
    y = np.asarray(measurements, float)
    if y.ndim != 2 or y.shape[1] != steps or sigma <= 0:
        raise ValueError("Probe observations/candidate time axes or sigma differ")
    if steps == 0:
        return np.full((len(y), count), 1. / count)
    # Algebraic SSE avoids a batch × grid × time residual allocation.
    sse = np.sum(y * y, axis=1)[:, None] + np.sum(force_candidates**2, axis=1)[None]
    sse -= 2. * (y @ force_candidates.T)
    sse = np.maximum(sse, 0.)
    logw = -(sse - sse.min(axis=1, keepdims=True)) / (2. * sigma**2)
    weights = np.exp(logw)
    return weights / weights.sum(axis=1, keepdims=True)


def summarize_posterior(weights, initial_grid, current_grid, future_force):
    """Separate estimates of old state, present state and future force."""
    cdf = np.cumsum(weights, axis=1)
    lo = np.argmax(cdf >= .05, axis=1)
    hi = np.argmax(cdf >= .95, axis=1)
    return dict(initial_mean=weights @ initial_grid,
                current_mean=weights @ current_grid,
                future_mean=weights @ future_force,
                initial_width90=initial_grid[hi] - initial_grid[lo],
                current_width90=current_grid[hi] - current_grid[lo],
                entropy=-np.sum(weights * np.log(np.maximum(weights, 1e-300)), axis=1))
