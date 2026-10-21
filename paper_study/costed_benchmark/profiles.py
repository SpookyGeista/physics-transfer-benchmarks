"""Public preparation-command distribution, without any material-law imports.

Random seeds/indices identify commands, not latent states. A consumer can draw a
separate prior bank and propagate those commands through its own frozen model.
The generator alone restricts actual trial generation to development seeds.
"""
from __future__ import annotations

import numpy as np

PROFILE_VERSION = "c2-waypoint-fixture-v1-2026-09-15"


def profile_commands(seed, profile_indices, *, dt=.0025, duration=4., waypoint_limit=.025):
    """C2 quintic segments with random interior waypoints and exact endpoints.

    Per-index RNG: SeedSequence([9152026, seed, index]). Each profile draws 3--8
    interior waypoints uniformly in [-limit,+limit], independently. Thus minor
    loops and their amplitudes vary without referring to material parameters.
    Segment durations differ by at most one canonical 2.5-ms interval so every
    turning point lies on that grid and its refinements. No material transition
    skips a within-step reversal. x[0]=x[-1]=v_endpoints=0 analytically.

    Returns time[T+1], x[n,T+1], v[n,T]=diff(x)/dt, normal_scale[n,T]=1,
    continuous_velocity[n,T+1] and rows. v is the work-equivalent constant
    interval velocity, not an assertion of constant physical fixture speed.
    """
    indices = np.asarray(profile_indices)
    if (isinstance(seed, bool) or int(seed) != seed or seed < 0 or indices.ndim != 1
            or not len(indices) or indices.dtype.kind not in "iu" or np.any(indices < 0)):
        raise ValueError("Nonnegative integer seed and nonempty profile indices required")
    if any(not np.isscalar(value) or not np.isfinite(value) or value <= 0
           for value in (dt, duration, waypoint_limit)):
        raise ValueError("Positive finite dt, duration and waypoint limit required")
    steps = round(duration / dt)
    if steps < 1 or not np.isclose(steps * dt, duration, atol=1e-12, rtol=0):
        raise ValueError("duration must be an integer number of steps")
    canonical_dt = .0025
    canonical_steps = round(duration/canonical_dt)
    if (not np.isclose(canonical_steps*canonical_dt, duration, rtol=0, atol=1e-12)
            or not np.isclose(canonical_dt/dt, round(canonical_dt/dt), rtol=0, atol=1e-10)):
        raise ValueError("dt must refine the fixed 2.5-ms waypoint grid and duration must lie on it")
    time = np.arange(steps+1) * dt
    trajectories, velocities, rows = [], [], []
    for raw_index in indices:
        index = int(raw_index)
        rng = np.random.default_rng(np.random.SeedSequence([9152026, int(seed), index]))
        interior_count = int(rng.integers(3, 9))
        waypoints = np.r_[0., rng.uniform(-waypoint_limit, waypoint_limit, interior_count), 0.]
        segment_count = len(waypoints)-1
        intervals = np.full(segment_count, canonical_steps//segment_count)
        intervals[:canonical_steps % segment_count] += 1
        if np.any(intervals < 1):
            raise ValueError("Preparation duration is too short for its declared waypoint distribution")
        knots = np.r_[0, np.cumsum(intervals)] * canonical_dt
        durations = np.diff(knots)
        segment = np.minimum(np.searchsorted(knots, time, side="right")-1, segment_count-1)
        phase = np.clip((time-knots[segment])/durations[segment], 0., 1.)
        blend = phase**3*(10.-15.*phase+6.*phase**2)
        derivative = 30.*phase**2*(1-phase)**2/durations[segment]
        change = waypoints[segment+1]-waypoints[segment]
        x = waypoints[segment]+change*blend
        velocity = change*derivative
        # Exact analytic endpoint evaluation, not a force/state or net-drift repair.
        x[0], x[-1], velocity[0], velocity[-1] = waypoints[0], waypoints[-1], 0., 0.
        trajectories.append(x)
        velocities.append(velocity)
        rows.append(dict(version=PROFILE_VERSION, seed=int(seed), index=index,
                         duration_s=float(duration), dt=float(dt), waypoints_m=waypoints.tolist(),
                         knot_times_s=knots.tolist(), segment_durations_s=durations.tolist(),
                         fixture="ideal imposed displacement at fixed normal load; not force-limited native preparation"))
    x = np.asarray(trajectories)
    return dict(time=time, x=x, v=np.diff(x, axis=1)/dt,
                normal_scale=np.ones((len(indices), steps)),
                continuous_velocity=np.asarray(velocities), rows=rows)


def factorial_design(seed, *, profile_count=6, amplitude=.003, geometry_ids=(0, 1, 2), signs=(-1, 1)):
    """Six independent profiles crossed with three geometry functions and two signs."""
    if profile_count != 6 or list(geometry_ids) != [0, 1, 2] or set(signs) != {-1, 1} or len(signs) != 2:
        raise ValueError("This development design requires 6 profiles x geometry IDs 0,1,2 x signs -1,+1")
    return [dict(episode_id=f"dev-p{seed}-h{index}-g{geometry}-s{int(sign):+d}",
                 preparation_seed=int(seed), preparation_index=index, profile_index=index,
                 geometry=int(geometry), amplitude=float(amplitude), excitation_sign=int(sign))
            for index in range(profile_count) for geometry in geometry_ids for sign in signs]
