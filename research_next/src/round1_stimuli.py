"""Reserved round-1 stimuli; no reference, model, or fitting imports."""
import itertools
import json
from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).resolve().parents[1]/"configs/round1.json"


def load_config():
    return json.loads(CONFIG_PATH.read_text())


def conditions(seed, config=None):
    cfg = load_config() if config is None else config
    if seed not in cfg["replicates"]:
        raise ValueError("Replicate is not in the declared round configuration")
    ranges = cfg["evaluation_ranges"]
    crosses = itertools.product((-1, 1), (-1, 1), (0., float(np.pi)), (-1, 1))
    rows = []
    for i, (prep_sign, cam_sign, phase, target_sign) in enumerate(crosses):
        stimulus_seed = int(cfg["evaluation_source_base"] + seed*1000 + i)
        rng = np.random.default_rng(stimulus_seed)
        # Fixed draw order is part of the stimulus definition and future lock.
        values = {key: float(rng.uniform(*ranges[key])) for key in (
            "preparation_duration", "preparation_amplitude", "cam_magnitude",
            "wave_amplitude", "wave_frequency", "velocity_amplitude")}
        rows.append(dict(episode=i, profile=i, id=f"round1-r{seed}-p{i}", replicate=seed,
                         stimulus_seed=stimulus_seed, preparation_sign=prep_sign,
                         cam_sign=cam_sign, phase=phase, target_sign=target_sign,
                         amplitude=cam_sign*values["cam_magnitude"], **values))
    if len(rows) != cfg["profiles_per_replicate"]["evaluation"]:
        raise ValueError("Declared case count differs from the full four-factor cross")
    return rows


def preparation(t, rows):
    """One C1 preload pulse followed by the remaining fixed-frame hold."""
    time = np.asarray(t, dtype=np.float64)
    scalar = time.ndim == 0
    if time.ndim > 1 or not np.isfinite(time).all():
        raise ValueError("Preparation time must be finite scalar or vector")
    time = np.atleast_1d(time)[None, :]
    duration = np.array([r["preparation_duration"] for r in rows])[:, None]
    amplitude = np.array([r["preparation_sign"]*r["preparation_amplitude"] for r in rows])[:, None]
    active = (time > 0) & (time < duration)
    phase = np.pi*time/duration
    delta = .02 + np.where(active, amplitude*np.sin(phase)**2, 0.)
    rate = np.where(active, amplitude*np.pi/duration*np.sin(2*phase), 0.)
    return (delta[:, 0], rate[:, 0]) if scalar else (delta, rate)


def commands(rows, config=None, extra_steps=None):
    """Known external compression and switching target; includes MPC lookahead.

    d is a sinusoid; its analytic rate begins immediately at movement onset.
    The derivative at the preparation-to-motion boundary is piecewise-defined.
    """
    cfg = load_config() if config is None else config
    dt = cfg["dt"]
    steps = round(cfg["motion_seconds"]/dt)
    extra = cfg["controller"]["horizon_steps"] if extra_steps is None else int(extra_steps)
    if extra < 0:
        raise ValueError("Nonnegative extra_steps required")
    t = np.arange(steps + extra)*dt
    phase = np.array([r["phase"] for r in rows])[:, None]
    angular = (2*np.pi*np.array([r["wave_frequency"] for r in rows]))[:, None]
    amplitude = np.array([r["wave_amplitude"] for r in rows])[:, None]
    d = .02 + amplitude*np.sin(angular*t[None, :] + phase)
    rate = amplitude*angular*np.cos(angular*t[None, :] + phase)
    switch_step = round(cfg["controller"]["switch_seconds"]/dt)
    target = (np.array([r["target_sign"] for r in rows])[:, None]
              * cfg["controller"]["target_magnitude"]
              * np.where(np.arange(len(t)) < switch_step, 1, -1))
    return d, rate, target


def velocity_profile(rows, config=None):
    """Two opposite C1 sin² pulses, sampled at all movement-step endpoints."""
    cfg = load_config() if config is None else config
    dt, duration = cfg["dt"], cfg["motion_seconds"]
    half = duration/2
    t = np.arange(round(duration/dt) + 1)*dt
    first = np.where((t > 0) & (t < half), np.sin(np.pi*t/half)**2, 0.)
    second = np.where((t > half) & (t < duration), np.sin(np.pi*(t-half)/half)**2, 0.)
    amplitude = np.array([r["target_sign"]*r["velocity_amplitude"] for r in rows])[:, None]
    return amplitude*(first-second)[None, :]
