"""Truth-side reachable preparations for an unknown-history material study.

Only the unchanged Reference.force transition creates material histories.  This
module is for environment generation/evaluation, never for an observer import.
The observer receives PreparedReference.initial_observation(), not this object,
its preparation plan, or truth_snapshot().  All quantities use SI units.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "study_v2" / "src"))
from decision_study.reference import Reference

DT = 0.0025
DELTA0 = 0.020
DURATION = 4.0
STEPS = 1600
WORLDS = ("maxwell", "boucwen")
SEED_RANGES = {"dev": (31001, 39999), "val": (41001, 49999), "test": (51001, 59999)}
_ALIASES = {"development": "dev", "validation": "val"}
RECIPES = ("rest", "recent_slip", "rest_after_slip", "pause_and_reverse",
           "load_modulated_reverse", "slow_reverse")
_DESIGN_VERSION = "reachable-history-v1-2026-09-15"


def _positive_integer(value, name):
    if isinstance(value, bool) or not np.isscalar(value) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _identity(n, seed, split):
    n = _positive_integer(n, "n")
    seed = _positive_integer(seed, "seed")
    split = _ALIASES.get(split, split)
    if split not in SEED_RANGES:
        raise ValueError(f"split must be one of {tuple(SEED_RANGES)}")
    lo, hi = SEED_RANGES[split]
    if not lo <= seed <= hi:
        raise ValueError(f"{split} requires seed in [{lo}, {hi}], got {seed}")
    return n, seed, split


def _readonly(array):
    array = np.asarray(array)
    array.setflags(write=False)
    return array


def preparation_design():
    """A source-frozen, score-independent description; generates no profiles."""
    return dict(version=_DESIGN_VERSION, dt=DT, duration=DURATION, steps=STEPS,
                delta=DELTA0, seed_ranges=copy.deepcopy(SEED_RANGES),
                recipes=list(RECIPES), end_x=0.0, end_v=0.0,
                normal_scale_bounds=[0.70, 1.30], peak_speed_bounds=[0.04, 0.32],
                normal_preparation="fresh declared Reference initialization",
                friction_preparation="Reference.force under imposed v and N",
                latent_state_injection=False,
                test_generation_requires_external_study_lock=True)


@dataclass(frozen=True)
class PreparationPlan:
    """Truth-only commands including the final boundary sample at t=4 s.

    normal_scale is relative to the declared initial normal reference force.
    x is the discrete semi-implicit integral of prescribed velocity; x(0)=x(T)=0.
    Smoothness refers to the continuous sin/cos segment construction.  The
    existing material transition consumes the left-end command each DT.
    """
    time: np.ndarray
    velocity: np.ndarray
    normal_scale: np.ndarray
    x: np.ndarray
    rows: tuple

    def commands(self, initial_normal):
        """Return independent arrays for truth replay; not observer features."""
        n = len(self.rows)
        normal = np.broadcast_to(np.asarray(initial_normal, dtype=float), (n,))
        if not np.isfinite(normal).all() or np.any(normal <= 0):
            raise ValueError("initial normal must be finite and positive")
        return dict(time=self.time.copy(), x=self.x.copy(), v=self.velocity.copy(),
                    normal=self.normal_scale * normal[:, None],
                    delta=np.full(self.velocity.shape, DELTA0),
                    rate=np.zeros(self.velocity.shape))


def _pulse(ramp_steps, plateau_steps):
    """C1 unit pulse: sin² ramp up, plateau, cos² ramp down, both endpoints zero."""
    phase = np.arange(ramp_steps + 1, dtype=float) / ramp_steps
    up = np.sin(.5 * np.pi * phase) ** 2
    down = up[::-1].copy()
    up[0], up[-1], down[0], down[-1] = 0., 1., 1., 0.
    return np.concatenate((up, np.ones(plateau_steps), down[1:]))


def preparation_plan(n, seed, split):
    """Deterministic plans independent of world, fitted parameters and scores.

    Per-index RNG streams make a batch prefix independent of requested n.
    Recipes are balanced by rotating (seed - split_start + index) modulo six.
    Reserved test plans must be generated only after the enclosing study lock;
    that orchestration check belongs to the study runner, not this source module.
    """
    n, seed, split = _identity(n, seed, split)
    velocity = np.zeros((n, STEPS + 1))
    normal_scale = np.ones_like(velocity)
    rows = []
    for index in range(n):
        rng = np.random.default_rng(np.random.SeedSequence([9152026, seed, index]))
        recipe_index = (seed - SEED_RANGES[split][0] + index) % len(RECIPES)
        recipe = RECIPES[recipe_index]
        row = dict(version=_DESIGN_VERSION, split=split, seed=seed, index=index,
                   recipe=recipe, duration=DURATION, dt=DT)
        if recipe == "rest":
            row.update(sign=0, peak_speed=0., ramp_steps=0, plateau_steps=0,
                       gap_steps=0, start_step=0, end_step=0, post_rest_steps=STEPS,
                       load_amplitude=0.)
            rows.append(row)
            continue
        ramp_steps = int(rng.integers(20, 41))  # 50--100 ms each smooth ramp
        plateau_steps = int(rng.integers(24, 161))  # 60--400 ms
        peak = float(rng.uniform(.12, .32))
        gap_steps = 0
        post_rest_steps = int(rng.integers(0, 9))  # 0--20 ms, before observing
        load_amplitude = 0.
        if recipe == "rest_after_slip":
            post_rest_steps = int(rng.integers(80, 601))  # 0.2--1.5 s
        elif recipe == "pause_and_reverse":
            gap_steps = int(rng.integers(20, 121))  # 50--300 ms between signs
            post_rest_steps = int(rng.integers(0, 41))
        elif recipe == "load_modulated_reverse":
            load_amplitude = float(rng.uniform(-.30, .30))
        elif recipe == "slow_reverse":
            peak = float(rng.uniform(.04, .10))
            plateau_steps = int(rng.integers(80, 241))
            gap_steps = int(rng.integers(0, 41))
        shape = _pulse(ramp_steps, plateau_steps)
        pulse_steps = len(shape) - 1
        active_steps = 2 * pulse_steps + gap_steps
        start = STEPS - post_rest_steps - active_steps
        if start < 0:
            raise RuntimeError("predeclared preparation exceeds fixed duration")
        sign = int(rng.choice([-1, 1]))
        second = start + pulse_steps + gap_steps
        velocity[index, start:start+len(shape)] += sign * peak * shape
        velocity[index, second:second+len(shape)] -= sign * peak * shape
        phase = np.arange(active_steps + 1, dtype=float) / active_steps
        load_wave = np.sin(np.pi * phase) ** 2
        load_wave[0] = load_wave[-1] = 0.
        normal_scale[index, start:start+active_steps+1] += load_amplitude * load_wave
        row.update(sign=sign, peak_speed=peak, ramp_steps=ramp_steps,
                   plateau_steps=plateau_steps, gap_steps=gap_steps,
                   start_step=start, end_step=start+active_steps,
                   post_rest_steps=post_rest_steps, load_amplitude=load_amplitude)
        rows.append(row)
    x = np.zeros_like(velocity)
    x[:, 1:] = DT * np.cumsum(velocity[:, 1:], axis=1)
    # No endpoint repair: paired equal/opposite pulses produce cancellation.
    if np.max(np.abs(x[:, -1])) > 1e-14:
        raise RuntimeError("preparation failed zero-net-displacement check")
    return PreparationPlan(_readonly(np.arange(STEPS + 1) * DT),
                           _readonly(velocity), _readonly(normal_scale),
                           _readonly(x), tuple(rows))


class PreparedReference:
    """Truth-side composed material with preserved physical end histories."""
    def __init__(self, normal, friction, plan, initial_normal):
        self._normal = normal
        self._friction = friction
        self._preparation = plan
        self._initial_normal = np.asarray(initial_normal, dtype=float).copy()
        self.n = normal._n
        self.normal_world = normal.world
        self.friction_world = friction.world
        self.world = friction.world
        self.updates = 0

    def initial_observation(self):
        """Only public start measurements; no history, age, ID or elapsed prep."""
        return dict(x=self._preparation.x[:, -1].copy(), v=np.zeros(self.n),
                    delta=np.full(self.n, DELTA0), compression_rate=np.zeros(self.n),
                    normal=self._initial_normal.copy())

    def force(self, delta, rate, v, normal_override=None, friction_enabled=True, dt=DT):
        """Return pre-update N/Ft and advance each active material once.

        Optional imposed N has the original Reference semantics: it changes the
        returned applied load and friction update, not the normal history law.
        Ordinary coupled motion should leave normal_override=None.
        """
        normal, _ = self._normal.force(delta, rate, 0., normal_override=normal_override,
                                       friction_enabled=False, dt=dt)
        _, friction = self._friction.force(delta, rate, v, normal_override=normal,
                                           friction_enabled=friction_enabled, dt=dt)
        self.updates += 1
        return normal, friction

    def clone(self):
        # Immutable command arrays may be shared; physical histories must not be.
        result = PreparedReference(copy.deepcopy(self._normal), copy.deepcopy(self._friction),
                                   self._preparation, self._initial_normal)
        result.updates = self.updates
        return result

    def observe(self, delta=DELTA0, rate=0., v=0., normal_override=None,
                friction_enabled=True, dt=DT):
        """Read current force on a clone; no hidden extra history transition."""
        return self.clone().force(delta, rate, v, normal_override=normal_override,
                                  friction_enabled=friction_enabled, dt=dt)


def referencepair(world, n, seed, split, *, normal_world=None):
    """Create physically reached, hidden friction histories and common normal state.

    world selects friction; normal_world defaults to world and may be crossed.
    No private material-history field is written.  Normal is freshly prepared
    under its declared initialization convention, independently of friction.
    Each friction history starts at the original Reference initial state and is
    advanced through all 1600 imposed-load/velocity steps.  Command arrays and
    diagnostic states must remain on the environment/evaluator side.
    """
    normal_world = world if normal_world is None else normal_world
    if world not in WORLDS or normal_world not in WORLDS:
        raise ValueError(f"world and normal_world must belong to {WORLDS}")
    plan = preparation_plan(n, seed, split)
    normal = Reference(normal_world, len(plan.rows), DELTA0)
    friction = Reference(world, len(plan.rows), DELTA0)
    initial_normal, _ = copy.deepcopy(normal).force(DELTA0, 0., 0., friction_enabled=False)
    for step in range(STEPS):
        friction.force(DELTA0, 0., plan.velocity[:, step],
                       normal_override=initial_normal * plan.normal_scale[:, step], dt=DT)
    return PreparedReference(normal, friction, plan, initial_normal)


def truth_preparation(prepared):
    """Full audit replay inputs, explicitly not an observer-facing API."""
    return dict(design=preparation_design(),
                normal_world=prepared.normal_world, friction_world=prepared.friction_world,
                rows=copy.deepcopy(list(prepared._preparation.rows)),
                commands=prepared._preparation.commands(prepared._initial_normal))


def truth_snapshot(prepared):
    """Copy diagnostic histories; reference coordinates, never estimator labels.

    Maxwell friction is coefficient M, not StateID's normalized mix.  A separate
    normalized coordinate is exported only as a truth diagnostic.  No fitted
    parameter or model is imported, and no history is ever injected here.
    """
    friction = prepared._friction._friction_history.copy()
    out = dict(normal_world=prepared.normal_world, friction_world=prepared.friction_world,
               normal_history=prepared._normal._normal_history.copy(),
               friction_history=friction, assembly_updates=prepared.updates)
    if prepared.friction_world == "maxwell":
        p = prepared._friction._params
        out["friction_coordinate"] = "coefficient_M"
        out["normalized_mix"] = (friction - p.mu_c) / (p.mu_s - p.mu_c)
    else:
        out["friction_coordinate"] = "age_seconds"
    return out
