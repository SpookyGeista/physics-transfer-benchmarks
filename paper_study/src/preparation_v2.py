"""Expand six unchanged fixture preparations into a crossed development batch.

This is truth-side generation code. Histories are reached using Reference.force
under imposed velocity/load in an ideal local material fixture; they are not
claimed reachable by the later force-limited native actuator. Normal preparation
is still fresh and known. No material-history field is assigned here.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

import preparation as original

RECIPES = original.RECIPES
TEST_LOCK_ROLE = "paper_study_native_v2_test_authorization"


def check_split_access(seed, split, test_lock=None):
    """Validate the seed partition before drawing a profile.

    The runner additionally verifies the authorization's config/source hashes.
    Direct wrapper calls for reserved seeds also require the explicit lock file.
    """
    _, seed, split = original._identity(1, seed, split)
    if split == "test":
        if test_lock is None:
            raise ValueError("Reserved test preparation requires an explicit test lock file")
        locked = json.loads(Path(test_lock).read_text())
        if (locked.get("role") != TEST_LOCK_ROLE or locked.get("status") != "locked"
                or locked.get("split") != "test"
                or seed not in locked.get("preparation_seeds", [])):
            raise ValueError("Test lock does not authorize this preparation seed")
    return seed, split


def factorial_design(seed, split, amplitudes, excitation_signs=(-1, 1), *, test_lock=None):
    """Return explicit IDs for six recipes x all amplitudes x both signs.

    Indices refer to the SAME six unchanged original profiles for this seed,
    including the original seed-dependent rotation of recipe names.
    """
    seed, split = check_split_access(seed, split, test_lock)
    amplitudes = np.asarray(amplitudes, dtype=float)
    signs = np.asarray(excitation_signs, dtype=float)
    if (amplitudes.ndim != 1 or not len(amplitudes) or not np.isfinite(amplitudes).all()
            or len(np.unique(amplitudes)) != len(amplitudes)):
        raise ValueError("amplitudes must be a finite, unique one-dimensional sequence")
    if signs.ndim != 1 or len(signs) != 2 or set(signs.tolist()) != {-1., 1.}:
        raise ValueError("Both excitation signs -1 and +1 must occur exactly once")
    base = original.preparation_plan(len(RECIPES), seed, split)
    rows = []
    for index, profile in enumerate(base.rows):
        for cam_index, amplitude in enumerate(amplitudes):
            for sign in signs:
                rows.append(dict(
                    episode_id=f"{split}-p{seed}-h{index}-c{cam_index}-s{int(sign):+d}",
                    preparation_seed=seed, preparation_index=index,
                    preparation_recipe=profile["recipe"], cam_index=cam_index,
                    amplitude=float(amplitude), excitation_sign=int(sign)))
    return rows


def indexed_plan(preparation_indices, seed, split, *, test_lock=None):
    """Copy selected commands; repeated indices share no writable array state."""
    seed, split = check_split_access(seed, split, test_lock)
    values = np.asarray(preparation_indices)
    if (values.ndim != 1 or not len(values) or values.dtype.kind not in "iu"
            or np.any(values < 0) or np.any(values >= len(RECIPES))):
        raise ValueError("preparation_indices must be nonempty integers in [0, 5]")
    indices = values.astype(int, copy=False)
    base = original.preparation_plan(len(RECIPES), seed, split)
    rows = []
    for episode, index in enumerate(indices):
        row = copy.deepcopy(base.rows[int(index)])
        row.update(preparation_index=int(index), batch_episode=episode)
        rows.append(row)
    return original.PreparationPlan(
        original._readonly(base.time.copy()),
        original._readonly(base.velocity[indices].copy()),
        original._readonly(base.normal_scale[indices].copy()),
        original._readonly(base.x[indices].copy()), tuple(rows))


def referencepair_indices(world, preparation_indices, seed, split, *, normal_world=None,
                          test_lock=None):
    """Reach every selected end history by replay, without latent injection.

    Repeated rows begin identically and then remain independent during coupled
    motion. Replaying the expanded command matrix is equivalent to cloning the
    six reached histories, without modifying private Reference state arrays.
    """
    normal_world = world if normal_world is None else normal_world
    if world not in original.WORLDS or normal_world not in original.WORLDS:
        raise ValueError(f"world and normal_world must belong to {original.WORLDS}")
    plan = indexed_plan(preparation_indices, seed, split, test_lock=test_lock)
    normal = original.Reference(normal_world, len(plan.rows), original.DELTA0)
    friction = original.Reference(world, len(plan.rows), original.DELTA0)
    initial_normal, _ = copy.deepcopy(normal).force(
        original.DELTA0, 0., 0., friction_enabled=False)
    for step in range(original.STEPS):
        friction.force(original.DELTA0, 0., plan.velocity[:, step],
                       normal_override=initial_normal * plan.normal_scale[:, step],
                       dt=original.DT)
    return original.PreparedReference(normal, friction, plan, initial_normal)


truth_preparation = original.truth_preparation
truth_snapshot = original.truth_snapshot
