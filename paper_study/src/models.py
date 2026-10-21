"""Fixed independently identified material heads; never imports a reference."""
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "research_next/src"))
from state_id import StateID, CompositeStateID


def model_paths(normal_world, friction_world, seed=1, noise="clean"):
    base = REPO / "research_next/results/round1/models/state_id"
    return tuple(base / f"{world}/r{seed}/{noise}/blind_selection/model.json"
                 for world in (normal_world, friction_world))


def load_model(normal_world, friction_world, seed=1, noise="clean"):
    normal_path, friction_path = model_paths(normal_world, friction_world, seed, noise)
    normal = StateID.load(normal_path).normal_model
    friction = StateID.load(friction_path).friction_model
    return CompositeStateID(normal, friction)


def initial_grid(model, grid, delta=.02):
    grid = np.asarray(grid)
    state = model.initial(np.full(len(grid), delta), np.zeros(len(grid)), np.zeros(len(grid)))
    key = "friction__mix" if model.friction_family == "maxwell" else "friction__age"
    state[key] = grid.copy()
    return state


def motion_step(model, state_xv, material_state, u, d, d_dot, amplitude, dt=.0025,
                normal_override=None):
    """Prediction model: pre-update contact forces and semi-implicit mass motion."""
    x, v = np.asarray(state_xv).T
    z = np.tanh(x / .04)
    slope = amplitude * (1 - z * z) / .04
    delta = d + amplitude * z
    rate = d_dot + slope * v
    normal, friction, next_material = model.step(delta, rate, v, material_state,
                                                normal_override=normal_override, dt=dt)
    next_v = v + dt * (u - slope * normal + friction) / 2.
    return np.column_stack((x + dt * next_v, next_v)), next_material, normal, friction
