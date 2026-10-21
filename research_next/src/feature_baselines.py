"""Causal polynomial material controls adapted from the existing diagnostics.

No reference-world, learner, simulator, or hardware imports. Force-labelled
feature_sequence is for isolated calibration/conditional diagnostics only.
The online step predicts normal force and feeds that prediction to friction.
"""
import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DT = .0025
VREG = .04
TAUS = {"old_flex": [.02, .08, .3],
        "path_ridge": [.005, .01, .02, .04, .08, .16, .32, .64, 1.28]}
LAMBDAS = {"old_flex": [1e-6, 1e-4, .01, 1., 100.],
           "path_ridge": [1e-8, 1e-6, 1e-4, .01, 1.]}


def vectors(*values):
    arrays = np.broadcast_arrays(*[np.atleast_1d(np.asarray(v, dtype=np.float64)) for v in values])
    if any(a.ndim != 1 or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Signals must broadcast to finite one-dimensional arrays")
    return arrays


def initial_state(kind, delta, velocity, normal_sensor):
    d, v, n = vectors(delta, velocity, normal_sensor)
    values = np.column_stack([d, v, n, np.abs(v)])
    return dict(history=np.repeat(values[:, None, :], len(TAUS[kind]), axis=1),
                initial_delta=d.copy(), delta_min=d.copy(), delta_max=d.copy(),
                travel_delta=np.zeros(len(d)), slip=np.zeros(len(d)), elapsed=np.zeros(len(d)))


def feature_step(kind, delta, rate, velocity, normal, state):
    """Current inputs plus pre-update history; this function does not mutate state."""
    d, r, v, n = vectors(delta, rate, velocity, normal)
    h = state["history"]
    if h.shape != (len(d), len(TAUS[kind]), 4):
        raise ValueError("State batch dimension/variant does not match signals")
    nf = np.column_stack([d/.02, r/.1, (d[:, None] - h[:, :, 0])/.02])
    ff = np.column_stack([v/.5, (v[:, None] - h[:, :, 1])/.5,
                          n/20, (n[:, None] - h[:, :, 2])/20])
    if kind == "path_ridge":
        nf = np.column_stack([nf, state["initial_delta"]/.02,
                              np.minimum(state["delta_min"], d)/.02,
                              np.maximum(state["delta_max"], d)/.02,
                              state["travel_delta"]/.02])
        ff = np.column_stack([ff, state["elapsed"]/2, state["slip"]/.1,
                              np.abs(v)/.5, h[:, :, 3]/.5])
    # Preserve the existing diagnostic's declared SI scales and float32 features.
    return nf.astype(np.float32), ff.astype(np.float32)


def advance_state(kind, delta, rate, velocity, normal, state, dt=DT):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Positive finite dt required")
    d, r, v, n = vectors(delta, rate, velocity, normal)
    h = state["history"]
    values = np.column_stack([np.maximum(d, 0), v, n, np.abs(v)])
    weight = (1 - np.exp(-dt/np.asarray(TAUS[kind])))[None, :, None]
    return dict(history=h + weight*(values[:, None, :] - h),
                initial_delta=state["initial_delta"].copy(),
                delta_min=np.minimum(state["delta_min"], d),
                delta_max=np.maximum(state["delta_max"], d),
                travel_delta=state["travel_delta"] + dt*np.abs(r),
                slip=state["slip"] + dt*np.abs(v), elapsed=state["elapsed"] + dt)


def feature_sequence(data, kind, dt=DT):
    """Measured-load calibration features, never the online coupled predictor."""
    d, r, v, n = data["delta"], data["rate"], data["state"][..., 1], data["normal"]
    state = initial_state(kind, d[:, 0], v[:, 0], n[:, 0])
    normal, friction = [], []
    for t in range(d.shape[1]):
        nf, ff = feature_step(kind, d[:, t], r[:, t], v[:, t], n[:, t], state)
        normal.append(nf)
        friction.append(ff)
        state = advance_state(kind, d[:, t], r[:, t], v[:, t], n[:, t], state, dt)
    return np.stack(normal, 1), np.stack(friction, 1)


@lru_cache(maxsize=None)
def monomial_indices(inputs, degree):
    return tuple(ix for order in range(1, degree + 1)
                 for ix in itertools.combinations_with_replacement(range(inputs), order))


@lru_cache(maxsize=None)
def monomial_layout(inputs, degree):
    return np.asarray([list(ix) + [inputs]*(degree-len(ix))
                       for ix in monomial_indices(inputs, degree)], dtype=np.int64)


def basis(values, head):
    x = np.asarray(values, dtype=np.float64)
    degree = 3 if head == "normal" else 2
    indices = monomial_indices(x.shape[1], degree)
    extra = x.shape[1] if head == "friction" else 0
    out = np.empty((len(x), 1 + len(indices) + extra), dtype=np.float64)
    out[:, 0] = 1
    padded = np.column_stack([x, np.ones(len(x))])
    out[:, 1:1+len(indices)] = np.prod(padded[:, monomial_layout(x.shape[1], degree)], axis=2)
    if extra:
        out[:, -extra:] = np.abs(x)
    return out


class FeatureBaseline:
    """Persistent external state, suitable for cloning into MPC candidates."""
    def __init__(self, kind, normal, friction, metadata=None, dt=DT):
        if kind not in TAUS:
            raise ValueError(kind)
        self.kind, self.dt = kind, dt
        self.normal, self.friction = normal, friction
        self.metadata = metadata or {}
        self._fitted = {head: {key: np.asarray(fit[key]) for key in ("mean", "scale", "coef")}
                        for head, fit in [("normal", normal), ("friction", friction)]}

    def initial(self, delta, v, N_sensor):
        return initial_state(self.kind, delta, v, N_sensor)

    def predict_head(self, values, head):
        fit = self._fitted[head]
        return ((basis(values, head) - fit["mean"])/fit["scale"]) @ fit["coef"]

    def _step(self, delta, rate, v, state, applied_normal=None, dt=None):
        d, r, velocity = vectors(delta, rate, v)
        nf, _ = feature_step(self.kind, d, r, velocity, np.zeros_like(d), state)
        normal = np.where(d > 0, np.maximum(20*self.predict_head(nf, "normal"), 0), 0)
        load = normal if applied_normal is None else vectors(applied_normal, d)[0]
        load = np.where(d > 0, np.maximum(load, 0), 0)
        _, ff = feature_step(self.kind, d, r, velocity, load, state)
        mu = np.maximum(self.predict_head(ff, "friction"), 0)
        friction = -load*mu*np.tanh(velocity/VREG)
        new = advance_state(self.kind, d, r, velocity, load, state, self.dt if dt is None else dt)
        return normal, friction, new

    def step(self, delta, rate, v, state, dt=None):
        """Predict N, couple it to Ft now, then advance once using predicted N."""
        return self._step(delta, rate, v, state, dt=dt)

    def conditional_step(self, delta, rate, v, normal_measured, state, dt=None):
        """Privileged diagnostic/calibration mode with an imposed measured load."""
        return self._step(delta, rate, v, state, applied_normal=normal_measured, dt=dt)

    def parameter_count(self):
        return len(self.normal["coef"]) + len(self.friction["coef"])

    def save(self, path):
        payload = dict(kind=self.kind, normal=self.normal, friction=self.friction,
                       metadata=self.metadata, dt=self.dt)
        with Path(path).open("x") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def fit_head(kind, head, x, target, vx, vtarget, factor=None, vfactor=None, block_size=4096):
    """Ridge fit using all rows, train-only scaling, complete held-out sequences.

    Force factors are multiplied into the design, so no division by low N/v
    or noisy force labels is needed. Ridge penalizes all except the intercept.
    """
    count = len(x)
    total, square = None, None
    for start in range(0, count, block_size):
        b = basis(x[start:start+block_size], head)
        if total is None:
            total, square = np.zeros(b.shape[1]), np.zeros(b.shape[1])
        total += b.sum(0)
        square += (b*b).sum(0)
    mean = total/count
    scale = np.sqrt(np.maximum(square/count - mean*mean, 0))
    mean[0], scale[0] = 0., 1.
    scale = np.where(scale < 1e-8, 1., scale)
    gram = np.zeros((len(mean), len(mean)))
    rhs = np.zeros(len(mean))
    target_scale = 20 if head == "normal" else 10
    for start in range(0, count, block_size):
        stop = start + block_size
        z = (basis(x[start:stop], head) - mean)/scale
        if head == "friction":
            z *= (factor[start:stop]/10)[:, None]
        gram += z.T@z/count
        rhs += z.T@(target[start:stop]/target_scale)/count
    reg = np.eye(len(mean))
    reg[0, 0] = 0
    candidates, coefficients = [], []
    for strength in LAMBDAS[kind]:
        coef = np.linalg.solve(gram + strength*reg + 1e-12*np.eye(len(mean)), rhs)
        coefficients.append(coef)
        error_sum = 0.
        for start in range(0, len(vx), block_size):
            stop = start + block_size
            raw = ((basis(vx[start:stop], head) - mean)/scale)@coef
            if head == "normal":
                pred = 20*np.maximum(raw, 0)*(vx[start:stop, 0] > 0)
            else:
                pred = vfactor[start:stop]*np.maximum(raw, 0)
            error_sum += np.sum((pred - vtarget[start:stop])**2)
        candidates.append(dict(lambda_=strength, val_rmse=float(np.sqrt(error_sum/len(vx)))))
    chosen = min(range(len(candidates)), key=lambda i: candidates[i]["val_rmse"])
    return dict(mean=mean.tolist(), scale=scale.tolist(), coef=coefficients[chosen].tolist(),
                selected_lambda=candidates[chosen]["lambda_"], candidates=candidates,
                selection="isolated conditional validation force RMSE; complete held-out trajectories",
                train_rows=count, validation_rows=len(vx))
