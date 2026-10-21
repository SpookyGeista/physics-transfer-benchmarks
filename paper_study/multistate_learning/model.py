"""Classical three-stop learner. No truth/reference/latent-label imports.

The declared family has six positive coefficients: three stiffnesses and three
force/normal-load capacities. State is three elastic deflections in meters.
The output is exact mean Ft for constant v/N over dt, including in-step yield.
"""
import json
from pathlib import Path

import numpy as np
from numba import njit


@njit(cache=True)
def _mean_deflection(z0, displacement, bound):
    z0 = min(max(z0, -bound), bound)
    if displacement == 0.:
        return z0, z0
    direction = 1. if displacement > 0. else -1.
    path = abs(displacement)
    elastic_path = min(path, max(bound - direction*z0, 0.))
    fraction = elastic_path/path
    mean = fraction*(z0+.5*direction*elastic_path) + (1.-fraction)*direction*bound
    end = min(max(z0+direction*elastic_path, -bound), bound)
    return mean, end


@njit(cache=True)
def basis(alpha, velocity, normal, dt):
    """Flat time-order design: Ft=sum(k_i * column_i), alpha_i=mu_i/k_i.

    Every episode starts from known relaxed preparation z=0. This function
    receives only measured velocities/loads and hypothesized coefficients.
    """
    n, steps = velocity.shape
    result = np.empty((n*steps, 3))
    for episode in range(n):
        state = np.zeros(3)
        for step in range(steps):
            dx = velocity[episode, step]*dt
            load = normal[episode, step]
            for element in range(3):
                mean, state[element] = _mean_deflection(state[element], dx, alpha[element]*load)
                result[episode*steps+step, element] = -mean
    return result


@njit(cache=True)
def _step(state, velocity, normal, dt, k, alpha):
    n = len(velocity)
    new = np.empty_like(state)
    force = np.zeros(n)
    for episode in range(n):
        for element in range(3):
            mean, new[episode, element] = _mean_deflection(
                state[episode, element], velocity[episode]*dt, alpha[element]*normal[episode])
            force[episode] -= k[element]*mean
    return force, new


class PIModel:
    """Pure online model with explicit clonable (n,3) state and fixed parameters."""
    def __init__(self, stiffness, friction_capacity, metadata=None):
        k, mu = np.asarray(stiffness, dtype=float), np.asarray(friction_capacity, dtype=float)
        if k.shape != (3,) or mu.shape != (3,) or not np.isfinite(k).all() or not np.isfinite(mu).all():
            raise ValueError("three finite stiffnesses and capacities required")
        if np.any(k <= 0) or np.any(mu <= 0):
            raise ValueError("coefficients must be positive")
        order = np.argsort(mu/k, kind='stable')
        self.k, self.mu = k[order].copy(), mu[order].copy()
        self.alpha = self.mu/self.k
        for a in (self.k, self.mu, self.alpha):
            a.setflags(write=False)
        self.metadata = metadata or {}

    def initial(self, n=1):
        """Known relaxed preparation. Unknown-history inference must be external."""
        if isinstance(n, bool) or int(n) != n or n < 1:
            raise ValueError("positive integer n required")
        return np.zeros((int(n), 3))

    def step(self, state, v, N, dt=.0025):
        """Return (mean_Ft, new_state), without mutating the supplied state."""
        state = np.asarray(state, dtype=float)
        if state.ndim != 2 or state.shape[1] != 3 or not np.isfinite(state).all():
            raise ValueError("state must be finite with shape (n,3)")
        v, N = [np.broadcast_to(np.asarray(a, dtype=float), (len(state),)).copy() for a in (v, N)]
        if not np.isfinite(v).all() or not np.isfinite(N).all() or np.any(N < 0):
            raise ValueError("finite v and nonnegative finite N required")
        if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("positive finite dt required")
        return _step(state, v, N, dt, self.k, self.alpha)

    def predict(self, v, N, dt=.0025):
        velocity, normal = np.asarray(v, dtype=float), np.asarray(N, dtype=float)
        if velocity.ndim != 2 or normal.shape != velocity.shape:
            raise ValueError("v and N must have identical (n,steps) shapes")
        if not np.isfinite(velocity).all() or not np.isfinite(normal).all() or np.any(normal < 0):
            raise ValueError("finite input arrays with nonnegative N required")
        if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("positive finite dt required")
        return (basis(self.alpha, velocity, normal, dt) @ self.k).reshape(velocity.shape)

    def save(self, path):
        data = dict(type='three_stop_PI', parameter_count=6,
                    stiffness_N_m=self.k.tolist(), friction_capacity=self.mu.tolist(),
                    yield_per_normal_m_N=self.alpha.tolist(), metadata=self.metadata)
        Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data['type'] != 'three_stop_PI':
            raise ValueError('wrong model type')
        return cls(data['stiffness_N_m'], data['friction_capacity'], data.get('metadata'))
