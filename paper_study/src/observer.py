"""Causal, fixed-parameter grid inference from position/velocity observations.

The unknown is the initial friction state, not a constitutive coefficient. Each
grid hypothesis keeps its own mechanical trajectory and complete material state.
An observation contributes ONE independent x/v measurement likelihood; neither
acceleration differences nor overlapping impulse residuals enter the objective.

Measured past N is a zero-order-held exogenous input until the next observation.
Its noise and hold error are NOT marginalized. The resulting likelihood is a
conditional approximation, not a calibrated posterior-coverage guarantee. N is
never a likelihood term or a supplied future forecast trajectory. Known initial
x/v and the normal model's declared fresh initialization are conditioning priors.
No truth/reference/preparation module is imported or accepted by this interface.
"""
from __future__ import annotations

import numpy as np


def initial_state_grid(friction_family, size=401):
    """Declared, fixed discrete prior support; no data-dependent endpoints.

    Uniform probability on a linear grid means uniform age (seconds), not a
    log-age prior. The Maxwell mix is its fitted law's normalized coordinate,
    not the reference law's friction coefficient.
    """
    if isinstance(size, bool) or int(size) != size or size < 2:
        raise ValueError("grid size must be an integer >= 2")
    if friction_family == "maxwell":
        return np.linspace(0., 1., int(size))
    if friction_family == "boucwen":
        return np.linspace(.02, 8., int(size))
    raise ValueError("friction family must be maxwell or boucwen")


def _copy_state(state):
    return {key: np.array(value, dtype=float, copy=True) for key, value in state.items()}


class GridObserver:
    """Vectorized grid observer, with batch order episode-major then grid.

    Public inputs use scalars or shape (episodes,); initial_xv is (episodes, 2)
    or (2,). Mechanical hypotheses are (episodes, grid, 2), model-state arrays
    have leading dimension episodes*grid, and weights are (episodes, grid).

    ``initial_d`` denotes LOCAL compression at initialization, whereas ``d`` in
    advance is the external command: delta=d+amplitude*tanh(x/.04). The initial
    normal history is exactly the injected frozen model's fresh convention.
    No unknown prehistory is accepted. The known initial sample has no likelihood
    contribution; observe is called after at least one dynamics step.

    sigma_N is recorded for provenance, not silently treated as a second force
    likelihood. sigma_x/v are combined in quadrature with positive, explicitly
    declared numerical floors, also for the ideal-sensor control. Extra process
    or model-error variance is not inferred or tuned here.
    """

    def __init__(self, model, grid, initial_xv, initial_d, initial_N, *,
                 dt=.0025, mass=2., cam_length=.04,
                 sigma_x=5e-5, sigma_v=.002, sigma_N=.15,
                 numerical_x=1e-6, numerical_v=1e-5,
                 assimilate=True, prior=None, state_key=None):
        self.model = model
        self.dt = self._positive(dt, "dt")
        self.mass = self._positive(mass, "mass")
        self.cam_length = self._positive(cam_length, "cam_length")
        self.numerical_x = self._positive(numerical_x, "numerical_x")
        self.numerical_v = self._positive(numerical_v, "numerical_v")
        self.sigma_x = self._nonnegative(sigma_x, "sigma_x")
        self.sigma_v = self._nonnegative(sigma_v, "sigma_v")
        self.sigma_N = self._nonnegative(sigma_N, "sigma_N")
        self.effective_sigma = np.hypot([self.sigma_x, self.sigma_v],
                                        [self.numerical_x, self.numerical_v])
        self.assimilate = bool(assimilate)
        self.grid = np.array(grid, dtype=float, copy=True)
        if (self.grid.ndim != 1 or self.grid.size == 0 or
                not np.isfinite(self.grid).all() or np.any(np.diff(self.grid) <= 0)):
            raise ValueError("grid must be a finite, strictly increasing 1D array")
        xv = np.asarray(initial_xv, dtype=float)
        if xv.shape == (2,):
            xv = xv[None, :]
        if xv.ndim != 2 or xv.shape[1] != 2 or not np.isfinite(xv).all():
            raise ValueError("initial_xv must be finite with shape (episodes, 2)")
        self.episodes, self.grid_size = len(xv), len(self.grid)
        if not self.episodes:
            raise ValueError("at least one episode required")
        self.grid_xv = np.repeat(xv[:, None, :], self.grid_size, axis=1)
        self.last_observed_N = self._episode(initial_N, "initial_N")
        initial_delta = self._repeat(initial_d, "initial_d")
        self.state = _copy_state(model.initial(initial_delta,
                                  self.grid_xv[..., 1].ravel(),
                                  np.repeat(self.last_observed_N, self.grid_size)))
        self._check_state(self.state)
        if state_key is None:
            candidates = [key for key in ("friction__mix", "friction__age")
                          if key in self.state]
            if len(candidates) != 1:
                raise ValueError("model must expose one friction__mix/age state")
            state_key = candidates[0]
        if state_key not in self.state or self.state[state_key].shape != (self.episodes*self.grid_size,):
            raise ValueError("initial state key must name a scalar friction state per hypothesis")
        self.state_key = state_key
        if state_key.endswith("age") and np.any(self.grid <= 0):
            raise ValueError("age grid must be strictly positive")
        if state_key.endswith("mix") and np.any((self.grid < 0) | (self.grid > 1)):
            raise ValueError("mix grid must be in [0, 1]")
        self.state[state_key] = np.tile(self.grid, self.episodes)
        if prior is None:
            prior = np.ones(self.grid_size)
        try:
            prior = np.broadcast_to(np.asarray(prior, dtype=float),
                                    (self.episodes, self.grid_size)).copy()
        except ValueError as exc:
            raise ValueError("prior must broadcast to (episodes, grid)") from exc
        if not np.isfinite(prior).all() or np.any(prior < 0) or np.any(prior.sum(axis=1) <= 0):
            raise ValueError("prior must have finite nonnegative positive-sum rows")
        self.prior = prior / prior.sum(axis=1, keepdims=True)
        self.weights = self.prior.copy()
        self.log_prior = np.full_like(self.prior, -np.inf)
        np.log(self.prior, out=self.log_prior, where=self.prior > 0)
        self.log_likelihood = np.zeros_like(self.prior)
        self.step_index = 0
        self.last_observed_step = 0
        self.observation_count = 0
        self.last_observed_xv = xv.copy()

    @staticmethod
    def _positive(value, name):
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
        return value

    @staticmethod
    def _nonnegative(value, name):
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be nonnegative and finite")
        return value

    def _episode(self, value, name):
        try:
            result = np.broadcast_to(np.asarray(value, dtype=float), (self.episodes,)).copy()
        except ValueError as exc:
            raise ValueError(f"{name} must be scalar or shape (episodes,)") from exc
        if not np.isfinite(result).all():
            raise ValueError(f"{name} must be finite")
        return result

    def _repeat(self, value, name):
        return np.repeat(self._episode(value, name), self.grid_size)

    def _check_state(self, state):
        for key, value in state.items():
            a = np.asarray(value)
            if a.ndim == 0 or a.shape[0] != self.episodes*self.grid_size or not np.isfinite(a).all():
                raise FloatingPointError(f"invalid material state array {key}")

    @property
    def time(self):
        return self.step_index * self.dt

    def advance(self, u, d, d_dot, amplitude, *, steps=1):
        """Propagate before the next observation using PREVIOUS measured N.

        Inputs are held constant for ``steps``; for varying known d/d_dot call
        once per microstep with steps=1. Forces use the pre-transition material
        state and mechanics, then material and semi-implicit mechanics advance
        once. Nonfinite trajectories raise rather than silently pruning a grid.
        """
        if isinstance(steps, bool) or int(steps) != steps or steps < 1:
            raise ValueError("steps must be a positive integer")
        force = self._repeat(u, "u")
        command = self._repeat(d, "d")
        command_rate = self._repeat(d_dot, "d_dot")
        amp = self._repeat(amplitude, "amplitude")
        held_N = np.repeat(self.last_observed_N, self.grid_size)
        for _ in range(int(steps)):
            x, v = self.grid_xv.reshape(-1, 2).T
            z = np.tanh(x / self.cam_length)
            slope = amp * (1-z*z) / self.cam_length
            delta = command + amp*z
            rate = command_rate + slope*v
            n, ft, next_state = self.model.step(delta, rate, v, self.state,
                                               normal_override=held_N, dt=self.dt)
            n, ft = np.asarray(n), np.asarray(ft)
            expected_shape = (self.episodes*self.grid_size,)
            if n.shape != expected_shape or ft.shape != expected_shape:
                raise ValueError("model outputs must have shape (episodes*grid,)")
            vn = v + self.dt * (force-slope*n+ft) / self.mass
            xn = x + self.dt*vn
            next_xv = np.column_stack((xn, vn)).reshape(self.episodes, self.grid_size, 2)
            if not np.isfinite(next_xv).all():
                raise FloatingPointError("nonfinite candidate mechanics; no hypotheses silently removed")
            self._check_state(next_state)
            self.grid_xv = next_xv
            self.state = next_state
            self.step_index += 1
        return self

    def observe(self, x, v, N):
        """Assimilate current x/v exactly once; hold the new N only afterward.

        The candidate trajectory is not reset to noisy observations. Conditional
        on all observed inputs, independent sensor noise permits this direct
        sequence likelihood. Model/N-hold errors need separate predictive checks.
        ``assimilate=False`` receives identical inputs and propagation but keeps
        the declared prior weights; it does not read hidden state or true forces.
        """
        if self.step_index <= self.last_observed_step:
            raise ValueError("observation requires a new time step; duplicate/initial sample rejected")
        observed = np.column_stack((self._episode(x, "x"), self._episode(v, "v")))
        next_N = self._episode(N, "N")
        residual = (observed[:, None, :] - self.grid_xv) / self.effective_sigma
        increment = -.5*np.sum(residual*residual, axis=2)
        if not np.isfinite(increment).all():
            raise FloatingPointError("nonfinite x/v likelihood")
        self.log_likelihood += increment
        if self.assimilate:
            log_weight = self.log_prior + self.log_likelihood
            log_weight -= np.max(log_weight, axis=1, keepdims=True)
            weight = np.exp(log_weight)
            self.weights = weight / weight.sum(axis=1, keepdims=True)
        self.last_observed_N = next_N
        self.last_observed_xv = observed
        self.last_observed_step = self.step_index
        self.observation_count += 1
        return self.weights.copy()

    def snapshot(self):
        """Independent arrays; old snapshots cannot change when filtering resumes."""
        return dict(weights=self.weights.copy(), prior=self.prior.copy(), grid=self.grid.copy(),
                    grid_xv=self.grid_xv.copy(), state=_copy_state(self.state),
                    log_likelihood=self.log_likelihood.copy(), step=self.step_index,
                    time=self.time, observation_count=self.observation_count,
                    last_observed_N=self.last_observed_N.copy(),
                    last_observed_xv=self.last_observed_xv.copy(),
                    state_key=self.state_key, effective_sigma=self.effective_sigma.copy(),
                    conditional_N_hold=True, sigma_N=self.sigma_N)

    def forecast_start(self, observed_xv=None):
        """Clone current material posterior; condition forecast mechanics equally.

        The caller applies the standard composed motion_step with NO future
        normal_override. No future measurement parameter exists here. Resetting
        every candidate to the same current measured x/v is a declared forecast
        condition and does not change either filtering trajectories or weights.
        Returns flat xv (episodes*grid,2), flat material state and weights(E,G).
        """
        if observed_xv is None:
            if self.last_observed_step != self.step_index:
                raise ValueError("a current observation or explicit current observed_xv is required")
            observed_xv = self.last_observed_xv
        observed_xv = np.asarray(observed_xv, dtype=float)
        if observed_xv.shape == (2,) and self.episodes == 1:
            observed_xv = observed_xv[None, :]
        if observed_xv.shape != (self.episodes, 2) or not np.isfinite(observed_xv).all():
            raise ValueError("observed_xv must be finite with shape (episodes,2)")
        return dict(xv=np.repeat(observed_xv, self.grid_size, axis=0),
                    state=_copy_state(self.state), weights=self.weights.copy(),
                    grid=self.grid.copy(), step=self.step_index, time=self.time)
