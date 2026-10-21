"""Full deterministic-model grid likelihood for independent x, v and N sensors.

Unlike the earlier conditional observer, normal force is predicted at every
microstep and its noisy measurement enters the likelihood. No measured N is
held as a noise-free dynamics input. This is an ordinary discrete Bayesian
state observer, not a proposed new learning algorithm. Identified constitutive
parameters and the fresh normal-state convention remain fixed conditioning
assumptions. Parameter/model uncertainty is not marginalized.
"""
import numpy as np
from observer import GridObserver


class JointGridObserver(GridObserver):
    def __init__(self, *args, numerical_N=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.numerical_N = self._positive(numerical_N, "numerical_N")
        self.effective_sigma_N = np.hypot(self.sigma_N, self.numerical_N)

    def advance(self, u, d, d_dot, amplitude, *, steps=1):
        if isinstance(steps, bool) or int(steps) != steps or steps < 1:
            raise ValueError("steps must be a positive integer")
        force, command, command_rate, amp = [self._repeat(a, name) for a, name in
            [(u, "u"), (d, "d"), (d_dot, "d_dot"), (amplitude, "amplitude")]]
        for _ in range(int(steps)):
            x, v = self.grid_xv.reshape(-1, 2).T
            z = np.tanh(x / self.cam_length)
            slope = amp * (1-z*z) / self.cam_length
            delta, rate = command + amp*z, command_rate + slope*v
            n, ft, state = self.model.step(delta, rate, v, self.state, dt=self.dt)
            vn = v + self.dt * (force-slope*n+ft) / self.mass
            self.grid_xv = np.column_stack((x+self.dt*vn, vn)).reshape(self.episodes, self.grid_size, 2)
            if not np.isfinite(self.grid_xv).all():
                raise FloatingPointError("nonfinite joint-grid mechanics")
            self._check_state(state)
            self.state = state
            self.step_index += 1
        return self

    def observe(self, x, v, N, *, d, d_dot, amplitude):
        if self.step_index <= self.last_observed_step:
            raise ValueError("observation requires a new time step")
        observed = np.column_stack((self._episode(x, "x"), self._episode(v, "v")))
        next_N = self._episode(N, "N")
        cx, cv = self.grid_xv.reshape(-1, 2).T
        amp = self._repeat(amplitude, "amplitude")
        z = np.tanh(cx / self.cam_length)
        slope = amp * (1-z*z) / self.cam_length
        delta = self._repeat(d, "d") + amp*z
        rate = self._repeat(d_dot, "d_dot") + slope*cv
        # Pure model.step returns a fresh state; discard it, never advance twice.
        predicted_N = self.model.step(delta, rate, cv, self.state, dt=self.dt)[0]
        residual = (observed[:, None, :] - self.grid_xv) / self.effective_sigma
        residual_N = (next_N[:, None] - predicted_N.reshape(self.episodes, self.grid_size)) / self.effective_sigma_N
        increment = -.5*(np.sum(residual*residual, axis=2) + residual_N*residual_N)
        if not np.isfinite(increment).all():
            raise FloatingPointError("nonfinite joint-grid likelihood")
        self.log_likelihood += increment
        if self.assimilate:
            log_weight = self.log_prior+self.log_likelihood
            log_weight -= np.max(log_weight, axis=1, keepdims=True)
            weight = np.exp(log_weight)
            self.weights = weight / weight.sum(axis=1, keepdims=True)
        self.last_observed_N = next_N
        self.last_observed_xv = observed
        self.last_observed_step = self.step_index
        self.observation_count += 1
        return self.weights.copy()

    def snapshot(self):
        return {**super().snapshot(), "conditional_N_hold": False,
                "effective_sigma_N": self.effective_sigma_N}
