"""Synthetic material targets, independently implemented from the predictors.

Only data generation and evaluation may import this module.  No learned model,
feature dictionary, optimizer, simulator, or robot interface is imported here.
All inputs use SI units; positive delta denotes compression.  Each ``force``
call returns forces at the current input and PRE-UPDATE history, then advances
that history by dt exactly once.  It is not a side-effect-free force query.

These are fixture-scale adaptations of established constitutive families, not
validated material cards.  Equations, parameters, initialization and verified
primary sources are recorded in study_v2/reports/reference_notes.md.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _MaxwellParameters:
    equilibrium_stiffness: float = 630.0  # N/m
    branch_stiffness: tuple = (210.0, 125.0, 80.0)  # N/m
    relaxation_time: tuple = (0.025, 0.17, 0.80)  # s
    normal_damping: float = 2.2  # N s/m
    mu_c: float = 0.29
    mu_s: float = 0.53
    stribeck_velocity: float = 0.11  # m/s
    stribeck_exponent: float = 1.4
    instantaneous_fraction: float = 0.65
    friction_lag: float = 0.050  # s
    friction_lag_per_load: float = 0.001  # s/N
    load_offset: float = 3.0  # N
    load_reference: float = 18.0  # N
    load_exponent: float = -0.055
    viscous_friction: float = 0.35  # N s/m
    velocity_regularization: float = 0.040  # m/s; declared to all predictors


@dataclass(frozen=True)
class _BoucWenParameters:
    equilibrium_stiffness: float = 680.0  # N/m
    hysteresis_force: float = 4.5  # N
    yield_displacement: float = 0.009  # m
    normal_damping: float = 1.8  # N s/m
    beta: float = 0.65
    gamma: float = 0.35
    exponent: int = 2
    open_relaxation_time: float = 0.12  # s; explicit contact adaptation
    mu_reference: float = 0.40
    rate_coefficient: float = 0.018
    state_coefficient: float = 0.026
    reference_velocity: float = 0.12  # m/s
    characteristic_slip: float = 0.008  # m; coarse fixture scale
    log_velocity_floor: float = 0.002  # m/s
    velocity_regularization: float = 0.040  # m/s; declared to all predictors
    viscous_friction: float = 0.40  # N s/m
    open_state_relaxation_time: float = 0.20  # s


class Reference:
    """Vectorized independent histories for ``n`` fixture replicas.

    ``initial_delta`` is a scalar or shape-(n,) array. Maxwell initialization is
    fully relaxed at that compression. Bouc-Wen initialization follows monotonic
    loading from zero, z=tanh(delta/yield_displacement); this is an explicit
    preparation convention, not an inference from the test trajectory.

    normal_override applies an ideal measured clamp load for calibration; it
    does not alter normal material history and cannot create contact at delta<=0.
    friction_enabled may be a boolean or a shape-(n,) boolean array.
    """

    def __init__(self, world, n, initial_delta):
        if world not in ("maxwell", "boucwen"):
            raise ValueError(f"Unknown reference world: {world!r}")
        if isinstance(n, bool) or int(n) != n or n <= 0:
            raise ValueError("n must be a positive integer")
        self._world = world
        self._n = int(n)
        initial = np.maximum(self._array(initial_delta, "initial_delta"), 0.0)
        if world == "maxwell":
            self._params = _MaxwellParameters()
            self._normal_history = np.repeat(initial[:, None], 3, axis=1)
            self._friction_history = np.full(self._n, self._params.mu_s)
        else:
            self._params = _BoucWenParameters()
            self._normal_history = np.tanh(initial / self._params.yield_displacement)
            self._friction_history = np.full(
                self._n,
                self._params.characteristic_slip / self._params.reference_velocity,
            )

    @property
    def world(self):
        return self._world

    def _array(self, value, name):
        a = np.asarray(value, dtype=np.float64)
        try:
            a = np.broadcast_to(a, (self._n,))
        except ValueError as exc:
            raise ValueError(f"{name} must be scalar or shape ({self._n},)") from exc
        if not np.isfinite(a).all():
            raise ValueError(f"{name} must be finite")
        return a

    def force(
        self, delta, rate, v, normal_override=None, friction_enabled=True, dt=0.0025
    ):
        """Return (normal, tangential) in N and advance history once.

        ``rate`` is compression rate in m/s, ``v`` is signed sliding velocity in
        m/s. Forces act on the moving fixture; Ft*v is never positive. This
        algebraic sign property does not establish total-system passivity.
        """
        if not np.isscalar(dt) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive scalar")
        d_in = self._array(delta, "delta")
        r_in = self._array(rate, "rate")
        velocity = self._array(v, "v")
        override = (
            None if normal_override is None
            else self._array(normal_override, "normal_override")
        )
        enabled = np.broadcast_to(np.asarray(friction_enabled, dtype=bool), (self._n,))
        contact = d_in > 0.0
        compression = np.maximum(d_in, 0.0)
        compression_rate = np.where(contact, r_in, 0.0)
        p = self._params

        if self.world == "maxwell":
            normal_raw = (
                p.equilibrium_stiffness * compression
                + np.sum(
                    np.asarray(p.branch_stiffness)
                    * (compression[:, None] - self._normal_history), axis=1
                )
                + p.normal_damping * compression_rate
            )
        else:
            normal_raw = (
                p.equilibrium_stiffness * compression
                + p.hysteresis_force * self._normal_history
                + p.normal_damping * compression_rate
            )
        normal = np.where(contact, np.maximum(normal_raw if override is None else override, 0.0), 0.0)
        loaded = contact & (normal > 0.0)

        if self.world == "maxwell":
            mu_target = p.mu_c + (p.mu_s - p.mu_c) * np.exp(
                -(np.abs(velocity) / p.stribeck_velocity) ** p.stribeck_exponent
            )
            load_factor = ((normal + p.load_offset) / p.load_reference) ** p.load_exponent
            mu = (
                p.instantaneous_fraction * mu_target
                + (1.0 - p.instantaneous_fraction) * self._friction_history
            ) * load_factor
        else:
            regularized_speed = np.hypot(velocity, p.log_velocity_floor)
            theta_reference = p.characteristic_slip / p.reference_velocity
            mu = np.maximum(
                p.mu_reference
                + p.rate_coefficient * np.log(regularized_speed / p.reference_velocity)
                + p.state_coefficient * np.log(self._friction_history / theta_reference),
                0.0,
            )
        tangential = np.where(
            loaded & enabled,
            -normal * mu * np.tanh(velocity / p.velocity_regularization)
            - p.viscous_friction * velocity,
            0.0,
        )

        # All outputs above use the same pre-update histories. Neither force
        # component updates the other component's history during evaluation.
        if self.world == "maxwell":
            weights = -np.expm1(-dt / np.asarray(p.relaxation_time))
            self._normal_history += weights * (compression[:, None] - self._normal_history)
            lag = p.friction_lag + p.friction_lag_per_load * normal
            target = np.where(loaded, mu_target, p.mu_s)
            self._friction_history += -np.expm1(-dt / lag) * (target - self._friction_history)
        else:
            self._advance_boucwen(compression_rate, contact, dt)
            speed = np.abs(velocity)
            inv_time = speed / p.characteristic_slip
            decay = np.exp(-dt * inv_time)
            # Exact aging-law integration for constant slip speed, including v=0.
            growth = np.divide(
                -np.expm1(-dt * inv_time), inv_time,
                out=np.full(self._n, dt), where=inv_time > 0.0,
            )
            theta_loaded = self._friction_history * decay + growth
            theta_open = theta_reference + (
                self._friction_history - theta_reference
            ) * np.exp(-dt / p.open_state_relaxation_time)
            self._friction_history[:] = np.where(loaded, theta_loaded, theta_open)
        return normal, tangential

    def _advance_boucwen(self, compression_rate, contact, dt):
        p = self._params
        normalized_rate = compression_rate / p.yield_displacement

        def derivative(z):
            return (
                normalized_rate
                - p.beta * np.abs(normalized_rate) * np.abs(z) ** (p.exponent - 1) * z
                - p.gamma * normalized_rate * np.abs(z) ** p.exponent
            )

        # RK4 with bounded normalized displacement per substep. This is an
        # integration choice, not an extra history update visible to the caller.
        substeps = max(1, int(np.ceil(np.max(np.abs(normalized_rate)) * dt / 0.05)))
        h = dt / substeps
        z = self._normal_history.copy()
        for _ in range(substeps):
            k1 = derivative(z)
            k2 = derivative(z + 0.5 * h * k1)
            k3 = derivative(z + 0.5 * h * k2)
            k4 = derivative(z + h * k3)
            z += h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        self._normal_history[:] = np.where(
            contact, z, self._normal_history * np.exp(-dt / p.open_relaxation_time)
        )
